import logging
import numpy as np
from scipy.sparse import coo_matrix
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.datasets import InferenceEmbeddingDataset
from data.dataloaders import (InferenceEmbeddingDataLoader,
                              TripletEmbeddingDataLoader)
from utils.comm import get_rank, all_gather, synchronize
from utils.misc import flatten, unique

from IPython import embed


logger = logging.getLogger(__name__)


class EmbeddingSubTrainer(object):
    """
    Class to help with training and evaluation processes.
    """
    def __init__(self, args, model, optimizer, scheduler):
        # we assume that `model` is a DDP pytorch model and is on the GPU
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.training_method == 'triplet':
            self.train_on_subset = self._train_triplet
        else:
            raise ValueError('training method not implemented yet')

    def get_embeddings(self, dataloader, evaluate=True):
        args = self.args
        self.model.eval()

        local_step = 0
        push_to_cpu_steps = 32
        idxs_list = []
        embeds_list = []
        master_idxs_list = []
        master_embeds_list = []

        def _synchronize_lists(_embeds_list, _idxs_list):
            gathered_data = all_gather({
                    'embeds_list': _embeds_list,
                    'idxs_list': _idxs_list,
                })
            if get_rank() == 0:
                _embeds_list = [d['embeds_list'] for d in gathered_data]
                _embeds_list = flatten(_embeds_list)
                _embeds_list = [x.cpu() for x in _embeds_list]
                _idxs_list = [d['idxs_list'] for d in gathered_data]
                _idxs_list = flatten(_idxs_list)
                _idxs_list = [x.cpu() for x in _idxs_list]
                master_embeds_list.extend(_embeds_list)
                master_idxs_list.extend(_idxs_list)
            synchronize()
            return [], []
            
        batch_iterator = tqdm(dataloader,
                              desc='Getting embeddings...',
                              disable=(not evaluate 
                                       or get_rank() != 0
                                       or args.disable_logging))
        for batch in batch_iterator:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids_a':      batch[1],
                          'attention_mask_a': batch[2],
                          'token_type_ids_a': batch[3]}
                embeds_list.append(self.model(**inputs))
                idxs_list.append(batch[0])
                local_step += 1
                if local_step % push_to_cpu_steps == 0:
                    embeds_list, idxs_list = _synchronize_lists(
                            embeds_list, idxs_list)

        embeds_list, idxs_list = _synchronize_lists(
                embeds_list, idxs_list)

        idxs, embeds = None, None
        if get_rank() == 0:
            idxs = torch.cat(master_idxs_list, dim=0).numpy()
            idxs, indices = np.unique(idxs, return_index=True)
            embeds = torch.cat(master_embeds_list, dim=0).numpy()
            embeds = embeds[indices]
        synchronize()
        return idxs, embeds

    def compute_scores_for_inference(self, clusters_mx, per_example_negs):
        # TODO: add description here
        args = self.args

        # get all of the unique examples 
        examples = clusters_mx.data.tolist()
        examples.extend(flatten(per_example_negs.tolist()))
        examples = unique(examples)

        # create dataset and dataloader
        dataset = InferenceEmbeddingDataset(
                args, examples, args.train_cache_dir)
        dataloader = InferenceEmbeddingDataLoader(args, dataset)

        # get the unique idxs and embeds for each idx
        idxs, embeds = self.get_embeddings(dataloader, evaluate=False)

        sparse_graph = None
        if get_rank() == 0:
            # create inverse index for mapping
            inverse_idxs = {v : k for k, v in enumerate(idxs)}

            ## make the list of pairs of dot products we need
            _row = clusters_mx.row
            # positives:
            local_pos_a, local_pos_b = np.where(
                    np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
            )
            a = clusters_mx.data[local_pos_a]
            b = clusters_mx.data[local_pos_b]
            # negatives:
            local_neg_a = np.tile(
                np.arange(per_example_negs.shape[0])[:, np.newaxis],
                (1, per_example_negs.shape[1])
            ).flatten()
            neg_a = clusters_mx.data[local_neg_a]
            neg_b = per_example_negs.flatten()

            # create subset of the sparse graph we care about
            a = np.concatenate((a, neg_a), axis=0)
            b = np.concatenate((b, neg_b), axis=0)
            edges = list(zip(a, b))
            affinities = [np.dot(embeds[inverse_idxs[i]],
                                 embeds[inverse_idxs[j]]) for i, j in edges]
            # convert to coo_matrix
            edges = np.asarray(edges).T
            affinities = np.asarray(affinities)
            _sparse_num = np.max(edges) + 1
            sparse_graph = coo_matrix((affinities, edges),
                                      shape=(_sparse_num, _sparse_num))

        synchronize()
        return sparse_graph

    def _train_triplet(self, dataset_list):
        args = self.args

        losses = [] 
        time_per_dataset = []
        dataset_sizes = []

        self.model.train()
        self.model.zero_grad()
        for dataset in dataset_list:
            _dataset_start_time = time.time()
            dataset_sizes.append(len(dataset))
            dataloader = TripletEmbeddingDataLoader(args, dataset)
            for batch in dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids_a':      batch[1],
                          'attention_mask_a': batch[2],
                          'token_type_ids_a': batch[3]}
                outputs = self.model(**inputs)

                pos_neg_dot_prods = torch.sum(
                        outputs[:,0:1,:] * outputs[:,1:,:],
                        dim=-1
                )
                loss = torch.mean(
                    F.relu(
                        pos_neg_dot_prods[:, 1]   # negative dot products
                        - pos_neg_dot_prods[:, 0] # positive dot products
                        + args.margin
                    )
                )
                losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        args.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
            time_per_dataset.append(time.time() - _dataset_start_time)

        total_time = sum(time_per_dataset)
        total_num_examples = 3 * sum(dataset_sizes) # because triplets
        return {'loss' : np.mean(losses),
                'time_per_example': total_time / total_num_examples,
                'num_examples': total_num_examples}

    def _train_sigmoid(self, dataset_list):
        pass
    
    def _train_softmax(self, dataset_list):
        pass

    def _train_accum_max_margin(self, dataset_list):
        pass

    def build_sparse_affinity_graph(self,
                                    midxs, 
                                    metadata,
                                    knn_index,
                                    build_coref_graph=False,
                                    build_linking_graph=False):
        # FIXME: this should probably go in `src/evaluation.py`
        assert build_coref_graph or build_linking_graph
        args = self.args

        coref_graph = None
        linking_graph = None
        if get_rank() == 0:
            # hack to get all of the embeddings
            idxs, embeds = knn_index.idxs, knn_index.X

            # create inverse index for mapping
            inverse_idxs = {v : k for k, v in enumerate(idxs)}

            mention_knn = None
            if build_coref_graph or args.available_entities == 'knn_candidates':
                # get all of the mention kNN
                mention_knn = knn_index.get_knn_limited_index(
                        midxs, include_index_idxs=midxs, k=args.k+1
                )
                mention_knn = mention_knn[:,1:]

            if build_coref_graph:
                # list of edges for sparse graph we will build
                coref_graph_edges = []

                # add mention-mention edges to 
                coref_graph_edges.extend(
                    [tuple(sorted((a, b)))
                        for a, l in zip(midxs, mention_knn) for b in l]
                )
                coref_graph_edges = unique(coref_graph_edges)
                affinities = [np.dot(embeds[inverse_idxs[i]],
                                     embeds[inverse_idxs[j]]) 
                                for i, j in coref_graph_edges]
                coref_graph_edges = np.asarray(coref_graph_edges).T
                _sparse_num = metadata.num_mentions + metadata.num_entities
                coref_graph = coo_matrix(
                            (affinities, coref_graph_edges),
                            shape=(_sparse_num, _sparse_num)
                )

            if build_linking_graph:
                # list of edges for sparse graph we will build
                linking_graph_edges = []

                # get mention-entity edges
                if args.available_entities == 'candidates_only':
                    for midx in midxs:
                        candidates = metadata.midx2cand.get(midx, [])
                        if len(candidates) > 0:
                            linking_graph_edges.extend(
                                [tuple(sorted((midx, eidx))) for eidx in candidates]
                            )
                elif args.available_entities == 'knn_candidates':
                    for i, midx in enumerate(midxs):
                        candidates = metadata.midx2cand.get(midx, [])
                        candidates.extend(
                                flatten([metadata.midx2cand.get(m, []) 
                                                for m in mention_knn[i]])
                        )
                        if len(candidates) > 0:
                            linking_graph_edges.extend(
                                [tuple(sorted((midx, eidx))) for eidx in candidates]
                            )
                else: # 'open_domain'
                    # use `knn_index` to do this efficiently
                    raise NotImplementedError('open domain not yet')

                # build the graph
                linking_graph_edges = unique(linking_graph_edges)
                affinities = [np.dot(embeds[inverse_idxs[i]],
                                     embeds[inverse_idxs[j]]) 
                                for i, j in linking_graph_edges]
                linking_graph_edges = np.asarray(linking_graph_edges).T
                _sparse_num = metadata.num_mentions + metadata.num_entities
                linking_graph = coo_matrix((affinities, linking_graph_edges),
                                              shape=(_sparse_num, _sparse_num))

        return coref_graph, linking_graph

    def save_model(self, global_step):
        self.model.module.save_model(suffix='checkpoint-{}'.format(global_step))
