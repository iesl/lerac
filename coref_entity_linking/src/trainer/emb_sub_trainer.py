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
                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'concat_input': False}
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
                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'concat_input': False}
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
        return {'embed_loss' : np.mean(losses),
                'embed_time_per_example': total_time / total_num_examples,
                'embed_num_examples': total_num_examples}

    def _train_sigmoid(self, dataset_list):
        pass
    
    def _train_softmax(self, dataset_list):
        pass

    def _train_accum_max_margin(self, dataset_list):
        pass


    def save_model(self, global_step):
        self.model.module.save_model(suffix='checkpoint-{}'.format(global_step))
