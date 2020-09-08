from collections import defaultdict
import logging
from functools import reduce
import numpy as np
import os
import pickle
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
import random

from clustering import (TripletDatasetBuilder,
                        SoftmaxDatasetBuilder,
                        ThresholdDatasetBuilder)
from data.datasets import MetaClusterDataset, InferenceEmbeddingDataset
from data.dataloaders import (MetaClusterDataLoader,
                              InferenceEmbeddingDataLoader)
from evaluation.evaluation import eval_wdoc, eval_xdoc
from model import MirrorEmbeddingModel, VersatileModel
from trainer.trainer import Trainer
from trainer.emb_sub_trainer import EmbeddingSubTrainer
from trainer.concat_sub_trainer import ConcatenationSubTrainer
from utils.comm import (all_gather,
                        broadcast,
                        get_world_size,
                        get_rank,
                        synchronize)
from utils.knn_index import WithinDocNNIndex, CrossDocNNIndex
from utils.misc import flatten, unique, dict_merge_with

if get_rank() == 0:
    import wandb

from IPython import embed


logger = logging.getLogger(__name__)


class ClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(ClusterLinkingTrainer, self).__init__(args)

        if hasattr(self, 'train_metadata'):
            args.num_entities = self.train_metadata.num_entities
        elif hasattr(self, 'val_metadata'):
            args.num_entities = self.val_metadata.num_entities
        elif hasattr(self, 'test_metadata'):
            args.num_entities = self.test_metadata.num_entities
        else:
            raise AttributeError('Must have a dataset metadata loaded and available')

        # create sub-trainers for models
        self.create_sub_trainers()

        # create knn_index and supervised clustering dataset builder
        if args.do_train or args.do_train_eval:
            #self.create_knn_index('train')
            if args.do_train:
                if 'triplet' in args.training_method:
                    self.dataset_builder = TripletDatasetBuilder(args)
                elif args.training_method == 'softmax':
                    self.dataset_builder = SoftmaxDatasetBuilder(args)
                elif args.training_method == 'threshold':
                    self.dataset_builder = ThresholdDatasetBuilder(args)
                else:
                    raise ValueError('unsupported training_method')
        if args.do_val: # or args.evaluate_during_training:
            pass
            #self.create_knn_index('val')
        if args.do_test:
            pass
            #self.create_knn_index('test')
    
    def create_models(self):
        logger.info('Creating models.')
        args = self.args
        self.models = {
                'affinity_model': VersatileModel(
                    args,
                    name='affinity_model'
                )
        }
        args.tokenizer = self.models['affinity_model'].tokenizer

    def create_sub_trainers(self):
        logger.info('Creating sub-trainers.')
        args = self.args
        for name, model in self.models.items():
            optimizer = self.optimizers[name] if args.do_train else None
            scheduler = self.schedulers[name] if args.do_train else None
            if isinstance(model.module, VersatileModel):
                self.embed_sub_trainer = EmbeddingSubTrainer(
                        args,
                        model,
                        optimizer,
                        scheduler
                )
                self.concat_sub_trainer = ConcatenationSubTrainer(
                        args,
                        model,
                        optimizer,
                        scheduler
                )
            else:
                raise ValueError('module not supported by a sub trainer')
    
    def create_train_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=False)

        # determine the set of gold clusters depending on the setting
        if args.clustering_domain == 'within_doc':
            clusters = flatten([list(doc.values()) 
                    for doc in self.train_metadata.wdoc_clusters.values()])
        elif args.clustering_domain == 'cross_doc':
            clusters = list(self.train_metadata.xdoc_clusters.values())
        else:
            raise ValueError('Invalid clustering_domain')

        self.train_dataset = MetaClusterDataset(clusters)
        self.train_dataloader = MetaClusterDataLoader(
                args, self.train_dataset)

    def create_train_eval_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=True)

        if args.available_entities in ['candidates_only', 'knn_candidates']:
            examples = flatten([[k] + v
                    for k, v in self.train_metadata.midx2cand.items()])
            examples.extend(self.train_metadata.midx2eidx.values())
        elif args.available_entities == 'open_domain':
            examples = list(self.train_metadata.idx2uid.keys())
        else:
            raise ValueError('Invalid available_entities')
        examples = unique(examples)

        self.train_eval_dataset = InferenceEmbeddingDataset(
                args, examples, args.train_cache_dir)
        self.train_eval_dataloader = InferenceEmbeddingDataLoader(
                args, self.train_eval_dataset)

    def create_val_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='val', evaluate=True)

        if args.available_entities in ['candidates_only', 'knn_candidates']:
            examples = flatten([[k] + v
                    for k, v in self.val_metadata.midx2cand.items()])
        elif args.available_entities == 'open_domain':
            examples = list(self.val_metadata.idx2uid.keys())
        else:
            raise ValueError('Invalid available_entities')
        examples = unique(examples)
        self.val_dataset = InferenceEmbeddingDataset(
                args, examples, args.val_cache_dir)
        self.val_dataloader = InferenceEmbeddingDataLoader(
                args, self.val_dataset)

    def create_test_dataloader(self):
        pass

    def create_knn_index(self, split=None):
        assert split == 'train' or split == 'val' or split == 'test'
        args = self.args

        NN_Index = (WithinDocNNIndex if args.clustering_domain == 'within_doc'
                        else CrossDocNNIndex)
        if split == 'train':
            self.train_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.train_eval_dataloader,
                    name='train'
            )
        elif split == 'val':
            self.val_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.val_dataloader,
                    name='val'
            )
        else:
            self.test_knn_index = NN_Index(
                    args,
                    self.embed_sub_trainer,
                    self.test_dataloader,
                    name='test'
            )

    def _build_temp_sparse_graph(self, clusters_mx, per_example_negs):
        args = self.args

        # get all of the unique examples 
        examples = clusters_mx.data.tolist()
        examples.extend(flatten(per_example_negs.tolist()))
        examples = unique(examples)
        examples = list(filter(lambda x : x >= 0, examples))

        sparse_graph = None
        if get_rank() == 0:
            ## make the list of pairs of dot products we need
            _row = clusters_mx.row
            # positives:
            local_pos_a, local_pos_b = np.where(
                    np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
            )
            pos_a = clusters_mx.data[local_pos_a]
            pos_b = clusters_mx.data[local_pos_b]
            # negatives:
            local_neg_a = np.tile(
                np.arange(per_example_negs.shape[0])[:, np.newaxis],
                (1, per_example_negs.shape[1])
            ).flatten()
            neg_a = clusters_mx.data[local_neg_a]
            neg_b = per_example_negs.flatten()

            neg_mask = (neg_b != -1)
            neg_a = neg_a[neg_mask]
            neg_b = neg_b[neg_mask]

            # create subset of the sparse graph we care about
            a = np.concatenate((pos_a, neg_a), axis=0)
            b = np.concatenate((pos_b, neg_b), axis=0)
            edges = list(zip(a, b))
            affinities = [0.0 for i, j in edges]

            # convert to coo_matrix
            edges = np.asarray(edges).T
            affinities = np.asarray(affinities)
            _sparse_num = np.max(edges) + 1
            sparse_graph = coo_matrix((affinities, edges),
                                      shape=(_sparse_num, _sparse_num))

        synchronize()
        return sparse_graph

    def train_step(self, batch):
        args = self.args

        # get the batch of clusters and approx negs for each individual example
        clusters_mx, per_example_negs = batch

        # compute scores using up-to-date model
        #sparse_graph = self.embed_sub_trainer.compute_scores_for_inference(
        #        clusters_mx, per_example_negs)
        #sparse_graph = self._build_temp_sparse_graph(
        #        clusters_mx, per_example_negs)

        # TODO: produce sparse graph w/ concat model in inference mode
        sparse_graph = self.concat_sub_trainer.compute_scores_for_inference(
                clusters_mx, per_example_negs)

        # create custom datasets for training
        embed_dataset_list = None
        concat_dataset_list = None
        dataset_metrics = None
        if get_rank() == 0:
            dataset_lists, dataset_metrics = self.dataset_builder(
                    clusters_mx, sparse_graph, self.train_metadata
            )
            embed_dataset_list, concat_dataset_list = dataset_lists
        dataset_metrics = broadcast(dataset_metrics, src=0)
        embed_dataset_list = broadcast(embed_dataset_list, src=0)
        concat_dataset_list = broadcast(concat_dataset_list, src=0)

        # take care of empty dataset list (should only happen when only considering m-m edges)
        if embed_dataset_list == None or concat_dataset_list == None:
            return {}

        ## train on datasets
        #embed_return_dict = self.embed_sub_trainer.train_on_subset(
        #        embed_dataset_list, self.train_metadata
        #)

        concat_return_dict = self.concat_sub_trainer.train_on_subset(
                concat_dataset_list, self.train_metadata
        )

        #embed_return_dict = broadcast(embed_return_dict, src=0)
        concat_return_dict = broadcast(concat_return_dict, src=0)

        return_dict = {}
        return_dict.update(dataset_metrics)
        #return_dict.update(embed_return_dict)
        return_dict.update(concat_return_dict)

        #if get_rank() == 0:
        #    embed()
        #synchronize()
        #exit()

        return return_dict

    def _neg_choosing_prep(self):
        args = self.args
        metadata = self.train_metadata

        # be able to map from midx to doc and back
        self.doc2midxs = defaultdict(list)
        self.midx2doc = {}
        for doc_id, wdoc_clusters in metadata.wdoc_clusters.items():
            mentions = flatten(wdoc_clusters.values())
            mentions = [x for x in mentions if x >= metadata.num_entities]
            self.doc2midxs[doc_id] = mentions
            for midx in mentions:
                self.midx2doc[midx] = doc_id

        # need to know the available entities in this case as well
        if args.available_entities not in ['candidates_only', 'knn_candidates']:
            self.avail_entity_idxs = list(range(num_entities))

    def _choose_negs(self, batch):
        args = self.args
        negatives_list = []
        clusters = [sorted(batch.getrow(i).data.tolist())
                        for i in range(batch.shape[0])]
        for c_idxs in clusters:
            if args.clustering_domain == 'within_doc':
                # get mention idxs within document
                doc_midxs = self.doc2midxs[self.midx2doc[c_idxs[1]]]

                # produce available negative mention idxs
                neg_midxs = [m for m in doc_midxs if m not in c_idxs]

                # determine the number of mention negatives
                num_m_negs = min(args.k // 2, len(neg_midxs))

                # initialize the negs tensors
                negs = np.tile(-1, (len(c_idxs), args.k))

                if args.mention_negatives == 'context_overlap':
                    # use overlapping context negative mentions
                    neg_midxs_objects = [
                            (midx, self.train_metadata.mentions[self.train_metadata.idx2uid[midx]])
                                for midx in neg_midxs
                    ]
                    for i, idx in enumerate(c_idxs):
                        if idx >= self.train_metadata.num_entities:
                            idx_object = self.train_metadata.mentions[self.train_metadata.idx2uid[idx]]
                            neg_context_dists = [(x[0], abs(idx_object['start_index'] - x[1]['start_index']))
                                                    for x in neg_midxs_objects]
                            neg_context_dists.sort(key=lambda x : x[1])
                            local_neg_midxs, _ = zip(*neg_context_dists)
                            negs[i,:num_m_negs] = np.asarray(local_neg_midxs[:num_m_negs])
                elif args.mention_negatives == 'random':
                    for i, idx in enumerate(c_idxs):
                        if idx >= self.train_metadata.num_entities:
                            negs[i,:num_m_negs] = random.sample(neg_midxs, num_m_negs)
                else:
                    # sample mention negatives according to embedding model
                    negs[:,:num_m_negs] = self.train_knn_index.get_knn_limited_index(
                        c_idxs,
                        include_index_idxs=neg_midxs,
                        k=num_m_negs
                    )

                # produce available negative entity idxs
                # NOTE: this doesn't allow negative e-e edges (there are never any positive ones)
                num_e_negs = args.k - num_m_negs
                if args.available_entities == 'candidates_only':
                    neg_eidxs = [
                        list(filter(
                            lambda x : x != c_idxs[0],
                            self.train_metadata.midx2cand.get(i, [])
                        ))[:num_e_negs]
                            for i in c_idxs
                                if i >= self.train_metadata.num_entities
                    ]
                    neg_eidxs = [
                        l + [-1] * (num_e_negs - len(l))
                            for l in neg_eidxs
                    ]
                    negs[1:, -num_e_negs:] = np.asarray(neg_eidxs)
                else:
                    if (args.clustering_domain == 'within_doc'
                            and args.available_entities == 'knn_candidates'):
                        # custom w/in doc negative available entities
                        self.avail_entity_idxs = flatten([
                            list(filter(
                                lambda x : x != c_idxs[0],
                                self.train_metadata.midx2cand.get(i, [])
                            ))
                                for i in (c_idxs + neg_midxs)
                                    if i >= self.train_metadata.num_entities
                        ])

                    _entity_knn_negs = self.train_knn_index.get_knn_limited_index(
                            c_idxs[1:],
                            include_index_idxs=self.avail_entity_idxs,
                            k=min(num_e_negs, len(self.avail_entity_idxs))
                    )

                    if _entity_knn_negs.shape[1] < num_e_negs:
                        assert _entity_knn_negs.shape[1] == len(self.avail_entity_idxs)
                        _buff = _entity_knn_negs.shape[1] - num_e_negs
                        negs[1:, -num_e_negs:_buff] = _entity_knn_negs
                    else:
                        negs[1:, -num_e_negs:] = _entity_knn_negs

                negatives_list.append(negs)

            else:
                raise NotImplementedError('xdoc neg sampling not implemented yet')
                # produce knn negatives for cluster and append to list

        negs = np.concatenate(negatives_list, axis=0)
        return negs

    def train(self):
        args = self.args

        # set up data structures for choosing available negatives on-the-fly
        if args.clustering_domain == 'within_doc':
            self._neg_choosing_prep()
        else:
            raise NotImplementedError('xdoc not implemented yet')


        global_step = 0
        log_return_dicts = []

        logger.info('Starting training...')

        batch = None
        for epoch in range(args.num_train_epochs):
            logger.info('********** [START] epoch: {} **********'.format(epoch))
            
            num_batches = None
            if get_rank() == 0:
                data_iterator = iter(self.train_dataloader)
                num_batches = len(data_iterator)
            num_batches = broadcast(num_batches, src=0)

            logger.info('num_batches: {}'.format(num_batches))

            for _ in trange(num_batches,
                            desc='Epoch: {} - Batches'.format(epoch),
                            disable=(get_rank() != 0 or args.disable_logging)):

                ### FIXME: hack for hyperparameter scheduling
                #if global_step > 400:
                #    args.training_edges_considered = 'all'
                #if global_step % 200 == 199:
                #    if get_rank() == 0:
                #        self.embed_sub_trainer.save_model(global_step)
                #    synchronize()
                #    val_metrics = self.evaluate(
                #            split='val',
                #            suffix='checkpoint-{}'.format(global_step)
                #    )
                #    if get_rank() == 0:
                #        wandb.log(val_metrics, step=global_step)
                #    synchronize()
                #    exit()


                # get batch from rank0 and broadcast it to the other processes
                if get_rank() == 0:
                    try:
                        next_batch = next(data_iterator)
                        # make sure the cluster_mx is sorted correctly
                        _row, _col, _data = [], [], []
                        current_row = 0
                        ctr = 0
                        for r, d in sorted(zip(next_batch.row, next_batch.data)):
                            if current_row != r:
                                current_row = r
                                ctr = 0
                            _row.append(r)
                            _col.append(ctr)
                            _data.append(d)
                            ctr += 1
                        next_batch = coo_matrix((_data, (_row, _col)), shape=next_batch.shape)
                        negs = self._choose_negs(next_batch)
                        batch = (next_batch, negs)
                    except StopIteration:
                        batch = None

                batch = broadcast(batch, src=0)
                
                if batch is None:
                    break

                # run train_step
                log_return_dicts.append(self.train_step(batch))
                global_step += 1

                # logging stuff for babysitting
                if global_step % args.logging_steps == 0:
                    avg_return_dict = reduce(dict_merge_with, log_return_dicts)
                    for stat_name, stat_value in avg_return_dict.items():
                        logger.info('Average %s: %s at global step: %s',
                                stat_name,
                                str(stat_value/args.logging_steps),
                                str(global_step)
                        )
                        if get_rank() == 0:
                            wandb.log({stat_name : stat_value/args.logging_steps}, step=global_step)
                    logger.info('Using {} edges for training'.format(args.training_edges_considered))
                    log_return_dicts = []

                # refresh the knn index 
                if args.knn_refresh_steps > 0 and global_step % args.knn_refresh_steps == 0:
                    logger.info('Refreshing kNN index...')
                    self.train_knn_index.refresh_index()
                    logger.info('Done.')

            # save the model at the end of every epoch
            if get_rank() == 0:
                #self.embed_sub_trainer.save_model(global_step)
                self.concat_sub_trainer.save_model(global_step)
            synchronize()

            logger.info('********** [END] epoch: {} **********'.format(epoch))

            # run full evaluation at the end of each epoch
            #if args.evaluate_during_training and epoch % 10 == 9:
            if args.evaluate_during_training:
                if args.do_train_eval:
                    train_eval_metrics = self.evaluate(
                            split='train',
                            suffix='checkpoint-{}'.format(global_step)
                    )
                    if get_rank() == 0:
                        wandb.log(train_eval_metrics, step=global_step)
                if args.do_val:
                    val_metrics = self.evaluate(
                            split='val',
                            suffix='checkpoint-{}'.format(global_step)
                    )
                    if get_rank() == 0:
                        wandb.log(val_metrics, step=global_step)

        logger.info('Training complete')
        if get_rank() == 0:
            embed()
        synchronize()
        exit()

    def evaluate(self, split='', suffix=''):
        assert split in ['train', 'val', 'test']
        args = self.args

        logger.info('********** [START] eval: {} **********'.format(split))

        if split == 'train':
            metadata = self.train_metadata
            #knn_index = self.train_knn_index
            example_dir = args.train_cache_dir
        elif split == 'val':
            metadata = self.val_metadata
            #knn_index = self.val_knn_index
            example_dir = args.val_cache_dir
        else:
            metadata = self.test_metadata
            #knn_index = self.test_knn_index
            example_dir = args.test_cache_dir

        ## refresh the knn index
        #knn_index.refresh_index()

        ## save the knn index
        #if get_rank() == 0:
        #    knn_save_fname = os.path.join(args.output_dir, suffix,
        #                                  'knn_index.' + split + '.debug_results.pkl')
        #    with open(knn_save_fname, 'wb') as f:
        #        pickle.dump((knn_index.idxs, knn_index.X), f, pickle.HIGHEST_PROTOCOL)

        embed_metrics = None
        concat_metrics = None
        if args.clustering_domain == 'within_doc':
            #embed_metrics = eval_wdoc(
            #    args, example_dir, metadata, knn_index, self.embed_sub_trainer,
            #    save_fname=os.path.join(args.output_dir, suffix,
            #                            'embed.' + split + '.debug_results.pkl')
            #)
            concat_metrics = eval_wdoc(
                args, example_dir, metadata, self.concat_sub_trainer,
                save_fname=os.path.join(args.output_dir, suffix,
                                        'concat.' + split + '.debug_results.pkl')
            )
            #concat_metrics = {}
        else:
            # FIXME: update after we finalize wdoc changes
            embed_metrics = eval_xdoc(
                args, example_dir, metadata, knn_index, self.embed_sub_trainer
            )
            concat_metrics = eval_xdoc(
                args, example_dir, metadata, knn_index, self.concat_sub_trainer
            )

        embed_metrics = broadcast(embed_metrics, src=0)
        concat_metrics = broadcast(concat_metrics, src=0)

        # pool all of the metrics into one dictionary
        #embed_metrics = {'embed_' + k : v for k, v in embed_metrics.items()}
        concat_metrics = {'concat_' + k : v for k, v in concat_metrics.items()}
        metrics = {}
        #metrics.update(embed_metrics)
        metrics.update(concat_metrics)
        metrics = {split + '_' + k : v for k, v in metrics.items()}

        logger.info(metrics)
        logger.info('********** [END] eval: {} **********'.format(split))

        if get_rank() == 0:
            embed()
        synchronize()
        exit()
        return metrics
