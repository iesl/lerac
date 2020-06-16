from collections import defaultdict
import logging
from functools import reduce
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange

from clustering import (TripletDatasetBuilder,
                        SigmoidDatasetBuilder,
                        SoftmaxDatasetBuilder,
                        AccumMaxMarginDatasetBuilder)
from data.datasets import MetaClusterDataset, InferenceEmbeddingDataset
from data.dataloaders import (MetaClusterDataLoader,
                              InferenceEmbeddingDataLoader)
from evaluation import eval_wdoc, eval_xdoc
from model import MirrorEmbeddingModel
from trainer.trainer import Trainer
from trainer.emb_sub_trainer import EmbeddingSubTrainer
from utils.comm import (all_gather,
                        broadcast,
                        get_world_size,
                        get_rank,
                        synchronize)
from utils.knn_index import WithinDocNNIndex, CrossDocNNIndex
from utils.misc import flatten, unique, dict_merge_with

from IPython import embed


logger = logging.getLogger(__name__)


class ClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(ClusterLinkingTrainer, self).__init__(args)

        # create sub-trainers for models
        self.create_sub_trainers()

        # create knn_index and supervised clustering dataset builder
        if args.do_train or args.do_train_eval:
            self.create_knn_index('train')
            if args.do_train:
                assert (args.pair_gen_method == 'all_pairs'
                        or args.training_method == 'accum_max_margin')
                if args.training_method == 'triplet':
                    self.dataset_builder = TripletDatasetBuilder(args)
                elif args.training_method == 'sigmoid':
                    self.dataset_builder = SigmoidDatasetBuilder(args)
                elif args.training_method == 'softmax':
                    self.dataset_builder = SoftmaxDatasetBuilder(args)
                else:
                    self.dataset_builder = AccumMaxMarginDatasetBuilder(args)
        if args.do_val: # or args.evaluate_during_training:
            self.create_knn_index('val')
        if args.do_test:
            self.create_knn_index('test')
    
    def create_models(self):
        logger.info('Creating models.')
        args = self.args
        self.models = {
                'embedding_model': MirrorEmbeddingModel(
                    args,
                    name='embedding_model'
                )
        }
        args.tokenizer = self.models['embedding_model'].tokenizer

    def create_sub_trainers(self):
        logger.info('Creating sub-trainers.')
        args = self.args
        self.sub_trainers = {}
        for name, model in self.models.items():
            optimizer = self.optimizers[name] if args.do_train else None
            scheduler = self.schedulers[name] if args.do_train else None
            if isinstance(model.module, MirrorEmbeddingModel):
                self.sub_trainers[name] = EmbeddingSubTrainer(
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
                    self.sub_trainers['embedding_model'],
                    self.train_eval_dataloader
            )
        elif split == 'val':
            self.val_knn_index = NN_Index(
                    args,
                    self.sub_trainers['embedding_model'],
                    self.val_dataloader
            )
        else:
            self.test_knn_index = NN_Index(
                    args,
                    self.sub_trainers['embedding_model'],
                    self.test_dataloader
            )

    def train_step(self, batch):
        args = self.args

        # get the batch of clusters and approx negs for each individual example
        clusters_mx, per_example_negs = batch

        # compute scores using up-to-date model
        sub_trainer = self.sub_trainers['embedding_model']
        sparse_graph = sub_trainer.compute_scores_for_inference(
                clusters_mx, per_example_negs)

        # create custom datasets for training
        dataset_list = None
        if get_rank() == 0:
            dataset_list = self.dataset_builder(clusters_mx, sparse_graph)
        dataset_list = broadcast(dataset_list, src=0)

        # train on datasets
        return_dict = sub_trainer.train_on_subset(dataset_list)

        return return_dict

    def _neg_choosing_prep(self):
        args = self.args
        metadata = self.train_metadata

        # be able to map from midx to doc and back
        self.doc2midxs = defaultdict(list)
        self.midx2doc = {}
        for doc_id, wdoc_clusters in metadata.wdoc_clusters.items():
            mentions = flatten(wdoc_clusters.values())
            self.doc2midxs[doc_id] = mentions
            for midx in mentions:
                self.midx2doc[midx] = doc_id

        # need to know the available entities in this case as well
        if args.available_entities not in ['candidates_only', 'knn_candidates']:
            self.avail_entity_idxs = list(range(num_entities))

    def _choose_negs(self, batch):
        args = self.args
        negatives_list = []
        clusters = [batch.getrow(i).data.tolist()
                        for i in range(batch.shape[0])]
        for c_idxs in clusters:
            if args.clustering_domain == 'within_doc':
                # get mention idxs within document
                doc_midxs = self.doc2midxs[self.midx2doc[c_idxs[0]]]

                # produce available negative idxs
                neg_midxs = [m for m in doc_midxs if m not in c_idxs]
                if args.available_entities in ['candidates_only', 'knn_candidates']:
                    neg_eidxs = flatten([self.train_metadata.midx2cand.get(m, [])
                                    for m in doc_midxs])
                    neg_eidxs = [x for x in neg_eidxs if x not in c_idxs]
                    avail_neg_idxs = neg_eidxs + neg_midxs
                else:
                    avail_neg_idxs = self.avail_entity_idxs + neg_midxs

                # produce knn negatives for cluster and append to list
                negatives_list.append(
                    self.train_knn_index.get_knn_limited_index(
                            c_idxs, include_index_idxs=avail_neg_idxs
                    )
                )
            else:
                # produce knn negatives for cluster and append to list
                negatives_list.append(
                    self.train_knn_index.get_knn_limited_index(
                            c_idxs, exclude_index_idxs=c_idxs
                    )
                )
        negs = np.concatenate(negatives_list, axis=0)
        return negs

    def train(self):
        args = self.args

        # set up data structures for choosing available negatives on-the-fly
        if args.clustering_domain == 'within_doc':
            self._neg_choosing_prep()

        global_step = 0
        log_return_dicts = []

        # FIXME: this only does one epoch as of now
        logger.info('Starting training...')

        for epoch in range(args.num_train_epochs):
            logger.info('********** [START] epoch: {} **********'.format(epoch))
            
            num_batches = None
            if get_rank() == 0:
                data_iterator = iter(self.train_dataloader)
                num_batches = len(data_iterator)
            num_batches = broadcast(num_batches, src=0)

            batch = None
            for _ in trange(num_batches,
                            desc='Epoch: {} - Batches'.format(epoch),
                            disable=(get_rank() != 0 or args.disable_logging)):
                # get batch from rank0 and broadcast it to the other processes
                if get_rank() == 0:
                    try:
                        next_batch = next(data_iterator)
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
                    log_return_dicts = []

                # refresh the knn index 
                if global_step % args.knn_refresh_steps == 0:
                    logger.info('Refreshing kNN index...')
                    self.train_knn_index.refresh_index()
                    logger.info('Done.')

                # save the model
                if global_step % args.save_steps == 0:
                    if get_rank() == 0:
                        for st in self.sub_trainers.values():
                            st.save_model(global_step)
                synchronize()

            logger.info('********** [END] epoch: {} **********'.format(epoch))

            # run full evaluation at the end of each epoch
            if args.evaluate_during_training:
                # TODO: add full evaluation
                if get_rank() == 0:
                    embed()
                synchronize()
                exit()

    def evaluate(self, split=''):
        assert split in ['train', 'val', 'test']
        args = self.args

        logger.info('********** [START] eval: {} **********'.format(split))

        sub_trainer = self.sub_trainers['embedding_model']
        if split == 'train':
            metadata = self.train_metadata
            knn_index = self.train_knn_index
        elif split == 'val':
            metadata = self.val_metadata
            knn_index = self.val_knn_index
        else:
            metadata = self.test_metadata
            knn_index = self.test_knn_index

        if args.clustering_domain == 'within_doc':
            eval_wdoc(args, metadata, knn_index, sub_trainer)
        else:
            eval_xdoc(args, metadata, knn_index, sub_trainer)

        logger.info('********** [END] eval: {} **********'.format(split))
