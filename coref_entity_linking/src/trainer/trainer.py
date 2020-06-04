import os
import sys
import time
import logging
import copy
import pickle
import random
from abc import ABC, abstractmethod
from tqdm import tqdm, trange
from collections import defaultdict
from itertools import combinations, count, product
from functools import reduce
import numpy as np
import faiss
import scipy
from scipy.special import expit
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, normalized_mutual_info_score, adjusted_rand_score
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset


from transformers import AdamW, get_linear_schedule_with_warmup

from model import (MirrorStackEmbeddingModel,
                   MirrorEmbeddingModel,
                   ConcatenationModel,
                   CrossEncoder,
                   PolyEncoder,
                   ScalarAffineModel)
from data.preprocessing import ZeshelPreprocessor
from data.data import (MentionClusteringProcessor,
                  LinkingProcessor,
                  XDocClusterLinkingProcessor,
                  WrappedTensorDataset,
                  LazyDataset,
                  LazyConcatDataset,
                  OnTheFlyConcatTrainDataset,
                  ClusterLinkingLazyTrainDataset,
                  ClusterLinkingLazyConcatTrainDataset,
                  bytes_to_id,
                  create_cluster_index,
                  get_mentions_and_entities,
                  get_index,
                  _read_candidates,
                  _read_mentions)
from comm import all_gather, broadcast
from utils import flatten, all_same, dict_merge_with

from IPython import embed


logger = logging.getLogger(__name__)


class Trainer(ABC):
    
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args

        # compute batch sizes
        args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        args.infer_batch_size = args.per_gpu_infer_batch_size * max(1, args.n_gpu)

        self.setup_model_and_dataloaders(args.model_name_or_path)

        logger.info("Training/evaluation parameters %s", args)

    def setup_model_and_dataloaders(self, model_name_or_path):
        args = self.args

        if args.local_rank not in [-1, 0]:
            dist.barrier()

        self.create_models()

        if args.local_rank == 0:
            dist.barrier()


        if args.do_train:
            # create training dataloader
            self.create_train_dataloader()

            if args.evaluate_during_training:
                self.create_train_eval_dataloader()
                #self.create_val_dataloader()

            # set optimizers and schedulers
            self.set_optimizers_and_schedulers()

        if args.do_train_eval:
            self.create_train_eval_dataloader()

        if args.do_val:
            self.create_val_dataloader()

        if args.do_test:
            self.create_test_dataloader()

        if args.task_name != 'xdoc_cluster_linking':
            for model_name in self.models.keys():
                self.models[model_name].to(args.device)

            if args.local_rank != -1:
                for model_name in self.models.keys():
                    self.models[model_name] = DDP( 
                            self.models[model_name],
                            device_ids=[args.local_rank],
                            output_device=args.local_rank,
                            find_unused_parameters=True
                    )

    def load_and_cache_examples(self, split=None, evaluate=False):

        args = self.args

        if args.local_rank not in [-1, 0]:
            dist.barrier()

        assert split == 'train' or split == 'val' or split == 'test'
        assert evaluate == True or split == 'train'

        cache_dir = os.path.join(args.data_dir, 'cache', split)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        if split == 'train':
            domains = args.train_domains
            args.train_cache_dir = cache_dir
        elif split == 'val':
            domains = args.val_domains
            args.val_cache_dir = cache_dir
        else:
            domains = args.test_domains
            args.test_cache_dir = cache_dir

        self.create_or_load_preprocessed_data(
                split, domains, cache_dir)

        if args.local_rank == 0:
            dist.barrier()

    def create_or_load_preprocessed_data(self,
                                         split,
                                         domains,
                                         cache_dir):
        args = self.args

        metadata_file = os.path.join(cache_dir, 'metadata.pt')

        if (os.path.exists(metadata_file) and not args.overwrite_cache):
            _metadata = torch.load(metadata_file)
        else:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            preprocessor = ZeshelPreprocessor(args)
            _metadata = preprocessor.preprocess_data(
                    args.data_dir,
                    split,
                    domains,
                    cache_dir,
                    args.max_seq_length,
                    args.tokenizer)

            torch.save(_metadata, metadata_file)

        if split == 'train':
            self.train_metadata = _metadata
        elif split == 'val':
            self.val_metadata = _metadata
        else:
            self.test_metadata = _metadata

    def set_optimizers_and_schedulers(self):
        args = self.args
        self.optimizers, self.schedulers = {}, {}

        # calculate hyperparameters for scheduler
        if args.max_steps > 0:
            args.t_total = args.max_steps
            args.num_train_epochs = (args.max_steps
                                     // (len(self.train_dataloader)
                                         // args.gradient_accumulation_steps)
                                     + 1)
        else:
            args.t_total = (len(self.train_dataloader)
                            // args.gradient_accumulation_steps
                            * args.num_train_epochs)

        # build optimizers and schedulers
        for model_name, model in self.models.items():
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() 
                             if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
                ]
            self.optimizers[model_name] = AdamW(
                    optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    eps=args.adam_epsilon
            )
            self.schedulers[model_name] = get_linear_schedule_with_warmup(
                    self.optimizers[model_name],
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=args.t_total
            )

    @abstractmethod
    def create_models(self):
        pass

    @abstractmethod
    def create_train_dataloader(self):
        pass

    @abstractmethod
    def create_train_eval_dataloader(self):
        pass
    
    @abstractmethod
    def create_val_dataloader(self):
        pass

    @abstractmethod
    def create_test_dataloader(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def evaluate(self, split):
        pass


class XDocClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(XDocClusterLinkingTrainer, self).__init__(args)

    def create_models(self):
        args = self.args
        self.models = {
                'mention_entity_score': CrossEncoder(
                    args,
                    name='mention_entity_score'
                ),
                'mention_embedding': MirrorStackEmbeddingModel(
                    args,
                    name='mention_embedding'
                )
        }

    def create_train_dataloader(self):
        self.train_dataloader = self.train_dataloader()

    def create_train_eval_dataloader(self):
        data = self.eval_data(split='train')
        (self.train_eval_cluster2mentions,
         self.train_eval_mention_dataset,
         self.train_eval_mention_entity_dataset) = data
    
    def create_val_dataloader(self):
        data = self.eval_data(split='val')
        (self.val_cluster2mentions,
         self.val_mention_dataset,
         self.val_mention_entity_dataset) = data

    def create_test_dataloader(self):
        data = self.eval_data(split='test')
        (self.test_cluster2mentions,
         self.test_mention_dataset,
         self.test_mention_entity_dataset) = data

    def _xdoc_train_collate_fn(self, batch):
        # get all of the cluster ids
        cluster_ids = flatten([[x['cluster_id']]
                   * len(x['mention_dataset']['indices'])
                                       for x in batch])
        # create mention dataset
        mention_dataset_dirs = [
            x['mention_dataset']['dir'] for x in batch
        ]
        mention_dataset_indices = flatten([
            x['mention_dataset']['indices'] for x in batch
        ])
        assert all_same(mention_dataset_dirs)
        mention_dataset = LazyDataset(
                mention_dataset_dirs[0],
                mention_dataset_indices
        )

        # create mention-entity scores dataset
        mention_entity_dirs = [
            x['mention_entity_dataset']['dir'] for x in batch
        ]
        mention_entity_indices = flatten([
            x['mention_entity_dataset']['indices'] for x in batch
        ])
        assert all_same(mention_entity_dirs)
        mention_entity_dataset = LazyConcatDataset(
                mention_entity_dirs[0],
                mention_entity_indices
        )

        return cluster_ids, mention_dataset, mention_entity_dataset

    def train_dataloader(self):
        args = self.args
        self.train_dataset = self.load_and_cache_examples(
                split='train', evaluate=False
        )
        return DataLoader(self.train_dataset,
                          batch_size=args.num_clusters_per_macro_batch,
                          collate_fn=self._xdoc_train_collate_fn,
                          num_workers=args.num_dataloader_workers,
                          pin_memory=False,
                          shuffle=False)

    def eval_data(self, split=None):
        assert split == 'train' or split == 'val' or split =='test'
        args = self.args
        return self.load_and_cache_examples(split=split, evaluate=True)


    def compute_emb_scores_inference(self, mention_dataset, evaluate=False):
        args = self.args
        
        self.models['mention_embedding'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_embedding'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )

        ddp_model.eval()

        mention_sampler = DistributedSampler(
                mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention_dataloader = DataLoader(
                mention_dataset,
                sampler=mention_sampler,
                batch_size=args.infer_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True,
            )

        mention_ids = []
        mention_indices = []
        mention_embeddings = []
        
        mention_batch_iterator = tqdm(mention_dataloader,
                                      desc='Mention',
                                      disable=(not evaluate
                                               or args.disable_logging))
        for batch in mention_batch_iterator:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            with torch.no_grad():
                serialized_mention_id = batch[0].cpu().numpy().tolist()
                _mention_ids = [bytes_to_id(x) for x in serialized_mention_id]
                _mention_indices = batch[1].cpu().numpy().tolist()
                inputs = {'input_ids_a':      batch[2],
                          'attention_mask_a': batch[3],
                          'token_type_ids_a': batch[4]}
                outputs = ddp_model(**inputs)
                mention_embeddings.append(outputs.cpu() if evaluate else outputs)
                mention_ids.extend(_mention_ids)
                mention_indices.extend(_mention_indices)   

        mention_embeddings = torch.cat(mention_embeddings, 0)

        self.models['mention_embedding'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        gathered_data = all_gather({
                'mention_ids': mention_ids,
                'mention_indices': mention_indices,
                'mention_embeddings': mention_embeddings
            })


        if args.local_rank not in [0, -1]:
            dist.barrier()
            return

        mention_ids = flatten([d['mention_ids'] for d in gathered_data])
        mention_ids, unique_indices = np.unique(mention_ids, return_index=True)
        mention_indices = flatten([d['mention_indices'] for d in gathered_data])
        mention_indices = np.asarray(mention_indices)[unique_indices]
        mention_embeddings = torch.cat([d['mention_embeddings']
                                            for d in gathered_data], 0)
        mention_embeddings = mention_embeddings[unique_indices]
        mention_embeddings = mention_embeddings.to(args.device) if evaluate else mention_embeddings

        mention2local_indices = {uid : i for i, uid in enumerate(mention_ids)}
        local_indices2mention = np.asarray(mention_ids)
        mention2dataset_indices = {uid : i for uid, i in zip(mention_ids, mention_indices)}

        if evaluate:
            dist.barrier()
            return (mention2local_indices,
                    local_indices2mention,
                    mention_embeddings)

        mention_mention_scores = 1.0-(mention_embeddings @ mention_embeddings.T)

        dist.barrier()
        return (mention2local_indices,
                local_indices2mention,
                mention2dataset_indices,
                mention_mention_scores)

    def compute_cat_scores_inference(self,
                                     mention2mention_dataset,
                                     num_mentions,
                                     evaluate=False):
        assert evaluate
        args = self.args
        
        self.models['mention_mention_score'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_mention_score'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )

        ddp_model.eval()

        num_mention2mention_pairs = len(mention2mention_dataset)

        mention2mention_sampler = DistributedSampler(
                mention2mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention2mention_dataloader = DataLoader(
                mention2mention_dataset,
                sampler=mention2mention_sampler,
                batch_size=args.infer_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True
            )

        mention2mention_scores = []

        total_loss = 0.0
        for batch in tqdm(mention2mention_dataloader,
                          desc='Mention2Mention',
                          disable=(not evaluate
                                   or args.disable_logging)):
            batch = tuple(t.to(args.device, non_blocking=True)
                                if type(t) != list else t
                            for t in batch)
            with torch.no_grad():
                mention_indices_a = batch[2].cpu().numpy().tolist()
                mention_indices_b = batch[4].cpu().numpy().tolist()
                inputs = {'input_ids':      batch[5].unsqueeze(1),
                          'attention_mask': batch[6].unsqueeze(1),
                          'token_type_ids': batch[7].unsqueeze(1),
                          'evaluate': True}
                _, scores = ddp_model(**inputs)
                scores = scores.squeeze(1).cpu().numpy().tolist()
                mention2mention_scores.extend(
                        zip(mention_indices_a, mention_indices_b, scores)
                )

        self.models['mention_mention_score'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        gathered_data = all_gather({
                'mention2mention_scores': mention2mention_scores,
            })

        if args.local_rank not in [0, -1]:
            dist.barrier()
            return

        mention2mention_scores = flatten([
                d['mention2mention_scores'] for d in gathered_data
        ])

        _row, _col, _data = zip(*mention2mention_scores)
        _row = np.asarray(_row)
        _col = np.asarray(_col)
        _data = np.asarray(_data)

        _sparse_indices = np.vstack((_row, _col))
        _sparse_indices, _data_indices = np.unique(
                _sparse_indices, axis=1, return_index=True
        )
        _sparse_values = _data[_data_indices]

        xdoc_sparse_mention_dists = csr_matrix(
                (_sparse_values, _sparse_indices),
                 shape=(num_mentions, num_mentions)
        )

        xdoc_sparse_mention_dists = (xdoc_sparse_mention_dists
                                     + xdoc_sparse_mention_dists.T) / 2
        xdoc_sparse_mention_dists = scipy.sparse.triu(
                xdoc_sparse_mention_dists, 0
        )

        dist.barrier()
        return xdoc_sparse_mention_dists

    def compute_me_scores_inference(self,
                                    mention_entity_dataset,
                                    evaluate=False):
        args = self.args
        self.models['mention_entity_score'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_entity_score'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )
        ddp_model.eval()

        sampler = DistributedSampler(
                mention_entity_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        dataloader = DataLoader(
                mention_entity_dataset,
                sampler=sampler,
                batch_size=args.infer_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True,
            )

        mention_ids = []
        mention_indices = []
        entity_ids = []
        entity_indices = []
        sparse_indices = []
        sparse_values = []
        sparse_tensor_size = None

        batch_iterator = tqdm(dataloader,
                              desc='Mention',
                              disable=(not evaluate or args.disable_logging))
        for batch in batch_iterator:

            batch = tuple(t.to(args.device, non_blocking=True)
                                if type(t) != list else t
                            for t in batch)
            with torch.no_grad():

                inputs = {'input_ids':      batch[4].unsqueeze(1),
                          'attention_mask': batch[5].unsqueeze(1),
                          'token_type_ids': batch[6].unsqueeze(1),
                          'evaluate': True}

                # compute the affine transformation of `sparse_values`
                _, _sparse_values = ddp_model(**inputs)
                _sparse_values = _sparse_values.squeeze(1)

                # do a bunch of bookkeeping
                _serialized_mention_id = batch[0].cpu().numpy().tolist()
                _mention_ids = [bytes_to_id(x) for x in _serialized_mention_id]
                _serialized_entity_id = batch[2].cpu().numpy().tolist()
                _entity_ids = [bytes_to_id(x) for x in _serialized_entity_id]

                _mention_indices = batch[1].cpu()
                _entity_indices = batch[3].cpu()

                sparse_indices.append(torch.stack((_mention_indices, _entity_indices), dim=1))
                sparse_values.append(_sparse_values.cpu())
                    
                sparse_tensor_size = (batch[7][0][0], batch[7][1][0])

                mention_ids.extend(_mention_ids)
                mention_indices.extend(_mention_indices.cpu().tolist())
                entity_ids.extend(_entity_ids)
                entity_indices.extend(_entity_indices.cpu().tolist())

        sparse_indices = torch.cat(sparse_indices, 0)
        sparse_values = torch.cat(sparse_values, 0)

        gathered_data = all_gather({
                'mention_ids': mention_ids,
                'mention_indices': mention_indices,
                'entity_ids': entity_ids,
                'entity_indices': entity_indices,
                'sparse_tensor_size': sparse_tensor_size,
                'sparse_indices': sparse_indices,
                'sparse_values': sparse_values,
            })

        self.models['mention_entity_score'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        if args.local_rank not in [0, -1]:
            dist.barrier()
            return

        sparse_tensor_size = tuple(gathered_data[0]['sparse_tensor_size'])
        sparse_indices = torch.cat([d['sparse_indices']
                                        for d in gathered_data], 0).t()
        sparse_indices = sparse_indices.numpy()
        sparse_indices, unique_indices = np.unique(sparse_indices,
                                                   return_index=True,
                                                   axis=1)
        sparse_values = torch.cat([d['sparse_values']
                                            for d in gathered_data], 0)
        sparse_values = sparse_values.numpy()
        sparse_values = sparse_values[unique_indices]

        mention_ids = flatten([d['mention_ids'] for d in gathered_data])
        entity_ids = flatten([d['entity_ids'] for d in gathered_data])
        mention_indices = flatten([d['mention_indices'] for d in gathered_data])
        entity_indices = flatten([d['entity_indices'] for d in gathered_data])

        mention_ids = np.asarray(mention_ids)[unique_indices]
        entity_ids = np.asarray(entity_ids)[unique_indices]
        mention_indices = np.asarray(mention_indices)[unique_indices]
        entity_indices = np.asarray(entity_indices)[unique_indices]

        dist.barrier()
        return (mention_ids,
                entity_ids,
                mention_indices,
                entity_indices,
                sparse_tensor_size,
                sparse_indices,
                sparse_values)

    def _aggregate_mm_and_me(self,
                             mention2local_indices,
                             local_indices2mention,
                             mention2dataset_indices,
                             mention_mention_scores,
                             me_mention_ids,
                             me_entity_ids,
                             me_mention_indices,
                             me_entity_indices,
                             sparse_me_tensor_size,
                             sparse_me_indices,
                             sparse_me_values):
        args = self.args

        num_mentions = sparse_me_tensor_size[0].item()
        num_entities = sparse_me_tensor_size[1].item()

        # make sure mention_mention_scores is symmetric
        mention_mention_scores = (mention_mention_scores
                                  + mention_mention_scores.t())/2.0
        mention_mention_scores = torch.triu(mention_mention_scores, diagonal=1)

        # convert the mention_mention_scores to sparse if not already
        if mention_mention_scores.layout == torch.strided:
            sparse_mm_scores = mention_mention_scores.to_sparse()
        else:
            sparse_mm_scores = mention_mention_scores

        # move the indices and values to cpu and convert to numpy
        sparse_mm_indices = sparse_mm_scores.indices().cpu().numpy()
        sparse_mm_values = sparse_mm_scores.values().cpu().numpy()

        # update indices to global joint indices
        reindex = lambda i : mention2dataset_indices[local_indices2mention[i]]
        v_reindex = np.vectorize(reindex)
        sparse_mm_indices = v_reindex(sparse_mm_indices)
        sparse_me_indices[1,:] += num_mentions
        me_entity_indices += num_mentions

        # merge mm and me sparse tensors
        sparse_indices = np.concatenate((sparse_mm_indices,
                                         sparse_me_indices), axis=1)
        sparse_values = np.concatenate((sparse_mm_values,
                                        sparse_me_values), axis=0)
        sparse_shape = (num_mentions, num_mentions + num_entities)
        sparse_score_matrix = coo_matrix((sparse_values,
                                          (sparse_indices[0],
                                           sparse_indices[1])),
                                         shape=sparse_shape)


        return (me_mention_ids,
                me_entity_ids,
                me_mention_indices,
                me_entity_indices,
                sparse_score_matrix)

    def build_sl_pairs_train_dataset(self,
                                     gold_entity_indices,
                                     me_mention_indices,
                                     me_entity_indices,
                                     sparse_score_matrix,
                                     mention_index2id,
                                     entity_index2id):
        args = self.args

        # shape stuff
        global_shape = sparse_score_matrix.shape
        num_mentions = global_shape[0]
        num_entities = global_shape[1] - global_shape[0]

        max_num_pairs = 48
        
        # store the tuples
        mention2mention_tuples = []
        mention_entity_tuples = []
        for entity_index in gold_entity_indices:

            # compute in cluster mask
            in_cluster_mentions = np.unique(
                    me_mention_indices[me_entity_indices == entity_index]
            )
            in_cluster_indices_mask = np.any(
                    (in_cluster_mentions[np.newaxis,:]
                       == sparse_score_matrix.row[:,np.newaxis]),
                    axis=1
            )

            if self.epoch > 0:
                in_cluster_indices_mask &= (
                        np.any(in_cluster_mentions[np.newaxis,:]
                               == sparse_score_matrix.col[:,np.newaxis],axis=1)
                        | (sparse_score_matrix.col == entity_index)
                )
            else:
                # training on all mention-entity pairs in the first epoch
                in_cluster_indices_mask &= (
                        np.any(in_cluster_mentions[np.newaxis,:]
                               == sparse_score_matrix.col[:,np.newaxis],axis=1)
                )

            # compute out cluster_mask
            out_cluster_indices_mask = ~in_cluster_indices_mask
            out_cluster_indices_mask &= (
                    np.any(in_cluster_mentions[np.newaxis,:]
                            == sparse_score_matrix.col[:,np.newaxis],
                           axis=1)
                    | np.any(in_cluster_mentions[np.newaxis,:]
                              == sparse_score_matrix.row[:,np.newaxis],
                             axis=1)
            )

            # compute in-cluster mst
            _row = sparse_score_matrix.row[in_cluster_indices_mask]
            _col = sparse_score_matrix.col[in_cluster_indices_mask]
            _data = sparse_score_matrix.data[in_cluster_indices_mask]
            in_cluster_csr = csr_matrix(
                    (_data, (_row, _col)),
                     shape=(global_shape[1], global_shape[1])
            )
            in_cluster_mst = minimum_spanning_tree(in_cluster_csr).tocoo()

            # compute positive edges
            pos_mst_indices_mask = (
                    in_cluster_mst.data > args.max_in_cluster_dist
            )
            #pick_pos_prob = max_num_pairs / (np.sum(pos_mst_indices_mask) + 1)
            pick_pos_prob = 1.0
            if np.sum(pos_mst_indices_mask) > 0.0:
                pos_index_generator = zip(
                        in_cluster_mst.row[pos_mst_indices_mask],
                        in_cluster_mst.col[pos_mst_indices_mask]
                )
                for row, col in pos_index_generator:
                    if col >= num_mentions:
                        if random.uniform(0, 1) < pick_pos_prob:
                            assert self.epoch > 0
                            mention_entity_tuples.append(
                                    (0.0, row, col-num_mentions)
                            )
                    else:
                        if random.uniform(0, 1) < pick_pos_prob:
                            mention2mention_tuples.append((0.0, row, col))

            if self.epoch <= 0:
                # NOTE: this always includes all positive mention-entity edges
                pos_me_indices_mask = (
                        np.any(
                            (in_cluster_mentions[np.newaxis,:]
                               == sparse_score_matrix.row[:,np.newaxis]),
                            axis=1
                        )
                        & (sparse_score_matrix.col == entity_index)
                )
                #pick_pos_prob = max_num_pairs / (np.sum(pos_me_indices_mask) + 1)
                pick_pos_prob = 1.0
                if np.sum(pos_me_indices_mask) > 0.0:
                    pos_index_generator = zip(
                            sparse_score_matrix.row[pos_me_indices_mask],
                            sparse_score_matrix.col[pos_me_indices_mask]
                    )
                    for row, col in pos_index_generator:
                        if random.uniform(0, 1) < pick_pos_prob:
                            assert col >= num_mentions
                            mention_entity_tuples.append(
                                    (0.0, row, col-num_mentions)
                            )

            # compute negative edges
            neg_indices_mask = (
                    (sparse_score_matrix.data 
                        < (args.max_in_cluster_dist + args.margin))
                    & out_cluster_indices_mask
            )
            pick_neg_prob = max_num_pairs / (np.sum(neg_indices_mask) + 1)
            if np.sum(neg_indices_mask) > 0.0:
                neg_index_generator = zip(
                        sparse_score_matrix.row[neg_indices_mask],
                        sparse_score_matrix.col[neg_indices_mask]
                )
                pick_neg_prob = ((pos_mst_indices_mask.size + 1)
                                 / np.sum(neg_indices_mask))

                for row, col in neg_index_generator:
                    if col >= num_mentions:
                        if random.uniform(0, 1) < pick_neg_prob:
                            mention_entity_tuples.append(
                                    (1.0, row, col-num_mentions)
                            )
                    else:
                        if random.uniform(0, 1) < pick_neg_prob:
                            mention2mention_tuples.append((1.0, row, col))

        return (mention2mention_tuples,
                mention_entity_tuples)

    def train_on_emb_pairs(self, mention2mention_dataset):
        args = self.args
        self.models['mention_embedding'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_embedding'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )
        ddp_model.train()
        ddp_model.zero_grad()

        num_mention2mention_pairs = len(mention2mention_dataset)

        mention2mention_sampler = DistributedSampler(
                mention2mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention2mention_dataloader = DataLoader(
                mention2mention_dataset,
                sampler=mention2mention_sampler,
                batch_size=args.train_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True
            )

        total_loss = 0.0
        for batch in mention2mention_dataloader:
            batch = tuple(t.to(args.device) for t in batch)

            _coeff = batch[0]
            _batch_size = _coeff.shape[0]

            mention_inputs = {'input_ids_a':      batch[3],
                              'attention_mask_a': batch[4],
                              'token_type_ids_a': batch[5],
                              'input_ids_b':      batch[8],
                              'attention_mask_b': batch[9],
                              'token_type_ids_b': batch[10]}

            scores = torch.diag(
                    ddp_model(**mention_inputs)
            )

            loss = torch.mean(torch.abs(_coeff - scores)) / len(mention2mention_dataloader)
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(),
                args.max_grad_norm
        )
        self.optimizers['mention_embedding'].step()
        self.schedulers['mention_embedding'].step()
        ddp_model.zero_grad()

        self.models['mention_embedding'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        return total_loss

    def train_on_cat_pairs(self, mention2mention_dataset):
        args = self.args
        self.models['mention_mention_score'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_mention_score'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )
        ddp_model.train()
        ddp_model.zero_grad()

        num_mention2mention_pairs = len(mention2mention_dataset)

        mention2mention_sampler = DistributedSampler(
                mention2mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention2mention_dataloader = DataLoader(
                mention2mention_dataset,
                sampler=mention2mention_sampler,
                batch_size=args.train_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True
            )

        total_loss = 0.0
        for batch in mention2mention_dataloader:
            batch = tuple(t.to(args.device, non_blocking=True)
                                if type(t) != list else t
                            for t in batch)
            coeff = batch[0]
            inputs = {'input_ids':      batch[5].unsqueeze(1),
                      'attention_mask': batch[6].unsqueeze(1),
                      'token_type_ids': batch[7].unsqueeze(1),
                      'evaluate': False,
                      'labels': coeff.unsqueeze(1)}

            loss, scores = ddp_model(**inputs)
            #loss = torch.mean(torch.abs(coeff - scores)) / len(mention2mention_dataloader)
            loss /= len(mention2mention_dataloader)
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
                self.models['mention_mention_score'].parameters(),
                args.max_grad_norm
        )
        self.optimizers['mention_mention_score'].step()
        self.schedulers['mention_mention_score'].step()
        ddp_model.zero_grad()

        self.models['mention_entity_score'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        return total_loss

    def train_sparse_me_scores(self, mention2entity_dataset):
        args = self.args
        self.models['mention_entity_score'].to(args.device)
        ddp_model = DDP( 
                self.models['mention_entity_score'],
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
        )
        ddp_model.train()
        ddp_model.zero_grad()

        num_mention2entity_pairs = len(mention2entity_dataset)

        mention2entity_sampler = DistributedSampler(
                mention2entity_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention2entity_dataloader = DataLoader(
                mention2entity_dataset,
                sampler=mention2entity_sampler,
                batch_size=args.train_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True
            )

        total_loss = 0.0
        for batch in mention2entity_dataloader:
            batch = tuple(t.to(args.device, non_blocking=True)
                                if type(t) != list else t
                            for t in batch)
            coeff = batch[0]
            inputs = {'input_ids':      batch[5].unsqueeze(1),
                      'attention_mask': batch[6].unsqueeze(1),
                      'token_type_ids': batch[7].unsqueeze(1),
                      'evaluate': False,
                      'labels': coeff.unsqueeze(1)}

            loss, scores = ddp_model(**inputs)
            #loss = torch.mean(torch.abs(coeff - scores)) / len(mention2entity_dataloader)
            loss /= len(mention2entity_dataloader)
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(
                self.models['mention_entity_score'].parameters(),
                args.max_grad_norm
        )
        self.optimizers['mention_entity_score'].step()
        self.schedulers['mention_entity_score'].step()
        self.models['mention_entity_score'].zero_grad()

        self.models['mention_entity_score'].to('cpu')
        ddp_model = None
        del ddp_model
        torch.cuda.empty_cache()

        return total_loss

    def train_step(self,
                   cluster_ids,
                   mention_dataset,
                   mention_entity_dataset):
        args = self.args

        # NOTE: *_outputs will be `None` for all local_rank != 0
        emb_outputs = self.compute_emb_scores_inference(mention_dataset)
        me_outputs = self.compute_me_scores_inference(
                mention_entity_dataset)

        # compute pairs for clustering on rank 0 process
        mention2mention_tuples = None
        mention_entity_tuples = None
        if args.local_rank == 0:
            agg_output = self._aggregate_mm_and_me(*(emb_outputs+me_outputs))
            (me_mention_ids,
             me_entity_ids,
             me_mention_indices,
             me_entity_indices,
             sparse_score_matrix) = agg_output

            entity_id2index = {_id : _index
                    for _id, _index in zip(me_entity_ids, me_entity_indices)}
            gold_entity_indices = map(lambda x : entity_id2index.get(x, -1),
                                      cluster_ids)
            gold_entity_indices = np.asarray(list(gold_entity_indices))
            gold_entity_indices = gold_entity_indices[gold_entity_indices != -1]
            gold_entity_indices = np.unique(gold_entity_indices).tolist()

            # for debugging
            mention_index2id = {_index : _id
                    for _id, _index in zip(me_mention_ids, me_mention_indices)}
            entity_index2id = {b : a for a, b in entity_id2index.items()}

            (mention2mention_tuples,
             mention_entity_tuples) = self.build_sl_pairs_train_dataset(
                     gold_entity_indices,
                     me_mention_indices,
                     me_entity_indices,
                     sparse_score_matrix,
                     mention_index2id, # for debugging purposes
                     entity_index2id
            )

        dist.barrier()

        # broadcast pairs to all processes
        mention2mention_tuples = broadcast(mention2mention_tuples, src=0)
        mention_entity_tuples = broadcast(mention_entity_tuples, src=0)

        # train on mention-entity pairs
        me_loss = 0.0
        num_me_pairs = len(mention_entity_tuples)
        if num_me_pairs > 0:
            mention2entity_dataset = ClusterLinkingLazyConcatTrainDataset(
                    mention_entity_tuples,
                    args.cached_train_mention_entity_dir
            )
            me_loss = self.train_sparse_me_scores(mention2entity_dataset)

        # train on mention-mention embedding pairs
        mm_emb_loss = 0.0
        num_mm_emb_pairs = len(mention2mention_tuples)
        if num_mm_emb_pairs > 0:
            # build the dataset on all ranks
            mention2mention_dataset = ClusterLinkingLazyTrainDataset(
                mention2mention_tuples, args.cached_train_mention_examples_dir,
                args.cached_train_mention_examples_dir)

            # distributed training on the pairs dataset we just built
            mm_emb_loss = self.train_on_emb_pairs(mention2mention_dataset)

        # train on mention-mention concat pairs
        mm_cat_loss = 0.0
        mention2mention_tuples.extend(
                [(c,b,a) for c,a,b in mention2mention_tuples]
        )
        if len(mention2mention_tuples) > 0:
            # build the dataset on all ranks
            mention2mention_dataset = OnTheFlyConcatTrainDataset(
                mention2mention_tuples, args.cached_train_mention_examples_dir,
                args.cached_train_mention_examples_dir)

            # distributed training on the concat dataset we just built
            mm_cat_loss = self.train_on_cat_pairs(mention2mention_dataset)

        return {'num_mm_emb_pairs': num_mm_emb_pairs,
                'num_me_pairs': num_me_pairs,
                'mm_emb_loss': mm_emb_loss,
                'mm_cat_loss': mm_cat_loss,
                'me_loss': me_loss}

    def get_custom_batches(self):
        args = self.args

        ### NOTE : just for development !!! remove before training !!!
        #return DataLoader(self.train_dataset,
        #                  batch_size=args.num_clusters_per_macro_batch,
        #                  collate_fn=self._xdoc_train_collate_fn,
        #                  num_workers=args.num_dataloader_workers,
        #                  pin_memory=False,
        #                  shuffle=False)

        # map mentions back to training examples
        mention_index2example = {}
        for example in self.train_dataset:
            for idx in example['mention_dataset']['indices']:
                mention_index2example[idx] = example

        logger.info("Computing mention embeddings...")
        emb_outputs = self.compute_emb_scores_inference(
                self.train_eval_mention_dataset,
                evaluate=True
        )

        custom_ordered_dataset = []
        if args.local_rank == 0:
            # extract output variables
            (mention2local_indices,
             local_indices2mention,
             mention_embeddings) = emb_outputs

            # convert to numpy
            mention_embeddings = mention_embeddings.cpu().numpy()
            num_mentions = mention_embeddings.shape[0]

            logger.info('Building the kNN index...')
            n_cells = 1000
            n_probe = 100
            k = 50
            d = mention_embeddings.shape[1]
            quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
            knn_index = faiss.IndexIVFFlat(
                    quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
            )
            knn_index.train(mention_embeddings)
            knn_index.add(mention_embeddings)
            knn_index.nprobe = n_probe

            logger.info('Querying the kNN index...')
            D, I = knn_index.search(mention_embeddings, k+1)

            _row = np.repeat(np.arange(I.shape[0]), k)
            _col = I[:,1:].reshape(-1,)
            _data = 1.0 - D[:,1:].reshape(-1,)

            # cluster using the MST graph
            logger.info('Computing MST...')
            xdoc_sparse_mention_dists = csr_matrix(
                    (_data, (_row, _col)),
                     shape=(num_mentions, num_mentions)
            )
            xdoc_mention_mst = minimum_spanning_tree(
                    xdoc_sparse_mention_dists).tocoo()

            logger.info('Choosing threshold...')
            for threshold in sorted(xdoc_mention_mst.data, reverse=True):
                pruned_mask = xdoc_mention_mst.data < threshold
                _data = xdoc_mention_mst.data[pruned_mask]
                _row = xdoc_mention_mst.row[pruned_mask]
                _col = xdoc_mention_mst.col[pruned_mask]
                pruned_mst = csr_matrix(
                        (_data, (_row, _col)),
                         shape=xdoc_mention_mst.shape
                )

                # produce clusters
                n_components, cluster_labels = connected_components(
                    csgraph=pruned_mst, directed=False, return_labels=True)

                if n_components >= len(self.train_dataloader):
                    logger.info('Found threshold...')
                    pred_cluster_map = defaultdict(list)
                    for i, label in enumerate(cluster_labels):
                        pred_cluster_map[label].append(i)

                    added_clusters = set()
                    for mention_cluster in pred_cluster_map.values():
                        for mention_index in mention_cluster:
                            example = mention_index2example[mention_index]
                            cluster_id = example['cluster_id']
                            if cluster_id in added_clusters:
                                continue
                            added_clusters.add(cluster_id)
                            custom_ordered_dataset.append(example)
                    break

        dist.barrier()

        logger.info('Broadcasting ordered dataset...')

        # broadcast the custom ordered dataset to the other processes
        custom_ordered_dataset = broadcast(custom_ordered_dataset, src=0)

        return DataLoader(custom_ordered_dataset,
                          batch_size=args.num_clusters_per_macro_batch,
                          collate_fn=self._xdoc_train_collate_fn,
                          num_workers=args.num_dataloader_workers,
                          pin_memory=False,
                          shuffle=False)

    def train(self):
        args = self.args

        logger.info("***** Running training *****")
        logger.info("  Num batches = %d", len(self.train_dataloader))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * args.gradient_accumulation_steps * (dist.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.t_total)

        args.disable_logging |= args.local_rank not in [-1, 0]

        global_step = 0
        log_return_dicts = []
        best_val_performance = None
        best_step = 0
        epoch_iterator = trange(int(args.num_train_epochs),
                                desc="Epoch",
                                disable=args.disable_logging)

        for epoch in epoch_iterator:

            #batch_iterator = tqdm(self.train_dataloader,
            #                      desc="Batch",
            #                      disable=args.disable_logging)
            logger.info('Creating custom batches for epoch {}'.format(epoch))
            batch_dataloader = self.get_custom_batches()
            batch_iterator = tqdm(batch_dataloader,
                                  desc="Batch",
                                  disable=args.disable_logging)
            self.epoch = epoch
            for batch in batch_iterator:
                dist.barrier()

                return_dict = self.train_step(*batch)
                log_return_dicts.append(copy.deepcopy(return_dict))
                global_step += 1

                # logging stuff for babysitting
                if global_step % args.logging_steps == 0:
                    avg_return_dict = reduce(dict_merge_with, log_return_dicts)
                    for stat_name, stat_value in avg_return_dict.items():
                        logger.info("Average %s: %s at global step: %s",
                                stat_name,
                                str(stat_value/args.logging_steps),
                                str(global_step)
                        )
                    log_return_dicts = []

                if global_step % args.save_steps == 0:
                    ## evaluate the model
                    #if args.evaluate_during_training:
                    #    logger.info('*** Evaluating checkpoint {} ***'.format(
                    #        global_step))
                    #    for split in ['train', 'val']:
                    #        results = self.evaluate(split=split)
                    #        if split == 'val':
                    #            if (best_val_performance is None
                    #                or results['avg_fmi'] > best_val_performance['avg_fmi']):
                    #                best_val_performance = results
                    #                best_step = global_step

                    # save the model
                    if args.local_rank in [0, -1]:
                        for key in self.models.keys():
                            self.models[key].save_model(
                                    suffix='checkpoint-{}'.format(global_step))
                    dist.barrier()
        
        logger.info('*** Training complete ***')
        if args.evaluate_during_training:
            logger.info('\tbest step: {}'.format(best_step))
            logger.info('\tbest val results:')
            for key, value in best_val_performance.items():
                logger.info('\t\t{}: {}'.format(key, value))

    def compute_linking_accuracy(self,
                                 mention_mst,
                                 mention_entity_dists,
                                 threshold,
                                 mentions,
                                 index2euid):
        # prune the mst using the threshold
        pruned_mask = mention_mst.data < threshold
        _data = mention_mst.data[pruned_mask]
        _row = mention_mst.row[pruned_mask]
        _col = mention_mst.col[pruned_mask]
        pruned_mst = csr_matrix(
                (_data, (_row, _col)),
                 shape=mention_mst.shape
        )

        # produce clusters
        n_components, cluster_labels = connected_components(
            csgraph=pruned_mst, directed=False, return_labels=True)

        pred_cluster_map = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            pred_cluster_map[label].append(i)

        mention_entity_dists = mention_entity_dists.tocsr()
        cluster2entity_map = {}
        for cluster_key, cluster_indices in pred_cluster_map.items():
            _closest_entities = []
            for _idx in cluster_indices:
                _row = mention_entity_dists.getrow(_idx).tocoo()
                if _row.size == 0:
                    continue
                _sparse_idx = np.argmin(_row.data)
                _closest_entities.append(
                        (_row.col[_sparse_idx], _row.data[_sparse_idx])
                )
            if len(_closest_entities) == 0:
                pred_entity_index = 0
            else:
                pred_entity_index, _ = min(_closest_entities, key=lambda x: x[1])
            cluster2entity_map[cluster_key] = pred_entity_index

        linking_preds = {
                m : cluster2entity_map[k] 
                    for k, v in pred_cluster_map.items() for m in v
        }

        hits = 0
        for m_idx, e_idx in linking_preds.items():
            if mentions[m_idx]['label_document_id'] == index2euid[e_idx]:
                hits += 1

        return hits / len(mentions)

    def evaluate(self, split='', threshold=None):
        assert split == 'train' or split == 'val' or split == 'test'

        args = self.args

        if split == 'train':
            cluster2mentions = self.train_eval_cluster2mentions
            mention_dataset = self.train_eval_mention_dataset
            mention_entity_dataset = self.train_eval_mention_entity_dataset
            domains = args.train_domains
        elif split == 'val':
            cluster2mentions = self.val_cluster2mentions
            mention_dataset = self.val_mention_dataset
            mention_entity_dataset = self.val_mention_entity_dataset
            domains = args.val_domains
        else:
            cluster2mentions = self.test_cluster2mentions
            mention_dataset = self.test_mention_dataset
            mention_entity_dataset = self.test_mention_entity_dataset
            domains = args.test_domains

        # load the mention objects for computing accuracy
        mentions, _, entity_documents = get_mentions_and_entities(
                args.data_dir, split, domains)
        index2euid = {v : k for k, v in get_index(entity_documents).items()}

        num_mentions = len(mentions)
        num_entities = len(index2euid)

        # Eval!
        logger.info("***** Running evaluation : {} *****".format(split))
        logger.info("  Num mentions = %d", len(mention_dataset))
        logger.info("  Batch size = %d", args.infer_batch_size)

        # compute all mention embeddings
        logger.info("Computing mention embeddings...")
        emb_outputs = self.compute_emb_scores_inference(
                mention_dataset,
                evaluate=True
        )

        #logger.info("Computing mention-entity scores...")
        #me_outputs = self.compute_me_scores_inference(
        #        mention_entity_dataset,
        #        evaluate=True
        #)

        # do kNN on rank 0 process
        mm_tuples = None
        if args.local_rank == 0:
            # extract output variables
            (mention2local_indices,
             local_indices2mention,
             mention_embeddings) = emb_outputs

            # convert to numpy
            mention_embeddings = mention_embeddings.cpu().numpy()

            # do some faiss stuff for sparsification
            logger.info('Building the kNN index...')
            n_cells = 200
            n_probe = 75
            k = 25
            d = mention_embeddings.shape[1]
            quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
            knn_index = faiss.IndexIVFFlat(
                    quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
            )
            knn_index.train(mention_embeddings)
            knn_index.add(mention_embeddings)
            knn_index.nprobe = n_probe

            logger.info('Querying the kNN index...')
            D, I = knn_index.search(mention_embeddings, k+1)

            # sparsified mention-mention distances
            _row = np.repeat(np.arange(I.shape[0]), k)
            _col = I[:,1:].reshape(-1,)
            _data = 1.0 - D[:,1:].reshape(-1,)

            ## for zeshel only
            #filter_row, filter_col, filter_data = [], [], []
            #for r, c, d in zip(_row, _col, _data):
            #    if mentions[r]['corpus'] == mentions[c]['corpus']:
            #        filter_row.append(r)
            #        filter_col.append(c)
            #        filter_data.append(d)

            #_row = filter_row
            #_col = filter_col
            #_data = filter_data

            mm_tuples = [(0, a, b) for a, b in list(set(zip(_row, _col)))]
            mm_tuples.extend([(c,b,a) for c,a,b in mm_tuples])
            mm_tuples = list(set(mm_tuples))

        # broadcast mm_tuples
        mm_tuples = broadcast(mm_tuples, src=0)

        # build the data set
        mention2mention_dataset = OnTheFlyConcatTrainDataset(
            mm_tuples, args.cached_mention_examples_dir,
            args.cached_mention_examples_dir)

        # compute the scores
        mm_outputs = self.compute_cat_scores_inference(mention2mention_dataset,
                                                       num_mentions,
                                                       evaluate=True)

        logger.info("Computing mention-entity scores...")
        me_outputs = self.compute_me_scores_inference(
                mention_entity_dataset,
                evaluate=True
        )

        if args.local_rank == 0:
            xdoc_sparse_mention_dists = mm_outputs
            _row = xdoc_sparse_mention_dists.row
            _col = xdoc_sparse_mention_dists.col
            _data = xdoc_sparse_mention_dists.data

            (mention_ids,
             entity_ids,
             mention_indices,
             entity_indices,
             sparse_tensor_size,
             me_sparse_indices,
             me_sparse_values) = me_outputs

            ## compute MST
            #logger.info('Computing MST...')
            #xdoc_sparse_mention_dists = csr_matrix(
            #        (_data, (_row, _col)),
            #         shape=(num_mentions, num_mentions)
            #)
            #xdoc_mention_mst = minimum_spanning_tree(
            #        xdoc_sparse_mention_dists).tocoo()

            #filter_xdoc_mention_mst = coo_matrix(
            #        (filter_data, (filter_row, filter_col)),
            #        shape=(num_mentions, num_mentions)
            #)

            ## create sparse mention-entity distance matrix
            #sparse_mention_entity_dists = csr_matrix(
            #        (sparse_values, sparse_indices),
            #         shape=(num_mentions, num_entities)
            #)

            ### Build MST over both mentions and entities
            mm_sparse_indices = np.vstack((np.asarray(_row), np.asarray(_col)))
            mm_sparse_values = np.asarray(_data)

            # adjust the me_sparse_indices by number of mentions
            me_sparse_indices[1, :] += num_mentions

            # build the joint sparse represenation of the graph
            sparse_indices = np.concatenate(
                    (mm_sparse_indices, me_sparse_indices), axis=1
            )
            sparse_values = np.concatenate(
                    (mm_sparse_values, me_sparse_values), axis=0
            )
            sparse_jme_dists = csr_matrix((sparse_values, sparse_indices),
                    shape=(num_mentions + num_entities,
                           num_mentions + num_entities)
            )

            # using the mst as the starting assignment graph for now
            # could also just use the sparse_jme_dists
            jme_assign_graph = minimum_spanning_tree(sparse_jme_dists).tocoo()
            #jme_assign_graph = sparse_jme_dists.tocoo()

            for i in tqdm(count(0), desc='JME MST Edges Removed'):
                # compute connected components
                _, vertex_labels = connected_components(
                        csgraph=jme_assign_graph, directed=False, return_labels=True
                )
                
                # find the components with more than one entity
                unique_labels, component_entity_counts = np.unique(
                        vertex_labels[num_mentions:], return_counts=True
                )
                prunable_components = unique_labels[component_entity_counts > 1]

                # if all the components have one entity or less, we're done
                if prunable_components.size == 0:
                    break

                # get edges and weights of current assignment graph
                jme_assign_indices = np.vstack(
                        (jme_assign_graph.row, jme_assign_graph.col)
                )
                jme_assign_values = jme_assign_graph.data

                # get vertices which can drop edges, i.e. `prunable_indices`
                vertex_mask = np.isin(vertex_labels, prunable_components)
                prunable_indices = np.where(vertex_mask)[0]
                unique_indices, counts = np.unique(
                        jme_assign_indices, return_counts=True
                )
                unique_indices_mask = np.where(
                        (counts == 1) * (unique_indices < num_mentions),
                        False,
                        True
                )
                prunable_indices_mask = np.isin(
                        prunable_indices, unique_indices[unique_indices_mask]
                )
                prunable_indices = prunable_indices[prunable_indices_mask]
 
                # get removable edge indices
                index_mask = np.isin(jme_assign_indices, prunable_indices)
                index_mask = index_mask[0] | index_mask[1]

                # find the max prunable edge
                edge_value_to_remove = np.max(jme_assign_values[index_mask])

                # mask away the max prunable edge
                to_keep_edge_mask = np.where(
                        (jme_assign_values == edge_value_to_remove)
                        * index_mask, False, True)
                jme_assign_indices = jme_assign_indices[:, to_keep_edge_mask]
                jme_assign_values = jme_assign_values[to_keep_edge_mask]

                # rebuild the graph
                jme_assign_graph = csr_matrix(
                        (jme_assign_values, jme_assign_indices),
                        shape=jme_assign_graph.shape
                ).tocoo()

            # compute coref accuracy
            pred_cluster_map = defaultdict(list)
            for i, label in enumerate(vertex_labels):
                pred_cluster_map[label].append(i)

            not_possible_mentions = 0
            possible_mentions = 0
            hits = 0
            for cluster in pred_cluster_map.values():
                cluster_mentions = list(filter(lambda x : x < num_mentions, cluster))
                cluster_entities = [x - num_mentions for x in filter(lambda x : x >= num_mentions, cluster)]
                assert len(cluster_entities) <= 1
                if len(cluster_mentions) == 0:  
                    continue                   
                if len(cluster_entities) == 0:
                    not_possible_mentions += len(cluster_mentions)
                else:
                    possible_mentions += len(cluster_mentions)
                    pred_entity = index2euid[cluster_entities[0]]
                    for idx in cluster_mentions:
                        if mentions[idx]['label_document_id'] == pred_entity:
                            hits += 1

            embed()
            exit()


            ##################################################################
            ## Probing experiment

            ## TODO: write probing experiment using ground truth clusters with
            ##       current mention-entity scores (and scaling) 
            ##       what is the recall of this?

            mention_entity_dists = sparse_mention_entity_dists.tocsr()

            _cluster_dict = defaultdict(list)
            for i, mention_obj in enumerate(mentions):
                _cluster_dict[mention_obj['label_document_id']].append(i)

            vanilla_hits = 0
            gold_coref_hits = 0
            gold_coref_recall_hits = 0
            for gt_euid, m_indices in _cluster_dict.items():
                _closest_entities = []
                for _idx in m_indices:
                    _row = mention_entity_dists.getrow(_idx).tocoo()
                    if _row.size == 0:
                        continue
                    _sparse_idx = np.argmin(_row.data)
                    _closest_entities.append(
                            (index2euid[_row.col[_sparse_idx]],
                             _row.data[_sparse_idx])
                    )
                if len(_closest_entities) > 0:
                    euids, entity_dists = zip(*_closest_entities)
                else:
                    continue

                # count vanilla hits
                vanilla_hits += sum([euid == gt_euid for euid in euids])

                # count coref hits
                min_dist_euid, _ = min(zip(euids, entity_dists), key=lambda x : x[1])
                if min_dist_euid == gt_euid:
                    gold_coref_hits += len(m_indices)
                if gt_euid in euids:
                    gold_coref_recall_hits += len(m_indices)

            logger.info('Vanilla Accuracy: {}'.format(vanilla_hits/len(mentions)))
            logger.info('Coref Recall: {}'.format(gold_coref_hits/len(mentions)))
            logger.info('Coref Recall Recall: {}'.format(gold_coref_recall_hits/len(mentions)))

            hits = 0
            external_hits = 0
            for indices in tqdm(I):
                _closest_entities = []
                for _idx in indices:
                    _row = mention_entity_dists.getrow(_idx).tocoo()
                    if _row.size == 0:
                        continue
                    _sparse_idx = np.argmin(_row.data)
                    _closest_entities.append(
                            (_row.col[_sparse_idx], _row.data[_sparse_idx])
                    )

                gt_euid = mentions[indices[0]]['label_document_id']
                entity_indices, entity_dists = zip(*_closest_entities)
                euids = list(map(lambda x : index2euid[x], entity_indices))

                if gt_euid == euids[0]:
                    hits += 1
                else:
                    gt_entity_dist = entity_dists[0]
                    if any([(euid == gt_euid and d < gt_entity_dist)
                            for euid, d in zip(euids[1:], entity_dists[1:])]):
                        external_hits += 1

            logger.info('Vanilla Accuracy: {}'.format(hits/len(mentions)))
            logger.info('Recall k={}: {}'.format(k, (hits + external_hits)/len(mentions)))


            ##################################################################


            if threshold is None:
                logger.info('Running k-means to generate list of thresholds'
                            ' to try...')
                #kmeans = KMeans(n_clusters=500, random_state=0)
                #kmeans.fit(xdoc_mention_mst.data.reshape(-1, 1))
                #thresholds = kmeans.cluster_centers_.reshape(-1,).tolist()
                thresholds = sorted(xdoc_mention_mst.data)[:500]

                scores = []
                for t in tqdm(thresholds, desc='Thresholds'):
                    s = self.compute_linking_accuracy(
                            filter_xdoc_mention_mst,
                            sparse_mention_entity_dists,
                            t,
                            mentions,
                            index2euid
                    )
                    scores.append((t, s))

                threshold, accuracy = max(scores, key=lambda x : x[1])
            else:
                accuracy = self.compute_linking_accuracy(
                        xdoc_mention_mst,
                        sparse_mention_entity_dists,
                        threshold,
                        mentions,
                        index2euid
                )

            logger.info('Threshold: {}'.format(threshold))
            logger.info('Accuracy: {}'.format(accuracy))

            embed()
            
        dist.barrier()
        exit()

class MentionClusteringTrainer(Trainer):

    def __init__(self, args):
        super(MentionClusteringTrainer, self).__init__(args)

    def create_model(self, args):
        self.model = MirrorStackEmbeddingModel(args)

    def create_train_dataloader(self, args):
        self.train_dataloader = self.train_dataloader()

    def create_train_eval_dataloader(self, args):
        (self.train_eval_document2mentions,
         self.train_eval_mention_dataset) = self.eval_data(split='train')
    
    def create_val_dataloader(self, args):
        (self.val_document2mentions,
         self.val_mention_dataset) = self.eval_data(split='val')

    def create_test_dataloader(self, args):
        (self.test_document2mentions,
         self.test_mention_dataset) = self.eval_data(split='test')

    def compute_scores_for_inference(self,
                                     mention_dataset,
                                     evaluate=False):
        args = self.args
        self.model.eval()

        mention_sampler = DistributedSampler(
                mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=False
            )
        mention_dataloader = DataLoader(
                mention_dataset,
                sampler=mention_sampler,
                batch_size=args.infer_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=False,
            )

        mention_ids = []
        mention_indices = []
        mention_embeddings = []
        
        mention_batch_iterator = tqdm(mention_dataloader,
                                      desc='Mention',
                                      disable=(not evaluate
                                               or args.disable_logging))
        for batch in mention_batch_iterator:
            batch = tuple(t.to(args.device, non_blocking=True) for t in batch)
            with torch.no_grad():
                serialized_mention_id = batch[0].cpu().numpy().tolist()
                _mention_ids = [bytes_to_id(x) for x in serialized_mention_id]
                _mention_indices = batch[1].cpu().numpy().tolist()
                inputs = {'input_ids_a':      batch[2],
                          'attention_mask_a': batch[3],
                          'token_type_ids_a': batch[4]}
                outputs = self.model(**inputs)
                mention_embeddings.append(outputs.cpu() if evaluate else outputs)
                mention_ids.extend(_mention_ids)
                mention_indices.extend(_mention_indices)   

        mention_embeddings = torch.cat(mention_embeddings, 0)

        gathered_data = all_gather({
                'mention_ids': mention_ids,
                'mention_indices': mention_indices,
                'mention_embeddings': mention_embeddings
            })

        if args.local_rank not in [0, -1]:
            dist.barrier()
            return

        mention_ids = flatten([d['mention_ids'] for d in gathered_data])
        mention_ids, unique_indices = np.unique(mention_ids, return_index=True)
        mention_indices = flatten([d['mention_indices'] for d in gathered_data])
        mention_indices = np.asarray(mention_indices)[unique_indices]
        mention_embeddings = torch.cat([d['mention_embeddings']
                                            for d in gathered_data], 0)
        mention_embeddings = mention_embeddings[unique_indices]
        mention_embeddings = mention_embeddings.to(args.device) if evaluate else mention_embeddings

        mention2local_indices = {uid : i for i, uid in enumerate(mention_ids)}
        local_indices2mention = np.asarray(mention_ids)
        mention2dataset_indices = {uid : i for uid, i in zip(mention_ids, mention_indices)}

        if evaluate:
            dist.barrier()
            return (mention2local_indices,
                    local_indices2mention,
                    mention_embeddings)

        mention_mention_scores = 1.0-(mention_embeddings @ mention_embeddings.T)

        dist.barrier()
        return (mention2local_indices,
                local_indices2mention,
                mention2dataset_indices,
                mention_mention_scores)

    def build_sl_pairs_train_dataset(self,
                                     mention2local_indices,
                                     local_indices2mention,
                                     mention2dataset_indices,
                                     mention_mention_scores,
                                     clusters):

        args = self.args
        mention_mention_scores = mention_mention_scores.cpu().numpy()

        mention2mention_tuples = []
        
        for _, muids in clusters.items():
            # create per cluster masks and indices
            in_cluster_indices = np.asarray([mention2local_indices[x] for x in muids])

            in_cluster_mask = np.zeros(mention_mention_scores.shape[0], dtype=bool)
            in_cluster_mask[in_cluster_indices] = True

            out_cluster_indices = np.where(~in_cluster_mask)[0].tolist()

            #################################
            ##### mention2mention pairs #####
            #################################
            in_cluster_adj_matrix = csr_matrix(
                    np.triu(mention_mention_scores[in_cluster_mask].T[in_cluster_mask].T, k=1))
            mst = minimum_spanning_tree(in_cluster_adj_matrix).toarray()
            sym_mst = mst + mst.T

            out_cluster_scores = mention_mention_scores[in_cluster_mask].T[~in_cluster_mask].T
            out_cluster_closest = np.argmin(out_cluster_scores, axis=1).tolist()

            pos_edge_to_add_mask = (np.min(out_cluster_scores, axis=1) < sym_mst)
            pos_edge_to_add_mask |= (np.min(out_cluster_scores, axis=1)
                                     < (1.0*args.max_in_cluster_dist))
            pos_edge_to_add_mask |= (sym_mst > args.max_in_cluster_dist)
            pos_edge_to_add_mask &= (sym_mst > 0)
            pos_edge_to_add_mask = pos_edge_to_add_mask | pos_edge_to_add_mask.T

            for i, row_mask in enumerate(pos_edge_to_add_mask):
                if np.any(row_mask):
                    _pos_tgt_indices = np.where(row_mask)[0]
                    mention2mention_tuples.extend([
                        (1.0,
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[x]]])
                            for x in _pos_tgt_indices])
                    mention2mention_tuples.append(
                            (-1.0,
                             mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                             mention2dataset_indices[local_indices2mention[out_cluster_indices[out_cluster_closest[i]]]]))

        return mention2mention_tuples

    def build_exp_link_pairs_train_dataset(self,
                                           mention2local_indices,
                                           local_indices2mention,
                                           mention2dataset_indices,
                                           mention_mention_scores,
                                           clusters):
        with torch.no_grad():
            args = self.args
            num_mentions = mention_mention_scores.shape[0]
            mention_mention_scores = mention_mention_scores[None,:,:]
            alpha_mention_mention_scores = args.alpha * mention_mention_scores
            mention2mention_tuples = []

            # convert the gold clusters to a nice binary format
            gold_clusters = []
            for _muids in clusters.values():
                _cluster_indices = [mention2local_indices[m] for m in _muids]
                _tmp_array = np.zeros(num_mentions, dtype=np.uint8)
                _tmp_array[_cluster_indices] = 1
                gold_clusters.append(_tmp_array)
            gold_clusters = np.asarray(gold_clusters)
            gold_clusters = torch.from_numpy(gold_clusters).to(args.device)

            # perform exp link HAC
            curr_clusters = torch.eye(num_mentions, dtype=torch.uint8, device=args.device)
            while True:
                # compute the best merge
                merges_gen = combinations(curr_clusters, 2)
                pure_merges, impure_merges = [], []
                for a, b in merges_gen:
                    a_b = a + b
                    if any((gold_clusters - a_b < 255).all(1)):
                        pure_merges.append((a, b))
                    else:
                        impure_merges.append((a, b))

                if len(pure_merges) == 0:
                    break

                pure_a, pure_b = zip(*pure_merges)
                pure_a, pure_b = torch.stack(pure_a), torch.stack(pure_b)
                impure_a, impure_b = zip(*impure_merges)
                impure_a, impure_b = torch.stack(impure_a), torch.stack(impure_b)

                pure_masks = pure_a[:,:,None] & pure_b[:,None,:]
                impure_masks = impure_a[:,:,None] & impure_b[:,None,:]

                # compute linkage scores for pure merges
                pure_log_weights = alpha_mention_mention_scores * pure_masks
                pure_max_amm_scores = mention_mention_scores + 5*(1 - pure_masks)
                pure_max_amm_scores = pure_max_amm_scores.view(pure_masks.shape[0], -1)
                pure_max_amm_scores = args.alpha * pure_max_amm_scores.min(1).values[:,None,None]
                pure_log_weights -= torch.log(
                    torch.sum(torch.exp((alpha_mention_mention_scores
                                  - pure_max_amm_scores) * pure_masks)
                                  * pure_masks,
                              dim=(1,2),
                              keepdim=True
                    )
                )
                pure_log_weights -= pure_max_amm_scores
                pure_log_weights *= pure_masks
                pure_log_exp_link = torch.log(mention_mention_scores + 1e-6) + pure_log_weights
                pure_weights = torch.exp(pure_log_weights) * pure_masks
                pure_exp_link = torch.sum(torch.exp(pure_log_exp_link) * pure_masks, axis=(1,2))
                best_pure_index = torch.argmin(pure_exp_link).item()
                best_pure_score = pure_exp_link[best_pure_index].item()

                # compute linkage scores for pure merges
                impure_log_weights = alpha_mention_mention_scores * impure_masks
                impure_max_amm_scores = mention_mention_scores + 5*(1 - impure_masks)
                impure_max_amm_scores = impure_max_amm_scores.view(impure_masks.shape[0], -1)
                impure_max_amm_scores = args.alpha * impure_max_amm_scores.min(1).values[:,None,None]
                impure_log_weights -= torch.log(
                    torch.sum(torch.exp((alpha_mention_mention_scores
                                  - impure_max_amm_scores) * impure_masks)
                                  * impure_masks,
                              dim=(1,2),
                              keepdim=True
                    )
                )
                impure_log_weights -= impure_max_amm_scores
                impure_log_weights *= impure_masks
                impure_log_exp_link = torch.log(mention_mention_scores + 1e-6) + impure_log_weights
                impure_weights = torch.exp(impure_log_weights) * impure_masks
                impure_exp_link = torch.sum(torch.exp(impure_log_exp_link) * impure_masks, axis=(1,2))
                best_impure_index = torch.argmin(impure_exp_link).item()
                best_impure_score = impure_exp_link[best_impure_index].item()

                if (best_pure_score + args.margin > best_impure_score
                    or best_pure_score > args.max_in_cluster_dist):

                    # add positive pairs
                    best_pure_a = np.nonzero(pure_a[best_pure_index])[0]
                    best_pure_b = np.nonzero(pure_b[best_pure_index])[0]

                    for index_a, index_b in product(best_pure_a, best_pure_b):
                        mention2mention_tuples.append(
                            (pure_weights[best_pure_index, index_a, index_b].item(),
                             mention2dataset_indices[local_indices2mention[index_a]],
                             mention2dataset_indices[local_indices2mention[index_b]])
                        )
                
                    # add negative pairs
                    best_impure_a = np.nonzero(impure_a[best_impure_index])[0]
                    best_impure_b = np.nonzero(impure_b[best_impure_index])[0]

                    for index_a, index_b in product(best_impure_a, best_impure_b):
                        mention2mention_tuples.append(
                            (-1.0 * impure_weights[best_impure_index, index_a, index_b].item(),
                             mention2dataset_indices[local_indices2mention[index_a]],
                             mention2dataset_indices[local_indices2mention[index_b]])
                        )

                # update current clusters
                old_cluster_a = pure_a[best_pure_index]
                old_cluster_b = pure_b[best_pure_index]
                row_cond = ~((curr_clusters == old_cluster_a).all(1)
                             | (curr_clusters == old_cluster_b).all(1))
                curr_clusters = torch.cat(
                        (curr_clusters[row_cond, :],
                         (old_cluster_a + old_cluster_b)[None,:])
                )

            return mention2mention_tuples

    def build_avg_link_pairs_train_dataset(self,
                                           mention2local_indices,
                                           local_indices2mention,
                                           mention2dataset_indices,
                                           mention_mention_scores,
                                           clusters):

        args = self.args

        mention2mention_tuples = []
        
        for _, muids in clusters.items():
            # create per cluster masks and indices
            in_cluster_indices = np.asarray([mention2local_indices[x] for x in muids])

            in_cluster_mask = np.zeros(mention_mention_scores.shape[0], dtype=bool)
            in_cluster_mask[in_cluster_indices] = True

            out_cluster_indices = np.where(~in_cluster_mask)[0].tolist()

            #################################
            ##### mention2mention pairs #####
            #################################
            in_cluster_scores = mention_mention_scores[in_cluster_mask].T[in_cluster_mask].T
            pos_edge_to_add_mask = in_cluster_scores > args.max_in_cluster_dist

            for i, row_mask in enumerate(pos_edge_to_add_mask):
                if np.any(row_mask):
                    _pos_tgt_indices = np.where(row_mask)[0]
                    mention2mention_tuples.extend([
                        (1.0,
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[x]]])
                            for x in _pos_tgt_indices])

            out_cluster_scores = mention_mention_scores[in_cluster_mask].T[~in_cluster_mask].T
            neg_edge_to_add_mask = out_cluster_scores < 2.0 * args.max_in_cluster_dist

            for i, row_mask in enumerate(neg_edge_to_add_mask):
                if np.any(row_mask):
                    _neg_tgt_indices = np.where(row_mask)[0]
                    mention2mention_tuples.extend([
                        (-1.0,
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                         mention2dataset_indices[local_indices2mention[out_cluster_indices[x]]])
                            for x in _neg_tgt_indices])

        return mention2mention_tuples
    
    def train_on_pairs(self, mention2mention_dataset):
        args = self.args
        self.model.train()
        self.model.zero_grad()

        num_mention2mention_pairs = len(mention2mention_dataset)

        mention2mention_sampler = DistributedSampler(
                mention2mention_dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=True
            )
        mention2mention_dataloader = DataLoader(
                mention2mention_dataset,
                sampler=mention2mention_sampler,
                batch_size=args.train_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=False
            )


        total_loss = 0.0

        for batch in mention2mention_dataloader:
            batch = tuple(t.to(args.device) for t in batch)

            _coeff = batch[0]
            _batch_size = _coeff.shape[0]

            # concat first and second mentions in pairs together
            mention_inputs = {'input_ids_a':      batch[3],
                              'attention_mask_a': batch[4],
                              'token_type_ids_a': batch[5],
                              'input_ids_b':      batch[8],
                              'attention_mask_b': batch[9],
                              'token_type_ids_b': batch[10]}

            scores = torch.diag(self.model(**mention_inputs))

            loss = torch.sum(_coeff * scores) / num_mention2mention_pairs
            total_loss += loss.item()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

        return total_loss

    def train_step(self, document_id, mention_dataset, clusters):
        args = self.args

        # NOTE: outputs will be `None` for all local_rank != 0
        outputs = self.compute_scores_for_inference(mention_dataset)

        mention2mention_tuples = None
        if args.local_rank == 0:
            (mention2local_indices,
             local_indices2mention,
             mention2dataset_indices,
             mention_mention_scores) = outputs

            # compute pairs for clustering on rank 0 process

            mention2mention_tuples = self.build_sl_pairs_train_dataset(
                     mention2local_indices,
                     local_indices2mention,
                     mention2dataset_indices,
                     mention_mention_scores,
                     clusters)
            #mention2mention_tuples = self.build_exp_link_pairs_train_dataset(
            #         mention2local_indices,
            #         local_indices2mention,
            #         mention2dataset_indices,
            #         mention_mention_scores,
            #         clusters)

            ## make the distributed sampler happy
            #while (len(mention2mention_tuples) > 0 
            #       and len(mention2mention_tuples) < args.world_size):
            #    mention2mention_tuples.extend(mention2mention_tuples)

        dist.barrier()

        # broadcast pairs to all processes
        mention2mention_tuples = broadcast(mention2mention_tuples, src=0)

        loss = 0.0
        if len(mention2mention_tuples) > 0:
            # build the dataset on all ranks
            mention2mention_dataset = ClusterLinkingLazyTrainDataset(
                mention2mention_tuples, args.cached_train_mention_examples_dir,
                args.cached_train_mention_examples_dir)

            # distributed training on the pairs dataset we just built
            loss = self.train_on_pairs(mention2mention_dataset)

        return {'num_pairs': len(mention2mention_tuples), 'loss': loss}

    def train(self):
        args = self.args

        logger.info("***** Running training *****")
        logger.info("  Num documents = %d", len(self.train_dataloader))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * args.gradient_accumulation_steps * (dist.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.t_total)

        args.disable_logging |= args.local_rank not in [-1, 0]

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        tr_m2m_pairs, logging_m2m_pairs = 0, 0
        best_val_performance = None
        best_step = 0
        epoch_iterator = trange(int(args.num_train_epochs),
                                desc="Epoch",
                                disable=args.disable_logging)

        # create cluster index for training
        cluster_index = create_cluster_index(args.data_dir, 'train')

        for epoch in epoch_iterator:
            document_iterator = tqdm(self.train_dataloader,
                                     desc="Document",
                                     disable=args.disable_logging)
            for document_id, mention_dataset in document_iterator:
                dist.barrier()

                # run step on this doc
                clusters = cluster_index[document_id]

                return_dict = self.train_step(document_id, mention_dataset, clusters)

                # for logging 
                tr_loss += return_dict['loss']
                tr_m2m_pairs += return_dict['num_pairs']
                global_step += 1

                # logging stuff for babysitting
                if global_step % args.logging_steps == 0:
                    logger.info("Average loss: %s at global step: %s",
                            str((tr_loss - logging_loss)/args.logging_steps),
                            str(global_step))
                    logger.info("Average num mention2mention pairs: %s at global step: %s",
                                str((tr_m2m_pairs - logging_m2m_pairs)/args.logging_steps),
                                str(global_step))
                    logging_loss = tr_loss
                    logging_m2m_pairs = tr_m2m_pairs

                if global_step % args.save_steps == 0:
                    # evaluate the model
                    if args.evaluate_during_training:
                        logger.info('*** Evaluating checkpoint {} ***'.format(
                            global_step))
                        for split in ['train', 'val']:
                            results = self.evaluate(split=split)
                            if split == 'val':
                                if (best_val_performance is None
                                    or results['avg_fmi'] > best_val_performance['avg_fmi']):
                                    best_val_performance = results
                                    best_step = global_step

                    # save the model
                    if args.local_rank in [0, -1]:
                        self.model.module.save_model(
                                suffix='checkpoint-{}'.format(global_step))
                    dist.barrier()
        
        logger.info('*** Training complete ***')
        if args.evaluate_during_training:
            logger.info('\tbest step: {}'.format(best_step))
            logger.info('\tbest val results:')
            for key, value in best_val_performance.items():
                logger.info('\t\t{}: {}'.format(key, value))

    def evaluate(self, split='', threshold=None, save_dir=None, dump_plots=False):
        assert split == 'train' or split == 'val' or split == 'test'

        if split == 'train':
            document2mentions = self.train_eval_document2mentions
            mention_dataset = self.train_eval_mention_dataset
        elif split == 'val':
            document2mentions = self.val_document2mentions
            mention_dataset = self.val_mention_dataset
        else:
            document2mentions = self.test_document2mentions
            mention_dataset = self.test_mention_dataset

        args = self.args

        # create cluster index for evaluation
        cluster_index = create_cluster_index(args.data_dir, split)

        # load the mention objects for computing accuracy
        mention_file = os.path.join(args.data_dir, 'mentions', split + '.json')
        candidate_file = os.path.join(args.data_dir, 'tfidf_candidates', split + '.json')
        mentions = _read_mentions(mention_file)
        tfidf_candidates = _read_candidates(candidate_file)
        mentions_dict = {m['mention_id']: m for m in mentions}

        # Eval!
        logger.info("***** Running evaluation : {} *****".format(split))
        logger.info("  Num documents = %d", len(document2mentions.keys()))
        logger.info("  Num mentions = %d", len(mention_dataset))
        logger.info("  Batch size = %d", args.infer_batch_size)

        # Iterate over all of the documents and record data structures
        logger.info("Computing embeddings...")
        outputs = self.compute_scores_for_inference(
                 mention_dataset, evaluate=True)

        # nothing for the other processes to do for a moment
        results = None
        if args.local_rank in [-1, 0]:
            (mention2local_indices,
             local_indices2mention,
             mention_embeddings) = outputs

            # for cluster-then-link scoring
            with open('vanilla_linker-experiment/checkpoint-150000/mention_entity_scores.val.pkl', 'rb') as f:
                logger.info('Loading mention entity scores')
                mention_entity_scores = pickle.load(f)

            def get_pred_clustering(mst,
                                    mention_local_indices,
                                    local_indices2mention,
                                    threshold):
                pruned_mst = copy.deepcopy(mst)
                pruned_mst[pruned_mst > threshold] = 0.0
                pruned_mst = csr_matrix(pruned_mst)
                mst = csr_matrix(mst)

                # produce clusters
                n_components, cluster_labels = connected_components(
                    csgraph=pruned_mst, directed=False, return_labels=True)

                pred_cluster_map = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    pred_cluster_map[label].append(local_indices2mention[mention_local_indices[i]])

                pred_mention_cluster_map = {
                        m : k for k, v in pred_cluster_map.items() for m in v
                }
                return pruned_mst, mst, pred_mention_cluster_map

            def get_single_linkage(mention_mention_scores):
                adj_matrix = csr_matrix(np.triu(mention_mention_scores, k=1))
                mst = minimum_spanning_tree(adj_matrix).toarray()
                edge_values = mst[mst != 0.0]
                return mst, edge_values

            def get_exp_linkage(mention_mention_scores):
                edge_values = []    # linkage edge values
                num_mentions = mention_mention_scores.shape[0]
                mention_mention_scores = mention_mention_scores[None,:,:]
                alpha_mention_mention_scores = args.alpha * mention_mention_scores

                mst = torch.zeros((num_mentions, num_mentions), device=args.device)
                curr_clusters = torch.eye(num_mentions, dtype=torch.uint8, device=args.device)
                while curr_clusters.shape[0] > 1:
                    # compute the best merge
                    merges_gen = combinations(curr_clusters, 2)
                    merges_a, merges_b = zip(*merges_gen)
                    merges_a = torch.stack(merges_a)
                    merges_b = torch.stack(merges_b)
                    merges_masks = merges_a[:,:,None] & merges_b[:,None,:]

                    # compute linkage scores for all merges
                    log_weights = alpha_mention_mention_scores * merges_masks
                    max_amm_scores = mention_mention_scores + 5*(1 - merges_masks)
                    max_amm_scores = max_amm_scores.view(merges_masks.shape[0], -1)
                    max_amm_scores = args.alpha * max_amm_scores.min(1).values[:,None,None]
                    log_weights -= torch.log(
                        torch.sum(torch.exp((alpha_mention_mention_scores
                                      - max_amm_scores) * merges_masks)
                                      * merges_masks,
                                  dim=(1,2),
                                  keepdim=True
                        )
                    )
                    log_weights -= max_amm_scores
                    log_weights *= merges_masks
                    log_exp_link = torch.log(mention_mention_scores + 1e-6) + log_weights
                    weights = torch.exp(log_weights) * merges_masks
                    exp_link = torch.sum(torch.exp(log_exp_link) * merges_masks, axis=(1,2))

                    # pick the best merge
                    best_index = torch.argmin(exp_link).item()
                    best_score = exp_link[best_index].item()

                    # add the best score to the edge_values list
                    edge_values.append(best_score)

                    # update mst
                    ravel_best_index = torch.argmax(weights[best_index]).item()
                    best_edge_indices = (ravel_best_index // weights.shape[1],
                                         ravel_best_index % weights.shape[1])
                    mst[best_edge_indices] = best_score

                    # update current clusters
                    old_cluster_a = merges_a[best_index]
                    old_cluster_b = merges_b[best_index]
                    row_cond = ~((curr_clusters == old_cluster_a).all(1)
                                 | (curr_clusters == old_cluster_b).all(1))
                    curr_clusters = torch.cat(
                            (curr_clusters[row_cond, :],
                             (old_cluster_a + old_cluster_b)[None,:])
                    )

                return mst.cpu().numpy(), edge_values
                    

            def get_all_mention2mention_scores():
                mst_edge_values = []
                for document_id in tqdm(document2mentions.keys(), desc='Documents'):
                    doc_mentions = document2mentions[document_id]
                    mention_local_indices = [mention2local_indices[m['mention_id']]
                                                for m in doc_mentions]
                    doc_mention_embeddings = mention_embeddings[mention_local_indices]
                    mention_mention_scores = 1 - (doc_mention_embeddings
                                                 @ doc_mention_embeddings.T)
                    _, edge_values = get_single_linkage(mention_mention_scores.cpu().numpy())
                    #_, edge_values = get_exp_linkage(mention_mention_scores)
                    mst_edge_values.append(edge_values)
                return np.hstack(mst_edge_values)

            def cluster_scores(thresholds):
                all_doc_ids = list(document2mentions.keys())
                doc2mst = {}
                for document_id in tqdm(all_doc_ids, desc='Documents'):
                    doc_mentions = document2mentions[document_id]
                    mention_local_indices = [mention2local_indices[m['mention_id']]
                                                for m in doc_mentions]
                    doc_mention_embeddings = mention_embeddings[mention_local_indices]
                    mention_mention_scores = 1 - (doc_mention_embeddings
                                                 @ doc_mention_embeddings.T)
                    mst, _ = get_single_linkage(mention_mention_scores.cpu().numpy())
                    #mst, _ = get_exp_linkage(mention_mention_scores)

                    doc2mst[document_id] = mst

                #failed_merge_mst_edges = defaultdict(list) # indexed by document
                #failed_split_mst_edges = defaultdict(list) # indexed by document
                #all_mst_edges = defaultdict(list) # indexed by docuemnt

                fmi_scores = defaultdict(list)
                nmi_scores = defaultdict(list)
                rand_index_scores = defaultdict(list)
                accuracy_scores = defaultdict(list)
                for threshold in tqdm(thresholds, desc='Thresholds'):
                    linking_hits = 0
                    for document_id in all_doc_ids:
                        doc_mentions = document2mentions[document_id]
                        mention_local_indices = [mention2local_indices[m['mention_id']]
                                                    for m in doc_mentions]
                        # ground truth clusters
                        clusters = cluster_index[document_id]
                        true_mention_cluster_map = {
                                m : k for k, v in clusters.items() for m in v
                        }

                        ## precomputed mst
                        #mst = doc2mst[document_id]
                        #heads, tails = np.nonzero(mst)
                        #heads = [local_indices2mention[mention_local_indices[i]] for i in heads]
                        #tails = [local_indices2mention[mention_local_indices[i]] for i in tails]
                        #edge_gen = zip(heads, tails, mst[mst != 0])

                        #for head, tail, weight in edge_gen:
                        #    head_cluster = true_mention_cluster_map[head]
                        #    tail_cluster = true_mention_cluster_map[tail]
                        #    if head_cluster == tail_cluster and weight > threshold:
                        #        # missed in cluster edge
                        #        failed_merge_mst_edges[document_id].append(
                        #                (head, tail, weight)
                        #        )
                        #    elif head_cluster != tail_cluster and weight <= threshold:
                        #        # missed cross cluster edge
                        #        failed_split_mst_edges[document_id].append(
                        #                (head, tail, weight)
                        #        )
                        #    all_mst_edges[document_id].append(
                        #            (head, tail, weight)
                        #    )

                        mst = doc2mst[document_id]
                        (pruned_mst,
                         mst,
                         pred_mention_cluster_map) = get_pred_clustering(
                                mst, mention_local_indices,
                                local_indices2mention, threshold
                        )


                        ### for cluster-then-link experiment
                        cluster_inverted_index = defaultdict(list)
                        for muid, cluster_id in pred_mention_cluster_map.items():
                            cluster_inverted_index[cluster_id].append(muid)

                        for cluster_id, muids in cluster_inverted_index.items():
                            tmp_dict = defaultdict(float)
                            for muid in muids:
                                tmp_dict = dict_merge_with(tmp_dict, mention_entity_scores[muid])
                            pred_entity, _ = max(tmp_dict.items(), key = lambda x : x[1])
                            for muid in muids:
                                if mentions_dict[muid]['label_document_id'] == pred_entity:
                                    linking_hits += 1
                        ###

                        # convert to lists of indices for assigments
                        true_labels, pred_labels = [], []
                        cluster_inverted_index = {}
                        cluster_counter = 0
                        for muid in sorted(true_mention_cluster_map.keys()):
                            pred_labels.append(pred_mention_cluster_map[muid])
                            _entity_label = true_mention_cluster_map[muid]
                            if _entity_label in cluster_inverted_index.keys():
                                true_labels.append(
                                        cluster_inverted_index[_entity_label])
                            else:
                                cluster_inverted_index[_entity_label] = cluster_counter
                                true_labels.append(cluster_counter)
                                cluster_counter += 1

                        # compute and save scores
                        fmi_scores[threshold].append(
                                fowlkes_mallows_score(true_labels, pred_labels))
                        nmi_scores[threshold].append(
                                normalized_mutual_info_score(true_labels, pred_labels))
                        rand_index_scores[threshold].append(
                                adjusted_rand_score(true_labels, pred_labels))
                        if dump_plots:
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            dot_file = os.path.join(
                                    save_dir,
                                    '.'.join([document_id, 'mst.dot'])
                            )
                            Graphviz.write_mst(dot_file,
                                               mentions_dict,
                                               mention_local_indices,
                                               local_indices2mention,
                                               mst,
                                               pruned_mst)

                    # compute and save accuracy
                    accuracy_scores[threshold].append(
                            linking_hits / len(mentions_dict.keys())) 
                    if dump_plots:
                        logger.info('Some MSTs to check out: {}'.format(
                                np.asarray(all_doc_ids)[
                                    np.argpartition(fmi_scores, 10)[:10]
                                ].tolist()
                            )
                        )

                #with open('mst_edges_data.val.pkl', 'wb') as f:
                #    pickle.dump(
                #        {
                #            'all_mst_edges' : all_mst_edges,
                #            'failed_merge_mst_edges' : failed_merge_mst_edges,
                #            'failed_split_mst_edges' : failed_split_mst_edges
                #        }, f, pickle.HIGHEST_PROTOCOL)

                results = {'fmi': fmi_scores,
                           'nmi': nmi_scores,
                           'rand_index': rand_index_scores,
                           'accuracy': accuracy_scores}

                return results

            if threshold is None:
                logger.info('Choosing threshold...')
                all_edge_scores = get_all_mention2mention_scores()
                logger.info('Running k-means...')
                kmeans = KMeans(n_clusters=50, random_state=0)
                kmeans.fit(all_edge_scores.reshape(-1, 1))
                thresholds = kmeans.cluster_centers_.reshape(-1,).tolist()
                logger.info('Computing cluster scores for each threshold...')
                results = cluster_scores(thresholds)
                results = {metric_name : {threshold : np.mean(values)
                                          for threshold, values in metric_values.items()}
                           for metric_name, metric_values in results.items()}
                #scores = [(threshold, results['fmi'][threshold]
                #                      * results['nmi'][threshold]
                #                      * results['rand_index'][threshold])
                #                for threshold in results['fmi'].keys()]
                scores = [(threshold, results['accuracy'][threshold])
                                for threshold in results['accuracy'].keys()]
                threshold, _ = max(scores, key = lambda x : x[1])
                logger.info('Chosen threshold: {}'.format(threshold))

            logger.info('Evalutating threshold: {}'.format(threshold))
            results = cluster_scores([threshold])
            results = {
                    'threshold': threshold,
                    'mean_fmi': np.mean(results['fmi'][threshold]),
                    'mean_nmi': np.mean(results['nmi'][threshold]),
                    'mean_rand_index': np.mean(results['rand_index'][threshold]),
                    'accuracy': results['accuracy'][threshold][0]
            }

            logger.info('Evaluation Results : {}'.format(split))
            for key, value in results.items():
                logger.info('\t{}: {}'.format(key, value))
            embed()

        dist.barrier()
        exit()
        results = broadcast(results, src=0)

        return results

    def train_dataloader(self):
        args = self.args
        dataset = self.load_and_cache_examples(split='train', evaluate=False)

        def collate_fn(batch):
            assert len(batch) == 1
            return batch[0]

        return DataLoader(dataset,
                          batch_size=1,
                          collate_fn=collate_fn,
                          num_workers=args.num_dataloader_workers,
                          pin_memory=True,
                          shuffle=True)

    def eval_data(self, split=None):
        assert split == 'train' or split == 'val' or split =='test'
        args = self.args
        return self.load_and_cache_examples(split=split, evaluate=True)

    def create_or_load_processed_data(self,
                                      split,
                                      domains,
                                      cache_dir,
                                      split_desc,
                                      evaluate):
        args = self.args

        cached_mention_examples_dir = os.path.join(
                cache_dir, split_desc + '_mention_examples')
        cached_document_indices_dir = os.path.join(
                cache_dir, split_desc + '_document_indices')

        if split == 'train':
            args.cached_train_mention_examples_dir = cached_mention_examples_dir

        if (os.path.exists(cached_document_indices_dir) 
            and not args.overwrite_cache):
            document_indices_files = os.listdir(cached_document_indices_dir)
            indices_dicts = []
            for filename in document_indices_files:
                indices_dicts.append(torch.load(os.path.join(cached_document_indices_dir, filename)))
        else:
            if not os.path.exists(cached_mention_examples_dir):
                os.makedirs(cached_mention_examples_dir)
            if not os.path.exists(cached_document_indices_dir):
                os.makedirs(cached_document_indices_dir)

            processor = MentionClusteringProcessor()

            indices_dicts = processor.get_document_datasets(
                args.data_dir,
                split,
                domains,
                cached_mention_examples_dir,
                cached_document_indices_dir,
                args.max_seq_length,
                self.model.tokenizer,
                evaluate=evaluate)

        if evaluate:
            _indices_dict = indices_dicts[0]
            document2mentions = _indices_dict['document2mentions']
            mention_dataset = LazyDataset(cached_mention_examples_dir,
                                          _indices_dict['mention_indices'])
            return (document2mentions,
                    mention_dataset)

        document_ids = []
        mention_datasets = []
        for _indices_dict in indices_dicts:
            document_ids.append(_indices_dict['document_id'])
            mention_datasets.append(LazyDataset(cached_mention_examples_dir,
                                                _indices_dict['mention_indices']))

        return list(zip(document_ids, mention_datasets))


class VanillaLinkingTrainer(Trainer):

    def __init__(self, args):
        super(VanillaLinkingTrainer, self).__init__(args)

    def create_models(self):
        args = self.args
        if args.task_name == 'vanilla_linking':
            self.models = {
                    'linking_model': ConcatenationModel(args)
            }
        elif args.task_name == 'poly_linking':
            self.models = {
                    'linking_model': PolyEncoder(args)
            }
        else:
            raise ValueError('Invalid task name: {}'.format(args.task_name))

    def create_train_dataloader(self):
        self.train_dataloader = self.create_dataloader(split='train',
                                                       evaluate=False)

    def create_train_eval_dataloader(self):
         self.train_eval_dataloader = self.create_dataloader(split='train')
    
    def create_val_dataloader(self):
        self.val_dataloader = self.create_dataloader(split='val')

    def create_test_dataloader(self):
         self.test_dataloader = self.create_dataloader(split='test')

    def create_dataloader(self, split=None, evaluate=True):
        args = self.args
        dataset = self.load_and_cache_examples(split=split, evaluate=evaluate)
        #dataset = [dataset[i] for i in range(10)] * 100
        sampler = DistributedSampler(
                dataset,
                num_replicas=args.world_size,
                rank=args.local_rank,
                shuffle=(not evaluate)
            )
        dataloader = DataLoader(
                dataset,
                sampler=sampler,
                batch_size=args.infer_batch_size if evaluate else args.train_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True
            )
        return dataloader

    def create_or_load_processed_data(self,
                                      split,
                                      domains,
                                      cache_dir,
                                      split_desc,
                                      evaluate):
        args = self.args

        cached_indices_file = os.path.join(cache_dir, split_desc + '.indices.pt')
        cached_examples_dir = os.path.join(cache_dir, split_desc + '_examples')

        if not os.path.exists(cached_examples_dir):
            os.makedirs(cached_examples_dir)

        # Load data features from cache or dataset file
        if os.path.exists(cached_indices_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_indices_file)
            indices = torch.load(cached_indices_file)
        else:
            processor = LinkingProcessor()

            logger.info("Creating features from dataset file at %s", args.data_dir)
            examples, documents = processor.get_examples(
                    args.data_dir, args.num_candidates, 
                    args.num_candidates_per_example,
                    split, domains,
                    evaluate, use_true_labels=(not evaluate))

            logger.info("Number of examples: %s", str(len(examples)))

            if args.task_name == 'vanilla_linking':
                convert_examples_to_features = processor.convert_examples_to_concatenation_features
            elif args.task_name == 'poly_linking':
                convert_examples_to_features = processor.convert_examples_to_poly_features

            indices = convert_examples_to_features(
                examples,
                documents,
                cached_examples_dir,
                args.max_seq_length,
                self.models['linking_model'].tokenizer,
                evaluate,
                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0
            )
            logger.info("Saving features into cached file %s", cached_indices_file)
            torch.save(indices, cached_indices_file)

        dataset = LazyDataset(cached_examples_dir, indices)

        return dataset

    def train(self):
        args = self.args
        self.models['linking_model'].train()
        self.models['linking_model'].zero_grad()

        logger.info("***** Running training *****")
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Num Examples = %d", len(self.train_dataloader))
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                       args.train_batch_size * args.gradient_accumulation_steps * (dist.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", args.t_total)

        args.disable_logging |= args.local_rank not in [-1, 0]
        epoch_iterator = trange(int(args.num_train_epochs),
                                desc="Epoch",
                                disable=args.disable_logging)
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0

        for epoch in epoch_iterator:
            batch_iterator = tqdm(self.train_dataloader,
                                  desc="Batch",
                                  disable=args.disable_logging)
            for batch in batch_iterator:
                self.models['linking_model'].train()
                batch = tuple(t.to(args.device) for t in batch)

                # concat first and second mentions in pairs together
                if args.task_name == 'vanilla_linking':
                    inputs = {'input_ids':      batch[3],
                              'attention_mask': batch[4],
                              'token_type_ids': batch[5],
                              'labels': batch[2]}
                elif args.task_name == 'poly_linking':
                    inputs = {'mention_input_ids':      batch[3],
                              'mention_attention_mask': batch[4],
                              'mention_token_type_ids': batch[5],
                              'entity_input_ids':      batch[6],
                              'entity_attention_mask': batch[7],
                              'entity_token_type_ids': batch[8],
                              'labels': batch[2]}

                loss, _ = self.models['linking_model'](**inputs)

                #if args.local_rank in [-1, 0]:
                #    embed()
                #dist.barrier()
                #exit()

                # for logging 
                tr_loss += loss.item()
                global_step += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                        self.models['linking_model'].parameters(),
                        args.max_grad_norm
                )
                self.optimizers['linking_model'].step()
                self.schedulers['linking_model'].step()
                self.models['linking_model'].zero_grad()

                # logging stuff for babysitting
                if global_step % args.logging_steps == 0:
                    logger.info("Average loss: %s at global step: %s",
                            str((tr_loss - logging_loss)/args.logging_steps),
                            str(global_step))
                    logging_loss = tr_loss

                # save the model
                if global_step % args.save_steps == 0:
                    # evaluate the model
                    if args.evaluate_during_training:
                        logger.info('*** Evaluating checkpoint {} ***'.format(
                            global_step))
                        for split in ['train', 'val']:
                            results = self.evaluate(split=split)
                            if split == 'val':
                                if (best_val_performance is None
                                    or results['accuracy'] > best_val_performance['accuracy']):
                                    best_val_performance = results
                                    best_step = global_step

                    if args.local_rank in [0, -1]:
                        logger.info('Saving checkpoint at global step: {}'.format(global_step))
                        self.models['linking_model'].module.save_model(
                                suffix='checkpoint-{}'.format(global_step))
                    dist.barrier()

        # save the final model
        if args.local_rank in [0, -1]:
            logger.info('Saving checkpoint at global step: {}'.format(global_step))
            self.models['linking_model'].module.save_model(
                    suffix='checkpoint-{}'.format(global_step))
        dist.barrier()

    def evaluate(self, split=''):
        args = self.args
        assert split == 'train' or split == 'val' or split == 'test'

        results = {}
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        if split == 'train':
            eval_dataloader = self.train_eval_dataloader
        elif split == 'val':
            eval_dataloader = self.val_dataloader
        else:
            eval_dataloader = self.test_dataloader

        # Eval!
        logger.info("***** Running evaluation {} *****".format(split))
        logger.info("  Batch size = %d", args.infer_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # for computing accuracy
        hit_by_mention_id = {}
        mention_entity_scores = defaultdict(dict)

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            self.models['linking_model'].eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                serialized_mention_id = batch[0].cpu().numpy().tolist()
                serialized_entity_id = batch[1].cpu().numpy().tolist()

                mention_ids = [[bytes_to_id(byte_set) for byte_set in example_mention_id]
                                for example_mention_id in serialized_mention_id]
                entity_ids = [[bytes_to_id(byte_set) for byte_set in example_entity_id]
                                for example_entity_id in serialized_entity_id]

                
                if args.task_name == 'vanilla_linking':
                    inputs = {'input_ids':      batch[3],
                              'attention_mask': batch[4],
                              'token_type_ids': batch[5],
                              'labels': batch[2],
                              'evaluate': True}
                elif args.task_name == 'poly_linking':
                    inputs = {'mention_input_ids':      batch[3],
                              'mention_attention_mask': batch[4],
                              'mention_token_type_ids': batch[5],
                              'entity_input_ids':      batch[6],
                              'entity_attention_mask': batch[7],
                              'entity_token_type_ids': batch[8],
                              'labels': batch[2],
                              'mention_ids': mention_ids,
                              'entity_ids': entity_ids,
                              'evaluate': True}

                loss, scores = self.models['linking_model'](**inputs)

                scores = scores.cpu().numpy()
                labels = batch[2].cpu().numpy()

                indices = np.expand_dims(np.argmax(scores, axis=1), axis=1)
                max_labels = np.take_along_axis(labels, indices, axis=1)
                max_labels = max_labels.squeeze().tolist()

                gathered_data = all_gather({
                        'mention_ids': mention_ids,
                        'entity_ids': entity_ids,
                        'max_labels': max_labels,
                        'scores': scores
                    })

                #if nb_eval_steps > 100:
                #    break

                if args.local_rank not in [0, -1]:
                    dist.barrier()
                    continue

                mention_ids = flatten([d['mention_ids'] for d in gathered_data])
                entity_ids = flatten([d['entity_ids'] for d in gathered_data])
                max_labels = flatten([d['max_labels'] for d in gathered_data])
                scores = np.concatenate([d['scores'] for d in gathered_data], axis=0)

                _mention_ids = [uid[0] for uid in mention_ids]
                for mention_id, max_label in zip(_mention_ids, max_labels):
                    if (mention_id not in hit_by_mention_id.keys() or
                            hit_by_mention_id[mention_id] == 1):
                        hit_by_mention_id[mention_id] = max_label

                for i in range(scores.shape[0]):
                    for j in range(scores.shape[1]):
                        mention_entity_scores[mention_ids[i][0]][entity_ids[i][j]] = scores[i][j]
                dist.barrier()

                nb_eval_steps += 1


        if args.local_rank in [0, -1]:


            hits = sum([v for _, v in hit_by_mention_id.items()])
            total = len(hit_by_mention_id.keys())
            accuracy = hits / total

            dump_file = os.path.join(args.output_dir, 'mention_entity_scores.{}.pkl'.format(split))
            logger.info('Dumping mention-entity linking {} scores to {}'.format(split, dump_file))
            with open(dump_file, 'wb') as f:
                pickle.dump(mention_entity_scores, f, pickle.HIGHEST_PROTOCOL)

            result = {"accuracy": accuracy}
            results.update(result)

            logger.info('Evaluation Results : {}'.format(split))
            for key, value in results.items():
                logger.info('\t{}: {}'.format(key, value))

        dist.barrier()

        results = broadcast(results, src=0)
        return results
