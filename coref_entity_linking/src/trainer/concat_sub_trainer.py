from collections import defaultdict
import logging
import numpy as np
from scipy.sparse import coo_matrix
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random

from data.datasets import PairsConcatenationDataset
from data.dataloaders import (PairsConcatenationDataLoader,
                              TripletConcatenationDataLoader,
                              SoftmaxConcatenationDataLoader,
                              ScaledPairsConcatenationDataLoader)
from utils.comm import get_rank, all_gather, synchronize, broadcast
from utils.misc import flatten, unique

from IPython import embed


logger = logging.getLogger(__name__)


class ConcatenationSubTrainer(object):
    """
    Class to help with training and evaluation processes.
    """
    def __init__(self, args, model, optimizer, scheduler):
        # we assume that `model` is a DDP pytorch model and is on the GPU
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler


    def compute_scores_for_inference(self, clusters_mx, per_example_negs):
        # TODO: add description here
        args = self.args

        edges = None
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
        edges = broadcast(edges, src=0)

        example_dir = args.train_cache_dir
        affinities = self.get_edge_affinities(edges, example_dir, None)

        sparse_graph = None
        if get_rank() == 0:
            # convert to coo_matrix
            edges = np.asarray(edges).T
            affinities = np.asarray(affinities)
            _sparse_num = np.max(edges) + 1
            sparse_graph = coo_matrix((affinities, edges),
                                      shape=(_sparse_num, _sparse_num))

        synchronize()
        return sparse_graph

    def train_on_subset(self, dataset_list, metadata):
        args = self.args
        if 'triplet' in args.training_method:
            return self._train_triplet(dataset_list, metadata)
        elif args.training_method == 'softmax':
            return self._train_softmax(dataset_list, metadata)
        elif args.training_method == 'threshold':
            return self._train_threshold(dataset_list, metadata)
        else:
            raise ValueError('unsupported training_method')

    def _train_triplet(self, dataset_list, metadata):
        args = self.args

        losses = [] 
        time_per_dataset = []
        dataset_sizes = []
        pos_m_neg_m_losses = []
        pos_m_neg_e_losses = []
        pos_e_neg_m_losses = []
        pos_e_neg_e_losses = []

        self.model.train()
        self.model.zero_grad()
        for dataset in dataset_list:
            _dataset_start_time = time.time()
            dataset_sizes.append(len(dataset))
            dataloader = TripletConcatenationDataLoader(args, dataset)
            for batch in dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                m_e_mask = torch.any(torch.any(batch[0] < args.num_entities, -1), -1).to(args.device)
                inputs = {'m_e_mask':       m_e_mask,
                          'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3]}
                outputs = self.model(**inputs)

                scores = torch.mean(outputs, -1)
                scores = torch.sigmoid(scores)

                if args.training_method == 'triplet_max_margin':
                    # max-margin
                    per_triplet_loss = F.relu(
                            scores[:, 1]   # negative dot products
                            - scores[:, 0] # positive dot products
                            + args.margin
                    )
                elif args.training_method == 'triplet_bpr':
                    # BPR
                    per_triplet_loss = torch.sigmoid(
                            scores[:, 1]   # negative dot products
                            - scores[:, 0] # positive dot products
                            + args.margin
                    )
                else:
                    raise ValueError('unsupported training_method')

                # record triplet specific losses
                _detached_per_triplet_loss = per_triplet_loss.clone().detach().cpu()

                _mask = batch[0] < metadata.num_entities
                pos_m_neg_m_mask = ~_mask[:, 0, 0, 1] & ~_mask[:, 1, 0, 1]
                pos_m_neg_e_mask = ~_mask[:, 0, 0, 1] & _mask[:, 1, 0, 1]
                pos_e_neg_m_mask = _mask[:, 0, 0, 1] & ~_mask[:, 1, 0, 1]
                pos_e_neg_e_mask = _mask[:, 0, 0, 1] & _mask[:, 1, 0, 1]

                pos_m_neg_m_losses.extend(_detached_per_triplet_loss[pos_m_neg_m_mask].numpy().tolist())
                pos_m_neg_e_losses.extend(_detached_per_triplet_loss[pos_m_neg_e_mask].numpy().tolist())
                pos_e_neg_m_losses.extend(_detached_per_triplet_loss[pos_e_neg_m_mask].numpy().tolist())
                pos_e_neg_e_losses.extend(_detached_per_triplet_loss[pos_e_neg_e_mask].numpy().tolist())
                loss = torch.mean(per_triplet_loss)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    args.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            time_per_dataset.append(time.time() - _dataset_start_time)

        gathered_data = all_gather({
                'pos_m_neg_m_losses' : pos_m_neg_m_losses,
                'pos_m_neg_e_losses' : pos_m_neg_e_losses,
                'pos_e_neg_m_losses' : pos_e_neg_m_losses,
                'pos_e_neg_e_losses' : pos_e_neg_e_losses
        })

        if get_rank() == 0:
            pos_m_neg_m_losses = flatten([d['pos_m_neg_m_losses'] for d in gathered_data])
            pos_m_neg_e_losses = flatten([d['pos_m_neg_e_losses'] for d in gathered_data])
            pos_e_neg_m_losses = flatten([d['pos_e_neg_m_losses'] for d in gathered_data])
            pos_e_neg_e_losses = flatten([d['pos_e_neg_e_losses'] for d in gathered_data])
            losses = pos_m_neg_m_losses + pos_m_neg_e_losses + pos_e_neg_m_losses + pos_e_neg_e_losses

            pos_m_neg_m_loss = 0.0 if len(pos_m_neg_m_losses) == 0 else np.mean(pos_m_neg_m_losses)
            pos_m_neg_e_loss = 0.0 if len(pos_m_neg_e_losses) == 0 else np.mean(pos_m_neg_e_losses)
            pos_e_neg_m_loss = 0.0 if len(pos_e_neg_m_losses) == 0 else np.mean(pos_e_neg_m_losses)
            pos_e_neg_e_loss = 0.0 if len(pos_e_neg_e_losses) == 0 else np.mean(pos_e_neg_e_losses)
            loss = np.mean(losses)

            synchronize()
            return {
                    'concat_loss' : loss,
                    'concat_num_examples': len(losses),
                    'concat_pos_m_neg_m_loss' : pos_m_neg_m_loss,
                    'concat_pos_m_neg_e_loss' : pos_m_neg_e_loss,
                    'concat_pos_e_neg_m_loss' : pos_e_neg_m_loss,
                    'concat_pos_e_neg_e_loss' : pos_e_neg_e_loss,
                    'concat_pos_m_neg_m_num_examples' : len(pos_m_neg_m_losses),
                    'concat_pos_m_neg_e_num_examples' : len(pos_m_neg_e_losses),
                    'concat_pos_e_neg_m_num_examples' : len(pos_e_neg_m_losses),
                    'concat_pos_e_neg_e_num_examples' : len(pos_e_neg_e_losses)
            }
        else:
            synchronize()
            return None


    def _train_softmax(self, dataset_list, metadata):
        args = self.args

        losses = [] 
        time_per_dataset = []
        dataset_sizes = []

        self.model.train()
        self.model.zero_grad()
        criterion = nn.CrossEntropyLoss()
        for dataset in dataset_list:
            _dataset_start_time = time.time()
            dataset_sizes.append(len(dataset))
            dataloader = SoftmaxConcatenationDataLoader(args, dataset)
            for batch in dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                m_e_mask = torch.any(torch.any(batch[0] < args.num_entities, -1), -1).reshape(-1,).to(args.device)
                inputs = {'m_e_mask':       m_e_mask,
                          'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3]}
                outputs = self.model(**inputs)
                scores = torch.mean(outputs, -1)
                target = torch.zeros(scores.shape[0], dtype=torch.long).cuda()
                loss = criterion(scores, target)
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

        gathered_data = all_gather({
                'losses' : losses,
        })

        if get_rank() == 0:
            losses = flatten([d['losses'] for d in gathered_data])
            loss = np.mean(losses)

            synchronize()
            return { 'concat_loss' : loss }
        else:
            synchronize()
            return None

    def _train_threshold(self, dataset_list, metadata):
        args = self.args

        losses = [] 
        time_per_dataset = []
        dataset_sizes = []

        self.model.train()
        self.model.zero_grad()
        random.shuffle(dataset_list)
        for dataset in dataset_list:
            _dataset_start_time = time.time()
            dataset_sizes.append(len(dataset))
            dataloader = ScaledPairsConcatenationDataLoader(args, dataset)
            agg_loss = 0.0
            for batch in dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                m_e_mask = torch.any(torch.any(batch[1] < args.num_entities, -1), -1).reshape(-1,).to(args.device)
                inputs = {'m_e_mask':       m_e_mask,
                          'input_ids':      batch[2],
                          'attention_mask': batch[3],
                          'token_type_ids': batch[4]}
                outputs = self.model(**inputs)
                scores = torch.mean(outputs, -1).reshape(-1,)
                loss = torch.mean(torch.abs(batch[0]) * F.relu(args.margin - (torch.sign(batch[0]) * scores)))
                #loss = torch.sum(F.relu(args.margin - (torch.sign(batch[0]) * scores))) * 8 / len(dataset)
                agg_loss += loss.item()
                loss.backward()

            losses.append(agg_loss)
            torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    args.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            time_per_dataset.append(time.time() - _dataset_start_time)

        gathered_data = all_gather({
                'losses' : losses,
                'dataset_sizes' : dataset_sizes,
        })

        if get_rank() == 0:
            losses = flatten([d['losses'] for d in gathered_data])
            dataset_sizes = flatten([d['dataset_sizes'] for d in gathered_data])
            loss = np.mean(losses)
            avg_dataset_size = np.mean(dataset_sizes)

            synchronize()
            return { 'concat_loss' : loss, 'dataset_size': avg_dataset_size, 'num_datasets': len(dataset_list) }
        else:
            synchronize()
            return None

    def get_edge_affinities(self, edges, example_dir, knn_index):
        args = self.args
        dataset = PairsConcatenationDataset(args, edges, example_dir)
        dataloader = PairsConcatenationDataLoader(args, dataset)

        edge2affinity = {}
        for batch in dataloader:
            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch)
                m_e_mask = torch.any(torch.any(batch[0] < args.num_entities, -1), -1).reshape(-1,).to(args.device)
                inputs = {'m_e_mask':       m_e_mask,
                          'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3]}
                outputs = self.model(**inputs)

                scores = torch.mean(outputs, -1).squeeze(-1)
                scores = torch.sigmoid(scores)
                scores = scores.cpu().numpy().tolist()

                idx_edge_pairs = batch[0].squeeze(1)[:,:,0].cpu().numpy()
                idx_edge_pairs = [tuple(p) for p in idx_edge_pairs.tolist()]

                try:
                    edge2affinity.update(
                            {k : v for k, v in zip(idx_edge_pairs, scores)}
                    )
                except:
                    if get_rank() == 0:
                        embed()
                    synchronize()
                    exit()

        gathered_edge2affinity = all_gather(edge2affinity)

        global_edge2affinity = {}
        for d in gathered_edge2affinity:
            global_edge2affinity.update(d)

        affinities = map(lambda e : global_edge2affinity[e], edges)
        return list(affinities)

    def save_model(self, global_step):
        self.model.module.save_model(suffix='checkpoint-{}'.format(global_step))
