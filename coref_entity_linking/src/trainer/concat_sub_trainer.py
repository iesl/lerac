from collections import defaultdict
import logging
import numpy as np
from scipy.sparse import coo_matrix
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.datasets import PairsConcatenationDataset
from data.dataloaders import (PairsConcatenationDataLoader,
                              TripletConcatenationDataLoader)
from utils.comm import get_rank, all_gather, synchronize
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
        if args.training_method == 'triplet':
            self.train_on_subset = self._train_triplet
        else:
            raise ValueError('training method not implemented yet')

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
                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'concat_input': True}
                outputs = self.model(**inputs)
                scores = torch.mean(outputs, -1)

                # max-margin
                per_triplet_loss = F.relu(
                        scores[:, 1]   # negative dot products
                        - scores[:, 0] # positive dot products
                        + args.margin
                )
                # BPR
                #per_triplet_loss = torch.sigmoid(
                #        scores[:, 1]   # negative dot products
                #        - scores[:, 0] # positive dot products
                #        + args.margin
                #)

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

    def get_edge_affinities(self, edges, example_dir, knn_index):
        args = self.args
        dataset = PairsConcatenationDataset(args, edges, example_dir)
        dataloader = PairsConcatenationDataLoader(args, dataset)

        edge2affinity = {}
        for batch in dataloader:
            with torch.no_grad():
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'concat_input': True}
                outputs = self.model(**inputs)
                scores = torch.mean(outputs, -1).squeeze(-1)
                scores = scores.cpu().numpy().tolist()

                idx_edge_pairs = batch[0].squeeze(1)[:,:,0].cpu().numpy()
                idx_edge_pairs = [tuple(p) for p in idx_edge_pairs.tolist()]

                edge2affinity.update(
                        {k : v for k, v in zip(idx_edge_pairs, scores)}
                )

        gathered_edge2affinity = all_gather(edge2affinity)

        global_edge2affinity = {}
        for d in gathered_edge2affinity:
            global_edge2affinity.update(d)

        affinities = map(lambda e : global_edge2affinity[e], edges)
        return list(affinities)

    def save_model(self, global_step):
        self.model.module.save_model(suffix='checkpoint-{}'.format(global_step))
