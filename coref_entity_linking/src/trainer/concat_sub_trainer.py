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
            dataloader = TripletConcatenationDataLoader(args, dataset)
            for batch in dataloader:
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3],
                          'concat_input': True}
                outputs = self.model(**inputs)
                scores = torch.sigmoid(torch.mean(outputs, -1))

                loss = torch.mean(
                    F.relu(
                        scores[:,1]
                        - scores[:,0]
                        + args.margin
                    )
                )
                #loss = torch.mean(
                #    torch.sigmoid(
                #        scores[:, 1]
                #        - scores[:, 0]
                #    )
                #)
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
        total_num_examples = 4 * sum(dataset_sizes) # because triplets
        return {'concat_loss' : np.mean(losses),
                'concat_time_per_example': total_time / total_num_examples,
                'concat_num_examples': total_num_examples}

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
