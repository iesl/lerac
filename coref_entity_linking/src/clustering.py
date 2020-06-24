from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from data.datasets import (TripletEmbeddingDataset,
                           TripletConcatenationDataset)
from utils.comm import get_rank, synchronize

from IPython import embed


logger = logging.getLogger(__name__)


class SupervisedClusteringDatasetBuilder(ABC):
    """ Abstract base class for building supervised clustering datasets. """
    def __init__(self, args):
        self.args = args
        if args.pair_gen_method == 'all_pairs':
            self.pairs_creator = AllPairsCreator()
        elif args.pair_gen_method == 'mst':
            self.pairs_creator = MstPairsCreator()
        else:
            self.pairs_creator = ExpLinkPairsCreator()

    @abstractmethod
    def __call__(self, clusters_mx, sparse_graph):
        pass


class TripletDatasetBuilder(SupervisedClusteringDatasetBuilder):

    def __init__(self, args):
        super(TripletDatasetBuilder, self).__init__(args)
    
    def __call__(self, clusters_mx, sparse_graph):
        args = self.args

        # get pairs_collection
        pairs_collection = self.pairs_creator(clusters_mx, sparse_graph)

        # build datasets
        embed_dataset_list = []
        concat_dataset_list = []
        triplets = []
        for cluster_pairs_list in pairs_collection:
            for joint_collection in cluster_pairs_list:
                anchor = joint_collection['anchor']
                pos_list = joint_collection['pos']
                neg_list = joint_collection['neg']
                min_len = min(len(pos_list), len(neg_list))
                if min_len == 0:
                    continue
                pos_list = pos_list[:min_len]
                neg_list = neg_list[:min_len]
                for p, n in zip(pos_list, neg_list):
                    triplets.append((anchor, p, n))

        # append one big dataset
        embed_dataset_list.append(
                TripletEmbeddingDataset(
                    args,
                    triplets,
                    args.train_cache_dir
                )
        )
        # append one big dataset
        concat_dataset_list.append(
                TripletConcatenationDataset(
                    args,
                    triplets,
                    args.train_cache_dir
                )
        )

        return embed_dataset_list, concat_dataset_list


class SigmoidDatasetBuilder(SupervisedClusteringDatasetBuilder):
    pass


class SoftmaxDatasetBuilder(SupervisedClusteringDatasetBuilder):
    pass


class AccumMaxMarginDatasetBuilder(SupervisedClusteringDatasetBuilder):
    pass


class PairsCreator(ABC):
    """ Abstract base class for generating training pairs. """
    @abstractmethod
    def __call__(self, clusters_mx, sparse_graph):
        pass


class AllPairsCreator(PairsCreator):
    """ Create all pairs collection. """
    def __call__(self, clusters_mx, sparse_graph):
        # FIXME: this is kinda slow, maybe fix with Cython??
        # check to make sure this is true
        _row = clusters_mx.row
        _data = clusters_mx.data
        assert np.all(np.isin(sparse_graph.row, _data))

        # get all of the edges
        all_edges = np.vstack((sparse_graph.row, sparse_graph.col))

        # get the positive and negative edges
        idx2cluster = {a : b for a, b in zip(_data, _row)}
        v_idx2cluster = np.vectorize(lambda x : idx2cluster.get(x, -1))
        all_cluster_assignments = v_idx2cluster(all_edges)
        pos_mask = (all_cluster_assignments[0] == all_cluster_assignments[1])
        neg_mask = ~pos_mask
        pos_edges = all_edges[:, pos_mask].T
        neg_edges = all_edges[:, neg_mask].T

        # organize pairs collection
        pairs_collection = []
        for cluster_index in range(clusters_mx.shape[0]):
            cluster_pairs_collection = []
            anchors = clusters_mx.data[_row == cluster_index]
            for anchor in anchors:
                pos = pos_edges[:, 1][pos_edges[:, 0] == anchor].tolist()
                neg = neg_edges[:, 1][neg_edges[:, 0] == anchor].tolist()
                cluster_pairs_collection.append(
                    {
                        'anchor' : anchor,
                        'pos' : pos,
                        'neg' : neg
                    }
                )
            pairs_collection.append(cluster_pairs_collection)

        return pairs_collection
    

class MstPairsCreator(PairsCreator):
    pass


class ExpLinkPairsCreator(PairsCreator):
    pass
