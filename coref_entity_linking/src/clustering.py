from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from data.datasets import TripletEmbeddingDataset
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
        dataset_list = []
        triplets = []
        for cluster_pairs_list in pairs_collection:
            for joint_collection in cluster_pairs_list:
                anchor = joint_collection['anchor']
                pos_list = joint_collection['pos']
                neg_list = joint_collection['neg']
                pos_len, neg_len = len(pos_list), len(neg_list)
                if pos_len > neg_len:
                    pos_list *= math.ceil(pos_len / neg_len)
                else:
                    neg_list *= math.ceil(neg_len / pos_len)

                for p, n in zip(pos_list, neg_list):
                    triplets.append((anchor, p, n))
        dataset_list.append(
                TripletEmbeddingDataset(
                    args,
                    triplets,
                    args.train_cache_dir
                )
        )

        return dataset_list


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
        assert np.all(np.isin(sparse_graph.row, clusters_mx.data))

        # get all of the edges
        all_edges = np.vstack((sparse_graph.row, sparse_graph.col)).T.tolist()

        # get the positive edges
        _row = clusters_mx.row
        local_pos_a, local_pos_b = np.where(
                np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
        )
        a = clusters_mx.data[local_pos_a]
        b = clusters_mx.data[local_pos_b]
        pos_edges = np.vstack((np.concatenate((a, b), axis=0),
                               np.concatenate((b, a), axis=0))).T.tolist()

        # get negative edges
        neg_edge_mask = [e not in pos_edges for e in all_edges]
        neg_edge_mask = np.asarray(neg_edge_mask)
        pos_edges = np.asarray(pos_edges)
        all_edges = np.asarray(all_edges)
        neg_edges = all_edges[neg_edge_mask]

        # randomly filter for speed up
        mask = np.full(pos_edges.shape[0], False)
        mask[:(2 * 3 * clusters_mx.data.size)] = True
        np.random.shuffle(mask)
        pos_edges = pos_edges[mask]

        # organize pairs collection
        pairs_collection = []
        for cluster_index in range(clusters_mx.shape[0]):
            cluster_pairs_collection = []
            anchors = clusters_mx.data[_row == cluster_index]
            for anchor in anchors:
                pos = pos_edges[:, 1][pos_edges[:, 0] == anchor].tolist()
                neg = neg_edges[:, 1][neg_edges[:, 0] == anchor].tolist()
                min_len = min(len(pos), len(neg))
                if min_len == 0:
                    continue
                pos = pos[:min_len]
                neg = neg[:min_len]
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
