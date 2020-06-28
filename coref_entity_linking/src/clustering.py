from abc import ABC, abstractmethod
import logging
import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from random import shuffle

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
    def __call__(self, clusters_mx, sparse_graph, num_entities):
        pass


class TripletDatasetBuilder(SupervisedClusteringDatasetBuilder):

    def __init__(self, args):
        super(TripletDatasetBuilder, self).__init__(args)
    
    def __call__(self, clusters_mx, sparse_graph, num_entities):
        args = self.args

        # get pairs_collection
        pairs_collection = self.pairs_creator(
                clusters_mx, sparse_graph, num_entities
        )

        # build datasets
        mention_entity_triplets = 0
        mention_mention_triplets = 0
        embed_dataset_list = []
        concat_dataset_list = []
        triplets = []
        for cluster_pairs_list in pairs_collection:
            for joint_collection in cluster_pairs_list:
                anchor = joint_collection['anchor']
                pos_list = joint_collection['pos']
                neg_list = joint_collection['neg']

                pos_m_e_list = [x for x in pos_list if x < num_entities]
                neg_m_e_list = [x for x in neg_list if x < num_entities]
                pos_m_m_list = [x for x in pos_list if x >= num_entities]
                neg_m_m_list = [x for x in neg_list if x >= num_entities]

                shuffle(pos_m_e_list)
                shuffle(neg_m_e_list)
                shuffle(pos_m_m_list)
                shuffle(neg_m_m_list)

                min_m_e_len = 4*min(len(pos_m_e_list), len(neg_m_e_list))
                while len(pos_m_e_list) < min_m_e_len:
                    pos_m_e_list.extend(pos_m_e_list)
                while len(neg_m_e_list) < min_m_e_len:
                    neg_m_e_list.extend(neg_m_e_list)
                pos_m_e_list = pos_m_e_list[:min_m_e_len]
                neg_m_e_list = neg_m_e_list[:min_m_e_len]

                min_m_m_len = min(len(pos_m_m_list), len(neg_m_m_list))
                while len(pos_m_m_list) < min_m_m_len:
                    pos_m_m_list.extend(pos_m_m_list)
                while len(neg_m_m_list) < min_m_m_len:
                    neg_m_m_list.extend(neg_m_m_list)
                pos_m_m_list = pos_m_m_list[:min_m_m_len]
                neg_m_m_list = neg_m_m_list[:min_m_m_len]

                mention_entity_triplets += len(pos_m_e_list)
                mention_mention_triplets += len(pos_m_m_list)

                pos_list = pos_m_e_list + pos_m_m_list
                neg_list = neg_m_e_list + neg_m_m_list
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

        dataset_metrics = {'mention_mention_triplets': mention_mention_triplets,
                           'mention_entity_triplets': mention_entity_triplets}

        return (embed_dataset_list, concat_dataset_list), dataset_metrics


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
    def __call__(self, clusters_mx, sparse_graph, num_entities):
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
        pos_edges = all_edges[:, pos_mask]
        pos_edges = np.concatenate((pos_edges, pos_edges[[1, 0]]), axis=1).T
        neg_edges = all_edges[:, neg_mask].T

        # organize pairs collection
        pairs_collection = []
        for cluster_index in range(clusters_mx.shape[0]):
            cluster_pairs_collection = []
            anchors = clusters_mx.data[_row == cluster_index] 
            for anchor in anchors:
                if anchor < num_entities:
                    continue
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
