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
    def __call__(self, clusters_mx, sparse_graph, metadata):
        pass


class TripletDatasetBuilder(SupervisedClusteringDatasetBuilder):

    def __init__(self, args):
        super(TripletDatasetBuilder, self).__init__(args)
    
    def __call__(self, clusters_mx, sparse_graph, metadata):
        args = self.args

        # get pairs_collection
        pairs_collection = self.pairs_creator(
                clusters_mx, sparse_graph, metadata
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

                if len(pos_list) == 0 or len(neg_list) == 0:
                    continue

                # actuall all pairs implementation
                #desired_num_pairs = min(
                #        max(len(pos_list), len(neg_list)),
                #        5*min(len(pos_list), len(neg_list))
                #)
                desired_num_pairs = max(len(pos_list), len(neg_list))
                while len(pos_list) < desired_num_pairs:
                    pos_list.extend(pos_list)
                while len(neg_list) < desired_num_pairs:
                    neg_list.extend(neg_list)
                pos_list = pos_list[:desired_num_pairs]
                neg_list = neg_list[:desired_num_pairs]

                shuffle(pos_list)
                shuffle(neg_list)

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

        dataset_metrics = {}

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
    def __call__(self, clusters_mx, sparse_graph, metadata):
        # FIXME: this is kinda slow, maybe fix with Cython??
        # check to make sure this is true
        _row = clusters_mx.row
        _data = clusters_mx.data
        assert np.all(np.isin(sparse_graph.row, _data))

        # get all of the edges
        all_edges = np.vstack((sparse_graph.row, sparse_graph.col))

        # get the positive and negative edges
        local_pos_a, local_pos_b = np.where(
                np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
        )
        pos_a = _data[local_pos_a]
        pos_b = _data[local_pos_b]
        _pos_edges = np.vstack((pos_a, pos_b)).T.tolist()

        pos_mask = np.asarray([x in _pos_edges for x in all_edges.T.tolist()])
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
                if anchor < metadata.num_entities:
                    continue
                pos = pos_edges[:, 1][pos_edges[:, 0] == anchor].tolist()

                # this line removes positive entity from anchor's pos list
                # if it doesn't appear in that mention's candidate set
                # NOTE: this is too harsh
                #pos = list(filter(lambda x : (x >= metadata.num_entities
                #                        or x in metadata.midx2cand[anchor]),
                #                  pos))

                neg = neg_edges[:, 1][neg_edges[:, 0] == anchor].tolist()
                neg = list(filter(lambda x : x >= 0, neg))

                # only use mention-entity links
                pos = list(filter(lambda x : x < metadata.num_entities, pos))
                neg = list(filter(lambda x : x < metadata.num_entities, neg))

                assert metadata.midx2eidx[anchor] in pos

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
