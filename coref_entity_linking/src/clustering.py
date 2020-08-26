from abc import ABC, abstractmethod
from collections import defaultdict
import logging
import math
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from random import shuffle

from data.datasets import (SoftmaxEmbeddingDataset,
                           SoftmaxConcatenationDataset,
                           TripletEmbeddingDataset,
                           TripletConcatenationDataset,
                           ScaledPairEmbeddingDataset,
                           ScaledPairConcatenationDataset)
from utils.comm import get_rank, synchronize

from IPython import embed


logger = logging.getLogger(__name__)


class SupervisedClusteringDatasetBuilder(ABC):
    """ Abstract base class for building supervised clustering datasets. """
    def __init__(self, args):
        self.args = args
        if args.pair_gen_method == 'all_pairs':
            self.pairs_creator = AllPairsCreator(args)
        elif args.pair_gen_method == 'mst':
            self.pairs_creator = MstPairsCreator(args)
        else:
            self.pairs_creator = ExpLinkPairsCreator(args)

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


class SoftmaxDatasetBuilder(SupervisedClusteringDatasetBuilder):

    def __init__(self, args):
        super(SoftmaxDatasetBuilder, self).__init__(args)
    
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
        examples = []
        for cluster_pairs_list in pairs_collection:
            for joint_collection in cluster_pairs_list:
                anchor = joint_collection['anchor']
                pos_list = joint_collection['pos']
                neg_list = joint_collection['neg']

                if len(pos_list) == 0 or len(neg_list) == 0:
                    continue

                num_negs = (args.per_gpu_train_batch_size // 2) - 1
                desired_num_examples = max(len(pos_list),
                                           math.ceil(len(neg_list) / num_negs))
                while len(pos_list) < desired_num_examples:
                    pos_list.extend(pos_list)
                while len(neg_list) // num_negs < desired_num_examples:
                    neg_list.extend(neg_list)
                pos_list = pos_list[:desired_num_examples]
                neg_list = neg_list[:num_negs*desired_num_examples]

                shuffle(pos_list)
                shuffle(neg_list)

                for i in range(desired_num_examples):
                    examples.append([anchor,
                                     pos_list[i],
                                     *neg_list[i*num_negs:(i+1)*num_negs]])

        if len(examples) == 0:
            return None, None, None

        # append one big dataset
        embed_dataset_list.append(
                SoftmaxEmbeddingDataset(
                    args,
                    examples,
                    args.train_cache_dir
                )
        )

        # append one big dataset
        concat_dataset_list.append(
                SoftmaxConcatenationDataset(
                    args,
                    examples,
                    args.train_cache_dir
                )
        )

        dataset_metrics = {}

        return (embed_dataset_list, concat_dataset_list), dataset_metrics


class ThresholdDatasetBuilder(SupervisedClusteringDatasetBuilder):

    def __init__(self, args):
        super(ThresholdDatasetBuilder, self).__init__(args)
    
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
        for cluster_pairs_list in pairs_collection:
            scaled_pairs = []
            for joint_collection in cluster_pairs_list:
                anchor = joint_collection['anchor']
                pos_list = joint_collection['pos']
                neg_list = joint_collection['neg']
                pos_scale_factor = 1.0
                if len(pos_list) > 0:
                    while len(pos_list) <= 0.5*len(neg_list):
                        pos_list.extend(pos_list)
                    pos_scale_factor = max(0.8*len(neg_list) / len(pos_list), 1.0)
                scaled_pairs.extend([(pos_scale_factor, anchor, p) for p in pos_list])
                scaled_pairs.extend([(-1.0, anchor, n) for n in neg_list])

            if len(scaled_pairs) == 0:
                continue

            embed_dataset_list.append(
                    ScaledPairEmbeddingDataset(
                        args,
                        scaled_pairs,
                        args.train_cache_dir
                    )
            )
            concat_dataset_list.append(
                    ScaledPairConcatenationDataset(
                        args,
                        scaled_pairs,
                        args.train_cache_dir
                    )
            )

        dataset_metrics = {}

        return (embed_dataset_list, concat_dataset_list), dataset_metrics


class PairsCreator(ABC):
    """ Abstract base class for generating training pairs. """
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def __call__(self, clusters_mx, sparse_graph):
        pass


class AllPairsCreator(PairsCreator):
    """ Create all pairs collection. """

    def __init__(self, args):
        super(AllPairsCreator, self).__init__(args)

    def __call__(self, clusters_mx, sparse_graph, metadata):
        args = self.args
        
        # FIXME: this is kinda slow, maybe fix with Cython??
        # check to make sure this is true
        _row = clusters_mx.row
        _data = clusters_mx.data
        assert np.all(np.isin(sparse_graph.row, _data))

        # get all of the edges
        all_edges = np.vstack((sparse_graph.row, sparse_graph.col))
        all_affinities = sparse_graph.data

        # get the positive and negative edges
        local_pos_a, local_pos_b = np.where(
                np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
        )
        pos_a = _data[local_pos_a]
        pos_b = _data[local_pos_b]
        _pos_edges = np.vstack((pos_a, pos_b)).T.tolist()

        pos_mask = np.asarray([x in _pos_edges for x in all_edges.T.tolist()])
        neg_mask = ~pos_mask

        #pos_edges = all_edges[:, pos_mask].T
        #pos_affinities = all_affinities[pos_mask]
        #pos_tuples_dict = defaultdict(list)
        #for (a, b), c in zip(pos_edges.tolist(), pos_affinities):
        #    pos_tuples_dict[a].append((b, c))
        #pos_edge_tuples = []
        #for a, pos_edges in pos_tuples_dict.items():
        #    pos_edge_tuples.extend([(a, b) for (b, c) in pos_edges if c < (args.margin)])
        #pos_edges = np.asarray(pos_edge_tuples).T
        pos_edges = all_edges[:, pos_mask]
        pos_edges = np.concatenate((pos_edges, pos_edges[[1, 0]]), axis=1).T

        # limit the number of negatives to only the most offending ones
        # this splits the quota evenly between m-m and m-e negs
        neg_edges = all_edges[:, neg_mask].T
        neg_affinities = all_affinities[neg_mask]
        neg_tuples_dict = defaultdict(list)
        for (a, b), c in zip(neg_edges.tolist(), neg_affinities):
            neg_tuples_dict[a].append((b, c))
        neg_edge_tuples = []
        for a, neg_edges in neg_tuples_dict.items():
            neg_m_m = [(b, c) for (b, c) in neg_edges if b >= args.num_entities]
            neg_m_m = sorted(neg_m_m, key=lambda x : x[1], reverse=True)
            neg_m_m = neg_m_m[:math.ceil(args.num_train_negs/2)]
            neg_m_e = [(b, c) for (b, c) in neg_edges if b < args.num_entities]
            neg_m_e = sorted(neg_m_e, key=lambda x : x[1], reverse=True)
            neg_m_e = neg_m_e[:math.floor(args.num_train_negs/2)]
            neg_edge_tuples.extend([(a, b) for b, _ in (neg_m_m + neg_m_e)])
        neg_edges = np.asarray(neg_edge_tuples)

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
                # NOTE: this might be too harsh
                pos = list(filter(lambda x : (x >= metadata.num_entities
                                        or x in metadata.midx2cand[anchor]),
                                  pos))

                neg = neg_edges[:, 1][neg_edges[:, 0] == anchor].tolist()
                neg = list(filter(lambda x : x >= 0, neg))

                if args.training_edges_considered == 'm-e':
                    # only use mention-entity links
                    pos = list(filter(lambda x : x < metadata.num_entities, pos))
                    neg = list(filter(lambda x : x < metadata.num_entities, neg))
                elif args.training_edges_considered == 'm-m':
                    # only use mention-mention links
                    pos = list(filter(lambda x : x >= metadata.num_entities, pos))
                    neg = list(filter(lambda x : x >= metadata.num_entities, neg))
                else:
                    assert args.training_edges_considered == 'all'

                #assert metadata.midx2eidx[anchor] in pos

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

    def __init__(self, args):
        super(MstPairsCreator, self).__init__(args)

    def __call__(self, clusters_mx, sparse_graph, metadata):
        args = self.args
        
        # FIXME: this is kinda slow, maybe fix with Cython??
        # check to make sure this is true
        _row = clusters_mx.row
        _data = clusters_mx.data
        assert np.all(np.isin(sparse_graph.row, _data))

        # get all of the edges
        all_edges = np.vstack((sparse_graph.row, sparse_graph.col))
        all_affinities = sparse_graph.data

        # get the positive and negative edges
        local_pos_a, local_pos_b = np.where(
                np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
        )
        pos_a = _data[local_pos_a]
        pos_b = _data[local_pos_b]
        if args.training_edges_considered == 'm-e':
            m_e_mask = pos_a < args.num_entities
            pos_a = pos_a[m_e_mask]
            pos_b = pos_b[m_e_mask]
        elif args.training_edges_considered == 'm-m':
            m_m_mask = pos_a >= args.num_entities
            pos_a = pos_a[m_m_mask]
            pos_b = pos_b[m_m_mask]
        _pos_edges = np.vstack((pos_a, pos_b)).T.tolist()
        pos_mask = np.asarray([x in _pos_edges for x in all_edges.T.tolist()])
                
        neg_mask = ~pos_mask
        pos_edges = all_edges[:, pos_mask]

        # limit the number of negatives to only the most offending ones
        # this splits the quota evenly between m-m and m-e negs
        neg_edges = all_edges[:, neg_mask].T
        neg_affinities = all_affinities[neg_mask]
        neg_tuples_dict = defaultdict(list)
        for (a, b), c in zip(neg_edges.tolist(), neg_affinities):
            neg_tuples_dict[a].append((b, c))
        neg_edge_tuples = []
        for a, neg_edges in neg_tuples_dict.items():
            neg_m_m = [(b, c) for (b, c) in neg_edges if b >= args.num_entities]
            neg_m_m = sorted(neg_m_m, key=lambda x : x[1], reverse=True)
            neg_m_m = neg_m_m[:math.ceil(args.num_train_negs/2)]
            neg_m_e = [(b, c) for (b, c) in neg_edges if b < args.num_entities]
            neg_m_e = sorted(neg_m_e, key=lambda x : x[1], reverse=True)
            neg_m_e = neg_m_e[:math.floor(args.num_train_negs/2)]
            neg_edge_tuples.extend([(a, b) for b, _ in (neg_m_m + neg_m_e)])
        neg_edges = np.asarray(neg_edge_tuples)

        # organize pairs collection
        pairs_collection = []
        for cluster_index in range(clusters_mx.shape[0]):
            cluster_pairs_collection = []
            anchors = clusters_mx.data[_row == cluster_index] 

            # get cluster specific edges based on MST
            in_cluster_mask = np.isin(pos_edges, anchors)
            in_cluster_mask = (in_cluster_mask[0] & in_cluster_mask[1])

            in_cluster_row = sparse_graph.row[pos_mask][in_cluster_mask]
            in_cluster_col = sparse_graph.col[pos_mask][in_cluster_mask]
            in_cluster_data = -1.0 * sparse_graph.data[pos_mask][in_cluster_mask]
            in_cluster_csr = csr_matrix(
                    (in_cluster_data, (in_cluster_row, in_cluster_col)),
                    shape=sparse_graph.shape
            )

            in_cluster_mst = minimum_spanning_tree(in_cluster_csr).tocoo()
            in_cluster_pos_edges = np.vstack((in_cluster_mst.row,
                                              in_cluster_mst.col))
            in_cluster_pos_edges = np.concatenate(
                    (in_cluster_pos_edges, in_cluster_pos_edges[[1, 0]]),
                    axis=1).T

            for anchor in anchors:
                if anchor < metadata.num_entities:
                    continue

                pos = in_cluster_pos_edges[:, 1][in_cluster_pos_edges[:, 0] == anchor].tolist()

                # this line removes positive entity from anchor's pos list
                # if it doesn't appear in that mention's candidate set
                # NOTE: this might be too harsh
                #pos = list(filter(lambda x : (x >= metadata.num_entities
                #                        or x in metadata.midx2cand[anchor]),
                #                  pos))

                neg = neg_edges[:, 1][neg_edges[:, 0] == anchor].tolist()
                neg = list(filter(lambda x : x >= 0, neg))

                if args.training_edges_considered == 'm-e':
                    # only use mention-entity links
                    pos = list(filter(lambda x : x < metadata.num_entities, pos))
                    neg = list(filter(lambda x : x < metadata.num_entities, neg))
                elif args.training_edges_considered == 'm-m':
                    # only use mention-mention links
                    pos = list(filter(lambda x : x >= metadata.num_entities, pos))
                    neg = list(filter(lambda x : x >= metadata.num_entities, neg))
                else:
                    assert args.training_edges_considered == 'all'

                cluster_pairs_collection.append(
                    {
                        'anchor' : anchor,
                        'pos' : pos,
                        'neg' : neg
                    }
                )
            pairs_collection.append(cluster_pairs_collection)


        return pairs_collection


class ExpLinkPairsCreator(PairsCreator):
    pass
