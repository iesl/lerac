import math
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

from data.datasets import TripletEmbeddingDataset
from utils.comm import get_rank, synchronize

from IPython import embed


def build_triplet_datasets(args, pairs_collection):
    datasets = []
    for cluster_pairs_list in pairs_collection:
        # create triplets for cluster
        triplets = []
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

        # create datasets from triplets for cluster
        datasets.append(
                TripletEmbeddingDataset(
                    args,
                    triplets,
                    args.train_cache_dir
                )
        )

    return datasets


def get_all_pairs(clusters_mx, sparse_graph):
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
