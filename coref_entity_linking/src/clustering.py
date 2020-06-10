import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from utils.comm import get_rank, synchronize

from IPython import embed


def build_triplet_dataset(clusters_mx, sparse_graph):

    # check to make sure this is true
    assert np.all(np.isin(sparse_graph.row, clusters_mx.data))

    # get the positive edges
    _row = clusters_mx.row
    local_pos_a, local_pos_b = np.where(
            np.triu(_row[np.newaxis, :] == _row[:, np.newaxis], k=1)
    )
    a = clusters_mx.data[local_pos_a]
    b = clusters_mx.data[local_pos_b]

    if get_rank() == 0:
        embed()
    synchronize()
    exit()
