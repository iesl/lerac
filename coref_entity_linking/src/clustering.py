from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from utils.comm import get_rank, synchronize

from IPython import embed


def build_triplet_dataset(clusters_mx, sparse_graph):
    if get_rank() == 0:
        embed()
    synchronize()
    exit()
