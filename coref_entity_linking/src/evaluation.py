import logging
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

from utils.comm import get_rank, synchronize

from IPython import embed


logger = logging.getLogger(__name__)


def eval_wdoc(metadata, knn_index, sub_trainer):
    logger.info('Building within doc sparse graphs...')
    doc_level_graphs = []
    for doc_clusters in tqdm(metadata.wdoc_clusters.values(), disable=(get_rank() != 0)):
        doc_mentions = np.asarray([x for k, v in doc_clusters.items()
                                        for x in v if x != k])
        doc_level_graphs.append(
            sub_trainer.build_sparse_affinity_graph(
                doc_mentions,
                metadata,
                knn_index,
                build_coref_graph=True,
                build_linking_graph=True
            )
        )
    logger.info('Done.')

    # don't need other processes at this point
    if get_rank() != 0:
        synchronize()
        return

    coref_graphs, linking_graphs, joint_graphs, = [], [], []
    for coref_graph, linking_graph in tqdm(doc_level_graphs, desc='joining graphs'):
        coref_graphs.append(coref_graph)
        linking_graphs.append(linking_graph)
        _joint_row = np.concatenate((coref_graph.row, linking_graph.row))
        _joint_col = np.concatenate((coref_graph.col, linking_graph.col))
        _joint_data = np.concatenate((coref_graph.data, linking_graph.data))
        joint_graphs.append(
                coo_matrix((_joint_data, (_joint_row, _joint_col)),
                           shape=linking_graph.shape)
        )

    coref_metrics = compute_coref_metrics(metadata, coref_graphs)
    #linking_metrics = compute_linking_metrics(metadata, linking_graphs)
    #joint_metrics = compute_joint_metrics(metadata, joint_graphs)

    synchronize()
    return


def eval_xdoc(metadata, knn_index, sub_trainer):
    pass


def compute_coref_metrics(metadata, coref_graphs):
    embed()
    return 


def compute_linking_metrics(metadata, linking_graphs):
    pass


def compute_joint_metrics(metadata, joint_graphs):
    pass
