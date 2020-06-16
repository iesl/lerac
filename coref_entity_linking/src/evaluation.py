import logging
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

from utils.comm import get_rank, synchronize
from utils.misc import unique

from IPython import embed


logger = logging.getLogger(__name__)


def eval_wdoc(args, metadata, knn_index, sub_trainer):
    logger.info('Building within doc sparse graphs...')
    doc_level_graphs = []
    for doc_clusters in tqdm(metadata.wdoc_clusters.values(), disable=(get_rank() != 0)):
        doc_mentions = np.asarray([x for k, v in doc_clusters.items()
                                        for x in v if x != k])
        doc_level_graphs.append(
            build_sparse_affinity_graph(
                args,
                doc_mentions,
                metadata,
                knn_index,
                sub_trainer,
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
    for coref_graph, linking_graph in doc_level_graphs:
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


def eval_xdoc(args, metadata, knn_index, sub_trainer):
    pass


def compute_coref_metrics(metadata, coref_graphs):
    embed()
    return 


def compute_linking_metrics(metadata, linking_graphs):
    pass


def compute_joint_metrics(metadata, joint_graphs):
    pass


def build_sparse_affinity_graph(args,
                                midxs, 
                                metadata,
                                knn_index,
                                sub_trainer,
                                build_coref_graph=False,
                                build_linking_graph=False):
    # FIXME: this should probably go in `src/evaluation.py`
    assert build_coref_graph or build_linking_graph

    coref_graph = None
    linking_graph = None
    if get_rank() == 0:
        # FIXME: hack to get all of the embeddings,
        #        should use sub_trainer to get scores
        idxs, embeds = knn_index.idxs, knn_index.X

        # create inverse index for mapping
        inverse_idxs = {v : k for k, v in enumerate(idxs)}

        mention_knn = None
        if build_coref_graph or args.available_entities == 'knn_candidates':
            # get all of the mention kNN
            mention_knn = knn_index.get_knn_limited_index(
                    midxs, include_index_idxs=midxs, k=args.k+1
            )
            mention_knn = mention_knn[:,1:]

        if build_coref_graph:
            # list of edges for sparse graph we will build
            coref_graph_edges = []

            # add mention-mention edges to 
            coref_graph_edges.extend(
                [tuple(sorted((a, b)))
                    for a, l in zip(midxs, mention_knn) for b in l]
            )
            coref_graph_edges = unique(coref_graph_edges)
            affinities = [np.dot(embeds[inverse_idxs[i]],
                                 embeds[inverse_idxs[j]]) 
                            for i, j in coref_graph_edges]
            coref_graph_edges = np.asarray(coref_graph_edges).T
            _sparse_num = metadata.num_mentions + metadata.num_entities
            coref_graph = coo_matrix(
                        (affinities, coref_graph_edges),
                        shape=(_sparse_num, _sparse_num)
            )

        if build_linking_graph:
            # list of edges for sparse graph we will build
            linking_graph_edges = []

            # get mention-entity edges
            if args.available_entities == 'candidates_only':
                for midx in midxs:
                    candidates = metadata.midx2cand.get(midx, [])
                    if len(candidates) > 0:
                        linking_graph_edges.extend(
                            [tuple(sorted((midx, eidx))) for eidx in candidates]
                        )
            elif args.available_entities == 'knn_candidates':
                for i, midx in enumerate(midxs):
                    candidates = metadata.midx2cand.get(midx, [])
                    candidates.extend(
                            flatten([metadata.midx2cand.get(m, []) 
                                            for m in mention_knn[i]])
                    )
                    if len(candidates) > 0:
                        linking_graph_edges.extend(
                            [tuple(sorted((midx, eidx))) for eidx in candidates]
                        )
            else: # 'open_domain'
                # use `knn_index` to do this efficiently
                raise NotImplementedError('open domain not yet')

            # build the graph
            linking_graph_edges = unique(linking_graph_edges)
            affinities = [np.dot(embeds[inverse_idxs[i]],
                                 embeds[inverse_idxs[j]]) 
                            for i, j in linking_graph_edges]
            linking_graph_edges = np.asarray(linking_graph_edges).T
            _sparse_num = metadata.num_mentions + metadata.num_entities
            linking_graph = coo_matrix((affinities, linking_graph_edges),
                                          shape=(_sparse_num, _sparse_num))

    return coref_graph, linking_graph
