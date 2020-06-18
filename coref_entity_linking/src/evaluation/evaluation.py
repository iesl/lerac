from collections import defaultdict
from copy import deepcopy
import logging
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import (minimum_spanning_tree,
                                  connected_components,
                                  breadth_first_tree)
from sklearn.cluster import KMeans
from sklearn.metrics import (fowlkes_mallows_score,
                             adjusted_rand_score)
from tqdm import tqdm

from utils.comm import get_rank, synchronize
from utils.misc import flatten, unique

from evaluation.special_partition import special_partition

from IPython import embed


logger = logging.getLogger(__name__)


def eval_wdoc(args, metadata, knn_index, sub_trainer):
    logger.info('Building within doc sparse graphs...')
    doc_level_graphs = []
    per_doc_coref_clusters = []
    for doc_clusters in tqdm(metadata.wdoc_clusters.values(), disable=(get_rank() != 0)):
        per_doc_coref_clusters.append(
                [[x for x in v if x != k] for k, v in doc_clusters.items()]
        )
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

    # build everything needed to compute metrics and compute them!
    coref_graphs, linking_graphs = [], []
    for coref_graph, linking_graph in doc_level_graphs:
        coref_graphs.append(coref_graph)
        linking_graphs.append(linking_graph)

    logger.info('Computing coref metrics...')
    coref_metrics = compute_coref_metrics(
            per_doc_coref_clusters, coref_graphs, args.eval_coref_threshold
    )
    logger.info('Done.')

    logger.info('Computing linking metrics...')
    linking_metrics, slim_linking_graph = compute_linking_metrics(
            metadata, linking_graphs
    )
    logger.info('Done.')
    
    logger.info('Computing joint metrics...')
    slim_coref_graph = _get_global_maximum_spanning_tree(coref_graphs)
    joint_metrics = compute_joint_metrics(metadata,
                                          [slim_linking_graph, slim_linking_graph])
    logger.info('Done.')

    synchronize()
    return


def eval_xdoc(args, metadata, knn_index, sub_trainer):
    pass


def compute_coref_metrics(gold_coref_clusters, coref_graphs, coref_threshold):
    global_gold_clustering = flatten(gold_coref_clusters)
    global_maximum_spanning_tree = _get_global_maximum_spanning_tree(
            coref_graphs
    )

    # compute metrics and choose threshold is one isn't specified
    if coref_threshold is None:
        logger.info('Generating candidate thresholds...')
        kmeans = KMeans(n_clusters=1000, random_state=0)
        kmeans.fit(global_maximum_spanning_tree.data.reshape(-1, 1))
        candidate_thresholds = kmeans.cluster_centers_.reshape(-1,).tolist()
        logger.info('Done.')

        logger.info('Choosing threshold...')
        threshold_results = []
        for _threshold in tqdm(candidate_thresholds):
            _metrics = _compute_coref_metrics_threshold(
                    global_gold_clustering,
                    global_maximum_spanning_tree,
                    _threshold
            )
            threshold_results.append((_threshold, _metrics))
        logger.info('Done.')
        max_threshold_results = max(threshold_results,
                                    key=lambda x : x[1]['rand_index'])

        coref_results = max_threshold_results[1]
        coref_results['threshold'] = max_threshold_results[0]
    else:
        coref_results = _compute_coref_metrics_threshold(
                    global_gold_clustering,
                    global_maximum_spanning_tree,
                    coref_threshold
        )
        coref_results['threshold'] = coref_threshold
        
    return coref_results



def compute_linking_metrics(metadata, linking_graphs):
    global_graph = _merge_sparse_graphs(linking_graphs)

    # compute recall
    _row = global_graph.row
    _col = global_graph.col
    mention2cand = defaultdict(list)
    for eidx, midx in zip(_row, _col):
        mention2cand[midx].append(eidx)

    recall_hits = 0
    recall_total = len(metadata.midx2eidx)
    no_candidates = []
    for midx in metadata.midx2eidx.keys():
        cands = mention2cand.get(midx, None)
        if cands is None:
            no_candidates.append(midx)
            continue
        if metadata.midx2eidx[midx] in cands:
            recall_hits += 1

    # entity linking decision is max row (entities) from each col (mentions)
    all_entity_choices = global_graph.argmax(axis=0)
    all_entity_choices = np.asarray(all_entity_choices).reshape(-1,)
    all_idxs = np.arange(all_entity_choices.size)
    # FIXME: this isn't completely correct if someone has no candidates
    mention_idx_mask = np.isin(all_idxs, list(mention2cand.keys()))
    pred_eidxs = all_entity_choices[mention_idx_mask]
    midxs = all_idxs[mention_idx_mask]

    # build slim global linking graph for joint linking inference
    max_affinities = global_graph.max(axis=0).data
    slim_global_graph = coo_matrix((max_affinities, (pred_eidxs, midxs)),
                                   shape=global_graph.shape)

    # compute linking accuracy
    linking_hits, linking_total = 0, 0
    for midx, pred_eidx in zip(midxs, pred_eidxs):
        if pred_eidx == metadata.midx2eidx[midx]:
            linking_hits += 1
        linking_total += 1

    results_dict = {'vanilla_recall' : recall_hits / recall_total,
                    'vanilla_accuracy' : linking_hits / linking_total,
                    'num_no_candidates' : len(no_candidates)}

    return results_dict, slim_global_graph


def compute_joint_metrics(metadata, joint_graphs):
    # get global joint graph
    global_joint_graph = _merge_sparse_graphs(joint_graphs)
    #global_joint_graph = _get_global_maximum_spanning_tree(joint_graphs)

    # reorder the data for the requirements of the `special_partition` function
    _row = global_joint_graph.row
    _col = global_joint_graph.col
    _data = global_joint_graph.data
    tuples = zip(_row, _col, _data)
    tuples = sorted(tuples, key=lambda x : (x[1], -x[0])) # sorted this way for nice DFS
    special_row, special_col, special_data = zip(*tuples)
    special_row = np.asarray(special_row, dtype=np.int)
    special_col = np.asarray(special_col, dtype=np.int)
    special_data = np.asarray(special_data)

    # order the edges in ascending order according to affinities
    ordered_edge_indices = np.argsort(special_data)

    # get the mask of edges we want to keep
    keep_edge_mask = special_partition(
            special_row,
            special_col,
            ordered_edge_indices,
            metadata.num_entities
    )

    # rebuild the graph in the correct order
    partitioned_joint_graph = coo_matrix(
            (special_data[keep_edge_mask],
             (special_row[keep_edge_mask], special_col[keep_edge_mask])),
            shape=global_joint_graph.shape
    )

    embed()
    return


def build_sparse_affinity_graph(args,
                                midxs, 
                                metadata,
                                knn_index,
                                sub_trainer,
                                build_coref_graph=False,
                                build_linking_graph=False):
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
                    for a, l in zip(midxs, mention_knn) for b in l if a != b]
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
                # get all of the mention kNN
                cand_gen_knn = knn_index.get_knn_limited_index(
                        midxs, exclude_index_idxs=midxs, k=args.k
                )
                linking_graph_edges.extend(
                    [tuple(sorted((a, b)))
                        for a, l in zip(midxs, cand_gen_knn) for b in l]
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


def _compute_coref_metrics_threshold(gold_clustering, mst, threshold):
    # get the true_labels 
    true_labels = [(i, x) for i, l in enumerate(gold_clustering) for x in l]
    true_labels = sorted(true_labels, key=lambda x : x[1])
    true_labels, true_midxs = zip(*true_labels)

    # prune mst (make sure to deepcopy!!!!)
    pruned_mst = deepcopy(mst)
    pruned_mask = pruned_mst.data > threshold
    pruned_mst.row = pruned_mst.row[pruned_mask]
    pruned_mst.col = pruned_mst.col[pruned_mask]
    pruned_mst.data = pruned_mst.data[pruned_mask]

    # get connected components to get clusters of `pruned_mst`
    n_components, labels = connected_components(
            csgraph=pruned_mst, directed=False, return_labels=True
    )
    pred_midxs = np.arange(labels.size)
    label_mask = np.isin(pred_midxs, true_midxs)
    pred_labels = zip(labels[label_mask], pred_midxs[label_mask])
    pred_labels = sorted(pred_labels, key=lambda x : x[1])
    pred_labels, _ = zip(*pred_labels)

    return {'fmi' : fowlkes_mallows_score(true_labels, pred_labels),
            'rand_index' : adjusted_rand_score(true_labels, pred_labels),
            'pred_labels' : pred_labels,
            'true_labels' : true_labels,
            'midxs' : true_midxs}


def _merge_sparse_graphs(graphs):
    global_graph_row = np.concatenate([graph.row for graph in graphs])
    global_graph_col = np.concatenate([graph.col for graph in graphs])
    global_graph_data = np.concatenate([graph.data for graph in graphs])
    global_graph = coo_matrix(
            (global_graph_data, (global_graph_row, global_graph_col)),
            shape=graphs[0].shape
    )
    return global_graph


def _get_global_maximum_spanning_tree(sparse_graph_list):
    # get MAXIMUM spanning trees by flipping affinities and computing
    # minimum spanning trees using these flipped affinities then flippling
    # the MST weights back
    for g in sparse_graph_list:
        g.data *= -1.0
    msts = [minimum_spanning_tree(g).tocoo() for g in sparse_graph_list]
    # make `msts` into MAXIMUM spanning trees 
    for mst in msts:
        mst.data *= -1.0

    # merge per doc things to global
    global_maximum_spanning_tree = _merge_sparse_graphs(msts)

    return global_maximum_spanning_tree
