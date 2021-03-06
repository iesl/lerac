from collections import defaultdict
from copy import deepcopy
import logging
import numpy as np
import pickle
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.csgraph import (minimum_spanning_tree,
                                  connected_components,
                                  breadth_first_tree)
from sklearn.cluster import KMeans
from sklearn.metrics import (fowlkes_mallows_score,
                             adjusted_rand_score)
from tqdm import tqdm

from utils.comm import get_rank, synchronize, broadcast
from utils.misc import flatten, unique

from evaluation.special_partition import special_partition

from IPython import embed


logger = logging.getLogger(__name__)


def eval_wdoc(args,
              example_dir,
              metadata,
              sub_trainer,
              save_fname=None):
    assert save_fname != None

    logger.info('Building within doc sparse graphs...')
    doc_level_graphs = []
    per_doc_coref_clusters = []
    for doc_clusters in tqdm(metadata.wdoc_clusters.values(), disable=(get_rank() != 0)):
        per_doc_coref_clusters.append(
                [[x for x in v if x != k] for k, v in doc_clusters.items()]
        )
        doc_mentions = np.asarray([x for k, v in doc_clusters.items()
                                        for x in v if x != k])
        doc_mentions = np.sort(doc_mentions)
        doc_level_graphs.append(
            build_sparse_affinity_graph(
                args,
                doc_mentions,
                example_dir,
                metadata,
                None,
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

    # build the joint whole graph
    joint_whole_graph = deepcopy(_merge_sparse_graphs(coref_graphs + linking_graphs))

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


    ########################################################################
    ## FIXME: hacking to get HAC working
    #
    #joint_whole_graph = _merge_sparse_graphs(coref_graphs + linking_graphs)

    #hierarchy_tree = np.full((2*joint_whole_graph.shape[0], 2), -1)
    #proposed_merges = np.vstack((joint_whole_graph.row, joint_whole_graph.col)).T

    #def _get_leaves(hierarchy_tree, internal_node):
    #    q = [internal_node]
    #    leaves = []
    #    while len(q) > 0:
    #        curr_node = q.pop()
    #        left_child = hierarchy_tree[curr_node][0]
    #        right_child = hierarchy_tree[curr_node][1]
    #        if left_child == -1:
    #            assert right_child == -1
    #            leaves.append(curr_node)
    #        else:
    #            q.append(left_child)
    #            q.append(right_child)
    #    return leaves

    #def _avg_linkage(joint_whole_graph, leaves_a, leaves_b):
    #    row_mask = np.isin(joint_whole_graph.row, leaves_a)\
    #               ^ np.isin(joint_whole_graph.row, leaves_b)
    #    col_mask = np.isin(joint_whole_graph.col, leaves_a)\
    #               ^ np.isin(joint_whole_graph.col, leaves_b)
    #    edge_weights = joint_whole_graph.data[row_mask & col_mask]
    #    if edge_weights.size == 0:
    #        return -np.inf
    #    return np.mean(edge_weights)
    #
    #merge_node_id = joint_whole_graph.shape[0] # start with the next possible index
    #valid_merge_exists = True
    #count = 0
    #while valid_merge_exists:
    #    valid_merge_exists = False
    #    max_linkage = 0.0
    #    max_a, max_b = None, None
    #    for pair in proposed_merges:
    #        a, b = tuple(pair)
    #        if a == b:
    #            continue
    #        valid_merge_exists = True
    #        leaves_a = _get_leaves(hierarchy_tree, a)
    #        leaves_b = _get_leaves(hierarchy_tree, b)
    #        linkage_score = _avg_linkage(joint_whole_graph, leaves_a, leaves_b)
    #        if linkage_score > max_linkage:
    #            max_a = a
    #            max_b = b
    #            max_linkage = linkage_score

    #    if not valid_merge_exists:
    #        continue

    #    # create new node in the hierarchy with id = `merge_node_id`
    #    hierarchy_tree[merge_node_id][0] = max_a
    #    hierarchy_tree[merge_node_id][1] = max_b

    #    # update all the relevant edges in `proposed_merges`
    #    join_mask = np.isin(proposed_merges, [max_a, max_b])
    #    proposed_merges[join_mask] = merge_node_id

    #    # increment for next merger
    #    merge_node_id += 1

    #    count += 1
    #    print(count)

    ########################################################################

    logger.info('Computing joint metrics...')
    slim_coref_graph = _get_global_maximum_spanning_tree(coref_graphs)
    joint_metrics = compute_joint_metrics(metadata,
                                          [slim_coref_graph, slim_linking_graph])
    logger.info('Done.')

    metrics = {
        'coref_fmi' : coref_metrics['fmi'],
        'coref_rand_index' : coref_metrics['rand_index'],
        'coref_threshold' : coref_metrics['threshold'],
        'vanilla_recall' : linking_metrics['vanilla_recall'],
        'vanilla_accuracy' : linking_metrics['vanilla_accuracy'],
        'joint_accuracy' : joint_metrics['joint_accuracy'],
        'joint_cc_recall' : joint_metrics['joint_cc_recall']
    }

    # save all of the predictions for later analysis
    save_data = {}
    save_data.update(coref_metrics)
    save_data.update(linking_metrics)
    save_data.update(joint_metrics)
    save_data.update({'metadata': metadata})
    save_data.update({'joint_whole_graph': joint_whole_graph})

    with open(save_fname, 'wb') as f:
        pickle.dump(save_data, f)

    synchronize()
    return metrics


def eval_xdoc(args, example_dir, metadata, knn_index, sub_trainer):
    pass


def compute_coref_metrics(gold_coref_clusters, coref_graphs, coref_threshold):
    global_gold_clustering = flatten(gold_coref_clusters)
    global_maximum_spanning_tree = _get_global_maximum_spanning_tree(
            coref_graphs
    )

    # compute metrics and choose threshold is one isn't specified
    if coref_threshold is None:
        logger.info('Generating candidate thresholds...')
        _edge_weights = global_maximum_spanning_tree.data.reshape(-1, 1)
        _num_thresholds = 1000
        if _edge_weights.shape[0] < _num_thresholds:
            candidate_thresholds = _edge_weights.reshape(-1,).tolist()
        else:
            kmeans = KMeans(n_clusters=_num_thresholds, random_state=0)
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
        assert eidx < midx
        mention2cand[midx].append(eidx)

    midxs = list(metadata.midx2eidx.keys())
    recall_hits = 0
    recall_total = len(midxs)
    no_candidates = []
    for midx in midxs:
        cands = mention2cand.get(midx, [])
        if len(cands) == 0:
            no_candidates.append(midx)
            continue
        if metadata.midx2eidx[midx] in cands:
            recall_hits += 1

    midxs = np.asarray(midxs)

    # build slim global linking graph for joint linking inference
    global_graph = global_graph.tocsc()
    def _get_slim_links(midx):
        col_entries = global_graph.getcol(midx).tocoo()
        if col_entries.nnz == 0:
            return (0, -np.inf)
        return max(zip(col_entries.row, col_entries.data), key=lambda x : x[1])
    v_max = np.vectorize(_get_slim_links)
    pred_eidxs, max_affinities = v_max(midxs)
    slim_global_graph = coo_matrix((max_affinities, (pred_eidxs, midxs)),
                                   shape=global_graph.shape)

    # compute linking accuracy
    missed_vanilla_midxs = []
    linking_hits, linking_total = 0, 0
    pred_midx2eidx = {m : e for m, e in zip(midxs, pred_eidxs)}
    for midx, true_eidx in metadata.midx2eidx.items():
        if true_eidx == pred_midx2eidx.get(midx, -1):
            linking_hits += 1
        elif true_eidx in metadata.midx2cand[midx]:
            missed_vanilla_midxs.append(midx)
        linking_total += 1

    results_dict = {
            'vanilla_recall' : recall_hits / recall_total,
            'vanilla_accuracy' : linking_hits / linking_total,
            'num_no_candidates' : len(no_candidates),
            'vanilla_pred_midx2eidx' : {m : e for m, e in zip(midxs, pred_eidxs)},
            'vanilla_slim_graph' : slim_global_graph
    }

    return results_dict, slim_global_graph


def compute_joint_metrics(metadata, joint_graphs):
    # get global joint graph
    global_joint_graph = _merge_sparse_graphs(joint_graphs)

    # compute recall at this stage
    _, cc_labels = connected_components(
            csgraph=global_joint_graph,
            directed=False,
            return_labels=True
    )
    cc_recall_hits, cc_recall_total = 0, 0
    for midx, true_eidx in metadata.midx2eidx.items():
        if cc_labels[midx] == cc_labels[true_eidx]:
            cc_recall_hits += 1
        cc_recall_total += 1
    cc_recall = cc_recall_hits / cc_recall_total

    # reorder the data for the requirements of the `special_partition` function
    _row = np.concatenate((global_joint_graph.row, global_joint_graph.col))
    _col = np.concatenate((global_joint_graph.col, global_joint_graph.row))
    _data = np.concatenate((global_joint_graph.data, global_joint_graph.data))
    tuples = zip(_row, _col, _data)
    tuples = sorted(tuples, key=lambda x : (x[1], -x[0])) # sorted this way for nice DFS
    special_row, special_col, special_data = zip(*tuples)
    special_row = np.asarray(special_row, dtype=np.int)
    special_col = np.asarray(special_col, dtype=np.int)
    special_data = np.asarray(special_data)

    # reconstruct the global joint graph shape
    global_joint_graph = coo_matrix(
            (special_data, (special_row, special_col)),
            shape=global_joint_graph.shape
    )

    # create siamese indices for easy lookups during partitioning
    edge_indices = {e : i for i, e in enumerate(zip(special_row, special_col))}
    siamese_indices = [edge_indices[(c, r)]
                            for r, c in zip(special_row, special_col)]
    siamese_indices = np.asarray(siamese_indices)

    # order the edges in ascending order according to affinities
    ordered_edge_indices = np.argsort(special_data)

    # partition the graph using the `keep_edge_mask`
    keep_edge_mask = special_partition(
            special_row,
            special_col,
            ordered_edge_indices,
            siamese_indices,
            metadata.num_entities
    )

    # build the partitioned graph
    partitioned_joint_graph = coo_matrix(
            (special_data[keep_edge_mask],
             (special_row[keep_edge_mask], special_col[keep_edge_mask])),
            shape=global_joint_graph.shape
    )

    # infer the linking decisions from clusters (connected compoents) of
    # the partitioned joint mention and entity graph 
    _, labels = connected_components(
            csgraph=partitioned_joint_graph,
            directed=False,
            return_labels=True
    )
    unique_labels, cc_sizes = np.unique(labels, return_counts=True)
    components = defaultdict(list)
    filtered_labels = unique_labels[cc_sizes > 1]
    for idx, cc_label in enumerate(labels):
        if cc_label in filtered_labels:
            components[cc_label].append(idx)
    pred_midx2eidx = {}
    for cluster_nodes in components.values():
        eidxs = [x for x in cluster_nodes if x < metadata.num_entities]
        midxs = [x for x in cluster_nodes if x >= metadata.num_entities]
        assert len(eidxs) == 1
        assert len(midxs) >= 1
        eidx = eidxs[0]
        for midx in midxs:
            pred_midx2eidx[midx] = eidx

    joint_hits, joint_total = 0, 0
    for midx, true_eidx in metadata.midx2eidx.items():
        if pred_midx2eidx.get(midx, -1) == true_eidx:
            joint_hits += 1
        joint_total += 1

    return {'joint_accuracy' : joint_hits / joint_total,
            'joint_pred_midx2eidx': pred_midx2eidx,
            'joint_cc_recall': cc_recall,
            'joint_slim_graph': global_joint_graph,
            'joint_keep_edge_mask': keep_edge_mask}


def build_sparse_affinity_graph(args,
                                midxs, 
                                example_dir,
                                metadata,
                                knn_index,
                                sub_trainer,
                                build_coref_graph=False,
                                build_linking_graph=False):

    assert build_coref_graph or build_linking_graph

    coref_graph = None
    linking_graph = None
    if get_rank() == 0:
        mention_knn = None
        if build_coref_graph or args.available_entities == 'knn_candidates':
            ## get all of the mention kNN
            #mention_knn = knn_index.get_knn_limited_index(
            #        midxs, include_index_idxs=midxs, k=args.k+1
            #)
            #mention_knn = mention_knn[:,1:]
            midx2doc = {}
            doc2midx = defaultdict(list)
            for doc_id, wdoc_clusters in metadata.wdoc_clusters.items():
                doc2midx[doc_id] = flatten(list(wdoc_clusters.values()))
                for midx in doc2midx[doc_id]:
                    midx2doc[midx] = doc_id
            mention_knn = []
            for midx in midxs:
                mention_knn.append([x for x in doc2midx[midx2doc[midx]] if x != midx and x >= args.num_entities])

    if build_coref_graph:
        # list of edges for sparse graph we will build
        coref_graph_edges = []
        if get_rank() == 0:
            # add mention-mention edges to list
            coref_graph_edges.extend(
                [tuple(sorted((a, b)))
                    for a, l in zip(midxs, mention_knn) for b in l if a != b]
            )
            coref_graph_edges = unique(coref_graph_edges)

        # broadcast edges to all processes to compute affinities
        coref_graph_edges = broadcast(coref_graph_edges, src=0)
        affinities = sub_trainer.get_edge_affinities(
                coref_graph_edges, example_dir, knn_index
        )

        # affinities are gathered to only rank 0 process
        if get_rank() == 0:
            # build the graph
            coref_graph_edges = np.asarray(coref_graph_edges).T
            _sparse_num = metadata.num_mentions + metadata.num_entities
            coref_graph = coo_matrix(
                        (affinities, coref_graph_edges),
                        shape=(_sparse_num, _sparse_num)
            )

    if build_linking_graph:
        # list of edges for sparse graph we will build
        linking_graph_edges = []
        if get_rank() == 0:
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
                if args.clustering_domain == 'within_doc':
                    for midx in midxs:
                        candidates = metadata.midx2cand.get(midx, [])
                        if len(candidates) > 0:
                            linking_graph_edges.extend(
                                [tuple(sorted((midx, eidx))) for eidx in candidates]
                            )
                elif args.clustering_domain == 'cross_doc':
                    raise NotImplementedError('unsupported clustering_domain')
                else:
                    raise ValueError('unsupported clustering_domain')
            else: # 'open_domain'
                # get all of the mention kNN
                cand_gen_knn = knn_index.get_knn_limited_index(
                        midxs,
                        include_index_idxs=np.arange(metadata.num_entities),
                        k=args.k
                )
                linking_graph_edges.extend(
                    [tuple(sorted((a, b)))
                        for a, l in zip(midxs, cand_gen_knn) for b in l]
                )

            # get all of the edges
            linking_graph_edges = unique(linking_graph_edges)

        # broadcast edges to all processes to compute affinities
        linking_graph_edges = broadcast(linking_graph_edges, src=0)
        affinities = sub_trainer.get_edge_affinities(
                linking_graph_edges, example_dir, knn_index
        )

        # affinities are gathered to only rank 0 process
        if get_rank() == 0:
            # build the graph
            linking_graph_edges = np.asarray(linking_graph_edges).T
            _sparse_num = metadata.num_mentions + metadata.num_entities
            linking_graph = coo_matrix((affinities, linking_graph_edges),
                                          shape=(_sparse_num, _sparse_num))

        if args.available_entities == 'knn_candidates':
            assert args.clustering_domain == 'within_doc'

            # pick expansion edges based on coref knn mentions
            expansion_factor = 5
            expansion_edges = []
            if get_rank() == 0:
                def _get_top_k(midx, graph, k):
                    row_entries = graph.getrow(midx).tocoo()
                    col_entries = graph.getcol(midx).tocoo()
                    entries = zip(
                        np.concatenate((row_entries.col, col_entries.row),
                                       axis=0),
                        np.concatenate((row_entries.data, col_entries.data),
                                       axis=0)
                    )
                    entries = list(entries)
                    if len(entries) == 0:
                        return []

                    sorted_data = sorted(entries, key=lambda x : x[1],
                                         reverse=True)
                    top_k, _ = zip(*sorted_data[:k])
                    return top_k

                top_k_coref = lambda i : _get_top_k(i, coref_graph,
                                                    expansion_factor)
                top_k_linking = lambda i : _get_top_k(i, linking_graph,
                                                      expansion_factor)
                for midx in midxs:
                    for coref_midx in top_k_coref(midx):
                        expansion_edges.extend([
                            tuple(sorted((x, midx)))
                                for x in top_k_linking(coref_midx)
                                    if x not in metadata.midx2cand[midx]
                        ])
                expansion_edges = unique(expansion_edges)

            # score the expanded candidate edges
            expansion_edges = broadcast(expansion_edges, src=0)
            expansion_affinities = sub_trainer.get_edge_affinities(
                    expansion_edges, example_dir, knn_index
            )

            if get_rank() == 0:
                # build the graph
                expansion_edges = np.asarray(expansion_edges).T
                linking_graph_edges = np.concatenate(
                        (linking_graph_edges, expansion_edges), axis=1
                )
                affinities += expansion_affinities
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

    # NOTE: !!!! have to flip back in memory stuff !!!!
    for g in sparse_graph_list:
        g.data *= -1.0

    # make `msts` into MAXIMUM spanning trees 
    for mst in msts:
        mst.data *= -1.0

    # merge per doc things to global
    global_maximum_spanning_tree = _merge_sparse_graphs(msts)

    return global_maximum_spanning_tree
