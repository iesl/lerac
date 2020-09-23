from collections import defaultdict
import faiss
import json
from tabulate import tabulate
from types import SimpleNamespace
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm

from src.evaluation.special_hac import special_hac

from IPython import embed


EXTERNAL_BASE_DIR=('/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/'
                   'experiments/mm_st21pv_long_entities/cluster_linking')
EXP_ID='exp_mst_2-3_hybrid'
CKPT_ID='checkpoint-3721' 

# TODO:
# - 

def load_pickle_file(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_data_files():
    concat_fname =  os.path.join(
            EXTERNAL_BASE_DIR,
            EXP_ID,
            CKPT_ID,
            'concat.val.debug_results.pkl'
    )

    concat_results_data = load_pickle_file(concat_fname)

    return concat_results_data


def compute_accuracy(results_data, metadata, pred_key=''):
    assert pred_key in results_data.keys()
    correct_midxs = []
    hits, total = 0, 0
    for midx, true_label in metadata.midx2eidx.items():
        pred_label = results_data[pred_key].get(midx, -1)
        try:
            assert 'joint' not in pred_key or pred_label != -1
        except:
            embed()
            exit()
        if true_label == pred_label:
            hits += 1
            correct_midxs.append(midx)
        total += 1
    return hits / total, correct_midxs


def compute_gold_clusters_recall(results_data, metadata, clusters):
    # `clusters` is a list of lists of midxs
    correct_midxs = []
    hits, total = 0, 0
    for c in clusters:
        _linked_eidxs = list(map(lambda midx : metadata.midx2eidx[midx], c))
        #assert all([x == _linked_eidxs[0] for x in _linked_eidxs])
        gold_eidx = _linked_eidxs[0]
        pred_eidxs = map(
                lambda x : results_data['vanilla_pred_midx2eidx'].get(x, -1), c
        )
        if gold_eidx in pred_eidxs:
            hits += len(c)
            correct_midxs.extend(c)
        total += len(c)
    return hits / total, correct_midxs


# single linkage linking
def compute_gold_clusters_accuracy(results_data, metadata, clusters):
    # `clusters` is a list of lists of midxs
    _row = results_data['vanilla_slim_graph'].row
    _col = results_data['vanilla_slim_graph'].col
    _data = results_data['vanilla_slim_graph'].data
    pred_midx2eidx_w_scores = {}
    for r, c, d in zip(_row, _col, _data):
        pred_midx2eidx_w_scores[c] = (r, d)

    correct_midxs = []
    hits, total = 0, 0
    for c in clusters:
        _linked_eidxs = list(map(lambda midx : metadata.midx2eidx[midx], c))
        #assert all([x == _linked_eidxs[0] for x in _linked_eidxs])
        gold_eidx = _linked_eidxs[0]
        pred_eidxs = map(
                lambda x : pred_midx2eidx_w_scores.get(x, (-1, 0.0)), c
        )
        if gold_eidx == max(pred_eidxs, key=lambda x : x[1])[0]:
            correct_midxs.extend(c)
            hits += len(c)
        total += len(c)
    return hits / total, correct_midxs


# average linkage linking
def compute_gold_clusters_avg_link_accuracy(results_data, metadata, clusters):
    # `clusters` is a list of lists of midxs
    _row = results_data['joint_whole_graph'].row
    _col = results_data['joint_whole_graph'].col
    _data = results_data['joint_whole_graph'].data
    pred_midx2eidx_w_scores = defaultdict(list)
    for r, c, d in zip(_row, _col, _data):
        pred_midx2eidx_w_scores[c].append((r, d))

    correct_midxs = []
    hits, total = 0, 0
    for c in clusters:
        _linked_eidxs = list(map(lambda midx : metadata.midx2eidx[midx], c))
        #assert all([x == _linked_eidxs[0] for x in _linked_eidxs])
        gold_eidx = _linked_eidxs[0]

        pred_eidxs = defaultdict(list)
        for x in c:
            for eidx, score in pred_midx2eidx_w_scores[x]:
                pred_eidxs[eidx].append(score)
        pred_eidxs = [(k, np.sum(v)/len(c)) for k, v in pred_eidxs.items() if k < metadata.num_entities]

        if len(pred_eidxs) != 0:

            individual_correct = []
            for x in c:
                scored_eidxs = pred_midx2eidx_w_scores[x]
                if len(scored_eidxs) == 0:
                    individual_correct.append(False)
                else:
                    individual_correct.append(gold_eidx == max(pred_midx2eidx_w_scores[x], key=lambda y : y[1])[0])

            if gold_eidx == max(pred_eidxs, key=lambda x : x[1])[0]:
                correct_midxs.extend(c)
                hits += len(c)
            elif np.sum(individual_correct) > 1:
                #embed()
                #exit()
                pass
        total += len(c)
    return hits / total, correct_midxs


def list_diff(list_a, list_b):
    a_not_b = [x for x in list_a if x not in list_b]
    b_not_a = [x for x in list_b if x not in list_a]
    return a_not_b, b_not_a
            

if __name__ == '__main__':

    print('Debugging experiment: {}, checkpoint: {}'.format(EXP_ID, CKPT_ID))

    print('Loading data...')
    concat_results_data = load_data_files()
    metadata = concat_results_data['metadata']
    print('Done.')

    results = SimpleNamespace()

    # compute list of lists of midxs for gold clusters analysis
    wdoc_clusters =  [
            [x for x in cluster if x >= metadata.num_entities] 
                for doc in metadata.wdoc_clusters.values()
                    for cluster in doc.values()
    ]

    pred_wdoc_clusters = defaultdict(list)
    for midx, label in zip(concat_results_data['midxs'], concat_results_data['pred_labels']):
        pred_wdoc_clusters[label].append(midx)
    pred_wdoc_clusters = list(pred_wdoc_clusters.values())

    print('Computing accuracies...')
    results.concat_vanilla_accuracy, cva_correct_midxs = compute_accuracy(
            concat_results_data, metadata, pred_key='vanilla_pred_midx2eidx'
    )

    results.concat_joint_accuracy, cja_correct_midxs = compute_accuracy(
            concat_results_data, metadata, pred_key='joint_pred_midx2eidx'
    )

    results.concat_gold_clusters_recall, cgcr_correct_midxs = compute_gold_clusters_recall(
            concat_results_data, metadata, wdoc_clusters
    )

    results.concat_gold_clusters_accuracy, cgca_correct_midxs = compute_gold_clusters_accuracy(
            concat_results_data, metadata, wdoc_clusters
    )

    results.concat_gold_clusters_avg_link_accuracy, _ = \
            compute_gold_clusters_avg_link_accuracy(concat_results_data, metadata, wdoc_clusters)

    results.concat_pred_clusters_recall, cgcr_correct_midxs = compute_gold_clusters_recall(
            concat_results_data, metadata, pred_wdoc_clusters
    )

    results.concat_pred_clusters_accuracy, cgca_correct_midxs = compute_gold_clusters_accuracy(
            concat_results_data, metadata, pred_wdoc_clusters
    )

    results.concat_pred_clusters_avg_link_accuracy, _ = \
            compute_gold_clusters_avg_link_accuracy(concat_results_data, metadata, pred_wdoc_clusters)
    print('Done.')

    print()
    rows = []
    for field in filter(lambda s : '__' not in s, dir(results)):
        rows.append([field, getattr(results, field)])
    print(tabulate(rows, headers=['Metric', 'Value']), '\n')


    ### special hac experimentation

    whole_graph = concat_results_data['joint_whole_graph']

    ## sparsify the graph
    def _get_top_k(midx, csr_graph, csc_graph, k):
        row_entries = csr_graph.getrow(midx).tocoo()
        col_entries = csc_graph.getcol(midx).tocoo()
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
        return sorted_data[:k]

    print('Starting sparsification...')
    k = 10
    sparse_row = []
    sparse_col = []
    sparse_data = []
    csr_whole_graph = whole_graph.tocsr()
    csc_whole_graph = whole_graph.tocsc()
    for i, midx in tqdm(enumerate(metadata.midx2cand.keys()), desc='sparsifying'):
        for nbr, weight in _get_top_k(midx, csr_whole_graph, csc_whole_graph, k):
            a, b = tuple(sorted([midx, nbr])) 
            sparse_row.append(a)
            sparse_col.append(b)
            sparse_data.append(weight)

    row, col, data = zip(*list(set(zip(sparse_row, sparse_col, sparse_data))))

    row = np.asarray(row, dtype=np.int)
    col = np.asarray(col, dtype=np.int)
    weights = np.asarray(data, dtype=np.float)
    
    mask = (weights > 0.5)
    row = row[mask]
    col = col[mask]
    weights = weights[mask]

    num_entities = metadata.num_entities
    num_mentions = metadata.num_mentions
    print('Done.')

    leaves = special_hac(row, col, weights, num_entities, num_mentions)

    # leaves with key > (num_mentions + num_entities) are our final clusters

    embed()
    exit()
