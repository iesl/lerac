import faiss
from tabulate import tabulate
from types import SimpleNamespace
import numpy as np
import os
import pickle
import torch

from IPython import embed


EXTERNAL_BASE_DIR=('/home/ds-share/data2/users/rangell/lerac/coref_entity_linking/'
                   'experiments/mm_st21pv_long_entities/cluster_linking')
EXP_ID='exp4'
CKPT_ID='checkpoint-7442' # will need this for later - right now pickle files saved in main folder not checkpoint folder

# TODO:
# - load pickle files from eval
# - load dataset metadata pytorch serialized file
# - change main codebase to also save knn_index???
# - 

def load_pickle_file(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_data_files():
    knn_index_fname =  os.path.join(
            EXTERNAL_BASE_DIR,
            EXP_ID,
            CKPT_ID,
            'knn_index.val.debug_results.pkl'
    )
    embed_fname =  os.path.join(
            EXTERNAL_BASE_DIR,
            EXP_ID,
            CKPT_ID,
            'embed.val.debug_results.pkl'
    )
    concat_fname =  os.path.join(
            EXTERNAL_BASE_DIR,
            EXP_ID,
            CKPT_ID,
            'concat.val.debug_results.pkl'
    )
    metadata_fname = 'data/mm_st21pv_long_entities/cache/val/metadata.pt'

    knn_index_tuple = load_pickle_file(knn_index_fname)
    embed_results_data = load_pickle_file(embed_fname)
    concat_results_data = load_pickle_file(concat_fname)
    metadata = torch.load(metadata_fname)

    return metadata, knn_index_tuple, embed_results_data, concat_results_data


def compute_accuracy(results_data, metadata, pred_key=''):
    assert pred_key in results_data.keys()
    hits, total = 0, 0
    for midx, true_label in metadata.midx2eidx.items():
        pred_label = results_data[pred_key].get(midx, -1)
        assert 'joint' not in pred_key or pred_label != -1
        if true_label == pred_label:
            hits += 1
        total += 1
    return hits / total


def compute_gold_clusters_linking_recall(results_data, metadata, clusters):
    # `clusters` is a list of lists of midxs
    hits, total = 0, 0
    for c in clusters:
        _linked_eidxs = list(map(lambda midx : metadata.midx2eidx[midx], c))
        assert all([x == _linked_eidxs[0] for x in _linked_eidxs])
        gold_eidx = _linked_eidxs[0]
        pred_eidxs = map(
                lambda x : results_data['vanilla_pred_midx2eidx'].get(x, -1), c
        )
        if gold_eidx in pred_eidxs:
            hits += len(c)
        total += len(c)
    return hits / total


def compute_gold_clusters_linking_accuracy(results_data,
                                           metadata,
                                           knn_idxs,
                                           knn_X,
                                           clusters,
                                           include_gold_eidxs=False):
    # `clusters` is a list of lists of midxs
    _row = results_data['vanilla_slim_graph'].row
    _col = results_data['vanilla_slim_graph'].col
    _data = results_data['vanilla_slim_graph'].data
    pred_midx2eidx_w_scores = {}
    for r, c, d in zip(_row, _col, _data):
        pred_midx2eidx_w_scores[c] = (r, d)

    added_gold_eidxs = 0
    if include_gold_eidxs:
        inverted_idxs = {midx : i for i, midx in enumerate(knn_idxs)}
        for midx, eidx in metadata.midx2eidx.items():
            if eidx in inverted_idxs.keys():
                eidx_score = np.dot(
                    knn_X[inverted_idxs[midx]],
                    knn_X[inverted_idxs[eidx]]
                )
                if (midx not in pred_midx2eidx_w_scores.keys()
                        or eidx_score > pred_midx2eidx_w_scores[midx][1]):
                    try:
                        assert eidx not in metadata.midx2cand[midx]
                    except:
                        embed()
                        exit()
                    pred_midx2eidx_w_scores[midx] = (eidx, eidx_score)
                    added_gold_eidxs += 1

    print('added gold eidxs {}'.format(added_gold_eidxs))

    hits, total = 0, 0
    for c in clusters:
        _linked_eidxs = list(map(lambda midx : metadata.midx2eidx[midx], c))
        assert all([x == _linked_eidxs[0] for x in _linked_eidxs])
        gold_eidx = _linked_eidxs[0]
        pred_eidxs = map(
                lambda x : pred_midx2eidx_w_scores.get(x, (-1, 0.0)), c
        )
        if gold_eidx == max(pred_eidxs, key=lambda x : x[1])[0]:
            hits += len(c)
        total += len(c)
    return hits / total


def knn_index_check(metadata, knn_idxs, knn_X, results):
    entity_mask = knn_idxs < metadata.num_entities
    entity_mask_index2idx = np.where(entity_mask)[0]
    mention_mask_index2idx = np.where(~entity_mask)[0]

    v_mention_mask_index2idx = np.vectorize(
            lambda i : knn_idxs[mention_mask_index2idx[i]]
    )
    v_entity_mask_index2idx = np.vectorize(
            lambda i : knn_idxs[entity_mask_index2idx[i]]
    )

    entity_reps = knn_X[entity_mask]
    mention_reps = knn_X[~entity_mask]

    d = mention_reps.shape[1]
    knn_index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
    knn_index.add(entity_reps)

    recall_dict = {}
    for k in [2**i for i in range(10)]:
        hits, total = 0, 0
        D, I = knn_index.search(mention_reps, k)
        for i, midx in enumerate(v_mention_mask_index2idx(np.arange(I.shape[0]))):
            nn_eidx = v_entity_mask_index2idx(I[i])
            if metadata.midx2eidx[midx] in nn_eidx:
                hits += 1
            total += 1
        recall_dict[k] = hits / total
    embed()
    exit()
            

if __name__ == '__main__':
    print('Loading data...')
    metadata, knn_index_tuple, embed_results_data, concat_results_data = load_data_files()
    print('Done.')

    results = SimpleNamespace()
    knn_idxs, knn_X = knn_index_tuple

    print('kNN index check...')
    knn_index_check(metadata, knn_idxs, knn_X, results)
    print('Done.')

    # compute list of lists of midxs for gold clusters analysis
    wdoc_clusters =  [
            [x for x in cluster if x >= metadata.num_entities] 
                for doc in metadata.wdoc_clusters.values()
                    for cluster in doc.values()
    ]


    print('Computing accuracies...')
    results.embed_vanilla_accuracy = compute_accuracy(
            embed_results_data, metadata, pred_key='vanilla_pred_midx2eidx'
    )
    results.concat_vanilla_accuracy = compute_accuracy(
            concat_results_data, metadata, pred_key='vanilla_pred_midx2eidx'
    )

    results.embed_joint_accuracy = compute_accuracy(
            embed_results_data, metadata, pred_key='joint_pred_midx2eidx'
    )
    results.concat_joint_accuracy = compute_accuracy(
            concat_results_data, metadata, pred_key='joint_pred_midx2eidx'
    )

    results.embed_gold_clusters_recall = compute_gold_clusters_linking_recall(
            embed_results_data, metadata, wdoc_clusters
    )
    results.concat_gold_clusters_recall = compute_gold_clusters_linking_recall(
            concat_results_data, metadata, wdoc_clusters
    )


    results.embed_gold_clusters_accuracy = compute_gold_clusters_linking_accuracy(
            embed_results_data, metadata, knn_idxs, knn_X, wdoc_clusters
    )
    results.concat_gold_clusters_accuracy = compute_gold_clusters_linking_accuracy(
            concat_results_data, metadata, knn_idxs, knn_X, wdoc_clusters
    )

    results.embed_gold_clusters_accuracy_w_gt_entity = compute_gold_clusters_linking_accuracy(
            embed_results_data, metadata, knn_idxs, knn_X, wdoc_clusters, include_gold_eidxs=True
    )
    #results.concat_gold_clusters_accuracy_w_gt_entity = compute_gold_clusters_linking_accuracy(
    #        concat_results_data, metadata, knn_idxs, knn_X, wdoc_clusters, include_gold_eidxs=True
    #)
            

    # TODO:
    # - what is the (vanilla and joint) embed model getting right that the concat model is getting wrong
    # - what is the (vanilla and joint) concat model getting right that the embed model is getting wrong
    # - 
    

    print('Done.')


    print()
    rows = []
    for field in filter(lambda s : '__' not in s, dir(results)):
        rows.append([field, getattr(results, field)])
    print(tabulate(rows, headers=['Metric', 'Value']), '\n')


    embed()
    exit()
