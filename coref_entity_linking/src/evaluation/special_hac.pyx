#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
import time
from tqdm import tqdm

BOOL = np.bool
INT = np.int
FLOAT = np.float

ctypedef np.int_t INT_t
ctypedef np.float_t FLOAT_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_leaves(np.ndarray[INT_t, ndim=2] hierarchy_tree,
                INT_t internal_node,
                INT_t num_entities):
    cdef list q = [internal_node]
    cdef list leaves = []
    cdef INT_t curr_node, left_child, right_child
    cdef bint has_entity = False
    while len(q) > 0:
        curr_node = q.pop()
        left_child = hierarchy_tree[curr_node][0]
        right_child = hierarchy_tree[curr_node][1]
        if left_child == -1:
            assert right_child == -1
            leaves.append(curr_node)
            if curr_node < num_entities:
                has_entity = True
        else:
            q.append(left_child)
            q.append(right_child)
    return leaves, has_entity


@cython.boundscheck(False)
@cython.wraparound(False)
def _avg_linkage(np.ndarray[INT_t, ndim=1] row, 
                 np.ndarray[INT_t, ndim=1] col,
                 np.ndarray[FLOAT_t, ndim=1] weights,
                 list leaves_a,
                 list leaves_b):
    cdef np.ndarray[BOOL_t, ndim=1] row_mask = np.isin(row, leaves_a) ^ np.isin(row, leaves_b)
    cdef np.ndarray[BOOL_t, ndim=1] col_mask = np.isin(col, leaves_a) ^ np.isin(col, leaves_b)
    cdef np.ndarray[FLOAT_t, ndim=1] edge_weights = weights[row_mask & col_mask]
    if edge_weights.size == 0:
        return -np.inf
    return np.sum(edge_weights) / (len(leaves_a) * len(leaves_b))



@cython.boundscheck(False)
@cython.wraparound(False)
def special_hac(np.ndarray[INT_t, ndim=1] row, 
                np.ndarray[INT_t, ndim=1] col,
                np.ndarray[FLOAT_t, ndim=1] weights,
                INT_t num_entities,
                INT_t num_mentions):
    assert row.shape[0] == col.shape[0]
    assert row.shape[0] == weights.shape[0]

    cdef np.ndarray[INT_t, ndim=2] hierarchy_tree = np.full(
            (2*(num_mentions + num_entities) - 1, 2), -1
    )
    cdef np.ndarray[INT_t, ndim=2] proposed_merges = np.vstack((row, col)).T
    cdef np.ndarray[BOOL_t, ndim=1] edges_out_a, edges_out_b

    cdef INT_t merge_node_id = num_mentions + num_entities # start with the next possible index
    cdef bint valid_merge_exists = True
    cdef INT_t count = 0
    cdef FLOAT_t max_linkage, linkage_score
    cdef INT_t max_a, max_b, a, b, tmp
    cdef list leaves_a, leaves_b
    cdef dict linkage_cache = {(r, c) : w for r, c, w in zip(row, col, weights)}
    cdef dict leaves, has_entity
    leaves = {r : [r] for r in row}
    leaves.update({c : [c] for c in col})
    has_entity = {x : x < num_entities for x in leaves.keys()}

    start_time = time.time()
    while valid_merge_exists:
        valid_merge_exists = False
        max_linkage = -np.inf
        for pair in proposed_merges:
            a, b = tuple(pair)

            # if the endpoints of this edge are already in the same cluster,
            # we cannot merge them
            if a == b:
                continue

            # make sure a < b always 
            if a > b:
                a, b = b, a

            if (a, b) not in linkage_cache.keys():
                # get the leaves of both ends of this edge
                leaves_a = leaves[a]
                leaves_b = leaves[b]
                has_entity_a = has_entity[a]
                has_entity_b = has_entity[b]

                # if the endpoints of this edge both are in clusters which have
                # an entity, we cannot merge them
                if has_entity_a and has_entity_b:
                    linkage_cache[(a, b)] = -np.inf
                    continue

                #linkage_score = _avg_linkage(row, col, weights, leaves_a, leaves_b)

                edges_out_a = (proposed_merges[:, 0] == a) ^ (proposed_merges[:, 1] == a)
                edges_out_b = (proposed_merges[:, 0] == b) ^ (proposed_merges[:, 1] == b)
                linkage_score = np.sum(weights[edges_out_a & edges_out_b]) / (len(leaves_a) * len(leaves_b))

                linkage_cache[(a, b)] = linkage_score

            linkage_score = linkage_cache[(a, b)]
            if linkage_score == -np.inf:
                continue

            valid_merge_exists = True

            if linkage_score > max_linkage:
                max_a = a
                max_b = b
                max_linkage = linkage_score

        if not valid_merge_exists:
            continue

        # create new node in the hierarchy with id = `merge_node_id`
        hierarchy_tree[merge_node_id][0] = max_a
        hierarchy_tree[merge_node_id][1] = max_b

        # update all the relevant edges in `proposed_merges`
        join_mask = np.isin(proposed_merges, [max_a, max_b])
        proposed_merges[join_mask] = merge_node_id

        # update helpful dictionaries
        leaves[merge_node_id] = leaves[max_a] + leaves[max_b]
        has_entity[merge_node_id] = has_entity[max_a] | has_entity[max_b]

        # delete non-needed keys
        del leaves[max_a]
        del leaves[max_b]
        del has_entity[max_a]
        del has_entity[max_b]

        # increment for next merger
        merge_node_id += 1

        count += 1
        if count % 100 == 0:
            print('{} merges in {} sec'.format(count, time.time() - start_time))

    return leaves
