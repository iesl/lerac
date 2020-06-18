#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import cython
import numpy as np
cimport numpy as np
from tqdm import tqdm

INT = np.int
BOOL = np.bool

ctypedef np.int_t INT_t
ctypedef np.npy_bool BOOL_t


@cython.boundscheck(False)
@cython.wraparound(False)
def _build_col_wise_adj_index(np.ndarray[INT_t, ndim=1] col,
                              INT_t col_max_value,
                              INT_t num_entities):
    # requires: sorted col ascending order
    cdef INT_t index_size = col_max_value - num_entities + 1
    cdef np.ndarray[INT_t, ndim=2] col_wise_adj_index = np.zeros([index_size, 2], dtype=INT)
    cdef INT_t adjusted_c

    cdef INT_t curr_col = col[0] - num_entities
    for i, c in enumerate(col):
        adjusted_c = c - num_entities
        if adjusted_c != curr_col:
            curr_col = adjusted_c
            col_wise_adj_index[curr_col, 0] = i
            col_wise_adj_index[curr_col, 1] = i+1
        else:
            col_wise_adj_index[curr_col, 1] += 1

    return col_wise_adj_index


@cython.boundscheck(False)
def _has_entity_in_component(list stack,
                             np.ndarray[INT_t, ndim=1] row,
                             np.ndarray[INT_t, ndim=2] col_wise_adj_index,
                             INT_t num_entities):
    # performs DFS and returns `True` whenever it hits an entity
    cdef bint has_entity = False
    cdef INT_t index_size = col_wise_adj_index.shape[0]
    cdef set visited = set()
    cdef INT_t curr_node
    
    while len(stack) > 0:
        # pop
        curr_node = stack[-1]
        stack = stack[:-1]

        # check if `curr_node` is an entity
        if curr_node < num_entities:
            has_entity = True
            break

        # check if we've visited `curr_node`
        if curr_node in visited:
            continue
        visited.add(curr_node)

        # get neighbors of `curr_node` and push them onto the stack
        curr_node -= num_entities
        start_index = col_wise_adj_index[curr_node, 0]
        end_index = col_wise_adj_index[curr_node, 1]
        stack.extend(row[start_index:end_index].tolist())
    
    return has_entity


@cython.boundscheck(False)
@cython.wraparound(False)
def special_partition(np.ndarray[INT_t, ndim=1] row, 
                      np.ndarray[INT_t, ndim=1] col,
                      np.ndarray[INT_t, ndim=1] ordered_indices,
                      INT_t num_entities):
    assert row.shape[0] == col.shape[0]
    assert row.shape[0] == ordered_indices.shape[0]

    cdef INT_t num_edges = row.shape[0]
    cdef np.ndarray[BOOL_t, ndim=1] keep_mask = np.ones([num_edges,], dtype=BOOL)
    cdef np.ndarray[INT_t, ndim=1] tmp_row, tmp_col
    cdef INT_t r, c
    cdef bint has_entity_r, has_entity_c
    cdef INT_t col_max_value = np.max(col)

    # has shape [N, 2]; [:,0] are starting indices and [:,1] are (exclusive) ending indices
    cdef np.ndarray[INT_t, ndim=2] col_wise_adj_index
    cdef INT_t adjusted_c
    col_wise_adj_index = _build_col_wise_adj_index(
            col, col_max_value, num_entities
    )

    for i in tqdm(ordered_indices, desc='dropping joint edges'):
        r = row[i]
        c = col[i]
        keep_mask[i] = False

        # create the temporary graph we want to check
        tmp_row = row[keep_mask]
        tmp_col = col[keep_mask]

        # update the adj list index
        adjusted_c = c - num_entities
        col_wise_adj_index[adjusted_c, 1] -= 1
        if adjusted_c + 1 < col_wise_adj_index.shape[0]:
            col_wise_adj_index[adjusted_c + 1, :] -= 1

        # check if we can remove the edge (r, c) 
        has_entity_r = _has_entity_in_component(
                [r], row, col_wise_adj_index, num_entities
        )
        has_entity_c = _has_entity_in_component(
                [c], row, col_wise_adj_index, num_entities
        )

        # add the edge back if we need it
        if not(has_entity_r and has_entity_c):
            keep_mask[i] = True
            adjusted_c = c - num_entities
            col_wise_adj_index[adjusted_c, 1] += 1
            if adjusted_c + 1 < col_wise_adj_index.shape[0]:
                col_wise_adj_index[adjusted_c + 1, :] += 1

    return keep_mask
