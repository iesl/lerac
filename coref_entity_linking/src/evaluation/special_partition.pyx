#cython: language_level=3
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np

DTYPE = np.int

ctypedef np.int_t DTYPE_t


def special_partition(np.ndarray f, np.ndarray g):
    assert f.dtype == DTYPE and g.dtype == DTYPE

    cdef int total_cells = (f.shape[0] * f.shape[1]) + (g.shape[0] * g.shape[1])

    return total_cells
