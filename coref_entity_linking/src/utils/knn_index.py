import os
import pickle
import logging
import faiss
import math
import numpy as np
import numpy.ma as ma
from tqdm import tqdm

from utils.comm import get_rank, synchronize

from IPython import embed


logger = logging.getLogger(__name__)


class NearestNeighborIndex(object):
    """
    Base class, but also treats all points the same -- no distinction b/w
    mentions and entities.
    """
    def __init__(self, args, sub_trainer, dataloader, name=''):
        self.args = args
        self.sub_trainer = sub_trainer
        self.dataloader = dataloader
        self.name = name
        self._compute_embeddings()

    def refresh_index(self):
        """ Refresh the index by recomputing the embeddings for all points. """
        synchronize()
        # TODO: add logger call here
        self._compute_embeddings()

    def get_knn_all(self, query_idxs, k=None):
        """ Consider all points in index when answering the query. """
        k = self.args.k if k is None else k
        assert get_rank() == 0
        in_query_mask = np.isin(self.idxs, query_idxs)
        assert np.sum(in_query_mask) == query_idxs.size
        in_query_X = self.X[in_query_mask]
        _, I = self._build_and_query_knn(self.X, in_query_X, k+1)
        remap = lambda i : self.idxs[i]
        v_remap = np.vectorize(remap)
        I = v_remap(I)
        return I[:,1:]

    def get_knn_restricted(self, query_idxs, restriction_map, shared_idxs=[]):
        """ Consider restricted (+shared) points when answering the query. """
        assert get_rank() == 0

        # compute unrestricted knn
        buffer_const = 3
        query_idxs = np.asarray(query_idxs)
        unrestricted_knn = self.get_knn_all(query_idxs,
                                            k=buffer_const*self.args.k)
        
        # create restricted knn
        shared_mask = np.isin(unrestricted_knn, shared_idxs)
        restricted_mask_list = []
        for i, q_idx in enumerate(query_idxs):
            restricted_mask_list.append(
                np.isin(unrestricted_knn[i], restriction_map[q_idx])
            )
        restricted_mask = np.vstack(restricted_mask_list) | shared_mask
        restricted_knn = ma.array(unrestricted_knn, mask=restricted_mask)

        return restricted_knn

    def get_knn_limited_index(self,
                              query_idxs,
                              include_index_idxs=None,
                              exclude_index_idxs=None,
                              k=None):
        """ Consider only out of cluster points when awnsering the query. """
        assert get_rank() == 0
        assert (include_index_idxs is None) ^ (exclude_index_idxs is None)
        k = self.args.k if k is None else k

        # build the masks
        query_mask = np.isin(self.idxs, query_idxs)
        assert np.sum(query_mask) == len(query_idxs)
        if include_index_idxs is not None:
            index_mask = np.isin(self.idxs, include_index_idxs, invert=False)
        else:
            index_mask = np.isin(self.idxs, exclude_index_idxs, invert=True)

        # get query and index representations
        query_X = self.X[query_mask]
        index_X = self.X[index_mask]

        # get index idxs
        index_idxs = self.idxs[index_mask]

        # compute limited index closest
        _, I = self._build_and_query_knn(
                index_X,
                query_X,
                k,
                n_cells=1,
                n_probe=1
        )

        # remap indices back to idxs
        v_remap = np.vectorize(lambda i : index_idxs[i])
        I = v_remap(I)
        return I

    def _compute_embeddings(self):
        # gather and save on rank 0 process
        # NOTES:
        #   - `self.X` : stacked embedding np array with shape: (N, D)
        #   - `self.idxs` : a np array of dataset idxs with shape: (N,)

        ## FIXME: only for testing
        #tmp_fname = '.'.join([self.name, 'knn_index.pkl'])
        #if os.path.exists(tmp_fname):
        #    if get_rank() == 0:
        #        logger.warn('!!!! LOADING PREVIOUSLY CACHED kNN INDEX !!!!')
        #        with open(tmp_fname, 'rb') as f:
        #            self.idxs, self.X = pickle.load(f)
        #else:
        #    self.idxs, self.X = self.sub_trainer.get_embeddings(self.dataloader)
        #    if get_rank() == 0:
        #        with open(tmp_fname, 'wb') as f:
        #            pickle.dump((self.idxs, self.X), f)

        self.idxs, self.X = self.sub_trainer.get_embeddings(self.dataloader)

    def _build_and_query_knn(self,
                             index_mx,
                             query_mx,
                             k,
                             n_cells=200,
                             n_probe=75):
        # Can change `n_cells`, `n_probe` as hyperparameters for knn search
        assert index_mx.shape[1] == query_mx.shape[1]
        d = index_mx.shape[1]
        if n_cells == 1:
            knn_index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        else:
            quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
            knn_index = faiss.IndexIVFFlat(
                    quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
            )
            knn_index.train(index_mx)
            knn_index.nprobe = n_probe
        knn_index.add(index_mx)
        D, I = knn_index.search(query_mx, k)
        return D, I


class WithinDocNNIndex(NearestNeighborIndex):
    pass


class CrossDocNNIndex(NearestNeighborIndex):
    pass
