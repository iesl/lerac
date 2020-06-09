import logging
import faiss
import numpy as np

from utils.comm import get_rank, synchronize

from IPython import embed


logger = logging.getLogger(__name__)


class NearestNeighborIndex(object):
    """
    Base class, but also treats all points the same -- no distinction b/w
    mentions and entities.
    """
    def __init__(self, args, sub_trainer, dataloader):
        self.args = args
        self.sub_trainer = sub_trainer
        self.dataloader = dataloader
        self._compute_embeddings()

    def refresh_index(self):
        """ Refresh the index by recomputing the embeddings for all points. """
        synchronize()
        # TODO: add logger call here
        self._compute_embeddings()

    def get_knn_all(self, query_idxs):
        """ Consider all points in index when answering the query. """
        assert get_rank() == 0
        pass

    def get_knn_negatives(self, query_clusters):
        """ Consider only out of cluster points when awnsering the query. """
        assert get_rank() == 0
        in_query_mask = np.isin(self.idxs, query_clusters.data)
        out_query_mask = ~in_query_mask

        # get in-query and out-query representations
        in_query_X = self.X[in_query_mask]
        out_query_X = self.X[out_query_mask]

        # get out-query idxs
        out_query_idxs = self.idxs[out_query_mask]

        # compute out-query closest
        _, I = self._build_and_query_knn(out_query_X, in_query_X, self.args.k)

        remap = lambda i : out_query_idxs[i]
        v_remap = np.vectorize(remap)
        I = v_remap(I)
        return I

    def _compute_embeddings(self):
        # gather and save on rank 0 process
        # NOTES:
        #   - `self.X` : stacked embedding np array with shape: (N, D)
        #   - `self.idxs` : a np array of dataset idxs with shape: (N,)
        self.idxs, self.X = self.sub_trainer.get_embeddings(self.dataloader)

    def _build_and_query_knn(self, index_mx, query_mx, k):
        assert index_mx.shape[1] == query_mx.shape[1]
        # Can change `n_cells`, `n_probe` as hyperparameters for knn search
        n_cells = 200
        n_probe = 75
        d = index_mx.shape[1]
        quantizer = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        knn_index = faiss.IndexIVFFlat(
                quantizer, d, n_cells, faiss.METRIC_INNER_PRODUCT
        )
        knn_index.train(index_mx)
        knn_index.add(index_mx)
        knn_index.nprobe = n_probe
        D, I = knn_index.search(query_mx, k)
        return D, I


class WithinDocNNIndex(NearestNeighborIndex):
    pass


class CrossDocNNIndex(NearestNeighborIndex):
    pass
