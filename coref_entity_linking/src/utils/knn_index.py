import faiss
import numpy as np

from utils.comm import get_rank, synchronize


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
        self._compute_embeddings()

    def get_knn_all(self, query_idxs):
        """ Consider all points in index when answering the query. """
        assert get_rank() == 0
        pass

    def get_knn_negatives(self, query_clusters):
        """ Consider only out of cluster points when awnsering the query. """
        assert get_rank() == 0
        pass

    def _compute_embeddings(self):
        # gather and save on rank 0 process
        # NOTES:
        #   - `self.X` : stacked embedding np array with shape: (N, D)
        #   - `self.idxs` : a np array of dataset idxs with shape: (N,)
        self.idxs, self.X = self.sub_trainer.get_embeddings(self.dataloader)


class WithinDocNNIndex(NearestNeighborIndex):
    pass


class CrossDocNNIndex(NearestNeighborIndex):
    pass
