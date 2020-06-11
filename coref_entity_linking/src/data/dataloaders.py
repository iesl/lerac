import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.comm import get_rank, get_world_size
from utils.misc import flatten

from IPython import embed


class MetaClusterDataLoader(DataLoader):
    """
    Custom DataLoader for MetaClusterDataset.
    """
    def __init__(self, args, dataset):
        self.max_cluster_size = dataset.max_cluster_size
        super(MetaClusterDataLoader, self).__init__(
                dataset,
                batch_size=args.num_clusters_per_macro_batch,
                collate_fn=self._custom_collate_fn,
                num_workers=args.num_dataloader_workers,
                pin_memory=True,
                shuffle=True
        )

    def _custom_collate_fn(self, batch):
        # builds a sparse matrix which represents the batch
        indices = flatten([list(zip([i] * len(c), list(range(len(c)))))
                                for i, c in enumerate(batch)])
        indices = np.asarray(indices).T
        values = np.asarray(flatten(batch))
        return coo_matrix((values, indices),
                          shape=(len(batch), self.max_cluster_size))


class InferenceEmbeddingDataLoader(DataLoader):
    """
    Custom DataLoader for InferenceEmbeddingDataset.
    """
    def __init__(self, args, dataset):
        super(InferenceEmbeddingDataLoader, self).__init__(
                dataset,
                sampler=self._custom_distributed_sampler(dataset),
                batch_size=args.infer_batch_size,
                num_workers=args.num_dataloader_workers,
                pin_memory=True,
        )

    def _custom_distributed_sampler(self, dataset):
        return DistributedSampler(
                dataset,
                num_replicas=get_world_size(),
                rank=get_rank(),
                shuffle=False
        )

