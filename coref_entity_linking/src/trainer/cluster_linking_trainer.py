from torch.utils.data import DataLoader, SequentialSampler

from comm import all_gather, broadcast, get_world_size, get_rank, synchronize
from data.datasets import MetaClusterDataset, InferenceEmbeddingDataset
from data.dataloaders import (MetaClusterDataLoader,
                              InferenceEmbeddingDataLoader)
from model import MirrorEmbeddingModel
from trainer.trainer import Trainer
from utils import flatten

from IPython import embed


class ClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(ClusterLinkingTrainer, self).__init__(args)
    
    def create_models(self):
        args = self.args
        self.models = {
                'embedding_model': MirrorEmbeddingModel(
                    args,
                    name='embedding_model'
                )
        }
        args.tokenizer = self.models['embedding_model'].tokenizer
    
    def create_train_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=False)

        if get_rank() == 0:
            # determine the set of gold clusters depending on the setting
            if args.clustering_domain == 'within_doc':
                clusters = flatten([list(doc.values()) 
                        for doc in self.train_metadata.wdoc_clusters.values()])
            elif args.clustering_domain == 'cross_doc':
                clusters = list(self.train_metadata.xdoc_clusters.values())
            else:
                raise ValueError('Invalid clustering_domain')

            self.train_dataset = MetaClusterDataset(clusters)
            self.train_dataloader = MetaClusterDataLoader(
                    args, self.train_dataset)

    def create_train_eval_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=True)

        if args.available_entities in ['candidates_only', 'knn_candidates']:
            examples = flatten([[k] + v
                    for k, v in self.train_metadata.midx2cand.items()])
        elif args.available_entities == 'open_domain':
            examples = list(self.train_metadata.idx2uid.keys())
        else:
            raise ValueError('Invalid available_entities')

        self.train_eval_dataset = InferenceEmbeddingDataset(
                args, examples, args.train_cache_dir)
        self.train_eval_dataloader = InferenceEmbeddingDataLoader(
                args, self.train_eval_dataset)
    
    def create_val_dataloader(self):
        pass

    def create_test_dataloader(self):
        pass

    def train(self):
        args = self.args

        # FIXME: this only does one epoch as of now
        if get_rank() == 0:
            data_iterator = iter(self.train_dataloader)
        
        batch = None
        while True:
            if get_rank() == 0:
                try:
                    next_batch = next(data_iterator)
                    # TODO: get negatives from index
                except StopIteration:
                    # FIXME: this should deal with the scatter
                    next_batch = None

            synchronize()
            # FIXME: this should be a scatter
            broadcast(next_batch, src=0)
            if next_batch is None:
                break

            # FIXME: This should all go in `train_step`
            #   TODO: run inference
            #   TODO: get supervised cluster dataset
            #   TODO: train on cluster dataset


            

    
    def evaluate(self, split):
        pass
