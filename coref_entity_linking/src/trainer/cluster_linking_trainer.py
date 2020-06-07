from torch.utils.data import DataLoader, SequentialSampler

from utils.comm import (all_gather,
                        broadcast,
                        get_world_size,
                        get_rank,
                        synchronize)
from data.datasets import MetaClusterDataset, InferenceEmbeddingDataset
from data.dataloaders import (MetaClusterDataLoader,
                              InferenceEmbeddingDataLoader)
from model import MirrorEmbeddingModel
from trainer.trainer import Trainer
from trainer.emb_sub_trainer import EmbeddingSubTrainer
from utils.knn_index import WithinDocNNIndex, CrossDocNNIndex
from utils.misc import flatten, unique

from IPython import embed


class ClusterLinkingTrainer(Trainer):

    def __init__(self, args):
        super(ClusterLinkingTrainer, self).__init__(args)
        self.create_sub_trainers()
        if args.do_train or args.do_train_eval:
            self.create_knn_index('train')
        if args.do_val: # or args.evaluate_during_training:
            self.create_knn_index('val')
        if args.do_test:
            self.create_knn_index('test')
    
    def create_models(self):
        args = self.args
        self.models = {
                'embedding_model': MirrorEmbeddingModel(
                    args,
                    name='embedding_model'
                )
        }
        args.tokenizer = self.models['embedding_model'].tokenizer

    def create_sub_trainers(self):
        args = self.args
        self.sub_trainers = {}
        for name, model in self.models.items():
            if isinstance(model.module, MirrorEmbeddingModel):
                self.sub_trainers[name] = EmbeddingSubTrainer(args, model)
            else:
                raise ValueError('module not supported by a sub trainer')
    
    def create_train_dataloader(self):
        args = self.args

        # load and cache examples and get the metadata for the dataset
        self.load_and_cache_examples(split='train', evaluate=False)

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
        examples = unique(examples)
        self.train_eval_dataset = InferenceEmbeddingDataset(
                args, examples, args.train_cache_dir)
        self.train_eval_dataloader = InferenceEmbeddingDataLoader(
                args, self.train_eval_dataset)

    def create_val_dataloader(self):
        pass

    def create_test_dataloader(self):
        pass

    def create_knn_index(self, split=None):
        assert split == 'train' or split == 'val' or split == 'test'
        args = self.args

        NN_Index = (WithinDocNNIndex if args.clustering_domain == 'within_doc'
                        else CrossDocNNIndex)
        if split == 'train':
            self.train_knn_index = NN_Index(
                    args,
                    self.sub_trainers['embedding_model'],
                    self.train_eval_dataloader
            )
        elif split == 'val':
            self.val_knn_index = NN_Index(
                    args,
                    self.sub_trainers['embedding_model'],
                    self.val_dataloader
            )
        else:
            self.test_knn_index = NN_Index(
                    args,
                    self.sub_trainers['embedding_model'],
                    self.test_dataloader
            )

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
                    ### TODO: !!! HERE !!!
                    # TODO: get negatives from knn index
                except StopIteration:
                    next_batch = None

            synchronize()
            broadcast(next_batch, src=0)
            if next_batch is None:
                break

            # FIXME: This should all go in `train_step`
            #   TODO: build inference dataset give next_batch
            #   TODO: run inference (can't just query the knn index!!!!)
            #   TODO: get supervised cluster dataset
            #   TODO: train on cluster dataset
    
    def evaluate(self, split):
        pass
