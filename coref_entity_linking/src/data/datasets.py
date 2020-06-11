import os
import random
from functools import lru_cache
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from abc import ABC, abstractmethod
import logging
import torch

from utils.comm import get_world_size

from IPython import embed


logger = logging.getLogger(__name__)


@lru_cache(maxsize=8388608)
def _load_example(_example_dir, _id):
    return torch.load(os.path.join(_example_dir, str(_id) + '.pt'))


def _first_arg(*args, **kwargs):
    return args[0]


@cached(LRUCache(maxsize=8388608), key=_first_arg)
def _create_bi_encoder_input(idx,
                             input_ids,
                             max_length,
                             tokenizer,
                             pad_token_segment_id=0,
                             pad_on_left=False,
                             pad_token=0,
                             mask_padding_with_zero=True):
    # NOTE: `idx` is purely for caching purposes

    cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')

    input_ids = [cls_token_id] + input_ids + [sep_token_id]
    token_type_ids = [0] * len(input_ids) # segment_ids

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask

    assert pad_on_left == False
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + (
            [0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + (
            [pad_token_segment_id] * padding_length)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(token_type_ids) == max_length

    return input_ids, attention_mask, token_type_ids


class MetaClusterDataset(torch.utils.data.Dataset):
    """
    Dataset which stores idx's but doesn't retrieve features
    """
    def __init__(self, examples):
        self.examples = examples
        self.max_cluster_size = max([len(c) for c in self.examples])

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]


class DistributedSafeDataset(torch.utils.data.Dataset):
    """
    Abstract super class for all distributed safe dataset classes below.
    """
    def __init__(self, args, examples):
        self.args = args
        self.examples = examples

    def __len__(self):
        return max(len(self.examples), get_world_size())
    
    def __getitem__(self, index):
        raise NotImplementedError('This is an abstract base class,'
                                  'not for explicit use')


class InferenceEmbeddingDataset(DistributedSafeDataset):
    """
    Single sequence examples for inference, evaluation, indexing,...
    """
    def __init__(self, args, examples, example_dir):
        super(InferenceEmbeddingDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self) - 1)
        _id = self.examples[index]
        _example_seq = _load_example(self.example_dir, _id).numpy().tolist()
        _example_features = _create_bi_encoder_input(
            _id, _example_seq, self.args.max_seq_length, self.args.tokenizer
        )
        input_ids, attention_mask, token_type_ids = _example_features

        example_idx = torch.tensor(_id, dtype=torch.long)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

        return (example_idx,
                input_ids,
                attention_mask,
                token_type_ids)


class TripletEmbeddingDataset(DistributedSafeDataset):
    """
    For triplet training.
    """
    def __init__(self, args, examples, example_dir):
        super(TripletDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        raise NotImplementedError('TODO')


class ScaledPairEmbeddingDataset(DistributedSafeDataset):
    """
    For any pair with coeff style dataset.
    """
    def __init__(self, args, examples, example_dir):
        super(ScaledPairDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        raise NotImplementedError('TODO')


class InferenceConcatDataset(InferenceEmbeddingDataset):
    pass
