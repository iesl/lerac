import os
import random
from functools import lru_cache
from cachetools import cached, LRUCache
from cachetools.keys import hashkey
from abc import ABC, abstractmethod
import logging
import torch

from utils.comm import get_world_size
from utils.misc import CLS_TOKEN, SEP_TOKEN

from IPython import embed


logger = logging.getLogger(__name__)


@lru_cache(maxsize=8388608)
def _load_example(_example_dir, _id):
    return torch.load(os.path.join(_example_dir, str(_id) + '.pt'))


def _first_arg(*args, **kwargs):
    return args[0]


def _first_and_second_args(*args, **kwargs):
    return (args[0], args[1])


@cached(LRUCache(maxsize=8388608), key=_first_arg)
def _create_bi_encoder_input(idx,
                             input_ids,
                             max_length,
                             tokenizer,
                             num_entities,
                             pad_token_segment_id=0,
                             pad_on_left=False,
                             pad_token=0,
                             mask_padding_with_zero=True):
    # NOTE: `idx` is purely for caching purposes

    cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    input_ids = [cls_token_id] + input_ids + [sep_token_id]
    token_type_ids = [idx < num_entities] * len(input_ids) # segment_ids

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


@cached(LRUCache(maxsize=8388608), key=_first_and_second_args)
def _create_cross_encoder_input(idx_a,
                                idx_b,
                                input_ids_a,
                                input_ids_b,
                                max_length,
                                tokenizer,
                                num_entities,
                                pad_token_segment_id=0,
                                pad_on_left=False,
                                pad_token=0,
                                mask_padding_with_zero=True):
    # NOTE: `idx_a` and `idx_b` are purely for caching purposes

    cls_token_id = tokenizer.convert_tokens_to_ids(CLS_TOKEN)
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    # concatenation of input ids
    input_ids = ([cls_token_id]
                 + input_ids_a
                 + [sep_token_id]
                 + input_ids_b
                 + [sep_token_id])

    # segment_ids
    #token_type_ids = ([0] * (len(input_ids_a) + 2)
    #                  + [1] * (len(input_ids_b) + 1))
    token_type_ids = ([idx_a < num_entities] * (len(input_ids_a) + 2)
                      + [idx_b < num_entities] * (len(input_ids_b) + 1))

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask

    # deal with padding
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
            index = random.randint(0, len(self.examples) - 1)
        _id = self.examples[index]
        _example_seq = _load_example(self.example_dir, _id).numpy().tolist()
        _example_features = _create_bi_encoder_input(
            _id, _example_seq, self.args.max_seq_length, self.args.tokenizer, self.args.num_entities
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
    For triplet training with embedding model.
    """
    def __init__(self, args, examples, example_dir):
        super(TripletEmbeddingDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self.examples) - 1)
        ids = self.examples[index]
        example_idx = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for _id in ids:
            _example_seq = _load_example(self.example_dir, _id).numpy().tolist()
            _example_features = _create_bi_encoder_input(
                _id, _example_seq, self.args.max_seq_length, self.args.tokenizer, self.args.num_entities
            )
            _input_ids, _attention_mask, _token_type_ids = _example_features

            example_idx.append(torch.tensor(_id, dtype=torch.long))
            input_ids.append(torch.tensor(_input_ids, dtype=torch.long))
            attention_mask.append(
                    torch.tensor(_attention_mask, dtype=torch.long))
            token_type_ids.append(
                    torch.tensor(_token_type_ids, dtype=torch.long))

        return (torch.stack(example_idx),
                torch.stack(input_ids),
                torch.stack(attention_mask),
                torch.stack(token_type_ids))


class TripletConcatenationDataset(DistributedSafeDataset):
    """
    For triplet training with concatenation model.
    """
    def __init__(self, args, examples, example_dir):
        super(TripletConcatenationDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self.examples) - 1)
        ids = self.examples[index]
        example_idxs = []
        input_ids = []
        attention_mask = []
        token_type_ids = []

        # anchor, positive, and negative
        anc, pos, neg = ids
        anc_seq = _load_example(self.example_dir, anc).numpy().tolist()
        pos_seq = _load_example(self.example_dir, pos).numpy().tolist()
        neg_seq = _load_example(self.example_dir, neg).numpy().tolist()
        seq_dict = {anc : anc_seq, pos : pos_seq, neg : neg_seq}

        for idx_a, idx_b in [(anc, pos), (anc, neg)]:
            # get the forward direction features
            fwd_feats = _create_cross_encoder_input(
                idx_a,
                idx_b,
                seq_dict[idx_a],
                seq_dict[idx_b],
                2*self.args.max_seq_length,
                self.args.tokenizer,
                self.args.num_entities
            )
            fwd_input_ids, fwd_attention_mask, fwd_token_type_ids = fwd_feats

            # get the backward direction features
            bwd_feats = _create_cross_encoder_input(
                idx_b,
                idx_a,
                seq_dict[idx_b],
                seq_dict[idx_a],
                2*self.args.max_seq_length,
                self.args.tokenizer,
                self.args.num_entities
            )
            bwd_input_ids, bwd_attention_mask, bwd_token_type_ids = bwd_feats

            # pack forward and backward features together
            example_idxs.append(
                torch.stack(
                    (torch.tensor((idx_a, idx_b), dtype=torch.long),
                     torch.tensor((idx_b, idx_a), dtype=torch.long))
                )
            )
            input_ids.append(
                torch.stack(
                    (torch.tensor(fwd_input_ids, dtype=torch.long),
                     torch.tensor(bwd_input_ids, dtype=torch.long))
                )
            )
            attention_mask.append(
                torch.stack(
                    (torch.tensor(fwd_attention_mask, dtype=torch.long),
                     torch.tensor(bwd_attention_mask, dtype=torch.long))
                )
            )
            token_type_ids.append(
                torch.stack(
                    (torch.tensor(fwd_token_type_ids, dtype=torch.long),
                     torch.tensor(bwd_token_type_ids, dtype=torch.long))
                )
            )

        # pack all off the pairs' data nicely
        return (torch.stack(example_idxs),
                torch.stack(input_ids),
                torch.stack(attention_mask),
                torch.stack(token_type_ids))


class SoftmaxEmbeddingDataset(DistributedSafeDataset):
    """
    For softmax training with embedding model.
    """
    def __init__(self, args, examples, example_dir):
        super(SoftmaxEmbeddingDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self.examples) - 1)
        ids = self.examples[index]
        example_idx = []
        input_ids = []
        attention_mask = []
        token_type_ids = []
        for _id in ids:
            _example_seq = _load_example(self.example_dir, _id).numpy().tolist()
            _example_features = _create_bi_encoder_input(
                _id, _example_seq, self.args.max_seq_length, self.args.tokenizer, self.args.num_entities
            )
            _input_ids, _attention_mask, _token_type_ids = _example_features

            example_idx.append(torch.tensor(_id, dtype=torch.long))
            input_ids.append(torch.tensor(_input_ids, dtype=torch.long))
            attention_mask.append(
                    torch.tensor(_attention_mask, dtype=torch.long))
            token_type_ids.append(
                    torch.tensor(_token_type_ids, dtype=torch.long))

        return (torch.stack(example_idx),
                torch.stack(input_ids),
                torch.stack(attention_mask),
                torch.stack(token_type_ids))


class SoftmaxConcatenationDataset(DistributedSafeDataset):
    """
    For softmax training with concatenation model.
    """
    def __init__(self, args, examples, example_dir):
        super(SoftmaxConcatenationDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self.examples) - 1)
        ids = self.examples[index]
        example_idxs = []
        input_ids = []
        attention_mask = []
        token_type_ids = []

        # anchor, positive, and negatives
        anc = ids[0]
        pos_and_neg = ids[1:]
        seq_dict = {x : _load_example(self.example_dir, x).numpy().tolist()
                    for x in ids}

        # NOTE: the one positive is always first in `pos_and_neg`,
        #        the rest of the ids are the negatives; this makes
        #        training much simpler to implement

        for pos_or_neg_idx in pos_and_neg:
            idx_a = anc
            idx_b = pos_or_neg_idx

            # get the forward direction features
            fwd_feats = _create_cross_encoder_input(
                idx_a,
                idx_b,
                seq_dict[idx_a],
                seq_dict[idx_b],
                2*self.args.max_seq_length,
                self.args.tokenizer,
                self.args.num_entities
            )
            fwd_input_ids, fwd_attention_mask, fwd_token_type_ids = fwd_feats

            # get the backward direction features
            bwd_feats = _create_cross_encoder_input(
                idx_b,
                idx_a,
                seq_dict[idx_b],
                seq_dict[idx_a],
                2*self.args.max_seq_length,
                self.args.tokenizer,
                self.args.num_entities
            )
            bwd_input_ids, bwd_attention_mask, bwd_token_type_ids = bwd_feats

            # pack forward and backward features together
            example_idxs.append(
                torch.stack(
                    (torch.tensor((idx_a, idx_b), dtype=torch.long),
                     torch.tensor((idx_b, idx_a), dtype=torch.long))
                )
            )
            input_ids.append(
                torch.stack(
                    (torch.tensor(fwd_input_ids, dtype=torch.long),
                     torch.tensor(bwd_input_ids, dtype=torch.long))
                )
            )
            attention_mask.append(
                torch.stack(
                    (torch.tensor(fwd_attention_mask, dtype=torch.long),
                     torch.tensor(bwd_attention_mask, dtype=torch.long))
                )
            )
            token_type_ids.append(
                torch.stack(
                    (torch.tensor(fwd_token_type_ids, dtype=torch.long),
                     torch.tensor(bwd_token_type_ids, dtype=torch.long))
                )
            )

        # pack all off the pairs' data nicely
        return (torch.stack(example_idxs),
                torch.stack(input_ids),
                torch.stack(attention_mask),
                torch.stack(token_type_ids))


class PairsConcatenationDataset(DistributedSafeDataset):
    """
    For triplet training with concatenation model.
    """
    def __init__(self, args, examples, example_dir):
        super(PairsConcatenationDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        if index >= len(self.examples):
            index = random.randint(0, len(self.examples) - 1)
        ids = self.examples[index]
        example_idxs = []
        input_ids = []
        attention_mask = []
        token_type_ids = []

        # anchor, positive, and negative
        idx_a, idx_b = ids
        seq_a = _load_example(self.example_dir, idx_a).numpy().tolist()
        seq_b = _load_example(self.example_dir, idx_b).numpy().tolist()

        # get the forward direction features
        fwd_feats = _create_cross_encoder_input(
            idx_a,
            idx_b,
            seq_a,
            seq_b,
            2*self.args.max_seq_length,
            self.args.tokenizer
        )
        fwd_input_ids, fwd_attention_mask, fwd_token_type_ids = fwd_feats

        # get the backward direction features
        bwd_feats = _create_cross_encoder_input(
            idx_a,
            idx_b,
            seq_a,
            seq_b,
            2*self.args.max_seq_length,
            self.args.tokenizer
        )
        bwd_input_ids, bwd_attention_mask, bwd_token_type_ids = bwd_feats

        # pack forward and backward features together
        example_idxs.append(
            torch.stack(
                (torch.tensor((idx_a, idx_b), dtype=torch.long),
                 torch.tensor((idx_b, idx_a), dtype=torch.long))
            )
        )
        input_ids.append(
            torch.stack(
                (torch.tensor(fwd_input_ids, dtype=torch.long),
                 torch.tensor(bwd_input_ids, dtype=torch.long))
            )
        )
        attention_mask.append(
            torch.stack(
                (torch.tensor(fwd_attention_mask, dtype=torch.long),
                 torch.tensor(bwd_attention_mask, dtype=torch.long))
            )
        )
        token_type_ids.append(
            torch.stack(
                (torch.tensor(fwd_token_type_ids, dtype=torch.long),
                 torch.tensor(bwd_token_type_ids, dtype=torch.long))
            )
        )

        # this is for correcting dimensionality
        return (torch.stack(example_idxs),
                torch.stack(input_ids),
                torch.stack(attention_mask),
                torch.stack(token_type_ids))


class ScaledPairEmbeddingDataset(DistributedSafeDataset):
    """
    For any pair with coeff style dataset.
    """
    def __init__(self, args, examples, example_dir):
        super(ScaledPairDataset, self).__init__(args, examples)
        self.example_dir = example_dir

    def __getitem__(self, index):
        raise NotImplementedError('TODO')
