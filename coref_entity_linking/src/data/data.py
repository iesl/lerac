# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import copy
import glob
import struct
import tqdm
import random
import math
import pickle
import operator
from collections import defaultdict
from typing import List
from functools import lru_cache
from scipy.sparse import coo_matrix

import torch
from transformers import PreTrainedTokenizer

from utils.comm import get_world_size
#from utils import START_MENTION_TOKEN, END_MENTION_TOKEN

from IPython import embed


logger = logging.getLogger(__name__)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def select_field(example, field, evaluate):
    func = lambda x : id_to_bytes(x) if field in ['muid', 'mention_id', 'entity_id', 'coref_mention_id'] else x
    return [func(choice[field]) for choice in example.features]


def id_to_bytes(uid):
    assert len(uid) <= 16
    num_pad = 16 - len(uid)
    uid_bytes = bytes(uid + num_pad * 'n', 'utf-8')
    uid_bytes = list(struct.unpack('=LLLL', uid_bytes))
    return uid_bytes


def bytes_to_id(uid_bytes):
    return struct.pack('=LLLL', *uid_bytes).decode('utf-8').replace('n', '')


def get_mention_context_tokens(context_tokens, start_index, end_index,
                               max_tokens, tokenizer):
    start_pos = start_index - max_tokens
    if start_pos < 0:
        start_pos = 0

    prefix = ' '.join(context_tokens[start_pos: start_index])
    suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
    prefix = tokenizer.tokenize(prefix)
    suffix = tokenizer.tokenize(suffix)
    mention = tokenizer.tokenize(
                ' '.join(context_tokens[start_index:end_index+1]))
    mention = [START_MENTION_TOKEN] + mention + [END_MENTION_TOKEN]

    assert len(mention) < max_tokens

    remaining_tokens = max_tokens - len(mention)
    half_remaining_tokens = int(math.ceil(1.0*remaining_tokens/2))

    mention_context = []

    if len(prefix) >= half_remaining_tokens and len(suffix) >= half_remaining_tokens:
        prefix_len = half_remaining_tokens
    elif len(prefix) >= half_remaining_tokens and len(suffix) < half_remaining_tokens:
        prefix_len = remaining_tokens - len(suffix)
    else:
        prefix_len = len(prefix)

    if prefix_len > len(prefix):
        prefix_len = len(prefix)

    prefix = prefix[-prefix_len:]

    mention_context = prefix + mention + suffix
    mention_context = mention_context[:max_tokens]

    return mention_context


def _read_mentions(mention_file):
    mentions = []
    with open(mention_file, encoding='utf-8') as fin:
        for line in fin:
            mentions.append(json.loads(line))
    return mentions


def _read_documents(document_files):
    documents = {}
    for fname in document_files:
        with open(fname, encoding='utf-8') as fin:
            for line in fin:
                doc_dict = json.loads(line)
                documents[doc_dict['document_id']] = doc_dict
    return documents


def _read_candidates(candidate_file):
    candidates = {}
    with open(candidate_file, encoding='utf-8') as fin:
        for line in fin:
            candidate_dict = json.loads(line)
            candidates[candidate_dict['mention_id']] = candidate_dict['tfidf_candidates']
    return candidates

def get_document2mentions(mentions):
    document2mentions = defaultdict(list)
    for m in mentions:
        document2mentions[m['context_document_id']].append(m)
    return document2mentions

def get_cluster2mentions(mentions):
    cluster2mentions = defaultdict(list)
    for m in mentions:
        cluster2mentions[m['label_document_id']].append(m)
    return cluster2mentions

def create_cluster_index(data_dir, split):
    # Load all of the mentions
    mention_file = os.path.join(data_dir, 'mentions', split + '.json')
    all_mentions = _read_mentions(mention_file)

    document2mentions = get_document2mentions(all_mentions)

    cluster_index = {}
    for document_id, mentions in document2mentions.items():
        _entity2mentions = defaultdict(list)
        for m in mentions:
            _entity2mentions[m['label_document_id']].append(m['mention_id'])
        cluster_index[document_id] = _entity2mentions

    return cluster_index


def get_mentions_and_entities(data_dir, split, domains):
    # Load all of the mentions
    mention_file = os.path.join(data_dir, 'mentions', split + '.json')
    mentions = _read_mentions(mention_file)

    # Load all of the documents for the mentions
    # `documents` is a dictionary with key 'document_id'
    #assert split in domains
    document_files = [os.path.join(data_dir, 'documents', domain + '.json')
                        for domain in domains]
    documents = _read_documents(document_files)
    entity_document_files = [os.path.join(data_dir, 'documents', domain + '.json')
                        for domain in domains if domain != split]
    entity_documents = _read_documents(entity_document_files)

    return mentions, documents, entity_documents


def get_index(dictionary):
    keys = sorted(list(dictionary.keys()))
    index = {key : i for i, key in enumerate(keys)}
    return index


@static_vars(cached_input_ids={})
def get_mention_input_ids(mention,
                          documents,
                          tokenizer,
                          mention_length,
                          cache_dataset_features=False):

    muid = mention['mention_id']

    if cache_dataset_features and muid in get_mention_input_ids.cached_input_ids.keys():
        input_ids = get_mention_input_ids.cached_input_ids[muid]
    else:
        context_document_id = mention['context_document_id']
        start_index = mention['start_index']
        end_index = mention['end_index']

        context_document = documents[context_document_id]['text']

        context_tokens = context_document.split()
        extracted_mention = context_tokens[start_index: end_index+1]
        extracted_mention = ' '.join(extracted_mention)
        assert extracted_mention == mention['text']

        mention_text_tokenized = tokenizer.tokenize(mention['text'])

        mention_context = get_mention_context_tokens(
                context_tokens, start_index, end_index,
                mention_length, tokenizer)

        input_ids = tokenizer.convert_tokens_to_ids(mention_context)

        if cache_dataset_features:
            get_mention_input_ids.cached_input_ids[muid] = input_ids

    return input_ids


@static_vars(cached_input_ids={})
def get_entity_input_ids(entity_id,
                         documents,
                         tokenizer,
                         entity_length,
                         cache_dataset_features=False):

    if cache_dataset_features and entity_id in get_entity_input_ids.cached_input_ids.keys():
        input_ids = get_entity_input_ids.cached_input_ids[entity_id]
    else:
        entity_text = documents[entity_id]['text']
        entity_tokens = tokenizer.tokenize(entity_text)
        entity_tokens = entity_tokens[:entity_length]

        input_ids = tokenizer.convert_tokens_to_ids(entity_tokens)

        if cache_dataset_features:
            get_entity_input_ids.cached_input_ids[entity_id] = input_ids

    return input_ids


def create_bi_encoder_input(input_ids,
                            max_length,
                            tokenizer,
                            pad_token_segment_id,
                            pad_on_left,
                            pad_token,
                            mask_padding_with_zero):

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


def create_cross_encoder_input(mention_context_ids,
                               entity_desc_ids,
                               max_length,
                               tokenizer,
                               pad_token_segment_id,
                               pad_on_left,
                               pad_token,
                               mask_padding_with_zero):
    cls_token_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')

    input_ids = [cls_token_id] + mention_context_ids + [sep_token_id]
    input_ids += entity_desc_ids + [sep_token_id]
    token_type_ids = ([0] * (len(mention_context_ids) + 2)
                      + [1] * (len(entity_desc_ids) + 1)) # segment_ids

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

    return (input_ids,
            attention_mask,
            token_type_ids)


@lru_cache(maxsize=1024)
def _load_example(_example_dir, _id):
    return torch.load(os.path.join(_example_dir, str(_id) + '.pt'))

class WrappedTensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor_dataset):
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return max(len(self.tensor_dataset), get_world_size())

    def __getitem__(self, index):
        if index >= len(self.tensor_dataset):
            index = random.randint(0, len(self.tensor_dataset) - 1)
        return self.tensor_dataset[index]
        

class LazyDataset(torch.utils.data.Dataset):

    def __init__(self, example_dir, example_ids):
        self.example_dir = example_dir
        self.example_ids = example_ids 
    
    def __len__(self):
        return max(len(self.example_ids), get_world_size())
    
    def __getitem__(self, index):
        if index >= len(self.example_ids):
            index = random.randint(0, len(self.example_ids) - 1)
        _id = self.example_ids[index]
        _example_features = _load_example(self.example_dir, _id)
        return _example_features


class LazyConcatDataset(LazyDataset):

    def __init__(self, example_dir, example_ids):
        super(LazyConcatDataset, self).__init__(example_dir, example_ids)
        self.example_ids = [str(x[0]) + '_' + str(x[1])
                                for x in self.example_ids]


class ClusterLinkingLazyTrainDataset(torch.utils.data.Dataset):

    def __init__(self, tuples, example_dir_a, example_dir_b):
        self.tuples = tuples
        self.example_dir_a = example_dir_a
        self.example_dir_b = example_dir_b
    
    def __len__(self):
        return max(len(self.tuples), get_world_size())
    
    def __getitem__(self, index):
        if index >= len(self.tuples):
            index = random.randint(0, len(self.tuples) - 1)

        # Select sample
        coeff, id_a, id_b = self.tuples[index]

        # Load example
        features_a = _load_example(self.example_dir_a, id_a)
        features_b = _load_example(self.example_dir_b, id_b)

        return (coeff, *features_a, *features_b)


class OnTheFlyConcatTrainDataset(ClusterLinkingLazyTrainDataset):

    def __init__(self, tuples, example_dir_a, example_dir_b):
        super(OnTheFlyConcatTrainDataset, self).__init__(
                tuples, example_dir_a, example_dir_b
        )

    def __getitem__(self, index):
        if index >= len(self.tuples):
            index = random.randint(0, len(self.tuples) - 1)

        # Select sample
        coeff, id_a, id_b = self.tuples[index]

        # Load example
        features_a = _load_example(self.example_dir_a, id_a)
        features_b = _load_example(self.example_dir_b, id_b)

        # concat on-the-fly!
        PAD_TOKEN_ID = 0

        input_ids_a, attention_mask_a, token_type_ids_a = features_a[2:]
        input_ids_b, attention_mask_b, token_type_ids_b = features_b[2:]

        final_seq_len = 2 * input_ids_a.numel() # 128 -> 256

        input_ids_a = input_ids_a[attention_mask_a.bool()].numpy().tolist()
        input_ids_b = input_ids_b[attention_mask_b.bool()].numpy().tolist()[1:]

        input_ids = input_ids_a + input_ids_b
        padding_length = final_seq_len - len(input_ids)
        assert padding_length >= 0
        attention_mask = [1] * len(input_ids) + [0] * padding_length
        token_type_ids = ([0] * len(input_ids_a)
                          + [1] * (len(input_ids_b) + padding_length))
        input_ids += [PAD_TOKEN_ID] * padding_length
    
        assert len(input_ids) == final_seq_len
        assert len(attention_mask) == final_seq_len
        assert len(token_type_ids) == final_seq_len

        return (coeff,
                features_a[0],
                features_a[1],
                features_b[0],
                features_b[1],
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
                torch.tensor(token_type_ids, dtype=torch.long))


class ClusterLinkingLazyConcatTrainDataset(torch.utils.data.Dataset):

    def __init__(self, tuples, example_dir):
        self.tuples = tuples
        self.example_dir = example_dir

    def __len__(self):
        return max(len(self.tuples), get_world_size())
    
    def __getitem__(self, index):
        if index >= len(self.tuples):
            index = random.randint(0, len(self.tuples) - 1)

        # Select sample
        coeff, m_idx, e_idx = self.tuples[index]

        # Load example
        features = _load_example(self.example_dir,
                                 str(m_idx) + '_' + str(e_idx))

        return (coeff, *features)

class XDocClusterLinkingProcessor(object):

    def get_cluster_datasets(
        self,
        data_dir: str,
        split: str,
        domains: List[str],
        cached_mention_examples_dir: str,       
        cached_mention_entity_examples_dir: str,       
        cached_cluster_indices_dir: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        evaluate=False,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True
    ):

        # load mentions and documents and entities
        mentions, documents, entity_documents = get_mentions_and_entities(
                data_dir, split, domains)
        mentions_dict = {m['mention_id']: m for m in mentions}

        # create entity_index map
        entity_index_map = get_index(entity_documents)

        # load candidates
        candidate_file = os.path.join(data_dir, 'tfidf_candidates', split + '.json')
        candidates = _read_candidates(candidate_file)

        # augment candidates with ground truth entity duing training
        if not evaluate:
            for m in mentions:
                muid = m['mention_id']
                gold_entity = m['label_document_id']
                if muid not in candidates.keys():
                    candidates[muid] = []
                if gold_entity not in candidates[muid]:
                    candidates[muid].append(gold_entity)

        # data structures for later
        cluster2mention_indices = defaultdict(list)
        cluster2mention_entity_indices = defaultdict(list)
        all_mention_indices = []
        all_mention_entity_indices = []

        # lengths for sequences 
        mention_length = max_length - 2
        entity_length = max_length - 1
        max_concat_length = 2 * max_length

        for mention_index, mention in tqdm.tqdm(enumerate(mentions), desc="convert mentions to features"):

            mention_id = mention['mention_id']
            cluster_id = mention['label_document_id']
            cluster2mention_indices[cluster_id].append(mention_index)
            all_mention_indices.append(mention_index)

            mention_context_ids = get_mention_input_ids(
                 mention, documents, tokenizer, mention_length)

            (mention_input_ids,
             mention_attention_mask,
             mention_token_type_ids) = create_bi_encoder_input(
                     mention_context_ids, max_length, tokenizer,
                     pad_token_segment_id, pad_on_left, pad_token,
                     mask_padding_with_zero)

            torch.save(
                (torch.tensor(id_to_bytes(mention['mention_id']),
                              dtype=torch.long),
                 torch.tensor(mention_index, dtype=torch.long),
                 torch.tensor(mention_input_ids, dtype=torch.long),
                 torch.tensor(mention_attention_mask, dtype=torch.long),
                 torch.tensor(mention_token_type_ids, dtype=torch.long)),
                os.path.join(cached_mention_examples_dir,
                             str(mention_index) + '.pt')
            )

            if mention_id not in candidates:
                assert evaluate
                continue

            for entity_id in candidates[mention_id]:
                entity_index = entity_index_map[entity_id]
                cluster2mention_entity_indices[cluster_id].append(
                        (mention_index, entity_index)
                )
                all_mention_entity_indices.append(
                        (mention_index, entity_index)
                )

                entity_desc_ids = get_entity_input_ids(
                        entity_id, documents, tokenizer,
                        entity_length, cache_dataset_features=True)

                (input_ids,
                 attention_mask,
                 token_type_ids) = create_cross_encoder_input(
                         mention_context_ids,
                         entity_desc_ids,
                         max_concat_length,
                         tokenizer,
                         pad_token_segment_id,
                         pad_on_left,
                         pad_token,
                         mask_padding_with_zero
                )

                torch.save(
                    (torch.tensor(id_to_bytes(mention['mention_id']),
                                  dtype=torch.long),
                     torch.tensor(mention_index, dtype=torch.long),
                     torch.tensor(id_to_bytes(entity_id),
                                  dtype=torch.long),
                     torch.tensor(entity_index, dtype=torch.long),
                     torch.tensor(input_ids, dtype=torch.long),
                     torch.tensor(attention_mask, dtype=torch.long),
                     torch.tensor(token_type_ids, dtype=torch.long),
                     torch.Size([len(mentions), len(entity_index_map)])),
                    os.path.join(cached_mention_entity_examples_dir,
                                 str(mention_index)
                                 + '_' + str(entity_index) + '.pt')
                )

        indices_dicts = []
        cluster2mentions = get_cluster2mentions(mentions)
        if evaluate:
            _indices_dict = {'cluster2mentions': cluster2mentions,
                             'mention_indices': all_mention_indices,
                             'mention_entity_indices': all_mention_entity_indices}
            torch.save(_indices_dict, os.path.join(cached_cluster_indices_dir, 'all.pt'))
            indices_dicts.append(_indices_dict)
        else:
            for cluster_id in cluster2mention_indices.keys():
                _indices_dict = {'cluster_id': cluster_id,
                                 'mention_indices': cluster2mention_indices[cluster_id],
                                 'mention_entity_indices': cluster2mention_entity_indices[cluster_id]}
                
                torch.save(_indices_dict, os.path.join(cached_cluster_indices_dir, str(cluster_id) + '.pt'))
                indices_dicts.append(_indices_dict)

        return indices_dicts


class MentionClusteringProcessor(object):
    """Processor for the mention affinity task."""

    def get_document_datasets(
        self,
        data_dir: str,
        split: str,
        domains: List[str],
        cached_mention_examples_dir: str,       
        cached_document_indices_dir: str,
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        evaluate=False,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True
    ):
        mentions, documents, entity_documents = get_mentions_and_entities(
                data_dir, split, domains)
        mentions_dict = {m['mention_id']: m for m in mentions}

        cluster2mention_indices = defaultdict(list)
        mention_length = max_length - 2
        for mention_index, mention in tqdm.tqdm(enumerate(mentions), desc="convert mentions to features"):
            document_id = mention['context_document_id']
            document2mention_indices[document_id].append(mention_index)

            mention_input_ids = get_mention_input_ids(
                 mention, documents, tokenizer, mention_length)

            (mention_input_ids,
             mention_attention_mask,
             mention_token_type_ids) = create_bi_encoder_input(
                     mention_input_ids, max_length, tokenizer,
                     pad_token_segment_id, pad_on_left, pad_token,
                     mask_padding_with_zero)

            torch.save(
                    (torch.tensor(id_to_bytes(mention['mention_id']), dtype=torch.long),
                     torch.tensor(mention_index, dtype=torch.long),
                     torch.tensor(mention_input_ids, dtype=torch.long),
                     torch.tensor(mention_attention_mask, dtype=torch.long),
                     torch.tensor(mention_token_type_ids, dtype=torch.long)),
                os.path.join(cached_mention_examples_dir, str(mention_index) + '.pt'))


        document2mentions = get_document2mentions(mentions)
        indices_dicts = []
        if evaluate:
            _indices_dict = {'document2mentions': document2mentions,
                             'mention_indices': list(range(len(mentions)))}
            torch.save(_indices_dict, os.path.join(cached_document_indices_dir, 'all.pt'))
            indices_dicts.append(_indices_dict)
        else:
            for doc_id in document2mention_indices.keys():
                _indices_dict = {'document_id': doc_id,
                                 'mention_indices': document2mention_indices[doc_id]}
                
                torch.save(_indices_dict, os.path.join(cached_document_indices_dir, str(doc_id) + '.pt'))
                indices_dicts.append(_indices_dict)

        return indices_dicts


class LinkingExample(object):
    """A single training example for mention affinity"""

    def __init__(self, mention, pos_entity, neg_entities):
        """Constructs a LinkingTrainExample.

        Args:
            mention: mention object (dict) of interest

        """
        self.mention = mention
        self.pos_entity = pos_entity
        self.neg_entities = neg_entities


class LinkingFeatures(object):

    def __init__(self, features):
        self.features = [
            {
                'mention_id': mention_id,
                'entity_id': entity_id,
                'label': label,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids
            }
            for mention_id, entity_id, label, input_ids, attention_mask, token_type_ids in features
        ]

class PolyLinkingFeatures(object):

    def __init__(self, features):
        self.features = [
            {
                'mention_id': mention_id,
                'entity_id': entity_id,
                'label': label,
                'mention_input_ids': mention_input_ids,
                'mention_attention_mask': mention_attention_mask,
                'mention_token_type_ids': mention_token_type_ids,
                'entity_input_ids': entity_input_ids,
                'entity_attention_mask': entity_attention_mask,
                'entity_token_type_ids': entity_token_type_ids
            }
            for (mention_id,
                 entity_id,
                 label,
                 mention_input_ids,
                 mention_attention_mask,
                 mention_token_type_ids,
                 entity_input_ids,
                 entity_attention_mask,
                 entity_token_type_ids) in features
        ]

class LinkingProcessor(object):
    """Processor for the linking task."""

    def get_examples(self, data_dir, num_candidates,
                      num_candidates_per_example,
                      split, domains, evaluate, use_true_labels=False):
        mentions, documents, entity_documents = get_mentions_and_entities(
                data_dir, split, domains)

        # Load the precomputed candidates for each mention
        # `candidates` is a dictionary with key 'mention_id'
        candidate_file = os.path.join(data_dir, 'tfidf_candidates', split + '.json')
        candidates = _read_candidates(candidate_file)

        #if evaluate:
        #    document2mentions = get_document2mentions(mentions)
        #    document2candidates = defaultdict(list)
        #    for doc_id, doc_mentions in document2mentions.items():
        #        for m in doc_mentions:
        #            muid = m['mention_id'] 
        #            if muid in candidates.keys():
        #                document2candidates[doc_id].extend(candidates[muid])
        #        for m in doc_mentions:
        #            muid = m['mention_id']
        #            candidates[muid] = document2candidates[doc_id]

        examples = []
        for m in mentions:
            mention_id = m['mention_id']
            label_document_id = m['label_document_id']

            assert label_document_id in documents.keys()

            #if evaluate:
            #    _num_candidates = (math.ceil(len(candidates[mention_id])
            #                                 / num_candidates) * num_candidates)
            #else:
            #    _num_candidates = num_candidates
            _num_candidates = num_candidates

            if use_true_labels:
                pos_entity = label_document_id
            elif (mention_id in candidates.keys()
                    and label_document_id in candidates[mention_id]):
                pos_entity = label_document_id
            else:
                pos_entity = None

            if (mention_id not in candidates.keys()
                    or len(candidates[mention_id]) == 0):
                neg_entities = random.choices(list(entity_documents.keys()),
                                              k=_num_candidates)
            else:
                neg_entities = candidates[mention_id]
            neg_entities = [x for x in neg_entities if x != pos_entity]
            assert len(neg_entities) != 0

            _num_neg_candidates = (math.ceil(_num_candidates / num_candidates_per_example)
                                   * (num_candidates_per_example - int(pos_entity is not None)))

            while len(neg_entities) < _num_neg_candidates:
                neg_entities.extend(neg_entities)

            # Handle multiple batches of neg entities for each pos entity
            for i in range(math.ceil(_num_candidates / num_candidates_per_example)):
                _num_neg_per_ex = num_candidates_per_example - int(pos_entity is not None)
                _neg_entities = neg_entities[i*_num_neg_per_ex:(i+1)*_num_neg_per_ex]
                assert (len(_neg_entities) + int(pos_entity is not None) == num_candidates_per_example)
                examples.append(LinkingExample(m, pos_entity, _neg_entities))

        return (examples, documents)

    def convert_examples_to_concatenation_features(
        self,
        examples: List[LinkingExample],
        documents: dict,
        cached_examples_dir: str,       
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) -> List[int]:

        # account for the tokens marking the mention '[CLS]', '[SEP]', and '[SEP]'
        _max_normal_tokens = max_length - 3
        mention_length = _max_normal_tokens // 2
        entity_length = _max_normal_tokens - mention_length
        indices = []

        CLS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[CLS]')
        SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SEP]')

        for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            example_features = []

            mention = example.mention
            mention_context_ids = get_mention_input_ids(
                    mention, documents, tokenizer,
                    mention_length, cache_dataset_features=True)

            # List candidate entities and labels
            if example.pos_entity is not None:
                candidates = [example.pos_entity] + example.neg_entities
                labels = [1] + len(example.neg_entities) * [0]
            else:
                candidate = example.neg_entities
                labels = len(example.neg_entities) * [0]

            for entity_id, label in zip(candidates, labels):
                entity_desc_ids = get_entity_input_ids(
                        entity_id, documents, tokenizer,
                        entity_length, cache_dataset_features=True)

                input_ids = [CLS_TOKEN_ID] + mention_context_ids + [SEP_TOKEN_ID]
                input_ids += entity_desc_ids + [SEP_TOKEN_ID]
                token_type_ids = ([0] * (len(mention_context_ids) + 2)
                                  + [1] * (len(entity_desc_ids) + 1)) # segment_ids

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

                example_features.append((mention['mention_id'], entity_id, label, input_ids, attention_mask, token_type_ids))

            featurized_example = LinkingFeatures(example_features)
            mention_ids = torch.tensor(
                select_field(featurized_example, 'mention_id', evaluate), dtype=torch.long)
            entity_ids = torch.tensor(
                select_field(featurized_example, 'entity_id', evaluate), dtype=torch.long)
            labels = torch.tensor(
                select_field(featurized_example, 'label', evaluate), dtype=torch.long)
            input_ids = torch.tensor(
                select_field(featurized_example, 'input_ids', evaluate), dtype=torch.long)
            attention_masks = torch.tensor(
                select_field(featurized_example, 'attention_mask', evaluate), dtype=torch.long)
            token_type_ids = torch.tensor(
                select_field(featurized_example, 'token_type_ids', evaluate),
                dtype=torch.long)

            torch.save(
                    (mention_ids,
                     entity_ids,
                     labels,
                     input_ids,
                     attention_masks,
                     token_type_ids),
                    os.path.join(cached_examples_dir, str(ex_index) + '.pt'))
            indices.append(ex_index)

        return indices

    def convert_examples_to_poly_features(
        self,
        examples: List[LinkingExample],
        documents: dict,
        cached_examples_dir: str,       
        max_length: int,
        tokenizer: PreTrainedTokenizer,
        evaluate: bool,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ) -> List[int]:
        """
        Loads a data file into a list of `MentionAffintityFeatures`
        """

        # account for the tokens marking the mention '[CLS]' and '[SEP]'
        # for both sequences
        mention_length = entity_length = max_length - 2
        indices = []

        CLS_TOKEN_ID = tokenizer.convert_tokens_to_ids('[CLS]')
        SEP_TOKEN_ID = tokenizer.convert_tokens_to_ids('[SEP]')

        for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            example_features = []

            mention = example.mention
            mention_context_ids = get_mention_input_ids(
                    mention, documents, tokenizer,
                    mention_length, cache_dataset_features=True)

            # List candidate entities and labels
            if example.pos_entity is not None:
                candidates = [example.pos_entity] + example.neg_entities
                labels = [1] + len(example.neg_entities) * [0]
            else:
                candidate = example.neg_entities
                labels = len(example.neg_entities) * [0]

            mention_input_ids = [CLS_TOKEN_ID] + mention_context_ids + [SEP_TOKEN_ID]
            mention_token_type_ids = [0] * (len(mention_context_ids) + 2)
            mention_attention_mask = [1 if mask_padding_with_zero else 0] * len(mention_input_ids) # input_mask

            assert pad_on_left == False
            mention_padding_length = mention_length + 2 - len(mention_input_ids)
            mention_input_ids = mention_input_ids + ([pad_token] * mention_padding_length)
            mention_attention_mask = mention_attention_mask + (
                    [0 if mask_padding_with_zero else 1] * mention_padding_length)
            mention_token_type_ids = mention_token_type_ids + (
                    [pad_token_segment_id] * mention_padding_length)

            for entity_id, label in zip(candidates, labels):
                entity_context_ids = get_entity_input_ids(
                        entity_id, documents, tokenizer,
                        entity_length, cache_dataset_features=True)

                entity_input_ids = [CLS_TOKEN_ID] + entity_context_ids + [SEP_TOKEN_ID]
                entity_token_type_ids = [0] * (len(entity_context_ids) + 2)

                # The mask has 1 for real tokens and 0 for padding tokens. Only real
                # tokens are attended to.
                entity_attention_mask = [1 if mask_padding_with_zero else 0] * len(entity_input_ids) # input_mask

                assert pad_on_left == False
                entity_padding_length = entity_length + 2 - len(entity_input_ids)
                entity_input_ids = entity_input_ids + ([pad_token] * entity_padding_length)
                entity_attention_mask = entity_attention_mask + (
                        [0 if mask_padding_with_zero else 1] * entity_padding_length)
                entity_token_type_ids = entity_token_type_ids + (
                        [pad_token_segment_id] * entity_padding_length)

                example_features.append((mention['mention_id'],
                                         entity_id,
                                         label,
                                         mention_input_ids,
                                         mention_attention_mask,
                                         mention_token_type_ids,
                                         entity_input_ids,
                                         entity_attention_mask,
                                         entity_token_type_ids))

            featurized_example = PolyLinkingFeatures(example_features)
            mention_ids = torch.tensor(
                select_field(featurized_example, 'mention_id', evaluate)[0:1], dtype=torch.long)
            entity_ids = torch.tensor(
                select_field(featurized_example, 'entity_id', evaluate), dtype=torch.long)
            labels = torch.tensor(
                select_field(featurized_example, 'label', evaluate), dtype=torch.long)
            mention_input_ids = torch.tensor(
                select_field(featurized_example, 'mention_input_ids', evaluate)[0:1], dtype=torch.long)
            mention_attention_masks = torch.tensor(
                select_field(featurized_example, 'mention_attention_mask', evaluate)[0:1], dtype=torch.long)
            mention_token_type_ids = torch.tensor(
                select_field(featurized_example, 'mention_token_type_ids', evaluate)[0:1], dtype=torch.long)
            entity_input_ids = torch.tensor(
                select_field(featurized_example, 'entity_input_ids', evaluate), dtype=torch.long)
            entity_attention_masks = torch.tensor(
                select_field(featurized_example, 'entity_attention_mask', evaluate), dtype=torch.long)
            entity_token_type_ids = torch.tensor(
                select_field(featurized_example, 'entity_token_type_ids', evaluate), dtype=torch.long)

            torch.save(
                    (mention_ids,
                     entity_ids,
                     labels,
                     mention_input_ids,
                     mention_attention_masks,
                     mention_token_type_ids,
                     entity_input_ids,
                     entity_attention_masks,
                     entity_token_type_ids),
                    os.path.join(cached_examples_dir, str(ex_index) + '.pt'))
            indices.append(ex_index)

        return indices


#class CorefLinkingExample():
#    """A single training example for coref linking"""
#
#    def __init__(self, mention, coref_mentions, labels):
#        """Constructs a CorefLinkingExample.
#
#        Args:
#            mention_id: mention id
#            coref_mention_ids: list of ids of proposed mentions that might be coreferent with `mention`
#
#        """
#        self.mention = mention
#        self.coref_mentions = coref_mentions
#        self.labels = labels
#
#
#class CorefLinkingFeatures():
#    def __init__(self, features):
#        self.features = [
#            {
#                'mention_id': mention_id,
#                'coref_mention_id': coref_mention_id,
#                'label': label,
#                'input_ids': input_ids,
#                'attention_mask': attention_mask,
#                'token_type_ids': token_type_ids
#            }
#            for mention_id, coref_mention_id, label, input_ids, attention_mask, token_type_ids in features
#        ]
#
#
#class CorefLinkingProcessor(DataProcessor):
#    """Processor for the coref linking task."""
#
#    def get_examples(self, args, data_dir, split, domains, evaluate=False):
#
#        logger.info("LOOKING AT {} | {} ".format(data_dir, split))
#
#        # Load all of the mentions
#        mention_file = os.path.join(data_dir, 'mentions', split + '.json')
#        mentions = _read_mentions(mention_file)
#        mention_dict = {m['mention_id']: m for m in mentions}
#
#        # Load all of the documents for the mentions
#        # `documents` is a dictionary with key 'document_id'
#        #assert split in domains
#        document_files = [os.path.join(data_dir, 'documents', domain + '.json')
#                            for domain in domains]
#        documents = _read_documents(document_files)
#
#        if split == 'train':
#            duid2muid = defaultdict(list)
#            for muid in mention_dict.keys():
#                duid2muid[muid.split('.')[0]].append(muid)
#
#            mention2coref_mentions = defaultdict(list)
#            mention2not_coref_mentions = defaultdict(list)
#            for muid in mention_dict.keys():
#                mention_label = mention_dict[muid]['label_document_id']
#                for uid in duid2muid[muid.split('.')[0]]:
#                    if uid == muid:
#                        continue
#                    elif mention_dict[uid]['label_document_id'] == mention_label:
#                        mention2coref_mentions[muid].append(uid)
#                    else:
#                        mention2not_coref_mentions[muid].append(uid)
#
#
#        coref_cand_file = os.path.join(args.coref_candidate_sets_dir,
#                'mention_coref_candidates.{}.k_{}.pkl'.format(split, args.k))
#        with open(coref_cand_file, 'rb') as f:
#            coref_candidates = pickle.load(f)
#
#        mention_entity_scores_file = os.path.join(args.mention_entity_scores_dir,
#                'mention_entity_scores.{}.pkl'.format(split))
#        with open(mention_entity_scores_file, 'rb') as f:
#            mention_entity_scores = pickle.load(f)
#
#        examples = []
#        for m in mentions:
#            muid = m['mention_id']
#            if split == 'train':
#                _coref_cands = mention2coref_mentions[muid] + mention2not_coref_mentions[muid]
#                coref_cands = [mention_dict[uid] for uid in _coref_cands[:args.k]]
#            else:
#                coref_cands = [mention_dict[uid] for uid in coref_candidates[muid]]
#            #coref_cands_entity_scores = [mention_entity_scores[uid]
#            #        for uid in coref_candidates[muid]]
#            #coref_cands_linking_decisions = [max(scores.items(), key=operator.itemgetter(1))[0]
#            #        for scores in coref_cands_entity_scores]
#            #labels = [int(x == m['label_document_id']) for x in coref_cands_linking_decisions]
#            labels = [int(cm['label_document_id'] == m['label_document_id'])
#                        for cm in coref_cands]
#            #if split == 'train' and sum(labels) == 0 and not evaluate:
#            #    continue
#
#            for coref_cand, label in zip(coref_cands, labels):
#                examples.append(CorefLinkingExample(m, [coref_cand], [label]))
#
#            #examples.append(CorefLinkingExample(m, coref_cands, labels))
#
#        return (examples, documents)
#
#    def convert_examples_to_features(
#        self,
#        examples: List[CorefLinkingExample],
#        documents: dict,
#        cached_examples_dir: str,       
#        max_length: int,
#        tokenizer: PreTrainedTokenizer,
#        evaluate: bool,
#        pad_token_segment_id=0,
#        pad_on_left=False,
#        pad_token=0,
#        mask_padding_with_zero=True,
#    ) -> List[int]:
#        """
#        Loads a data file into a list of `MentionAffintityFeatures`
#        """
#
#        # account for the tokens marking the mention '[CLS]', '[SEP]', and '[SEP]'
#        _max_normal_tokens = max_length - 3
#        mention_length = _max_normal_tokens // 2
#        coref_mention_length = _max_normal_tokens - mention_length
#        indices = []
#
#        for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
#            if ex_index % 10000 == 0:
#                logger.info("Writing example %d of %d" % (ex_index, len(examples)))
#
#            example_features = []
#
#            mention = example.mention
#            context_document_id = mention['context_document_id']
#            start_index = mention['start_index']
#            end_index = mention['end_index']
#
#            context_document = documents[context_document_id]['text']
#
#            context_tokens = context_document.split()
#            extracted_mention = context_tokens[start_index: end_index+1]
#            extracted_mention = ' '.join(extracted_mention)
#            assert extracted_mention == mention['text']
#
#            mention_text_tokenized = tokenizer.tokenize(mention['text'])
#
#            mention_context = get_mention_context_tokens(
#                    context_tokens, start_index, end_index,
#                    mention_length, tokenizer)
#
#            for coref_mention, label in zip(example.coref_mentions, example.labels):
#
#                context_document_id = coref_mention['context_document_id']
#                start_index = coref_mention['start_index']
#                end_index = coref_mention['end_index']
#
#                context_document = documents[context_document_id]['text']
#
#                context_tokens = context_document.split()
#                extracted_mention = context_tokens[start_index: end_index+1]
#                extracted_mention = ' '.join(extracted_mention)
#                assert extracted_mention == coref_mention['text']
#
#                coref_mention_text_tokenized = tokenizer.tokenize(coref_mention['text'])
#
#                coref_mention_context = get_mention_context_tokens(
#                        context_tokens, start_index, end_index,
#                        coref_mention_length, tokenizer)
#
#                input_tokens = ['[CLS]'] + mention_context + ['[SEP]']
#                input_tokens += coref_mention_context + ['[SEP]']
#                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#                token_type_ids = ([0] * (len(mention_context) + 2)
#                                  + [1] * (len(coref_mention_context) + 1)) # segment_ids
#
#                # The mask has 1 for real tokens and 0 for padding tokens. Only real
#                # tokens are attended to.
#                attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask
#
#                assert pad_on_left == False
#                padding_length = max_length - len(input_ids)
#                input_ids = input_ids + ([pad_token] * padding_length)
#                attention_mask = attention_mask + (
#                        [0 if mask_padding_with_zero else 1] * padding_length)
#                token_type_ids = token_type_ids + (
#                        [pad_token_segment_id] * padding_length)
#
#                assert len(input_ids) == max_length
#                assert len(attention_mask) == max_length
#                assert len(token_type_ids) == max_length
#
#                example_features.append((mention['mention_id'], coref_mention['mention_id'], label, input_ids, attention_mask, token_type_ids))
#
#            featurized_example = CorefLinkingFeatures(example_features)
#            mention_ids = torch.tensor(
#                select_field(featurized_example, 'mention_id', evaluate), dtype=torch.long)
#            coref_mention_ids = torch.tensor(
#                select_field(featurized_example, 'coref_mention_id', evaluate), dtype=torch.long)
#            labels = torch.tensor(
#                select_field(featurized_example, 'label', evaluate), dtype=torch.long)
#            input_ids = torch.tensor(
#                select_field(featurized_example, 'input_ids', evaluate), dtype=torch.long)
#            attention_masks = torch.tensor(
#                select_field(featurized_example, 'attention_mask', evaluate), dtype=torch.long)
#            token_type_ids = torch.tensor(
#                select_field(featurized_example, 'token_type_ids', evaluate),
#                dtype=torch.long)
#
#            torch.save(
#                    (mention_ids,
#                     coref_mention_ids,
#                     labels,
#                     input_ids,
#                     attention_masks,
#                     token_type_ids),
#                    os.path.join(cached_examples_dir, str(ex_index) + '.pt'))
#            indices.append(ex_index)
#
#        return indices
#
#
#class CandidateGenerationTrainFeatures():
#    def __init__(self, features):
#        self.features = [
#            {
#                'mention_id': mention_id,
#                'entity_id': entity_id,
#                'label': label,
#                'input_ids': input_ids,
#                'attention_mask': attention_mask,
#                'token_type_ids': token_type_ids
#            }
#            for mention_id, entity_id, label, input_ids, attention_mask, token_type_ids in features
#        ]
#
#
#class CandidateGenerationProcessor(DataProcessor):
#    """Processor for the linking task."""
#
#    def _get_mentions_and_entities(self, data_dir, split, domains):
#        # Load all of the mentions
#        mention_file = os.path.join(data_dir, 'mentions', split + '.json')
#        mentions = _read_mentions(mention_file)
#
#        # Load all of the documents for the mentions
#        # `documents` is a dictionary with key 'document_id'
#        #assert split in domains
#        document_files = [os.path.join(data_dir, 'documents', domain + '.json')
#                            for domain in domains]
#        documents = _read_documents(document_files)
#        entity_document_files = [os.path.join(data_dir, 'documents', domain + '.json')
#                            for domain in domains if domain != split]
#        entity_documents = _read_documents(entity_document_files)
#
#        return mentions, documents, entity_documents
#
#    def _get_mention_features(self,
#                              mention,
#                              documents,
#                              tokenizer,
#                              mention_length,
#                              max_length,
#                              pad_token_segment_id,
#                              pad_on_left,
#                              pad_token,
#                              mask_padding_with_zero):
#
#        context_document_id = mention['context_document_id']
#        start_index = mention['start_index']
#        end_index = mention['end_index']
#
#        context_document = documents[context_document_id]['text']
#
#        context_tokens = context_document.split()
#        extracted_mention = context_tokens[start_index: end_index+1]
#        extracted_mention = ' '.join(extracted_mention)
#        assert extracted_mention == mention['text']
#
#        mention_text_tokenized = tokenizer.tokenize(mention['text'])
#
#        mention_context = get_mention_context_tokens(
#                context_tokens, start_index, end_index,
#                mention_length, tokenizer)
#
#        input_tokens = ['[CLS]'] + mention_context + ['[SEP]']
#        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#        token_type_ids = [0] * len(input_ids) # segment_ids
#
#        # The mask has 1 for real tokens and 0 for padding tokens. Only real
#        # tokens are attended to.
#        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask
#
#        assert pad_on_left == False
#        padding_length = max_length - len(input_ids)
#        input_ids = input_ids + ([pad_token] * padding_length)
#        attention_mask = attention_mask + (
#                [0 if mask_padding_with_zero else 1] * padding_length)
#        token_type_ids = token_type_ids + (
#                [pad_token_segment_id] * padding_length)
#
#        assert len(input_ids) == max_length
#        assert len(attention_mask) == max_length
#        assert len(token_type_ids) == max_length
#
#        return input_ids, attention_mask, token_type_ids
#
#    def _get_entity_features(self,
#                             entity_id,
#                             documents,
#                             tokenizer,
#                             entity_length,
#                             max_length,
#                             pad_token_segment_id,
#                             pad_on_left,
#                             pad_token,
#                             mask_padding_with_zero):
#
#        entity_text = documents[entity_id]['text']
#        entity_tokens = tokenizer.tokenize(entity_text)
#        entity_tokens = entity_tokens[:entity_length]
#
#        input_tokens = ['[CLS]'] + entity_tokens + ['[SEP]']
#        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#        token_type_ids = [0] * len(input_ids) # segment_ids
#
#        # The mask has 1 for real tokens and 0 for padding tokens. Only real
#        # tokens are attended to.
#        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids) # input_mask
#
#        assert pad_on_left == False
#        padding_length = max_length - len(input_ids)
#        input_ids = input_ids + ([pad_token] * padding_length)
#        attention_mask = attention_mask + (
#                [0 if mask_padding_with_zero else 1] * padding_length)
#        token_type_ids = token_type_ids + (
#                [pad_token_segment_id] * padding_length)
#
#        assert len(input_ids) == max_length
#        assert len(attention_mask) == max_length
#        assert len(token_type_ids) == max_length
#
#        return input_ids, attention_mask, token_type_ids
#
#    def get_train_features(
#        self,
#        data_dir: str,
#        split: str,
#        domains: List[str],
#        cached_pos_pairs_examples_dir: str,       
#        cached_neg_entities_examples_dir: str,       
#        max_length: int,
#        tokenizer: PreTrainedTokenizer,
#        pad_token_segment_id=0,
#        pad_on_left=False,
#        pad_token=0,
#        mask_padding_with_zero=True
#    ) -> List[int]:
#        """See base class."""
#        assert split == 'train'
#        logger.info("LOOKING AT {} train".format(data_dir))
#
#        mentions, documents, entity_documents = self._get_mentions_and_entities(
#                data_dir, split, domains)
#        
#        # account for the tokens marking the mention '[CLS]' and '[SEP]'
#        mention_length = entity_length = max_length - 2
#        pos_pairs_indices = []
#        pos_entities = []
#
#        for (mention_index, mention) in tqdm.tqdm(enumerate(mentions), desc="convert examples to features"):
#            if mention_index % 10000 == 0:
#                logger.info("Writing example %d of %d" % (mention_index, len(mentions)))
#
#            pos_entities.append(mention['label_document_id'])
#
#            (mention_input_ids,
#             mention_attention_mask,
#             mention_token_type_ids) = self._get_mention_features(
#                 mention, documents, tokenizer, mention_length, max_length,
#                 pad_token_segment_id, pad_on_left, pad_token,
#                 mask_padding_with_zero)
#
#            (entity_input_ids,
#             entity_attention_mask,
#             entity_token_type_ids) = self._get_entity_features(
#                 mention['label_document_id'], documents, tokenizer,
#                 entity_length, max_length, pad_token_segment_id, pad_on_left,
#                 pad_token, mask_padding_with_zero)
#
#            torch.save(
#                    (torch.tensor(mention_input_ids, dtype=torch.long),
#                     torch.tensor(mention_attention_mask, dtype=torch.long),
#                     torch.tensor(mention_token_type_ids, dtype=torch.long),
#                     torch.tensor(entity_input_ids, dtype=torch.long),
#                     torch.tensor(entity_attention_mask, dtype=torch.long),
#                     torch.tensor(entity_token_type_ids, dtype=torch.long)),
#                os.path.join(cached_pos_pairs_examples_dir, str(mention_index) + '.pt'))
#            pos_pairs_indices.append(mention_index)
#
#        pos_entities = set(pos_entities)
#        neg_entities_indices = []
#
#        for (entity_index, entity) in tqdm.tqdm(enumerate(entity_documents.keys()), desc="convert entities to features"):
#            if entity_index % 10000 == 0:
#                logger.info("Writing example %d of %d" % (entity_index, len(entity_documents.keys())))
#
#            # don't include any of the paired entities in this index set
#            if entity in pos_entities:
#                continue
#
#            (entity_input_ids,
#             entity_attention_mask,
#             entity_token_type_ids) = self._get_entity_features(
#                 entity, documents, tokenizer, entity_length, max_length,
#                 pad_token_segment_id, pad_on_left, pad_token,
#                 mask_padding_with_zero)
#
#            torch.save(
#                    (torch.tensor(id_to_bytes(entity), dtype=torch.long),
#                     torch.tensor(entity_input_ids, dtype=torch.long),
#                     torch.tensor(entity_attention_mask, dtype=torch.long),
#                     torch.tensor(entity_token_type_ids, dtype=torch.long)),
#                os.path.join(cached_neg_entities_examples_dir, str(entity_index) + '.pt'))
#            neg_entities_indices.append(entity_index)
#
#        return pos_pairs_indices, neg_entities_indices
#
#
#    def get_eval_features(
#        self,
#        data_dir: str,
#        split: str,
#        domains: List[str],
#        cached_mention_examples_dir: str,       
#        cached_entity_examples_dir: str,       
#        max_length: int,
#        tokenizer: PreTrainedTokenizer,
#        pad_token_segment_id=0,
#        pad_on_left=False,
#        pad_token=0,
#        mask_padding_with_zero=True
#    ) -> List[int]:
#        """See base class."""
#        logger.info("LOOKING AT {} train".format(data_dir))
#
#        mentions, documents, entity_documents = self._get_mentions_and_entities(
#                data_dir, split, domains)
#        
#        # account for the tokens marking the mention '[CLS]' and '[SEP]'
#        mention_length = entity_length = max_length - 2
#        mention_indices = []
#        entity_indices = []
#
#        for (mention_index, mention) in tqdm.tqdm(enumerate(mentions), desc="convert mentions to features"):
#            if mention_index % 10000 == 0:
#                logger.info("Writing example %d of %d" % (mention_index, len(mentions)))
#
#            (mention_input_ids,
#             mention_attention_mask,
#             mention_token_type_ids) = self._get_mention_features(
#                 mention, documents, tokenizer, mention_length, max_length,
#                 pad_token_segment_id, pad_on_left, pad_token,
#                 mask_padding_with_zero)
#
#            torch.save(
#                    (torch.tensor(id_to_bytes(mention['mention_id']), dtype=torch.long),
#                     torch.tensor(mention_input_ids, dtype=torch.long),
#                     torch.tensor(mention_attention_mask, dtype=torch.long),
#                     torch.tensor(mention_token_type_ids, dtype=torch.long)),
#                os.path.join(cached_mention_examples_dir, str(mention_index) + '.pt'))
#            mention_indices.append(mention_index)
#
#        for (entity_index, entity) in tqdm.tqdm(enumerate(entity_documents.keys()), desc="convert entities to features"):
#            if entity_index % 10000 == 0:
#                logger.info("Writing example %d of %d" % (entity_index, len(entity_documents.keys())))
#
#            (entity_input_ids,
#             entity_attention_mask,
#             entity_token_type_ids) = self._get_entity_features(
#                 entity, documents, tokenizer, entity_length, max_length,
#                 pad_token_segment_id, pad_on_left, pad_token,
#                 mask_padding_with_zero)
#
#            torch.save(
#                    (torch.tensor(id_to_bytes(entity), dtype=torch.long),
#                     torch.tensor(entity_input_ids, dtype=torch.long),
#                     torch.tensor(entity_attention_mask, dtype=torch.long),
#                     torch.tensor(entity_token_type_ids, dtype=torch.long)),
#                os.path.join(cached_entity_examples_dir, str(entity_index) + '.pt'))
#            entity_indices.append(entity_index)
#
#        return mention_indices, entity_indices
#
#
#processors = {
#    "mention_affinity": MentionAffinityProcessor,
#    "candidate_generation": CandidateGenerationProcessor,
#    "coref_linking": CorefLinkingProcessor,
#    "linking": LinkingProcessor,
#    "cluster_linking": ClusterLinkingProcessor
#}
