import os
import json
import math
import tqdm
from collections import defaultdict
from types import SimpleNamespace
import torch

from utils.misc import (START_MENTION_HGHLGHT_TOKEN,
                        END_MENTION_HGHLGHT_TOKEN,
                        START_ENTITY_HGHLGHT_TOKEN,
                        END_ENTITY_HGHLGHT_TOKEN)

from IPython import embed


class ZeshelPreprocessor(object):
    """
    Preprocess data in the zero-shot format
    """

    def __init__(self, args):
        self.args = args

    def read_mentions(self, mention_file):
        mentions = []
        with open(mention_file, encoding='utf-8') as fin:
            for line in fin:
                mentions.append(json.loads(line))
        return mentions

    def read_documents(self, document_files):
        documents = {}
        for fname in document_files:
            with open(fname, encoding='utf-8') as fin:
                for line in fin:
                    doc_dict = json.loads(line)
                    documents[doc_dict['document_id']] = doc_dict
        return documents

    def read_candidates(self, candidate_file):
        candidates = {}
        with open(candidate_file, encoding='utf-8') as fin:
            for line in fin:
                candidate_dict = json.loads(line)
                candidates[candidate_dict['mention_id']] = candidate_dict['tfidf_candidates']
        return candidates

    def get_mention_context_tokens(self,
                                   context_tokens,
                                   start_index,
                                   end_index,
                                   max_tokens,
                                   tokenizer):
        start_pos = start_index - max_tokens
        if start_pos < 0:
            start_pos = 0

        prefix = ' '.join(context_tokens[start_pos: start_index])
        suffix = ' '.join(context_tokens[end_index+1: end_index+max_tokens+1])
        prefix = tokenizer.tokenize(prefix)
        suffix = tokenizer.tokenize(suffix)
        mention = tokenizer.tokenize(
                    ' '.join(context_tokens[start_index:end_index+1]))
        mention = [START_MENTION_HGHLGHT_TOKEN] + mention + [END_MENTION_HGHLGHT_TOKEN]

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

    def get_mentions_and_entities(self, data_dir, split, domains):
        # Load all of the mentions
        mention_file = os.path.join(data_dir, 'mentions', split + '.json')
        mentions = self.read_mentions(mention_file)

        # Load all of the documents for the mentions
        # `documents` is a dictionary with key 'document_id'
        document_files = [os.path.join(data_dir, 'documents', domain + '.json')
                            for domain in domains]
        documents = self.read_documents(document_files)
        entity_document_files = [os.path.join(data_dir, 'documents', domain + '.json')
                            for domain in domains if domain != split]
        entity_documents = self.read_documents(entity_document_files)

        return mentions, documents, entity_documents

    def get_mention_input_ids(self,
                              mention,
                              documents,
                              tokenizer,
                              mention_length):

        muid = mention['mention_id']

        context_document_id = mention['context_document_id']
        start_index = mention['start_index']
        end_index = mention['end_index']

        context_document = documents[context_document_id]['text']

        context_tokens = context_document.split()
        extracted_mention = context_tokens[start_index: end_index+1]
        extracted_mention = ' '.join(extracted_mention)
        assert extracted_mention == mention['text']

        mention_text_tokenized = tokenizer.tokenize(mention['text'])

        mention_context = self.get_mention_context_tokens(
                context_tokens, start_index, end_index,
                mention_length, tokenizer)

        input_ids = tokenizer.convert_tokens_to_ids(mention_context)

        return input_ids

    def get_entity_input_ids(self,
                             entity_id,
                             documents,
                             tokenizer,
                             entity_length):

        entity_text = documents[entity_id]['text']
        entity_tokens = tokenizer.tokenize(entity_text)
        entity_tokens = entity_tokens[:entity_length-2]
        entity_tokens = [START_ENTITY_HGHLGHT_TOKEN] + entity_tokens + [END_ENTITY_HGHLGHT_TOKEN]
        input_ids = tokenizer.convert_tokens_to_ids(entity_tokens)
        return input_ids

    def preprocess_data(self,
                        data_dir,
                        split,
                        domains,
                        cache_dir,
                        max_seq_length,
                        tokenizer,
                        evaluate=False,
                        pad_token_segment_id=0,
                        pad_on_left=False,
                        pad_token=0,
                        mask_padding_with_zero=True):

        # load mentions and entities
        mentions, documents, entities = self.get_mentions_and_entities(
                data_dir, split, domains)
        mentions = {m['mention_id']: m for m in mentions}

        # load precomputed candidates
        candidate_file = os.path.join(data_dir,
                                      'tfidf_candidates', split + '.json')
        candidates = self.read_candidates(candidate_file)

        # adjust length for special tokens
        max_seq_length -= 2

        # global index
        global_idx = 0

        # metadata
        num_mentions = 0
        num_entities = 0
        uid2idx = defaultdict(lambda : None)
        idx2uid = defaultdict(lambda : None)
        midx2cand = defaultdict(list)
        midx2eidx = defaultdict(lambda : None)
        midx2type = defaultdict(lambda : None)
        wdoc_clusters = defaultdict(lambda : defaultdict(set))
        xdoc_clusters = defaultdict(set)

        # preprocess entities
        for entity in tqdm.tqdm(entities.values(),
                                desc="convert entities to features"):
            uid = entity['document_id']
            
            # track metadata
            num_entities += 1
            uid2idx[uid] = global_idx
            idx2uid[global_idx] = uid

            # preprocess entity sequence
            entity_desc_ids = self.get_entity_input_ids(
                    uid, documents, tokenizer, max_seq_length
            )

            # save sequence
            torch.save(
                    torch.tensor(entity_desc_ids, dtype=torch.long),
                    os.path.join(cache_dir,
                                 str(global_idx) + '.pt')
            )
            global_idx += 1

        train_no_label_count = 0

        # preprocess mentions
        for mention in tqdm.tqdm(mentions.values(),
                                 desc="convert mentions to features"):
            uid = mention['mention_id']
            ctxt_uid = mention['context_document_id']
            label_uid = mention['label_document_id']
            if isinstance(label_uid, list):
                label_idx = [uid2idx[x] for x in label_uid]
                label_idx = list(filter(lambda x : x is not None, label_idx))
            else:
                label_idx = uid2idx[label_uid]
            mention_type = mention['category']

            # ignore all BC5CDR training examples which have no gold label
            if split == 'train' and isinstance(label_idx, list) \
                    and label_idx == [None]:
                train_no_label_count += 1
                continue

            # track metadata
            num_mentions += 1
            uid2idx[uid] = global_idx
            idx2uid[global_idx] = uid
            midx2cand[global_idx] = list(map(lambda uid : uid2idx[uid],
                                             candidates.get(uid, [])))
            midx2eidx[global_idx] = label_idx
            midx2type[global_idx] = mention_type
            if isinstance(label_idx, list): # multiple possibilities for BC5CDR
                for idx in label_idx:
                    assert idx is not None
                    wdoc_clusters[ctxt_uid][idx].add(idx)
                    wdoc_clusters[ctxt_uid][idx].add(global_idx)
                    xdoc_clusters[idx].add(global_idx)
                    xdoc_clusters[idx].add(idx)
            else:
                wdoc_clusters[ctxt_uid][label_idx].add(label_idx)
                wdoc_clusters[ctxt_uid][label_idx].add(global_idx)
                xdoc_clusters[label_idx].add(global_idx)
                xdoc_clusters[label_idx].add(label_idx)

            # preprocess mention sequence
            mention_context_ids = self.get_mention_input_ids(
                 mention, documents, tokenizer, max_seq_length
            )

            #save_sequence
            torch.save(
                    torch.tensor(mention_context_ids, dtype=torch.long),
                    os.path.join(cache_dir,
                                 str(global_idx) + '.pt')
            )
            global_idx += 1

        metadata_dict = {
            'num_mentions' : num_mentions,
            'num_entities' : num_entities,
            'uid2idx' : uid2idx,
            'idx2uid' : idx2uid,
            'midx2cand' : midx2cand,
            'midx2eidx' : midx2eidx,
            'midx2type' : midx2type,
            'wdoc_clusters' : wdoc_clusters,
            'xdoc_clusters' : xdoc_clusters,
            'mentions' : mentions,
            'entities' : entities
        }

        def _recursively_recast(d):
            _new_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    _new_dict[k] = _recursively_recast(dict(v))
                elif isinstance(v, set):
                    _new_dict[k] = list(v)
                else:
                    _new_dict[k] = v
            return _new_dict

        metadata_dict = _recursively_recast(metadata_dict)

        return SimpleNamespace(**metadata_dict)
