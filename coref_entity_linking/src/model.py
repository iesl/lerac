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

import os
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from transformers import (BertConfig,
                          BertModel,
                          BertTokenizer,
                          BertPreTrainedModel,
                          DistilBertModel)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils.comm import get_rank, all_gather, synchronize
from utils.misc import (CLS_TOKEN,
                        SEP_TOKEN,
                        START_MENTION_HGHLGHT_TOKEN,
                        END_MENTION_HGHLGHT_TOKEN,
                        START_ENTITY_HGHLGHT_TOKEN,
                        END_ENTITY_HGHLGHT_TOKEN,
                        flatten,
                        DistributedCache)

from IPython import embed


class VersatileModel(nn.Module):

    def __init__(self, args, name=''):
        super(VersatileModel, self).__init__()

        # for saving purposes
        if name is not '':
            name += '_'
        self.name = name

        self.args = args
        if args.model_name_or_path is not None:
            assert args.trained_model_dir is None
            self._create_model_from_scratch()
        else:
            assert args.trained_model_dir is not None
            self._load_pretrained_model()

    def _create_model_from_scratch(self):
        args = self.args
        config = BertConfig.from_pretrained(
                args.config_name 
                    if args.config_name
                    else args.model_name_or_path,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir
                    if args.cache_dir
                    else None)

        self.tokenizer = BertTokenizer.from_pretrained(
               args.tokenizer_name
                   if args.tokenizer_name
                   else args.model_name_or_path,
               do_lower_case=args.do_lower_case,
               cache_dir=args.cache_dir
                   if args.cache_dir
                   else None)

        # Add some custom variables to the config
        config.start_mention_id = self.tokenizer.convert_tokens_to_ids(START_MENTION_HGHLGHT_TOKEN)
        config.end_mention_id = self.tokenizer.convert_tokens_to_ids(END_MENTION_HGHLGHT_TOKEN)
        config.start_entity_id = self.tokenizer.convert_tokens_to_ids(START_ENTITY_HGHLGHT_TOKEN)
        config.end_entity_id = self.tokenizer.convert_tokens_to_ids(END_ENTITY_HGHLGHT_TOKEN)
        config.cls_id = self.tokenizer.convert_tokens_to_ids(CLS_TOKEN)
        config.sep_id = self.tokenizer.convert_tokens_to_ids(SEP_TOKEN)

        m_e_config = copy.deepcopy(config)
        m_e_config.concat_pooling_strategy = 'cls'
        self.m_e_model = BertSequenceScoringModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=m_e_config,
                cache_dir=args.cache_dir if args.cache_dir else None)

        m_m_config = copy.deepcopy(config)
        m_m_config.concat_pooling_strategy = 'pool_highlighted_outputs'
        self.m_m_model = BertSequenceScoringModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=m_m_config,
                cache_dir=args.cache_dir if args.cache_dir else None)

    def _load_pretrained_model(self):
        args = self.args
        self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model'))
        self.m_e_model = BertSequenceScoringModel.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model', 'm_e_model'))
        self.m_m_model = BertSequenceScoringModel.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model', 'm_m_model'))

    def save_model(self, suffix=None):
        # suffix should be a string that describes the model, often a checkpoint
        assert suffix is not None
        args = self.args
        save_dir = os.path.join(args.output_dir, suffix, self.name + 'model')
        m_e_save_dir = os.path.join(save_dir, 'm_e_model')
        m_m_save_dir = os.path.join(save_dir, 'm_m_model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(m_e_save_dir):
            os.makedirs(m_e_save_dir)
        if not os.path.exists(m_m_save_dir):
            os.makedirs(m_m_save_dir)
        self.m_e_model.save_pretrained(m_e_save_dir)
        self.m_m_model.save_pretrained(m_m_save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def forward(self,
                m_e_mask=None,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                head_mask=None,
                inputs_embeds=None,
                evaluate=False):
        
        batch_size, num_seq, sym_dim, seq_len = input_ids.shape
        outputs = torch.zeros((batch_size*num_seq, 1, sym_dim)).to(self.args.device)
        m_e_mask = m_e_mask.reshape(batch_size*num_seq,)
        input_ids = input_ids.reshape(batch_size*num_seq, 1, sym_dim, seq_len)
        attention_mask = attention_mask.reshape(batch_size*num_seq, 1, sym_dim, seq_len)
        token_type_ids = token_type_ids.reshape(batch_size*num_seq, 1, sym_dim, seq_len)
        outputs[m_e_mask] = self.m_e_model(input_ids=input_ids[m_e_mask],
                                           attention_mask=attention_mask[m_e_mask],
                                           token_type_ids=token_type_ids[m_e_mask],
                                           evaluate=evaluate)
        outputs[~m_e_mask] = self.m_m_model(input_ids=input_ids[~m_e_mask],
                                            attention_mask=attention_mask[~m_e_mask],
                                            token_type_ids=token_type_ids[~m_e_mask],
                                            evaluate=evaluate)
        outputs = outputs.reshape((batch_size, num_seq, sym_dim))
        
        return outputs


class MirrorEmbeddingModel(nn.Module):
    
    def __init__(self, args, name=''):
        super(MirrorEmbeddingModel, self).__init__()

        # for saving purposes
        if name is not '':
            name += '_'
        self.name = name

        self.args = args
        if args.model_name_or_path is not None:
            assert args.trained_model_dir is None
            self._create_model_from_scratch()
        else:
            assert args.trained_model_dir is not None
            self._load_pretrained_model()

    def _create_model_from_scratch(self):
        args = self.args
        config = BertConfig.from_pretrained(
                args.config_name 
                    if args.config_name
                    else args.model_name_or_path,
                finetuning_task=args.task_name,
                cache_dir=args.cache_dir
                    if args.cache_dir
                    else None)

        self.tokenizer = BertTokenizer.from_pretrained(
               args.tokenizer_name
                   if args.tokenizer_name
                   else args.model_name_or_path,
               do_lower_case=args.do_lower_case,
               cache_dir=args.cache_dir
                   if args.cache_dir
                   else None)

        # Add some custom variables to the config
        config.start_mention_id = self.tokenizer.convert_tokens_to_ids(START_MENTION_HGHLGHT_TOKEN)
        config.end_mention_id = self.tokenizer.convert_tokens_to_ids(END_MENTION_HGHLGHT_TOKEN)
        config.start_entity_id = self.tokenizer.convert_tokens_to_ids(START_ENTITY_HGHLGHT_TOKEN)
        config.end_entity_id = self.tokenizer.convert_tokens_to_ids(END_ENTITY_HGHLGHT_TOKEN)
        config.pooling_strategy = args.pooling_strategy
        config.seq_embed_dim = args.seq_embed_dim
        self.model = SequenceEmbeddingModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None)

    def _load_pretrained_model(self):
        args = self.args
        self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model'))
        self.model = SequenceEmbeddingModel.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model'))

    def save_model(self, suffix=None):
        # suffix should be a string that describes the model, often a checkpoint
        assert suffix is not None
        args = self.args
        save_dir = os.path.join(args.output_dir, suffix, self.name + 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

    def forward(self,
                input_ids_a=None,
                attention_mask_a=None,
                token_type_ids_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                token_type_ids_b=None,
                head_mask=None,
                inputs_embeds=None,
                evaluate=False):

        _squeeze_batch_dim = False
        if len(input_ids_a.shape) == 2:
            _squeeze_batch_dim = True
            input_ids_a = input_ids_a.unsqueeze(0)
            attention_mask_a = attention_mask_a.unsqueeze(0)
            token_type_ids_a = token_type_ids_a.unsqueeze(0)

        if input_ids_b is not None:
            if _squeeze_batch_dim:
                input_ids_b = input_ids_b.unsqueeze(0)
                attention_mask_b = attention_mask_b.unsqueeze(0)
                token_type_ids_b = token_type_ids_b.unsqueeze(0)

            assert len(input_ids_a.shape) == len(input_ids_b.shape)
            assert input_ids_a.shape[0] == input_ids_b.shape[0]
            num_seq_a = input_ids_a.shape[1]

            input_ids = torch.cat((input_ids_a, input_ids_b), 1)
            attention_mask = torch.cat((attention_mask_a, attention_mask_b), 1)
            token_type_ids = torch.cat((token_type_ids_a, token_type_ids_b), 1)
        else:
            input_ids = input_ids_a
            attention_mask = attention_mask_a
            token_type_ids = token_type_ids_a
        
        embeddings = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                evaluate=evaluate)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        if input_ids_b is None:
            if _squeeze_batch_dim:
                embeddings = embeddings.squeeze(0)
            return embeddings
        else:
            embeddings_a = embeddings[:,:num_seq_a,:]
            embeddings_b = torch.transpose(embeddings[:,num_seq_a:,:], 1, 2)
            scores = 1.0 - torch.bmm(embeddings_a, embeddings_b).squeeze(0)

            return scores


###################################
#### Custom BERT Model Classes ####
###################################


class SequenceEmbeddingModel(BertPreTrainedModel):

    def __init__(self, config):
        super(SequenceEmbeddingModel, self).__init__(config)
        self.config = config
        self.start_mention_id = config.start_mention_id
        self.end_mention_id = config.end_mention_id

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embedding_layer = nn.Linear(config.hidden_size, config.seq_embed_dim)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False
    ):

        with torch.no_grad():
            batch_size, num_seq, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

            if self.config.pooling_strategy == 'pool_highlighted_outputs':
                index_tensor = torch.arange(input_ids.shape[1])
                index_tensor = index_tensor.repeat(input_ids.shape[0], 1)
                index_tensor = index_tensor.to(input_ids.device)
                start_indices = (input_ids == self.start_mention_id).nonzero()[:,1:]
                end_indices = (input_ids == self.end_mention_id).nonzero()[:,1:]
                mask = (index_tensor > start_indices) & (index_tensor < end_indices)
                mask.unsqueeze_(-1)
            elif self.config.pooling_strategy == 'pool_all_outputs':
                mask = attention_mask[:,:,None]
            else:
                raise ValueError("Invalid pooling strategy")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = (torch.sum(outputs[0] * mask, 1)
                         / torch.sum(mask, 1))

        pooled_output = self.dropout(pooled_output)
        seq_embedding = self.embedding_layer(pooled_output)
        seq_embedding = seq_embedding.reshape(batch_size, num_seq, -1)

        return seq_embedding


class BertSequenceScoringModel(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSequenceScoringModel, self).__init__(config)
        self.config = config
        self.start_mention_id = config.start_mention_id
        self.end_mention_id = config.end_mention_id
        self.start_entity_id = config.start_entity_id
        self.end_entity_id = config.end_entity_id
        self.cls_id = config.cls_id
        self.sep_id = config.sep_id

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # for concatenation pool_highlighted_outputs mode
        if self.config.concat_pooling_strategy == 'pool_highlighted_outputs':
            self.concatenation_pool_layer1 = nn.Linear(2*config.hidden_size, config.hidden_size)
            self.concatenation_pool_layer2 = nn.Linear(config.hidden_size, config.hidden_size)
            self.concatenation_pool_layer3 = nn.Linear(config.hidden_size, 1)

        # for concatenation cls mode
        if self.config.concat_pooling_strategy == 'cls':
            self.concatenation_cls_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
            self.concatenation_cls_layer2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False
    ):

        with torch.no_grad():
            batch_size, num_seq, sym_dim, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

            if self.config.concat_pooling_strategy == 'pool_highlighted_outputs':
                index_tensor = torch.arange(input_ids.shape[1])
                index_tensor = index_tensor.repeat(input_ids.shape[0], 1)
                index_tensor = index_tensor.to(input_ids.device)

                start_indices = torch.nonzero((input_ids == self.start_mention_id) | (input_ids == self.start_entity_id),
                                              as_tuple=False)[:,1:]
                end_indices = torch.nonzero((input_ids == self.end_mention_id) | (input_ids == self.end_entity_id),
                                            as_tuple=False)[:,1:]

                start_indices_a = start_indices[::2,:]
                end_indices_a = end_indices[::2,:]
                start_indices_b = start_indices[1::2,:]
                end_indices_b = end_indices[1::2,:]
                
                mask_a = (index_tensor > start_indices_a) & (index_tensor < end_indices_a)
                mask_b = (index_tensor > start_indices_b) & (index_tensor < end_indices_b)
                mask_a.unsqueeze_(-1)
                mask_b.unsqueeze_(-1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        if self.config.concat_pooling_strategy == 'pool_highlighted_outputs':
            pooled_output_a = (torch.sum(outputs[0] * mask_a, 1)
                             / torch.sum(mask_a, 1))
            pooled_output_a = self.dropout(pooled_output_a)
            pooled_output_b = (torch.sum(outputs[0] * mask_b, 1)
                             / torch.sum(mask_b, 1))
            pooled_output_b = self.dropout(pooled_output_b)
            pooled_output = torch.cat((pooled_output_a, pooled_output_b), dim=1)
            output = F.relu(self.concatenation_pool_layer1(pooled_output))
            output = F.relu(self.concatenation_pool_layer2(output))
            output = self.concatenation_pool_layer3(output)
            output = output.reshape(batch_size, num_seq, sym_dim)
        elif self.config.concat_pooling_strategy == 'cls':
            output = outputs[0][:, 0, :]
            output = F.relu(self.concatenation_cls_layer1(output))
            output = self.concatenation_cls_layer2(output)
            output = output.reshape(batch_size, num_seq, sym_dim)

        output = torch.sigmoid(output)

        return output


class VersatileBertModel(BertPreTrainedModel):

    def __init__(self, config):
        super(VersatileBertModel, self).__init__(config)
        self.config = config
        self.start_mention_id = config.start_mention_id
        self.end_mention_id = config.end_mention_id
        self.start_entity_id = config.start_entity_id
        self.end_entity_id = config.end_entity_id
        self.cls_id = config.cls_id
        self.sep_id = config.sep_id

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # for embedding mode
        self.embedding_layer = nn.Linear(config.hidden_size, config.seq_embed_dim)

        # for concatenation pool_highlighted_outputs mode
        self.concatenation_pool_layer1 = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.concatenation_pool_layer2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.concatenation_pool_layer3 = nn.Linear(config.hidden_size, 1)

        # for concatenation cls mode
        self.concatenation_cls_layer1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.concatenation_cls_layer2 = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        concat_input=False,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False
    ):

        inputs = {
            'input_ids' : input_ids,
            'attention_mask' : attention_mask,
            'token_type_ids' : token_type_ids,
            'position_ids' : position_ids,
            'head_mask' : head_mask,
            'inputs_embeds' : inputs_embeds,
            'evaluate' : evaluate
        }

        if not concat_input:
            return self._forward_embed(**inputs)
        else:
            return self._forward_concat(**inputs)

    def _forward_embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False
    ):
        with torch.no_grad():
            _squeeze_batch_dim = False
            if len(input_ids.shape) == 2:
                _squeeze_batch_dim = True
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)
                token_type_ids = token_type_ids.unsqueeze(0)

            batch_size, num_seq, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

            if self.config.embed_pooling_strategy == 'pool_highlighted_outputs':
                index_tensor = torch.arange(input_ids.shape[1])
                index_tensor = index_tensor.repeat(input_ids.shape[0], 1)
                index_tensor = index_tensor.to(input_ids.device)
                #start_indices = (input_ids == self.start_mention_id).nonzero()[:,1:]
                #end_indices = (input_ids == self.end_mention_id).nonzero()[:,1:]
                start_indices = torch.nonzero((input_ids == self.start_mention_id) | (input_ids == self.start_entity_id),
                                              as_tuple=False)[:,1:]
                end_indices = torch.nonzero((input_ids == self.end_mention_id) | (input_ids == self.end_entity_id),
                                            as_tuple=False)[:,1:]
                mask = (index_tensor > start_indices) & (index_tensor < end_indices)
                mask.unsqueeze_(-1)
            elif self.config.embed_pooling_strategy == 'pool_all_outputs':
                mask = attention_mask[:,:,None]
            else:
                raise ValueError("Invalid pooling strategy")

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = (torch.sum(outputs[0] * mask, 1)
                         / torch.sum(mask, 1))
        pooled_output = self.dropout(pooled_output)
        seq_embedding = self.embedding_layer(pooled_output)
        seq_embedding = F.normalize(seq_embedding, p=2, dim=-1)
        seq_embedding = seq_embedding.reshape(batch_size, num_seq, -1)
        if _squeeze_batch_dim:
            seq_embedding = seq_embedding.squeeze(0)
        return seq_embedding

    def _forward_concat(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        evaluate=False
    ):

        #assert self.config.concat_pooling_strategy == 'pool_highlighted_outputs'
        
        with torch.no_grad():
            batch_size, num_seq, sym_dim, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

            index_tensor = torch.arange(input_ids.shape[1])
            index_tensor = index_tensor.repeat(input_ids.shape[0], 1)
            index_tensor = index_tensor.to(input_ids.device)

            #_start_indices = (input_ids == self.start_mention_id).nonzero()
            #_end_indices = (input_ids == self.end_mention_id).nonzero()

            start_indices = torch.nonzero((input_ids == self.start_mention_id) | (input_ids == self.start_entity_id),
                                          as_tuple=False)[:,1:]
            end_indices = torch.nonzero((input_ids == self.end_mention_id) | (input_ids == self.end_entity_id),
                                        as_tuple=False)[:,1:]

            start_indices_a = start_indices[::2,:]
            end_indices_a = end_indices[::2,:]
            start_indices_b = start_indices[1::2,:]
            end_indices_b = end_indices[1::2,:]
            
            mask_a = (index_tensor > start_indices_a) & (index_tensor < end_indices_a)
            mask_b = (index_tensor > start_indices_b) & (index_tensor < end_indices_b)
            mask_a.unsqueeze_(-1)
            mask_b.unsqueeze_(-1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        #if self.config.concat_pooling_strategy == 'pool_highlighted_outputs':
        pooled_output_a = (torch.sum(outputs[0] * mask_a, 1)
                         / torch.sum(mask_a, 1))
        pooled_output_a = self.dropout(pooled_output_a)
        pooled_output_b = (torch.sum(outputs[0] * mask_b, 1)
                         / torch.sum(mask_b, 1))
        pooled_output_b = self.dropout(pooled_output_b)
        pooled_output = torch.cat((pooled_output_a, pooled_output_b), dim=1)
        output = F.relu(self.concatenation_pool_layer1(pooled_output))
        output = F.relu(self.concatenation_pool_layer2(output))
        output = self.concatenation_pool_layer3(output)
        pooled_hghlght_output = output.reshape(batch_size, num_seq, sym_dim)
            #output = torch.sigmoid(output)
        #elif self.config.concat_pooling_strategy == 'cls':
        cls_output = outputs[0][:, 0, :]
        cls_output = F.relu(self.concatenation_cls_layer1(cls_output))
        cls_output = self.concatenation_cls_layer2(cls_output)
        cls_output = cls_output.reshape(batch_size, num_seq, sym_dim)
        return cls_output, pooled_hghlght_output
