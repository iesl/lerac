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

from utils.misc import (START_HGHLGHT_TOKEN,
                        END_HGHLGHT_TOKEN,
                        flatten,
                        DistributedCache)

from IPython import embed


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
        config.start_mention_id = self.tokenizer.convert_tokens_to_ids(START_HGHLGHT_TOKEN)
        config.end_mention_id = self.tokenizer.convert_tokens_to_ids(END_HGHLGHT_TOKEN)
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

