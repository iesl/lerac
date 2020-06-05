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

    def get_embeddings(self, dataset):
        pass

    def compute_scores_for_inference(self, dataset):
        pass

    def train_on_subset(self, dataset):
        pass

    def compute_topk_score_evaluation(self, dataset):
        pass



class ScalarAffineModel(nn.Module):

    def __init__(self, args, name=''):
        super(ScalarAffineModel, self).__init__()
        self.args = args

        # for saving purposes
        if name is not '':
            name += '_'
        self.name = name

        self.affine_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.affine_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        #self.activation = nn.Softplus()
        self.activation = nn.Sigmoid()

        if args.model_name_or_path is None:
            assert args.trained_model_dir is not None
            model_save_path = os.path.join(args.trained_model_dir,
                                           self.name + 'model',
                                           'pytorch_model.bin')
            model_state_dict = torch.load(model_save_path)
            self.load_state_dict(model_state_dict)

    def forward(self, tensor):
        x = (self.affine_weight * tensor) + self.affine_bias
        x = self.activation(x)
        return x

    def save_model(self, suffix=None):
        assert suffix is not None
        args = self.args
        save_dir = os.path.join(args.output_dir, suffix, self.name + 'model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.state_dict(),
                   os.path.join(save_dir, 'pytorch_model.bin'))


class MirrorStackEmbeddingModel(nn.Module):
    
    def __init__(self, args, name=''):
        super(MirrorStackEmbeddingModel, self).__init__()

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

    def get_scores(self, embeddings_a, embeddings_b):
        pass


class ConcatenationModel(nn.Module):
    
    def __init__(self, args, name=''):
        super(ConcatenationModel, self).__init__()

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

        self.criterion = nn.CrossEntropyLoss()

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

        self.model = SequenceScoringModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None)

    def _load_pretrained_model(self):
        args = self.args
        self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(args.trained_model_dir, self.name + 'model'))
        self.model = SequenceScoringModel.from_pretrained(
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
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                head_mask=None,
                inputs_embeds=None,
                evaluate=False):

        with torch.no_grad():
            batch_size, num_seq, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

        scores = self.model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

        scores = scores.reshape(batch_size, num_seq)
        loss = 0.0
        if not evaluate:
            labels = labels.nonzero(as_tuple=True)[1]
            loss = self.criterion(scores, labels)

        return loss, scores


class CrossEncoder(ConcatenationModel):

    def __init__(self, args, name):
        super(CrossEncoder, self).__init__(args, name=name)

        self.criterion = nn.BCEWithLogitsLoss()
        self.activation = nn.Sigmoid() 

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                labels=None,
                head_mask=None,
                inputs_embeds=None,
                evaluate=False):

        args = self.args

        with torch.no_grad():
            batch_size, num_seq, seq_len = input_ids.shape

            input_ids = input_ids.reshape(-1, seq_len)
            attention_mask = attention_mask.reshape(-1, seq_len)
            token_type_ids = token_type_ids.reshape(-1, seq_len)

        scores = self.model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)

        scores = scores.reshape(batch_size, num_seq)

        loss = 0.0
        if not evaluate:
            loss = self.criterion(scores, labels)
        scores = self.activation(scores)

        return loss, scores

class PolyEncoder(nn.Module):
    
    def __init__(self, args):
        super(PolyEncoder, self).__init__()

        self.args = args
        if args.model_name_or_path is not None:
            assert args.trained_model_dir is None
            self._create_model_from_scratch()
        else:
            assert args.trained_model_dir is not None
            self._load_pretrained_model()

        self.criterion = nn.CrossEntropyLoss()
        
        # setup cache for inference
        self.mention_cache = DistributedCache(args)
        self.entity_cache = DistributedCache(args)

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

        # add extra config variables
        config.start_mention_id = self.tokenizer.convert_tokens_to_ids(START_HGHLGHT_TOKEN)
        config.end_mention_id = self.tokenizer.convert_tokens_to_ids(END_HGHLGHT_TOKEN)
        config.pooling_strategy = args.pooling_strategy
        config.num_context_codes = args.num_context_codes

        self.mention_model = SequenceMultiEmbeddingModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None)
        self.entity_model = SequenceEmbeddingModel.from_pretrained(
                args.model_name_or_path,
                from_tf=bool('.ckpt' in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None)

    def _load_pretrained_model(self):
        args = self.args

        self.tokenizer = BertTokenizer.from_pretrained(
                os.path.join(args.trained_model_dir, 'model'))
        self.mention_model = SequenceMultiEmbeddingModel.from_pretrained(
                os.path.join(args.trained_model_dir, 'model', 'mention_model'))
        self.entity_model = SequenceEmbeddingModel.from_pretrained(
                os.path.join(args.trained_model_dir, 'model', 'entity_model'))

    def save_model(self, suffix=None):
        # suffix should be a string that describes the model, often a checkpoint
        assert suffix is not None
        args = self.args
        
        save_dir = os.path.join(args.output_dir, suffix, 'model')
        mention_model_save_dir = os.path.join(save_dir, 'mention_model')
        entity_model_save_dir = os.path.join(save_dir, 'entity_model')
        _dirs = [save_dir, mention_model_save_dir, entity_model_save_dir]
        for d in _dirs:
            if not os.path.exists(d):
                os.makedirs(d)
        self.tokenizer.save_pretrained(save_dir)
        self.mention_model.save_pretrained(mention_model_save_dir)
        self.entity_model.save_pretrained(entity_model_save_dir)

    def empty_cache(self):
        self.mention_cache.empty_cache()
        self.entity_cache.empty_cache()

    def forward(self,
                mention_input_ids=None,
                mention_attention_mask=None,
                mention_token_type_ids=None,
                entity_input_ids=None,
                entity_attention_mask=None,
                entity_token_type_ids=None,
                mention_ids=None,
                entity_ids=None,    
                labels=None,
                head_mask=None,
                inputs_embeds=None,
                evaluate=False):

        args = self.args

        mention_reps = self.mention_model(
                mention_input_ids,
                mention_attention_mask,
                mention_token_type_ids
        )

        entity_reps = self.entity_model(
                entity_input_ids,
                entity_attention_mask,
                entity_token_type_ids
        )

        mention_reps = F.normalize(mention_reps, p=2, dim=-1)
        entity_reps = F.normalize(entity_reps, p=2, dim=-1)

        context_weights = torch.bmm(
                entity_reps, torch.transpose(mention_reps, 1, 2)
        )
        softmax_context_weights = F.softmax(context_weights, dim=2)
        context_mention_reps = torch.bmm(
                softmax_context_weights, mention_reps
        )

        context_mention_reps = F.normalize(context_mention_reps, p=2, dim=-1)

        scores = torch.sum(context_mention_reps * entity_reps, dim=2)

        loss = 0.0
        if not evaluate:
            labels = labels.nonzero(as_tuple=True)[1]
            loss = self.criterion(scores, labels)

        return loss, scores


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
        self.embedding_layer = nn.Linear(config.hidden_size, config.hidden_size)

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


class SequenceMultiEmbeddingModel(BertPreTrainedModel):

    def __init__(self, config):
        super(SequenceMultiEmbeddingModel, self).__init__(config)
        self.start_mention_id = config.start_mention_id
        self.end_mention_id = config.end_mention_id

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context_codes = nn.Linear(
                config.hidden_size , config.num_context_codes, bias=False
        )

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        all_hidden_outputs = outputs[0]
        attention_weights = self.context_codes(all_hidden_outputs)
        attention_weights = F.softmax(attention_weights, dim=1)
        contextualized_outputs = torch.bmm(
                torch.transpose(attention_weights, 1, 2),
                all_hidden_outputs
        )

        return contextualized_outputs


class SequenceScoringModel(BertPreTrainedModel):

    def __init__(self, config):
        super(SequenceScoringModel, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.score_layer = nn.Linear(config.hidden_size, 1)

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

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        scores = self.score_layer(pooled_output)

        return scores


class BertDssmModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context_fc = nn.Linear(config.hidden_size, 64)
        self.response_fc = nn.Linear(config.hidden_size, 64)

    def forward(self,
                context_input_ids,
                context_segment_ids,
                context_input_masks,
                responses_input_ids,
                responses_segment_ids, responses_input_masks, labels=None):
        ## only select the first response (whose lbl==1)
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)

        if isinstance(self.bert, DistilBertModel):
            context_vec = self.bert(context_input_ids, context_input_masks)[-1]  # [bs,dim]
            context_vec = context_vec[:, 0]
        else:
            context_vec = self.bert(context_input_ids, context_input_masks, context_segment_ids)[-1]    # [bs,dim]

        batch_size, res_cnt, seq_length = responses_input_ids.shape
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_segment_ids = responses_segment_ids.view(-1, seq_length)

        if isinstance(self.bert, DistilBertModel):
            responses_vec = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs,dim]
            responses_vec = responses_vec[:, 0]
        else:
            responses_vec = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[
                -1]  # [bs,dim]
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        context_vec = self.context_fc(self.dropout(context_vec))
        context_vec = F.normalize(context_vec, 2, -1)

        responses_vec = self.response_fc(self.dropout(responses_vec))
        responses_vec = F.normalize(responses_vec, 2, -1)

        if labels is not None:
            responses_vec = responses_vec.squeeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.t())  # [bs, bs]
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product * 5, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()

            return loss
        else:
            context_vec = context_vec.unsqueeze(1)
            dot_product = torch.matmul(context_vec, responses_vec.permute(0, 2, 1))  # take this as logits
            dot_product.squeeze_(1)
            cos_similarity = (dot_product + 1) / 2
            return cos_similarity


class BertPolyDssmModel(BertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context_fc = nn.Linear(config.hidden_size, 64)
        self.response_fc = nn.Linear(config.hidden_size, 64)

        self.vec_dim = 64

        self.poly_m = 16
        self.poly_code_embeddings = nn.Embedding(self.poly_m + 1, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.context_fc = nn.Linear(config.hidden_size, self.vec_dim)
        self.response_fc = nn.Linear(config.hidden_size, self.vec_dim)

    def forward(self, context_input_ids, context_segment_ids, context_input_masks,
                            responses_input_ids, responses_segment_ids, responses_input_masks, labels=None):
        ## only select the first response (whose lbl==1)
        if labels is not None:
            responses_input_ids = responses_input_ids[:, 0, :].unsqueeze(1)
            responses_segment_ids = responses_segment_ids[:, 0, :].unsqueeze(1)
            responses_input_masks = responses_input_masks[:, 0, :].unsqueeze(1)
        batch_size, res_cnt, seq_length = responses_input_ids.shape

        ## poly context encoder
        if isinstance(self.bert, DistilBertModel):
            state_vecs = self.bert(context_input_ids, context_input_masks)[-1]  # [bs, length, dim]
        else:
            state_vecs = self.bert(context_input_ids, context_input_masks, context_segment_ids)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long, device=context_input_ids.device)
        poly_code_ids += 1
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        context_vecs = dot_attention(poly_codes, state_vecs, state_vecs, context_input_masks, self.dropout)

        ## response encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)
        responses_input_masks = responses_input_masks.view(-1, seq_length)
        responses_segment_ids = responses_segment_ids.view(-1, seq_length)
        if isinstance(self.bert, DistilBertModel):
            state_vecs = self.bert(responses_input_ids, responses_input_masks)[-1]  # [bs, length, dim]
        else:
            state_vecs = self.bert(responses_input_ids, responses_input_masks, responses_segment_ids)[0]    # [bs, length, dim]
        poly_code_ids = torch.zeros(batch_size * res_cnt, 1, dtype=torch.long, device=context_input_ids.device)
        poly_codes = self.poly_code_embeddings(poly_code_ids)
        responses_vec = dot_attention(poly_codes, state_vecs, state_vecs, responses_input_masks, self.dropout)
        responses_vec = responses_vec.view(batch_size, res_cnt, -1)

        ## norm vectors
        context_vecs = self.context_fc(self.dropout(context_vecs))
        context_vecs = F.normalize(context_vecs, 2, -1)  # [bs, m, dim]
        responses_vec = self.response_fc(self.dropout(responses_vec))
        responses_vec = F.normalize(responses_vec, 2, -1)

        ## poly final context vector aggregation
        if labels is not None:
            responses_vec = responses_vec.view(1, batch_size, -1).expand(batch_size, batch_size, self.vec_dim)
        final_context_vec = dot_attention(responses_vec, context_vecs, context_vecs, None, self.dropout)
        final_context_vec = F.normalize(final_context_vec, 2, -1)  # [bs, res_cnt, dim], res_cnt==bs when training

        dot_product = torch.sum(final_context_vec * responses_vec, -1)  # [bs, res_cnt], res_cnt==bs when training
        if labels is not None:
            mask = torch.eye(context_input_ids.size(0)).to(context_input_ids.device)
            loss = F.log_softmax(dot_product * 5, dim=-1) * mask
            loss = (-loss.sum(dim=1)).mean()

            return loss
        else:
            cos_similarity = (dot_product + 1) / 2
            return cos_similarity

def dot_attention(q, k, v, v_mask=None, dropout=None):
    attention_weights = torch.matmul(q, k.transpose(-1, -2))
    if v_mask is not None:
        attention_weights *= v_mask.unsqueeze(1)
    attention_weights = F.softmax(attention_weights, -1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, v)
    return output
