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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import struct
from collections import OrderedDict, defaultdict

from IPython import embed

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from sklearn.cluster import KMeans
import faiss
import torch
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertModel, BertTokenizer,
                                  XLNetConfig, XLNetForMultipleChoice,
                                  XLNetTokenizer, RobertaConfig,
                                  RobertaForMultipleChoice, RobertaTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from dataset_utils import processors, LazyDataset, ClusterLinkingLazyTrainDataset, bytes_to_id, _read_mentions, _read_candidates, create_cluster_index
from modeling import BertForCandidateGeneration
from gp import bayesian_optimisation

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForCandidateGeneration, BertTokenizer),
}

def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def compute_scores_for_inference(args,
                                 mention_coref_model,
                                 mention_linking_model,
                                 entity_linking_model,
                                 mention_dataset,
                                 entity_dataset,
                                 evaluate=False):
    mention_coref_model.eval()
    mention_linking_model.eval()
    entity_linking_model.eval()

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    mention_sampler = SequentialSampler(mention_dataset)
    mention_dataloader = DataLoader(mention_dataset, sampler=mention_sampler, batch_size=args.eval_batch_size, num_workers=10)
    entity_sampler = SequentialSampler(entity_dataset)
    entity_dataloader = DataLoader(entity_dataset, sampler=entity_sampler, batch_size=args.eval_batch_size, num_workers=10)

    mention_ids = []
    entity_ids = []

    mention_indices = []
    entity_indices = []

    mention_coref_embeddings = []
    mention_linking_embeddings = []
    entity_linking_embeddings = []
    
    for batch in tqdm(mention_dataloader, desc='Mentions', disable=(not evaluate)):
        batch = tuple(t.to(args.device, non_blocking=True) for t in batch)

        with torch.no_grad():
            serialized_mention_id = batch[0].cpu().numpy().tolist()
            _mention_ids = [bytes_to_id(x) for x in serialized_mention_id]
            _mention_indices = batch[1].cpu().numpy().tolist()
            inputs = {'input_ids':      batch[2],
                      'attention_mask': batch[3],
                      'token_type_ids': batch[4] if args.model_type in ['bert', 'xlnet'] else None}
            outputs_coref = mention_coref_model(**inputs)
            outputs_linking = mention_linking_model(**inputs)
            
            mention_coref_embeddings.append(outputs_coref.cpu().numpy())
            mention_linking_embeddings.append(outputs_linking.cpu().numpy())
            mention_ids.extend(_mention_ids)
            mention_indices.extend(_mention_indices)

    for batch in tqdm(entity_dataloader, desc='Entities', disable=(not evaluate)):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            serialized_entity_id = batch[0].cpu().numpy().tolist()
            _entity_ids = [bytes_to_id(x) for x in serialized_entity_id]
            _entity_indices = batch[1].cpu().numpy().tolist()
            inputs = {'input_ids':      batch[2],
                      'attention_mask': batch[3],
                      'token_type_ids': batch[4] if args.model_type in ['bert', 'xlnet'] else None}
            outputs = entity_linking_model(**inputs)
            #outputs = mention_linking_model(**inputs)
            
            entity_linking_embeddings.append(outputs.cpu().numpy())
            entity_ids.extend(_entity_ids)
            entity_indices.extend(_entity_indices)

    mention_coref_embeddings = np.vstack(mention_coref_embeddings)
    mention_linking_embeddings = np.vstack(mention_linking_embeddings)
    entity_linking_embeddings = np.vstack(entity_linking_embeddings)

    mention2local_indices = {uid : i for i, uid in enumerate(mention_ids)}
    entity2local_indices = {uid : i for i, uid in enumerate(entity_ids)}

    local_indices2mention = np.asarray(mention_ids)
    local_indices2entity = np.asarray(entity_ids)

    mention2dataset_indices = {uid : i for uid, i in zip(mention_ids, mention_indices)}
    entity2dataset_indices = {uid : i for uid, i in zip(entity_ids, entity_indices)}

    if evaluate:
        return (mention2local_indices,
                entity2local_indices,
                local_indices2mention,
                local_indices2entity,
                mention_coref_embeddings,
                mention_linking_embeddings,
                entity_linking_embeddings)

    mention_mention_scores = 1 - (mention_coref_embeddings @ mention_coref_embeddings.T)
    mention_entity_scores = 1 - (mention_linking_embeddings @ entity_linking_embeddings.T)

    return (mention2local_indices,
            entity2local_indices,
            local_indices2mention,
            local_indices2entity,
            mention2dataset_indices,
            entity2dataset_indices,
            mention_mention_scores,
            mention_entity_scores)


def build_sl_pairs_train_dataset(args,
                                 mention2local_indices,
                                 entity2local_indices,
                                 local_indices2mention,
                                 local_indices2entity,
                                 mention2dataset_indices,
                                 entity2dataset_indices,
                                 mention_mention_scores,
                                 mention_entity_scores,
                                 clusters):

    mention2mention_tuples = []
    mention2entity_tuples = []
    
    for euid, muids in clusters.items():
        # create per cluster masks and indices
        in_cluster_indices = np.asarray([mention2local_indices[x] for x in muids])

        in_cluster_mask = np.zeros(mention_mention_scores.shape[0], dtype=bool)
        in_cluster_mask[in_cluster_indices] = True

        out_cluster_indices = np.where(~in_cluster_mask)[0].tolist()

        ground_truth_entity_index = entity2local_indices[euid]
        entity_mask = np.zeros(mention_entity_scores.shape[1], dtype=bool)
        entity_mask[ground_truth_entity_index] = True
        non_entity_indices = np.where(~entity_mask)[0].tolist()

        #################################
        ##### mention2mention pairs #####
        #################################
        in_cluster_adj_matrix = csr_matrix(
                np.triu(mention_mention_scores[in_cluster_mask].T[in_cluster_mask].T, k=1))
        mst = minimum_spanning_tree(in_cluster_adj_matrix).toarray()
        sym_mst = mst + mst.T

        out_cluster_scores = mention_mention_scores[in_cluster_mask].T[~in_cluster_mask].T
        out_cluster_closest = np.argmin(out_cluster_scores, axis=1).tolist()

        pos_edge_to_add_mask = (np.min(out_cluster_scores, axis=1) < sym_mst)
        pos_edge_to_add_mask |= (np.min(out_cluster_scores, axis=1) < (2.0*args.max_in_cluster_dist))
        pos_edge_to_add_mask |= (sym_mst > args.max_in_cluster_dist)
        pos_edge_to_add_mask &= (sym_mst > 0)
        pos_edge_to_add_mask = pos_edge_to_add_mask | pos_edge_to_add_mask.T

        for i, row_mask in enumerate(pos_edge_to_add_mask):
            if np.any(row_mask):
                _pos_tgt_indices = np.where(row_mask)[0]
                mention2mention_tuples.extend([
                    (1.0,
                     mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                     mention2dataset_indices[local_indices2mention[in_cluster_indices[x]]])
                        for x in _pos_tgt_indices])
                mention2mention_tuples.append(
                        (-1.0,
                         mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                         mention2dataset_indices[local_indices2mention[out_cluster_indices[out_cluster_closest[i]]]]))

        ################################
        ##### mention2entity pairs #####
        ################################
        cluster_mention_euid_scores = mention_entity_scores[in_cluster_mask].T[entity_mask].T
        min_pos_score = np.min(cluster_mention_euid_scores)
        closest_mention_to_euid = np.argmin(cluster_mention_euid_scores)

        negatives_matrix = mention_entity_scores[in_cluster_mask].T[~entity_mask].T
        per_mention_mins = np.min(negatives_matrix, axis=1)
        neg_edge_to_add_mask = negatives_matrix < (min_pos_score + args.margin)
        neg_edge_to_add_mask &= (negatives_matrix == per_mention_mins.reshape(-1, 1))

        if np.any(neg_edge_to_add_mask):
            mention2entity_tuples.append(
                    (1.0,
                     mention2dataset_indices[local_indices2mention[in_cluster_indices[closest_mention_to_euid]]],
                     entity2dataset_indices[euid]))
            #mention2entity_tuples.extend(
            #        [(1.0,
            #         mention2dataset_indices[local_indices2mention[i]],
            #         entity2dataset_indices[euid])
            #             for i in in_cluster_indices])

            ### old scoring: `-negatives_matrix[i,entity_index]/np.sum(negatives_matrix[neg_edge_to_add_mask])`

            for i, row_mask in enumerate(neg_edge_to_add_mask):
                if np.any(row_mask):
                    _neg_tgt_entity_indices = np.where(row_mask)[0]
                    mention2entity_tuples.extend([
                            (-negatives_matrix[i,entity_index]/np.sum(negatives_matrix[neg_edge_to_add_mask]),
                             mention2dataset_indices[local_indices2mention[in_cluster_indices[i]]],
                             entity2dataset_indices[local_indices2entity[non_entity_indices[entity_index]]])
                        for entity_index in _neg_tgt_entity_indices]
                    )

    mention2mention_dataset = ClusterLinkingLazyTrainDataset(
        mention2mention_tuples, args.cached_train_mention_examples_dir,
        args.cached_train_mention_examples_dir)

    mention2entity_dataset = ClusterLinkingLazyTrainDataset(
        mention2entity_tuples, args.cached_train_mention_examples_dir,
        args.cached_train_entity_examples_dir)

    return mention2mention_dataset, mention2entity_dataset


def train(args, document_ids, mention_datasets, entity_datasets,
          mention_coref_model, mention_linking_model, entity_linking_model,
          tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    assert len(document_ids) == len(entity_datasets)
    assert len(mention_datasets) == len(entity_datasets)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(pos_pairs_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(mention_datasets) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    # mention_coref_model optimizer and scheduler
    mention_coref_optimizer_grouped_parameters = [
        {'params': [p for n, p in mention_coref_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in mention_coref_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    mention_coref_optimizer = AdamW(mention_coref_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    mention_coref_scheduler = get_linear_schedule_with_warmup(mention_coref_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        mention_coref_model, mention_coref_optimizer = amp.initialize(mention_coref_model, mention_coref_optimizer, opt_level=args.fp16_opt_level)

    # mention_linking_model optimizer and scheduler
    mention_linking_optimizer_grouped_parameters = [
        {'params': [p for n, p in mention_linking_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in mention_linking_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    mention_linking_optimizer = AdamW(mention_linking_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    mention_linking_scheduler = get_linear_schedule_with_warmup(mention_linking_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        mention_linking_model, mention_linking_optimizer = amp.initialize(mention_linking_model, mention_linking_optimizer, opt_level=args.fp16_opt_level)
        
    # entity_linking_model optimizer and scheduler
    entity_linking_optimizer_grouped_parameters = [
        {'params': [p for n, p in entity_linking_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in entity_linking_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    entity_linking_optimizer = AdamW(entity_linking_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    entity_linking_scheduler = get_linear_schedule_with_warmup(entity_linking_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        entity_linking_model, entity_linking_optimizer = amp.initialize(entity_linking_model, entity_linking_optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(mention_coref_model, torch.nn.DataParallel):
        mention_coref_model = torch.nn.DataParallel(mention_coref_model)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(mention_linking_model, torch.nn.DataParallel):
        mention_linking_model = torch.nn.DataParallel(mention_linking_model)

    if args.n_gpu > 1 and not isinstance(entity_linking_model, torch.nn.DataParallel):
        entity_linking_model = torch.nn.DataParallel(entity_linking_model)

    ## Distributed training (should be after apex fp16 initialization)
    #if args.local_rank != -1:
    #    mention_model = torch.nn.parallel.DistributedDataParallel(mention_model, device_ids=[args.local_rank],
    #                                                      output_device=args.local_rank,
    #                                                      find_unused_parameters=True)
    #    entity_model = torch.nn.parallel.DistributedDataParallel(entity_model, device_ids=[args.local_rank],
    #                                                      output_device=args.local_rank,
    #                                                      find_unused_parameters=True)

    # create cluster index for training
    cluster_index = create_cluster_index(args.data_dir, 'train')

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num documents = %d", len(mention_datasets))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    tr_m2m_pairs, logging_m2m_pairs = 0, 0
    tr_m2e_pairs, logging_m2e_pairs = 0, 0
    best_val_accuracy = 0.0
    best_steps = 0
    mention_coref_model.zero_grad()
    mention_linking_model.zero_grad()
    entity_linking_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    loss_criterion = nn.CrossEntropyLoss()
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)

    for epoch in train_iterator:

        doc_level_datasets = list(zip(document_ids, mention_datasets, entity_datasets))
        random.shuffle(doc_level_datasets)

        ## NOTE: This is only for testing!
        #doc_level_datasets = [doc_level_datasets[0]] * 100

        # Iterate over all of the documents
        for document_id, mention_dataset, entity_dataset in tqdm(doc_level_datasets, desc="Documents"):
            (mention2local_indices,
             entity2local_indices,
             local_indices2mention,
             local_indices2entity,
             mention2dataset_indices,
             entity2dataset_indices,
             mention_mention_scores,
             mention_entity_scores) = compute_scores_for_inference(
                     args, mention_coref_model, mention_linking_model,
                     entity_linking_model, mention_dataset, entity_dataset)

            clusters = cluster_index[document_id]

            (mention2mention_dataset,
             mention2entity_dataset) = build_sl_pairs_train_dataset(
                 args,
                 mention2local_indices,
                 entity2local_indices,
                 local_indices2mention,
                 local_indices2entity,
                 mention2dataset_indices,
                 entity2dataset_indices,
                 mention_mention_scores,
                 mention_entity_scores,
                 clusters)

            mention2mention_sampler = SequentialSampler(mention2mention_dataset)
            mention2mention_dataloader = DataLoader(mention2mention_dataset, sampler=mention2mention_sampler, batch_size=args.train_batch_size, num_workers=10)

            mention2entity_sampler = SequentialSampler(mention2entity_dataset)
            mention2entity_dataloader = DataLoader(mention2entity_dataset, sampler=mention2entity_sampler, batch_size=args.train_batch_size, num_workers=10)

            #logger.info("\n")
            #logger.info("mention2mention pairs: {}\tmention2entity pairs: {}\n".format(
            #    len(mention2mention_dataset), len(mention2entity_dataset)))

            tr_m2m_pairs += len(mention2mention_dataset)
            tr_m2e_pairs += len(mention2entity_dataset)

            ######################################
            ### TRAIN MENTION-TO-MENTION COREF ###
            ######################################
            if len(mention2mention_dataset) > 0:
                num_mention2mention_pairs = len(mention2mention_dataset)

                for batch in iter(mention2mention_dataloader):
                    mention_coref_model.train()

                    batch = tuple(t.to(args.device) for t in batch)

                    _coeff = batch[0]
                    _batch_size = _coeff.shape[0]

                    # concat first and second mentions in pairs together
                    mention_inputs = {'input_ids':      torch.cat((batch[3], batch[8]), 0),
                                      'attention_mask': torch.cat((batch[4], batch[9]), 0),
                                      'token_type_ids': torch.cat((batch[5], batch[10]), 0) if args.model_type in ['bert', 'xlnet'] else None}

                    mention_reps = mention_coref_model(**mention_inputs)

                    scores = 1 - torch.bmm(mention_reps[:_batch_size].view(_batch_size, 1, -1), 
                                           mention_reps[_batch_size:].view(_batch_size, -1, 1)).squeeze()

                    #loss = torch.mean(_coeff * scores) # `_coeff * scores` is to be maximized
                    loss = torch.sum(_coeff * scores) / num_mention2mention_pairs

                    tr_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(mention_coref_model.parameters(), args.max_grad_norm)

                mention_coref_optimizer.step()
                mention_coref_scheduler.step()  # Update learning rate schedule
                mention_coref_model.zero_grad()

            #######################################
            ### TRAIN MENTION-TO-ENTITY LINKING ###
            #######################################
            if len(mention2entity_dataset) > 0:
                num_mention2entity_pairs = len(mention2entity_dataset)

                for batch in iter(mention2entity_dataloader):
                    mention_linking_model.train()
                    entity_linking_model.train()

                    batch = tuple(t.to(args.device) for t in batch)

                    _coeff = batch[0]
                    _batch_size = _coeff.shape[0]

                    mention_inputs = {'input_ids':      batch[3],
                                      'attention_mask': batch[4],
                                      'token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None}

                    entity_inputs = {'input_ids':      batch[8],
                                     'attention_mask': batch[9],
                                     'token_type_ids': batch[10] if args.model_type in ['bert', 'xlnet'] else None}

                    mention_reps = mention_linking_model(**mention_inputs)
                    entity_reps = entity_linking_model(**entity_inputs)


                    scores = 1 - torch.bmm(mention_reps.view(_batch_size, 1, -1), 
                                           entity_reps.view(_batch_size, -1, 1)).squeeze()

                    #inputs = {'input_ids':      torch.cat((batch[3], batch[8]), 0),
                    #          'attention_mask': torch.cat((batch[4], batch[9]), 0),
                    #          'token_type_ids': torch.cat((batch[5], batch[10]), 0) if args.model_type in ['bert', 'xlnet'] else None}

                    #reps = mention_linking_model(**inputs)

                    #scores = 1 - torch.bmm(reps[:_batch_size].view(_batch_size, 1, -1), 
                    #                       reps[_batch_size:].view(_batch_size, -1, 1)).squeeze()

                    #loss = torch.mean(_coeff * scores)
                    loss = torch.sum(_coeff * scores) / num_mention2entity_pairs

                    tr_loss += loss.item()

                    loss.backward()

                torch.nn.utils.clip_grad_norm_(mention_linking_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(entity_linking_model.parameters(), args.max_grad_norm)

                mention_linking_optimizer.step()
                entity_linking_optimizer.step()
                mention_linking_scheduler.step()  # Update learning rate schedule
                entity_linking_scheduler.step()  # Update learning rate schedule
                mention_linking_model.zero_grad()
                entity_linking_model.zero_grad()

            global_step += 1

            ## NOTE: For testing
            #_ = evaluate(args, mention_coref_model, mention_linking_model, entity_linking_model, tokenizer, split='train')

            # For tracking 
            if global_step % args.tracking_steps == 0:
                logger.info("Average loss: %s at global step: %s",
                        str((tr_loss - logging_loss)/args.tracking_steps),
                        str(global_step))
                logger.info("Average num mention2mention pairs: %s at global step: %s",
                            str((tr_m2m_pairs - logging_m2m_pairs)/args.tracking_steps),
                            str(global_step))
                logger.info("Average num mention2entity pairs: %s at global step: %s",
                        str((tr_m2e_pairs - logging_m2e_pairs)/args.tracking_steps),
                        str(global_step))
                logging_loss = tr_loss
                logging_m2m_pairs = tr_m2m_pairs
                logging_m2e_pairs = tr_m2e_pairs

            # Logging and evaluation
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # Log metrics
                if args.evaluate_during_training:
                    _ = evaluate(args,
                                 mention_coref_model,
                                 mention_linking_model,
                                 entity_linking_model,
                                 tokenizer,
                                 split='train')
                    results = evaluate(args,
                                       mention_coref_model,
                                       mention_linking_model,
                                       entity_linking_model,
                                       tokenizer,
                                       split='val')
                    for key, value in results.items():
                        tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    if results["cluster_linking_accuracy"] > best_val_accuracy:

                        best_val_accuracy = results["cluster_linking_accuracy"]
                        best_steps = global_step
                        if args.do_test:
                            results_test = evaluate(args, mention_model, entity_model, tokenizer, test=True)
                            for key, value in results_test.items():
                                tb_writer.add_scalar('test_{}'.format(key), value, global_step)
                            logger.info("test acc: %s, loss: %s, global steps: %s", str(results_test['eval_acc']), str(results_test['eval_loss']), str(global_step))

                #tb_writer.add_scalar('mention_lr', mention_scheduler.get_lr()[0], global_step)
                #tb_writer.add_scalar('entity_lr', entity_scheduler.get_lr()[0], global_step)
                #tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                #logger.info("Average loss: %s at global step: %s", str((tr_loss - logging_loss)/args.logging_steps), str(global_step))
                #logging_loss = tr_loss

            if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                mention_coref_output_dir = os.path.join(args.mention_coref_output_dir, 'checkpoint-{}'.format(global_step))
                mention_linking_output_dir = os.path.join(args.mention_linking_output_dir, 'checkpoint-{}'.format(global_step))
                entity_linking_output_dir = os.path.join(args.entity_linking_output_dir, 'checkpoint-{}'.format(global_step))

                if not os.path.exists(mention_coref_output_dir):
                    os.makedirs(mention_coref_output_dir)
                if not os.path.exists(mention_linking_output_dir):
                    os.makedirs(mention_linking_output_dir)
                if not os.path.exists(entity_linking_output_dir):
                    os.makedirs(entity_linking_output_dir)

                model_to_save = mention_coref_model.module if hasattr(mention_coref_model, 'module') else mention_coref_model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(mention_coref_output_dir)
                tokenizer.save_vocabulary(mention_coref_output_dir)
                torch.save(args, os.path.join(mention_coref_output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", mention_coref_output_dir)

                model_to_save = mention_linking_model.module if hasattr(mention_linking_model, 'module') else mention_linking_model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(mention_linking_output_dir)
                tokenizer.save_vocabulary(mention_linking_output_dir)
                torch.save(args, os.path.join(mention_linking_output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", mention_linking_output_dir)

                model_to_save = entity_linking_model.module if hasattr(entity_linking_model, 'module') else entity_linking_model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(entity_linking_output_dir)
                tokenizer.save_vocabulary(entity_linking_output_dir)
                torch.save(args, os.path.join(entity_linking_output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", entity_linking_output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, mention_coref_model, mention_linking_model, entity_linking_model, tokenizer, prefix="", split=None):
    assert split == 'train' or split == 'val' or split == 'test'

    eval_task_names = (args.task_name,)
    eval_mention_coref_outputs_dirs = (args.mention_coref_output_dir,)
    eval_mention_linking_outputs_dirs = (args.mention_linking_output_dir,)
    eval_entity_linking_outputs_dirs = (args.entity_linking_output_dir,)

    task_iter = zip(eval_task_names, eval_mention_coref_outputs_dirs, eval_mention_linking_outputs_dirs, eval_entity_linking_outputs_dirs)

    results = {}
    for eval_task, eval_mention_coref_output_dir, eval_mention_linking_output_dir, eval_entity_linking_output_dir in task_iter: 
        (document2mentions,
         document2entities,
         mention_dataset,
         entity_dataset) = load_and_cache_examples(
                args, args.task_name, tokenizer, split=split, evaluate=True)

        if not os.path.exists(eval_mention_coref_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_mention_coref_output_dir)
        if not os.path.exists(eval_mention_linking_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_mention_linking_output_dir)
        if not os.path.exists(eval_entity_linking_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_entity_linking_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(mention_coref_model, torch.nn.DataParallel):
            mention_coref_model = torch.nn.DataParallel(mention_coref_model)

        if args.n_gpu > 1 and not isinstance(mention_linking_model, torch.nn.DataParallel):
            mention_linking_model = torch.nn.DataParallel(mention_linking_model)

        if args.n_gpu > 1 and not isinstance(entity_linking_model, torch.nn.DataParallel):
            entity_linking_model = torch.nn.DataParallel(entity_linking_model)

        # create cluster index for training
        cluster_index = create_cluster_index(args.data_dir, split)

        # load the mention objects for computing accuracy
        mention_file = os.path.join(args.data_dir, 'mentions', split + '.json')
        candidate_file = os.path.join(args.data_dir, 'tfidf_candidates', split + '.json')
        mentions = _read_mentions(mention_file)
        tfidf_candidates = _read_candidates(candidate_file)
        mentions_dict = {m['mention_id']: m for m in mentions}

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num documents = %d", len(document2mentions.keys()))
        logger.info("  Num mentions = %d", len(mention_dataset))
        logger.info("  Num entities = %d", len(entity_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # Iterate over all of the documents and record data structures
        logger.info("Computing embeddings...")
        (mention2local_indices,
         entity2local_indices,
         local_indices2mention,
         local_indices2entity,
         mention_coref_embeddings,
         mention_linking_embeddings,
         entity_linking_embeddings) = compute_scores_for_inference(
                 args, mention_coref_model, mention_linking_model, entity_linking_model,
                 mention_dataset, entity_dataset, evaluate=True)

        logger.info("Done.")

        def get_all_mention2mention_scores():
            mst_edge_values = []
            for document_id in document2mentions.keys():
                doc_mentions = document2mentions[document_id]
                mention_local_indices = [mention2local_indices[m['mention_id']]
                                            for m in doc_mentions]
                doc_mention_coref_embeddings = mention_coref_embeddings[mention_local_indices]
                mention_mention_scores = 1 - (doc_mention_coref_embeddings
                                             @ doc_mention_coref_embeddings.T)
                adj_matrix = csr_matrix(np.triu(mention_mention_scores, k=1))
                mst = minimum_spanning_tree(adj_matrix).toarray()
                edge_values = mst[mst != 0.0]
                mst_edge_values.append(edge_values)
            return np.hstack(mst_edge_values)

        def cluster_linking_score(threshold):
            mention_level_hits = 0
            for document_id in document2mentions.keys():

                # grab mentions and entities for documents
                doc_mentions = document2mentions[document_id]
                doc_entities = document2entities[document_id]

                # ground truth clusters
                clusters = cluster_index[document_id]

                # slice out doc-level mention and entity reps and pair-wise scores
                mention_local_indices = [mention2local_indices[m['mention_id']]
                                            for m in doc_mentions]
                entity_local_indices = [entity2local_indices[e]
                                            for e in doc_entities]

                doc_mention_coref_embeddings = mention_coref_embeddings[mention_local_indices]
                doc_mention_linking_embeddings = mention_linking_embeddings[mention_local_indices]
                doc_entity_linking_embeddings = entity_linking_embeddings[entity_local_indices]
                
                mention_mention_scores = 1 - (doc_mention_coref_embeddings
                                             @ doc_mention_coref_embeddings.T)
                mention_entity_scores = 1 - (doc_mention_linking_embeddings
                                             @ doc_entity_linking_embeddings.T)

                # compute pruned MST
                adj_matrix = csr_matrix(np.triu(mention_mention_scores, k=1))
                mst = minimum_spanning_tree(adj_matrix).toarray()
                edge_values = mst[mst != 0.0]
                mst[mst > threshold] = 0.0
                pruned_mst = csr_matrix(mst)

                # produce clusters
                n_components, cluster_labels = connected_components(
                    csgraph=pruned_mst, directed=False, return_labels=True)

                pred_cluster_map = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    pred_cluster_map[label].append(local_indices2mention[mention_local_indices[i]])

                # compute linking decisions
                for label in range(n_components):
                    cluster_mask = (cluster_labels == label)
                    cluster_entity_scores = mention_entity_scores[cluster_mask]
                    _, entity_index = np.unravel_index(cluster_entity_scores.argmin(),
                                                       cluster_entity_scores.shape)
                    pred_cluster_entity = local_indices2entity[entity_local_indices[entity_index]]
                    for muid in pred_cluster_map[label]:
                        if mentions_dict[muid]['label_document_id'] == pred_cluster_entity:
                            mention_level_hits += 1

            return mention_level_hits / len(mentions)

        logger.info('Choosing threshold...')
        all_edge_scores = get_all_mention2mention_scores()
        kmeans = KMeans(n_clusters=100, random_state=0)
        kmeans.fit(all_edge_scores.reshape(-1, 1))
        _thresholds = kmeans.cluster_centers_.reshape(-1,).tolist()
        threshold, cluster_linking_accuracy = max(
            [(t, cluster_linking_score(t)) for t in _thresholds],
            key = lambda x : x[1]
        )
        
        logger.info('Done.')

        result = {'threshold' : threshold,
                  'vanilla_linking_accuracy': cluster_linking_score(0.0),
                  'cluster_linking_accuracy': cluster_linking_accuracy}
        results.update(result)

        output_eval_files = [os.path.join(eval_mention_coref_output_dir, split + "_eval_results.txt"),
                             os.path.join(eval_mention_linking_output_dir, split + "_eval_results.txt"),
                             os.path.join(eval_entity_linking_output_dir, split + "_eval_results.txt")]

        for output_eval_file in output_eval_files:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(str(prefix) + " | split: " + str(split)))
                writer.write("mention coref model           =%s\n" % str(args.mention_coref_model_name_or_path))
                writer.write("mention linking model           =%s\n" % str(args.mention_linking_model_name_or_path))
                writer.write("entity linking model           =%s\n" % str(args.entity_linking_model_name_or_path))
                writer.write("total batch size=%d\n" % (args.per_gpu_train_batch_size * args.gradient_accumulation_steps *
                             (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
                writer.write("train num epochs=%d\n" % args.num_train_epochs)
                writer.write("fp16            =%s\n" % args.fp16)
                writer.write("max seq length  =%d\n" % args.max_seq_length)
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, split=None, evaluate=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    assert split == 'train' or split == 'val' or split == 'test'
    assert evaluate == True or split == 'train'

    processor = processors[task]()

    cache_dir = os.path.join(args.data_dir, 'cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if split == 'train' and evaluate:
        cached_mode = 'train_eval'
    else:
        cached_mode = split

    split_desc = 'cached_{}_{}_{}'.format(
        cached_mode,
        str(args.max_seq_length),
        str(task))

    cached_mention_examples_dir = os.path.join(cache_dir, split_desc + '_mention_examples')
    cached_entity_examples_dir = os.path.join(cache_dir, split_desc + '_entity_examples')
    cached_document_indices_dir = os.path.join(cache_dir, split_desc + '_document_indices')

    if split == 'train':
        args.cached_train_mention_examples_dir = cached_mention_examples_dir
        args.cached_train_entity_examples_dir = cached_entity_examples_dir

    if (os.path.exists(cached_document_indices_dir) 
        and not args.overwrite_cache):
        document_indices_files = os.listdir(cached_document_indices_dir)
        indices_dicts = []
        for filename in document_indices_files:
            indices_dicts.append(torch.load(os.path.join(cached_document_indices_dir, filename)))
    else:
        if split == 'train':
            domains = args.train_domains
        elif split == 'val':
            domains = args.val_domains
        else:
            domains = args.test_domains

        if not os.path.exists(cached_mention_examples_dir):
            os.makedirs(cached_mention_examples_dir)
        if not os.path.exists(cached_entity_examples_dir):
            os.makedirs(cached_entity_examples_dir)
        if not os.path.exists(cached_document_indices_dir):
            os.makedirs(cached_document_indices_dir)

        indices_dicts = processor.get_document_datasets(
            args.data_dir,
            split,
            domains,
            cached_mention_examples_dir,
            cached_entity_examples_dir,
            cached_document_indices_dir,
            args.max_seq_length,
            tokenizer,
            evaluate=evaluate,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    if evaluate:
        _indices_dict = indices_dicts[0]
        document2mentions = _indices_dict['document2mentions']
        document2entities = _indices_dict['document2entities']
        mention_dataset = LazyDataset(cached_mention_examples_dir,
                                      _indices_dict['mention_indices'])
        entity_dataset = LazyDataset(cached_entity_examples_dir,
                                     _indices_dict['entity_indices'])
        return (document2mentions,
                document2entities,
                mention_dataset,
                entity_dataset)

    document_ids = []
    mention_datasets = []
    entity_datasets = []
    for _indices_dict in indices_dicts:
        document_ids.append(_indices_dict['document_id'])
        mention_datasets.append(LazyDataset(cached_mention_examples_dir,
                                            _indices_dict['mention_indices']))
        entity_datasets.append(LazyDataset(cached_entity_examples_dir,
                                           _indices_dict['entity_indices']))

    return document_ids, mention_datasets, entity_datasets


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--mention_coref_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained mention model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--mention_linking_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained mention model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--entity_linking_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained entity model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--mention_coref_output_dir", default=None, type=str, required=True,
                        help="The output directory where the mention coref model predictions and checkpoints will be written.")
    parser.add_argument("--mention_linking_output_dir", default=None, type=str, required=True,
                        help="The output directory where the mention linking model predictions and checkpoints will be written.")
    parser.add_argument("--entity_linking_output_dir", default=None, type=str, required=True,
                        help="The output directory where the entity model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the val set.")
    parser.add_argument("--do_test", action='store_true', help='Whether to run test on the test set')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_domains", default=None, nargs='+',
                        help="list of train domains of documents")
    parser.add_argument("--val_domains", default=None, nargs='+',
                        help="list of val domains of documents")
    parser.add_argument("--test_domains", default=None, nargs='+',
                        help="list of test domains of documents")
    parser.add_argument("--num_candidates", default=None, type=int,
                        help="number of candidates per mention")
    parser.add_argument("--num_candidates_per_example", default=None, type=int,
                        help="number of candidates per example for a particular example")
    parser.add_argument("--margin", default=0.0, type=float,
                        help="For linking loss")
    parser.add_argument("--max_in_cluster_dist", default=0.0, type=float,
                        help="For clustering loss")
    parser.add_argument("--sequence_embedding_size", default=None, type=int,
                        help="size of embedding to output from BERT")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--tracking_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.mention_coref_output_dir) and os.listdir(args.mention_coref_output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.mention_coref_output_dir))

    if os.path.exists(args.mention_linking_output_dir) and os.listdir(args.mention_linking_output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.mention_linking_output_dir))

    if os.path.exists(args.entity_linking_output_dir) and os.listdir(args.entity_linking_output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.entity_linking_output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s  -  %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.mention_coref_model_name_or_path,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.seq_embed_size = args.sequence_embedding_size
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.mention_coref_model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    mention_coref_model = model_class.from_pretrained(args.mention_coref_model_name_or_path,
                                                from_tf=bool('.ckpt' in args.mention_coref_model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    mention_linking_model = model_class.from_pretrained(args.mention_linking_model_name_or_path,
                                                from_tf=bool('.ckpt' in args.mention_linking_model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    entity_linking_model = model_class.from_pretrained(args.entity_linking_model_name_or_path,
                                                from_tf=bool('.ckpt' in args.entity_linking_model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    mention_coref_model.to(args.device)
    mention_linking_model.to(args.device)
    entity_linking_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        document_ids, mention_datasets, entity_datasets = load_and_cache_examples(
                args, args.task_name, tokenizer, split='train', evaluate=False)
        global_step, tr_loss, best_steps = train(
                args, document_ids, mention_datasets, entity_datasets,
                mention_coref_model, mention_linking_model,
                entity_linking_model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.mention_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.mention_output_dir)
        if not os.path.exists(args.entity_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.entity_output_dir)

        logger.info("Saving mention_model checkpoint to %s", args.mention_output_dir)
        logger.info("Saving entity_model checkpoint to %s", args.entity_output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = mention_model.module if hasattr(mention_model, 'module') else mention_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.mention_output_dir)
        tokenizer.save_pretrained(args.mention_output_dir)
        model_to_save = entity_model.module if hasattr(entity_model, 'module') else entity_model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.entity_output_dir)
        tokenizer.save_pretrained(args.entity_output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.mention_output_dir, 'training_args.bin'))
        torch.save(args, os.path.join(args.entity_output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        mention_model = model_class.from_pretrained(args.mention_output_dir)
        mention_model.to(args.device)
        entity_model = model_class.from_pretrained(args.entity_output_dir)
        entity_model.to(args.device)
        tokenizer = tokenizer_class.from_pretrained(args.mention_output_dir)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.mention_output_dir = args.mention_model_name_or_path
            args.entity_output_dir = args.entity_model_name_or_path
        checkpoints = [(args.mention_output_dir, args.entity_output_dir)]
        if args.eval_all_checkpoints:
            mention_checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.mention_output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            entity_checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.entity_output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            checkpoints = zip(mention_checkpoints, entity_checkpoints)
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for mention_checkpoint, entity_checkpoint in checkpoints:
            global_step = mention_checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = mention_checkpoint.split('/')[-1] if mention_checkpoint.find('checkpoint') != -1 else ""

            mention_model = model_class.from_pretrained(mention_checkpoint)
            mention_model.to(args.device)
            entity_model = model_class.from_pretrained(entity_checkpoint)
            entity_model.to(args.device)
            result = evaluate(args, mention_model, entity_model, tokenizer, prefix=prefix, split='val')
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.mention_output_dir = args.mention_model_name_or_path
            args.entity_output_dir = args.entity_model_name_or_path
        checkpoints = [(args.mention_output_dir, args.entity_output_dir)]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for mention_checkpoint, entity_checkpoint in checkpoints:
            global_step = mention_checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = mention_checkpoint.split('/')[-1] if mention_checkpoint.find('checkpoint') != -1 else ""

            mention_model = model_class.from_pretrained(mention_checkpoint)
            mention_model.to(args.device)
            entity_model = model_class.from_pretrained(entity_checkpoint)
            entity_model.to(args.device)
            result = evaluate(args, mention_model, entity_model, tokenizer, prefix=prefix, split='test')
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
