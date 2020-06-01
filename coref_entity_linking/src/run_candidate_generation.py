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
from collections import OrderedDict

from IPython import embed

import numpy as np
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

from dataset_utils import processors, LazyDataset, bytes_to_id, _read_mentions, _read_candidates
from modeling import BertForCandidateGeneration

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


def train(args, pos_pairs_dataset, neg_entities_dataset, mention_model, entity_model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    pos_pairs_sampler = RandomSampler(pos_pairs_dataset) if args.local_rank == -1 else DistributedSampler(pos_pairs_dataset)
    pos_pairs_dataloader = DataLoader(pos_pairs_dataset, sampler=pos_pairs_sampler, batch_size=2*args.train_batch_size, num_workers=1)

    neg_entities_sampler = RandomSampler(neg_entities_dataset) if args.local_rank == -1 else DistributedSampler(neg_entities_dataset)
    neg_entities_dataloader = DataLoader(neg_entities_dataset, sampler=neg_entities_sampler, batch_size=3*args.train_batch_size, num_workers=1)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(pos_pairs_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(pos_pairs_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    # mention_model optimizer and scheduler
    mention_optimizer_grouped_parameters = [
        {'params': [p for n, p in mention_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in mention_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    mention_optimizer = AdamW(mention_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    mention_scheduler = get_linear_schedule_with_warmup(mention_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        mention_model, mention_optimizer = amp.initialize(mention_model, mention_optimizer, opt_level=args.fp16_opt_level)
        
    # entity_model optimizer and scheduler
    entity_optimizer_grouped_parameters = [
        {'params': [p for n, p in entity_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in entity_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    entity_optimizer = AdamW(entity_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    entity_scheduler = get_linear_schedule_with_warmup(entity_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        entity_model, entity_optimizer = amp.initialize(entity_model, entity_optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1 and not isinstance(mention_model, torch.nn.DataParallel):
        mention_model = torch.nn.DataParallel(mention_model)

    if args.n_gpu > 1 and not isinstance(entity_model, torch.nn.DataParallel):
        entity_model = torch.nn.DataParallel(entity_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        mention_model = torch.nn.parallel.DistributedDataParallel(mention_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
        entity_model = torch.nn.parallel.DistributedDataParallel(entity_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num mentions = %d", len(pos_pairs_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_val_accuracy = 0.0
    best_steps = 0
    mention_model.zero_grad()
    entity_model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    
    loss_criterion = nn.CrossEntropyLoss()

    for epoch in train_iterator:
        epoch_iterator = tqdm(pos_pairs_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        neg_entities_iterator = iter(neg_entities_dataloader)
        for step, batch in enumerate(epoch_iterator):
            mention_model.train()
            entity_model.train()

            neg_batch = next(neg_entities_iterator, None)
            assert neg_batch is not None

            batch = tuple(t.to(args.device) for t in batch)
            neg_batch = tuple(t.to(args.device) for t in neg_batch)

            mention_inputs = {'input_ids':      batch[0],
                              'attention_mask': batch[1],
                              'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None}
            entity_inputs = {'input_ids':      batch[3],
                             'attention_mask': batch[4],
                             'token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None}
            #if epoch < 2:
            #    entity_inputs = {'input_ids':      batch[3],
            #                     'attention_mask': batch[4],
            #                     'token_type_ids': batch[5] if args.model_type in ['bert', 'xlnet'] else None}
            #else:
            #    entity_inputs = {'input_ids':      torch.cat((batch[3], neg_batch[1]), 0),
            #                     'attention_mask': torch.cat((batch[4], neg_batch[2]), 0),
            #                     'token_type_ids': torch.cat((batch[5], neg_batch[3]), 0) if args.model_type in ['bert', 'xlnet'] else None}

            # Gather all of the reps from the model
            mention_reps = mention_model(**mention_inputs)
            entity_reps = entity_model(**entity_inputs)

            # Compute the loss
            scores = torch.matmul(mention_reps, torch.t(entity_reps))
            target = torch.arange(scores.shape[0], device=scores.device)
            loss = loss_criterion(scores, target)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mention_model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(entity_model.parameters(), args.max_grad_norm)

            ## TEST: evaluation
            #_ = evaluate(args, mention_model, entity_model, tokenizer, split='train')

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                mention_optimizer.step()
                entity_optimizer.step()
                mention_scheduler.step()  # Update learning rate schedule
                entity_scheduler.step()  # Update learning rate schedule
                mention_model.zero_grad()
                entity_model.zero_grad()
                global_step += 1

                print_steps = 100
                if global_step % print_steps == 0:
                    logger.info("\n\nLoss: %s at global step: %s\n\n", str(loss.item()), str(global_step))
                    #if global_step > 5 * print_steps:
                    #    embed()
                    #    exit()

                #if global_step % 2 == 0:
                #    epoch_iterator.close()
                #    break

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        _ = evaluate(args, mention_model, entity_model, tokenizer, split='train')
                        results = evaluate(args, mention_model, entity_model, tokenizer, split='val')
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                        if results["Recall@1"] > best_val_accuracy:

                            best_val_accuracy = results["Recall@1"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, mention_model, entity_model, tokenizer, test=True)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar('test_{}'.format(key), value, global_step)
                                logger.info("test acc: %s, loss: %s, global steps: %s", str(results_test['eval_acc']), str(results_test['eval_loss']), str(global_step))

                    tb_writer.add_scalar('mention_lr', mention_scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('entity_lr', entity_scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info("Average loss: %s at global step: %s", str((tr_loss - logging_loss)/args.logging_steps), str(global_step))
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    mention_output_dir = os.path.join(args.mention_output_dir, 'checkpoint-{}'.format(global_step))
                    entity_output_dir = os.path.join(args.entity_output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(mention_output_dir):
                        os.makedirs(mention_output_dir)
                    if not os.path.exists(entity_output_dir):
                        os.makedirs(entity_output_dir)

                    model_to_save = mention_model.module if hasattr(mention_model, 'module') else mention_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(mention_output_dir)
                    tokenizer.save_vocabulary(mention_output_dir)
                    torch.save(args, os.path.join(mention_output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", mention_output_dir)

                    model_to_save = entity_model.module if hasattr(entity_model, 'module') else entity_model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(entity_output_dir)
                    tokenizer.save_vocabulary(entity_output_dir)
                    torch.save(args, os.path.join(entity_output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", entity_output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                neg_entities_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, mention_model, entity_model, tokenizer, prefix="", split=None):
    assert split == 'train' or split == 'val' or split == 'test'

    eval_task_names = (args.task_name,)
    eval_mention_outputs_dirs = (args.mention_output_dir,)
    eval_entity_outputs_dirs = (args.entity_output_dir,)

    task_iter = zip(eval_task_names, eval_mention_outputs_dirs, eval_entity_outputs_dirs)

    results = OrderedDict()
    for eval_task, eval_mention_output_dir, eval_entity_output_dir in task_iter: 
        mention_dataset, entity_dataset = load_and_cache_examples(
                args, eval_task, tokenizer, split=split, evaluate=True)

        if not os.path.exists(eval_mention_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_mention_output_dir)
        if not os.path.exists(eval_entity_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_entity_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        mention_sampler = SequentialSampler(mention_dataset)
        mention_dataloader = DataLoader(mention_dataset, sampler=mention_sampler, batch_size=args.eval_batch_size, num_workers=10)
        entity_sampler = SequentialSampler(entity_dataset)
        entity_dataloader = DataLoader(entity_dataset, sampler=entity_sampler, batch_size=args.eval_batch_size, num_workers=10)

        # multi-gpu evaluate
        if args.n_gpu > 1 and not isinstance(mention_model, torch.nn.DataParallel):
            mention_model = torch.nn.DataParallel(mention_model)

        if args.n_gpu > 1 and not isinstance(entity_model, torch.nn.DataParallel):
            entity_model = torch.nn.DataParallel(entity_model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(mention_dataset) + len(entity_dataset))
        logger.info("  Num mentions = %d", len(mention_dataset))
        logger.info("  Num entities = %d", len(entity_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        # Load the mention objects for computing recall
        mention_file = os.path.join(args.data_dir, 'mentions', split + '.json')
        candidate_file = os.path.join(args.data_dir, 'tfidf_candidates', split + '.json')
        mentions = _read_mentions(mention_file)
        tfidf_candidates = _read_candidates(candidate_file)
        mentions_dict = {m['mention_id']: m for m in mentions}

        mention_ids = []
        entity_ids = []
        mention_embeddings = []
        entity_embeddings = []

        for batch in tqdm(mention_dataloader, desc="Getting mention embeddings"):
            mention_model.eval()
            entity_model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                serialized_mention_id = batch[0].cpu().numpy().tolist()
                _mention_ids = [bytes_to_id(x) for x in serialized_mention_id]

                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None}
                outputs = mention_model(**inputs)
                
                mention_embeddings.append(outputs.cpu().numpy())
                mention_ids.extend(_mention_ids)

        for batch in tqdm(entity_dataloader, desc="Getting entity embeddings"):
            mention_model.eval()
            entity_model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                serialized_entity_id = batch[0].cpu().numpy().tolist()
                _entity_ids = [bytes_to_id(x) for x in serialized_entity_id]

                inputs = {'input_ids':      batch[1],
                          'attention_mask': batch[2],
                          'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None}
                outputs = entity_model(**inputs)
                
                entity_embeddings.append(outputs.cpu().numpy())
                entity_ids.extend(_entity_ids)

        #Q = np.vstack(mention_embeddings)
        #X = np.vstack(entity_embeddings)

        #logger.info('\n\nCreating k-NN index...')
        #index = faiss.IndexFlatL2(X.shape[1])
        #index.add(X)

        #logger.info('Querying the index...')
        #D, I = index.search(Q, args.k)

        #logger.info('Computing candidate generation recall...')
        #recall_at_k = {}
        #ks = [1 << i for i in range(len(bin(args.k)) - 3)] + [args.k] 
        #for _k in ks:
        #    hits = 0
        #    total = 0

        #    for mention_index, entity_indices in enumerate(I[:, :_k]):
        #        _mention_id = mention_ids[mention_index]
        #        _entity_ids = [entity_ids[x] for x in entity_indices]
        #        if mentions_dict[_mention_id]['label_document_id'] in _entity_ids:
        #            hits += 1
        #        total += 1
        #    recall_at_k[_k] = hits / total

        mention_embeddings = np.vstack(mention_embeddings)
        entity_embeddings = np.vstack(entity_embeddings)

        mention_index = {muid: i for i, muid in enumerate(mention_ids)}
        entity_index = {euid: i for i, euid in enumerate(entity_ids)}

        recall_at_k = {}
        ks = [1 << i for i in range(len(bin(args.k)) - 3)] + [args.k] 
        for _k in ks:
            hits = 0
            total = 0

            for mention_index, mention_embedding in enumerate(mention_embeddings):
                _mention_id = mention_ids[mention_index]
                if _mention_id in tfidf_candidates.keys():
                    _entity_indices = [entity_index[euid] for euid in tfidf_candidates[_mention_id]]
                    _entity_embeddings = entity_embeddings[_entity_indices]
                    _scores = _entity_embeddings @ mention_embedding
                    if _k >= _scores.size:
                        _subset_indices = range(_scores.size)
                    else:
                        _subset_indices = np.argpartition(1-_scores, _k)[:_k]
                    try:
                        _entity_ids = [tfidf_candidates[_mention_id][i] for i in _subset_indices]
                    except Exception:
                        embed()
                        exit()
                    if mentions_dict[_mention_id]['label_document_id'] in _entity_ids:
                        hits += 1
                total += 1
            recall_at_k[_k] = hits / total

        result = OrderedDict({'Recall@' + str(k) : v for k, v in recall_at_k.items()})
        results.update(result)

        output_eval_files = [os.path.join(eval_mention_output_dir, split + "_eval_results.txt"),
                             os.path.join(eval_entity_output_dir, split + "_eval_results.txt")]

        for output_eval_file in output_eval_files:
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(str(prefix) + " | split: " + str(split)))
                writer.write("mention model           =%s\n" % str(args.mention_model_name_or_path))
                writer.write("entity model           =%s\n" % str(args.entity_model_name_or_path))
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

    split_desc = 'cached_{}_{}_{}_{}'.format(
        cached_mode,
        list(filter(None, args.mention_model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task))

    if not evaluate:
        cached_pos_pairs_indices_file = os.path.join(cache_dir, split_desc + '.pos_pairs.indices.pt')
        cached_pos_pairs_examples_dir = os.path.join(cache_dir, split_desc + 'pos_pairs_examples')
        cached_neg_entities_indices_file = os.path.join(cache_dir, split_desc + '.neg_entities.indices.pt')
        cached_neg_entities_examples_dir = os.path.join(cache_dir, split_desc + 'neg_entities_examples')

        if not os.path.exists(cached_pos_pairs_examples_dir):
            os.makedirs(cached_pos_pairs_examples_dir)
        if not os.path.exists(cached_neg_entities_examples_dir):
            os.makedirs(cached_neg_entities_examples_dir)

        # Load data features from cache or dataset file
        if (os.path.exists(cached_pos_pairs_indices_file) 
            and os.path.exists(cached_neg_entities_indices_file)
            and not args.overwrite_cache):
            logger.info("Loading features from cached file %s", cached_pos_pairs_indices_file)
            pos_pairs_indices = torch.load(cached_pos_pairs_indices_file)
            logger.info("Loading features from cached file %s", cached_neg_entities_indices_file)
            neg_entities_indices = torch.load(cached_neg_entities_indices_file)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            pos_pairs_indices, neg_entities_indices = processor.get_train_features(
                args.data_dir,
                split,
                args.train_domains,
                cached_pos_pairs_examples_dir,
                cached_neg_entities_examples_dir,
                args.max_seq_length,
                tokenizer,
                pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0
            )

            if args.local_rank in [-1, 0]:
                logger.info("Saving indices into cached file %s", cached_pos_pairs_indices_file)
                torch.save(pos_pairs_indices, cached_pos_pairs_indices_file)
                logger.info("Saving indices into cached file %s", cached_neg_entities_indices_file)
                torch.save(neg_entities_indices, cached_neg_entities_indices_file)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        pos_pairs_dataset = LazyDataset(cached_pos_pairs_examples_dir, pos_pairs_indices)
        neg_entities_dataset = LazyDataset(cached_neg_entities_examples_dir, neg_entities_indices)
        return pos_pairs_dataset, neg_entities_dataset

    else:
        cached_mention_indices_file = os.path.join(cache_dir, split_desc + '.mention.indices.pt')
        cached_mention_examples_dir = os.path.join(cache_dir, split_desc + '_mention_examples')
        cached_entity_indices_file = os.path.join(cache_dir, split_desc + '.entity.indices.pt')
        cached_entity_examples_dir = os.path.join(cache_dir, split_desc + '_entity_examples')

        if not os.path.exists(cached_mention_examples_dir):
            os.makedirs(cached_mention_examples_dir)

        if not os.path.exists(cached_entity_examples_dir):
            os.makedirs(cached_entity_examples_dir)

        if split == 'train':
            domains = args.train_domains
        elif split == 'val':
            domains = args.val_domains
        elif split == 'test':
            domains = args.test_domains
        else:
            raise ValueError("Invalid split")

        if (os.path.exists(cached_mention_indices_file)
                and os.path.exists(cached_entity_indices_file)
                and not args.overwrite_cache):
            logger.info("Loading mentions from cached file %s", cached_mention_indices_file)
            mention_indices = torch.load(cached_mention_indices_file)
            logger.info("Loading entities from cached file %s", cached_entity_indices_file)
            entity_indices = torch.load(cached_entity_indices_file)
        else:
            mention_indices, entity_indices = processor.get_eval_features(
                args.data_dir,
                split,
                domains,
                cached_mention_examples_dir,
                cached_entity_examples_dir,
                args.max_seq_length,
                tokenizer,
                pad_on_left=False,
                pad_token_segment_id=0
            )

            if args.local_rank in [-1, 0]:
                logger.info("Saving mention indices into cached file %s", cached_mention_indices_file)
                torch.save(mention_indices, cached_mention_indices_file)
                logger.info("Saving entity indices into cached file %s", cached_entity_indices_file)
                torch.save(entity_indices, cached_entity_indices_file)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

        mention_dataset = LazyDataset(cached_mention_examples_dir, mention_indices)
        entity_dataset = LazyDataset(cached_entity_examples_dir, entity_indices)

        return mention_dataset, entity_dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--mention_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained mention model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--entity_model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained entity model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--mention_output_dir", default=None, type=str, required=True,
                        help="The output directory where the mention model predictions and checkpoints will be written.")
    parser.add_argument("--entity_output_dir", default=None, type=str, required=True,
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
    parser.add_argument("--k", default=64, type=int,
                        help="for candidate generation recall")
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

    if os.path.exists(args.mention_output_dir) and os.listdir(args.mention_output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.mention_output_dir))
    if os.path.exists(args.entity_output_dir) and os.listdir(args.entity_output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.entity_output_dir))

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
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
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
    config = config_class.from_pretrained(args.config_name if args.config_name else args.mention_model_name_or_path,
                                          finetuning_task=args.task_name,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.seq_embed_size = args.sequence_embedding_size
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.mention_model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    mention_model = model_class.from_pretrained(args.mention_model_name_or_path,
                                                from_tf=bool('.ckpt' in args.mention_model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    entity_model = model_class.from_pretrained(args.entity_model_name_or_path,
                                                from_tf=bool('.ckpt' in args.entity_model_name_or_path),
                                                config=config,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    mention_model.to(args.device)
    entity_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        pos_pairs_dataset, neg_entities_dataset = load_and_cache_examples(
                args, args.task_name, tokenizer, split='train', evaluate=False)
        global_step, tr_loss, best_steps = train(
                args, pos_pairs_dataset, neg_entities_dataset,
                mention_model, entity_model, tokenizer)
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
