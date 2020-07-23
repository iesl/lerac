# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
import pickle

from IPython import embed

import numpy as np
import faiss
import torch
import torch.distributed as dist
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

from trainer.trainer import (MentionClusteringTrainer,
                             VanillaLinkingTrainer,
                             XDocClusterLinkingTrainer)
from trainer.cluster_linking_trainer import ClusterLinkingTrainer
from utils.misc import initialize_exp

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="")
    parser.add_argument("--trained_model_dir", default=None, type=str,
                        help="")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default=None, type=str, required=True,
                        help="The global log directory for safety")

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
    parser.add_argument("--do_train_eval", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_val", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_test", action='store_true', help='Whether to run test on the test set')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--dump_coref_candidate_sets", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_domains", default=None, nargs='+',
                        help="list of train domains of documents")
    parser.add_argument("--val_domains", default=None, nargs='+',
                        help="list of val domains of documents")
    parser.add_argument("--test_domains", default=None, nargs='+',
                        help="list of test domains of documents")
    parser.add_argument("--train_mention_entity_scores", default=None,
                        help="path to serialized mention-entity scores file")
    parser.add_argument("--val_mention_entity_scores", default=None,
                        help="path to serialized mention-entity scores file")
    parser.add_argument("--test_mention_entity_scores", default=None,
                        help="path to serialized mention-entity scores file")
    parser.add_argument("--sample_size", default=None, type=int,
                        help="size of sets to sample for ranking loss")
    parser.add_argument("--margin", default=None, type=float,
                        help="margin for max-margin loss")
    parser.add_argument("--alpha", default=None, type=float,
                        help="alpha for exponential linkage")
    parser.add_argument("--k", default=32, type=int,
                        help="k for kNN for eval of coref candidates")
    parser.add_argument("--max_in_cluster_dist", default=0.0, type=float,
                        help="For clustering loss")
    parser.add_argument("--seq_embed_dim",
                        default=128,
                        type=int,
                        help="output dimension of embedding model")
    parser.add_argument("--pooling_strategy",
                        default='pool_all_outputs',
                        choices = ['pool_all_outputs', 'pool_highlighted_outputs'],
                        type=str,
                        help="how to pool the output layer representations.")
    parser.add_argument("--num_context_codes", default=4, type=int,
                        help="number of context codes for poly-encoder")
    parser.add_argument("--num_candidates", default=64, type=int,
                        help="number of candidates per mention for training concatenation model.")
    parser.add_argument("--num_candidates_per_example", default=16, type=int,
                        help="number of candidates per example for concatenation model.")
    parser.add_argument("--clustering_domain",
                        default='within_doc',
                        choices = ['within_doc', 'cross_doc'],
                        type=str,
                        help="do within document clustering"
                             "or cross-document clustering")
    parser.add_argument("--available_entities",
                        default='candidates_only',
                        choices = ['candidates_only',
                                   'knn_candidates',
                                   'open_domain'],
                        type=str,
                        help="what to consider when choosing negative entities")
    parser.add_argument("--pair_gen_method",
                        default='all_pairs',
                        choices = ['all_pairs', 'mst', 'explink'],
                        type=str,
                        help="method for generating pairs")
    parser.add_argument("--training_method",
                        default='triplet',
                        choices = ['triplet', 'sigmoid', 'softmax', 'accum_max_margin'],
                        type=str,
                        help="method of training on pairs")
    parser.add_argument("--training_edges_considered",
                        default='all',
                        choices = ['all', 'm-e', 'm-m'],
                        type=str,
                        help="which types of edges to consider when training")
    parser.add_argument("--eval_coref_threshold", default=None, type=float,
                        help="For clustering loss")

    parser.add_argument("--num_clusters_per_macro_batch", default=8, type=int,
                        help="num clusters to consider in outer-loop batch")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_infer_batch_size", default=8, type=int,
                        help="batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--knn_refresh_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--num_dataloader_workers', type=int, default=0,
                        help='Number of workers dataloader should use.')

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

    parser.add_argument('--disable_logging', action='store_true',
                        help="Disable tqdm logging")

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    # set logger filename
    if args.do_train:
        logger_filename = "train.log"
    elif args.do_train_eval:
        logger_filename = "train_eval.log"
    elif args.do_val:
        logger_filename = "val.log"
    elif args.do_test:
        logger_filename = "test.log"

    # initialize experiment, including setting up logger
    initialize_exp(args, logger_filename)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # construct trainer object give task
    if args.task_name == 'mention_clustering':
        trainer = MentionClusteringTrainer(args)
    elif args.task_name in ['vanilla_linking', 'poly_linking']:
        trainer = VanillaLinkingTrainer(args)
    elif args.task_name == 'xdoc_cluster_linking':
        trainer = XDocClusterLinkingTrainer(args)
    elif args.task_name == 'cluster_linking':
        trainer = ClusterLinkingTrainer(args)
    else:
        raise ValueError("Invalid task name: {}".format(args.task_name))
    logger.info('Successfully created trainer object')

    if args.do_train:
        trainer.train()
    if args.do_train_eval:
        trainer.evaluate(split='train')
    if args.do_val:
        trainer.evaluate(split='val')
    if args.do_test:
        trainer.evaluate(split='test')


if __name__ == "__main__":
    main()
