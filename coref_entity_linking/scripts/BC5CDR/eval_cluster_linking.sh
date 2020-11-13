#!/bin/bash

# README:
# - always call from the root directory of this repo

set -eux

python -m torch.distributed.launch \
    --nproc_per_node 4 \
    src/main.py \
        --data_dir '/local/coref_entity_linking/data/BC5CDR/' \
        --model_type 'bert' \
        --trained_model_dir '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/experiments/BC5CDR/cluster_linking/exp_0/checkpoint-398/' \
        --task_name 'cluster_linking' \
        --output_dir '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/experiments/BC5CDR/cluster_linking/exp_0/checkpoint-398/' \
        --log_dir '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/logs/' \
        --do_val \
        --max_seq_length 128 \
        --seq_embed_dim 128 \
        --embed_pooling_strategy 'pool_highlighted_outputs' \
        --concat_pooling_strategy 'pool_highlighted_outputs' \
        --clustering_domain 'within_doc' \
        --available_entities 'candidates_only' \
        --k 16 \
        --per_gpu_infer_batch 256 \
        --logging_steps 25 \
        --train_domains 'train' 'entity_documents' \
        --val_domains 'val' 'entity_documents'
