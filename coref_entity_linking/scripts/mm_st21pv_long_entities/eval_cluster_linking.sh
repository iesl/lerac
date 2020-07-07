#!/bin/bash

# README:
# - always call from the root directory of this repo

set -eux

#repo_root_dir='coref_entity_linking'
#parent_pwd=$(basename `readlink -ve "/proc/$PPID/cwd"`)
#
#if [ $parent_pwd != $repo_root_dir ]; then
#     echo "Script must be run from the root directory of the repo"
#     exit 1;
#fi

#mkdir -p exp 
#

#BASE_DIR="/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking"
BASE_DIR="/home/meta-powerusers/lerac/coref_entity_linking"
EXTERNAL_BASE_DIR="/home/ds-share/data2/users/rangell/lerac/coref_entity_linking"
DATASET="mm_st21pv_long_entities"

train_domains=( "train" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )
val_domains=( "val" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )
test_domains=( "test" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/main.py \
        --data_dir data/${DATASET}/ \
        --model_type bert \
        --trained_model_dir ${EXTERNAL_BASE_DIR}/experiments/${DATASET}/cluster_linking/exp4/checkpoint-7442/ \
        --task_name cluster_linking \
        --output_dir ${EXTERNAL_BASE_DIR}/experiments/${DATASET}/cluster_linking/exp4/checkpoint-7442/ \
        --log_dir ${EXTERNAL_BASE_DIR}/logs/ \
        --do_val \
        --max_seq_length 128 \
        --seq_embed_dim 128 \
        --pooling_strategy 'pool_highlighted_outputs' \
        --clustering_domain 'within_doc' \
        --available_entities 'candidates_only' \
        --k 32 \
        --per_gpu_infer_batch 256 \
        --evaluate_during_training \
        --logging_steps 25 \
        --train_domains ${train_domains[@]} \
        --val_domains ${val_domains[@]} \
        --test_domains ${test_domains[@]} \
