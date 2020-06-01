#!/bin/bash

# README:
# - always call from the root directory of this repo

set -euxo pipefail

#repo_root_dir='coref_entity_linking'
#parent_pwd=$(basename `readlink -ve "/proc/$PPID/cwd"`)
#
#if [ $parent_pwd != $repo_root_dir ]; then
#     echo "Script must be run from the root directory of the repo"
#     exit 1;
#fi

#mkdir -p exp 
#

BASE_DIR="/mnt/nfs/scratch1/rangell/coref_entity_linking"
DATASET="zeshel"

train_domains=("american_football" "doctor_who" "fallout" "final_fantasy" "military" "pro_wrestling" "starwars" "world_of_warcraft")
val_domains=("coronation_street" "elder_scrolls" "ice_hockey" "muppets")
test_domains=("forgotten_realms" "lego" "star_trek" "yugioh")

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/run.py \
        --data_dir data/${DATASET}/ \
        --model_type bert \
        --trained_model_dir ${BASE_DIR}/experiments/${DATASET}/xdoc_cluster_linking/checkpoint-10200/ \
        --task_name xdoc_cluster_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/xdoc_cluster_linking/checkpoint-10200/ \
        --log_dir ${BASE_DIR}/logs/ \
        --do_val \
        --max_seq_length 128 \
        --pooling_strategy 'pool_highlighted_outputs' \
        --per_gpu_infer_batch 128 \
        --train_domains ${train_domains[@]} \
        --train_mention_entity_scores ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-73914/mention_entity_scores.train.pkl \
        --val_domains ${val_domains[@]} \
        --val_mention_entity_scores ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-73914/mention_entity_scores.val.pkl \
        --test_domains ${test_domains[@]} \
        --test_mention_entity_scores ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-73914/mention_entity_scores.test.pkl \
