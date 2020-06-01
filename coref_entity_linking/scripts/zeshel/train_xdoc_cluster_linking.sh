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
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --task_name xdoc_cluster_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/xdoc_cluster_linking/ \
        --log_dir ${BASE_DIR}/logs/ \
        --do_train \
        --max_seq_length 128 \
        --pooling_strategy 'pool_highlighted_outputs' \
        --max_in_cluster_dist 0.1 \
        --margin 0.8 \
        --warmup_steps 0 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --num_clusters_per_macro_batch 16 \
        --per_gpu_train_batch 6 \
        --per_gpu_infer_batch 96 \
        --num_train_epochs 6.0 \
        --logging_steps 25 \
        --save_steps 300 \
        --train_domains ${train_domains[@]} \
        --train_mention_entity_scores ${BASE_DIR}/experiments/zeshel/vanilla_linking/checkpoint-73914/mention_entity_scores.train.pkl \
