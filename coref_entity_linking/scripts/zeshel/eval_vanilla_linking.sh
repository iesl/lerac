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
        --trained_model_dir ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-73914/ \
        --do_lower_case \
        --task_name vanilla_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-73914/ \
        --log_dir ${BASE_DIR}/logs/ \
        --do_test \
        --max_seq_length 256 \
        --num_candidates 64 \
        --num_candidates_per_example 16 \
        --per_gpu_infer_batch 32 \
        --train_domains ${train_domains[@]} \
        --val_domains ${val_domains[@]} \
        --test_domains ${test_domains[@]}
