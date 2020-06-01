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
DATASET="mm_st21pv_long_entities"

train_domains=("train" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")
val_domains=("val" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")
test_domains=("test" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/run.py \
        --data_dir data/mm_st21pv_long_entities/ \
        --model_type bert \
        --trained_model_dir ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-178341/ \
        --task_name vanilla_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/checkpoint-178341/ \
        --log_dir ${BASE_DIR}/logs/ \
        --do_test \
        --max_seq_length 256 \
        --num_candidates 64 \
        --num_candidates_per_example 16 \
        --per_gpu_infer_batch 32 \
        --train_domains ${train_domains[@]} \
        --val_domains ${val_domains[@]} \
        --test_domains ${test_domains[@]}
