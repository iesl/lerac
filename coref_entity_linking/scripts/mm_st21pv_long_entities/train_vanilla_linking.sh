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
        --model_name_or_path models/biobert_v1.1_pubmed/ \
        --task_name vanilla_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/vanilla_linking/  \
        --log_dir ${BASE_DIR}/logs/ \
        --do_train \
        --max_seq_length 256 \
        --num_candidates 64 \
        --num_candidates_per_example 16 \
        --warmup_steps 1000 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --per_gpu_train_batch=1 \
        --per_gpu_infer_batch=32 \
        --num_train_epochs 3.0 \
        --logging_steps 10 \
        --save_steps 10000 \
        --train_domains ${train_domains[@]} \
