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
train_domains=("train" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")
eval_domains=("val" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")
test_domains=("test" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204")

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/run.py \
        --data_dir data/mm_st21pv_long_entities/ \
        --model_type bert \
        --trained_model_dir poly_linker-experiment/checkpoint-89172/ \
        --task_name poly_linking \
        --output_dir poly_linker-experiment/checkpoint-89172/ \
        --do_eval \
        --max_seq_length 256 \
        --num_candidates 64 \
        --num_candidates_per_example 32 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --per_gpu_train_batch 1 \
        --per_gpu_infer_batch 24 \
        --logging_steps 500 \
        --save_steps 5000 \
        --train_domains ${train_domains[@]} \
        --val_domains ${eval_domains[@]}
