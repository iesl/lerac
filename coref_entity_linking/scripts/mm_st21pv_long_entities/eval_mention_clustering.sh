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
        --trained_model_dir mc_pool_hightlighted-experiment/checkpoint-15810/ \
        --task_name mention_clustering \
        --output_dir mc_pool_hightlighted-experiment/checkpoint-15810/ \
        --pooling_strategy 'pool_highlighted_outputs' \
        --do_eval \
        --max_seq_length 128 \
        --per_gpu_infer_batch=384 \
        --train_domains ${train_domains[@]} \
        --val_domains ${eval_domains[@]}
