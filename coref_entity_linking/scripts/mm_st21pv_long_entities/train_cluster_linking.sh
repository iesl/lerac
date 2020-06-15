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
DATASET="mm_st21pv_long_entities"

train_domains=( "train" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )
val_domains=( "val" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )
test_domains=( "test" "T005" "T007" "T017" "T022" "T031" "T033" "T037" "T038" "T058" "T062" "T074" "T082" "T091" "T092" "T097" "T098" "T103" "T168" "T170" "T201" "T204" )

python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/main.py \
        --data_dir data/${DATASET}/ \
        --model_type bert \
        --model_name_or_path models/biobert_v1.1_pubmed/ \
        --task_name cluster_linking \
        --output_dir ${BASE_DIR}/experiments/${DATASET}/cluster_linking/ \
        --log_dir ${BASE_DIR}/logs/ \
        --do_train \
        --max_seq_length 128 \
        --seq_embed_dim 128 \
        --pooling_strategy 'pool_highlighted_outputs' \
        --clustering_domain 'within_doc' \
        --available_entities 'candidates_only' \
        --training_method 'triplet' \
        --pair_gen_method 'all_pairs' \
        --k 16 \
        --max_in_cluster_dist 0.1 \
        --margin 0.5 \
        --warmup_steps 0 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --num_clusters_per_macro_batch 16 \
        --per_gpu_train_batch 32 \
        --per_gpu_infer_batch 256 \
        --num_train_epochs 3 \
        --evaluate_during_training \
        --logging_steps 25 \
        --knn_refresh_steps 100 \
        --save_steps 300 \
        --train_domains ${train_domains[@]} \
        --val_domains ${val_domains[@]} \
