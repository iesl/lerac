#!/bin/bash

# README:
# - always call from the `coref_entity_linking directory` of this repo

set -eux


# TODO:
# - change absolute paths to be variable


python -m torch.distributed.launch \
    --nproc_per_node 8 \
    src/main.py \
        --data_dir '/local/coref_entity_linking/data/mm_st21pv_long_entities/' \
        --model_type 'bert' \
        --model_name_or_path 'models/biobert_v1.1_pubmed/' \
        --task_name 'cluster_linking' \
        --output_dir '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/experiments/mm_st21pv_long_entities/cluster_linking/exp_0_m-m/' \
        --log_dir '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/logs/' \
        --do_train \
        --max_seq_length '128' \
        --seq_embed_dim '128' \
        --embed_pooling_strategy 'pool_highlighted_outputs' \
        --concat_pooling_strategy 'pool_highlighted_outputs' \
        --clustering_domain 'within_doc' \
        --available_entities 'candidates_only' \
        --mention_negatives 'random' \
        --training_method 'triplet_max_margin' \
        --pair_gen_method 'mst' \
        --training_edges_considered 'm-m' \
        --k '128' \
        --num_train_negs '24' \
        --margin '0.7' \
        --warmup_steps '100' \
        --learning_rate '5e-5' \
        --max_grad_norm '1.0' \
        --num_clusters_per_macro_batch '16' \
        --per_gpu_train_batch_size '16' \
        --per_gpu_infer_batch_size '256' \
        --num_train_epochs '6' \
        --logging_steps '25' \
        --knn_refresh_steps '-1' \
        --train_domains 'train' 'T005' 'T007' 'T017' 'T022' 'T031' 'T033' 'T037' 'T038' 'T058' 'T062' 'T074' 'T082' 'T091' 'T092' 'T097' 'T098' 'T103' 'T168' 'T170' 'T201' 'T204' \
        --val_domains 'val' 'T005' 'T007' 'T017' 'T022' 'T031' 'T033' 'T037' 'T038' 'T058' 'T062' 'T074' 'T082' 'T091' 'T092' 'T097' 'T098' 'T103' 'T168' 'T170' 'T201' 'T204'
