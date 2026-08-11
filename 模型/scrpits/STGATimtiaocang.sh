#!/usr/bin/env bash
set -euo pipefail

/data/zhangyudong/.conda/envs/quantyd/bin/python -u run.py \
    --random_seed 2025 \
    --is_training 0 \
    --inference 1 \
    --infer_date 2025-12-21 \
    --fast_inference 1 \
    --alpha 158 \
    --instruments all \
    --model STGATim \
    --seq_len 30 \
    --horizon 5 \
    --input_size 158 \
    --hidden_size 128 \
    --num_layers 1 \
    --gat_hidden_size 32 \
    --gat_heads 4 \
    --gat_layers 2 \
    --gat_topk 20 \
    --dropout 0.1 \
    --train_begin_date 2015-01-02 \
    --valid_begin_date 2025-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2026-03-01 \
    --train_epochs 200 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --use_gpu True \
    --gpu 1 \
    --checkpoint /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/checkpoint/STGATim \
    --result /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/STGATim \
    --log /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/logs/STGATim \
