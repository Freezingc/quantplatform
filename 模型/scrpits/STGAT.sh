#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/zhangyudong/FDU/Quant-Platform-Prod-new/模型"
cd "$PROJECT_DIR"

STDOUT_DIR="$PROJECT_DIR/data/logs/_stdout"
mkdir -p "$STDOUT_DIR"
RUN_TAG="STGAT_$(date +%Y%m%d_%H%M%S)"
OUT_LOG="$STDOUT_DIR/${RUN_TAG}.out"
PID_FILE="$STDOUT_DIR/${RUN_TAG}.pid"

nohup python -u run.py \
    --random_seed 2025 \
    --is_training 1 \
    --alpha 158 \
    --instruments csi300 \
    --model STGAT \
    --seq_len 30 \
    --horizon 5 \
    --input_size 158 \
    --hidden_size 128 \
    --num_layers 1 \
    --stgat_kernel_size 3 \
    --gat_hidden_size 32 \
    --gat_heads 4 \
    --gat_layers 2 \
    --gat_topk 20 \
    --dropout 0.1 \
    --train_begin_date 2015-01-02 \
    --valid_begin_date 2024-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2026-03-01 \
    --train_epochs 200 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --use_gpu True \
    --gpu 1 \
    --checkpoint /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/checkpoint/STGAT \
    --result /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/STGAT \
    --log /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/logs/STGAT \
    > "$OUT_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[INFO] STGAT 已在后台启动"
echo "[INFO] PID: $PID"
echo "[INFO] 输出日志: $OUT_LOG"
echo "[INFO] PID 文件: $PID_FILE"
echo "[INFO] 实时查看日志: tail -f \"$OUT_LOG\""
echo "[INFO] 停止训练: kill $PID"
