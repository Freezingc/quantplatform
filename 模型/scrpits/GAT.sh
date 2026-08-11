#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/zhangyudong/FDU/Quant-Platform-Prod-new/模型"
cd "$PROJECT_DIR"

STDOUT_DIR="$PROJECT_DIR/data/logs/_stdout"
mkdir -p "$STDOUT_DIR"
RUN_TAG="GAT_$(date +%Y%m%d_%H%M%S)"
OUT_LOG="$STDOUT_DIR/${RUN_TAG}.out"
PID_FILE="$STDOUT_DIR/${RUN_TAG}.pid"

nohup python -u run.py \
    --random_seed 2025 \
    --is_training 1 \
    --instruments all \
    --alpha 158 \
    --model GAT \
    --seq_len 30 \
    --horizon 5 \
    --input_size 158\
    --hidden_size 32 \
    --dropout 0.1 \
    --fill_method fast \
    --train_begin_date 2020-01-01 \
    --valid_begin_date 2025-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2026-03-01 \
    --train_epochs 200 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --use_gpu True \
    --gpu 2 \
    --checkpoint /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/checkpoint/GAT \
    --result /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/GAT \
    --log /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/logs/GAT \
    > "$OUT_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[INFO] GAT 已在后台启动"
echo "[INFO] PID: $PID"
echo "[INFO] 输出日志: $OUT_LOG"
echo "[INFO] PID 文件: $PID_FILE"
echo "[INFO] 实时查看日志: tail -f \"$OUT_LOG\""
echo "[INFO] 停止训练: kill $PID"
