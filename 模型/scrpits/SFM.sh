# ========================================
# 训练配置说明
# ========================================
# --resume: 从checkpoint继续训练
# --lambda_turnover: 换手率控制参数（广义损失函数）
#   0.0  : 不控制换手率（默认，纯IC优化）
#   0.1  : 轻度控制（推荐，华泰研报最优值）
#   0.2  : 中度控制
#   0.3  : 强力控制（可能降低收益）
# ========================================
#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/zhangyudong/FDU/Quant-Platform-Prod-new/模型"
cd "$PROJECT_DIR"

STDOUT_DIR="$PROJECT_DIR/data/logs/_stdout"
mkdir -p "$STDOUT_DIR"
RUN_TAG="SFM_$(date +%Y%m%d_%H%M%S)"
OUT_LOG="$STDOUT_DIR/${RUN_TAG}.out"
PID_FILE="$STDOUT_DIR/${RUN_TAG}.pid"


nohup python -u run.py \
	--random_seed 2025 \
    --is_training 1 \
    --data_path /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/test_data.parquet \
    --model SFM \
    --seq_len 30 \
    --horizon 5 \
    --input_dim 103 \
    --output_dim 1 \
    --hidden_dim 64 \
    --freq_dim 10 \
    --train_begin_date 2015-01-03 \
    --valid_begin_date 2025-01-01 \
    --test_begin_date 2025-01-01 \
    --test_end_date 2026-03-01 \
    --train_epochs 20 \
    --batch_size 1 \
    --learning_rate 0.0001 \
    --use_gpu True \
    --gpu 2 \
    --checkpoint /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/checkpoint/SFM \
    --result /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/result/SFM \
    --log /data/zhangyudong/FDU/Quant-Platform-Prod-new/模型/data/logs/SFM \
    > "$OUT_LOG" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[INFO] SFM 已在后台启动"
echo "[INFO] PID: $PID"
echo "[INFO] 输出日志: $OUT_LOG"
echo "[INFO] PID 文件: $PID_FILE"


