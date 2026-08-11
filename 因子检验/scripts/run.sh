#!/bin/bash
# run.sh - 一键运行因子下载、分析与上传流程

PROJECT_DIR=~/quant/work/FactorEval
cd $PROJECT_DIR || exit 1

echo "[INFO] 当前目录: $(pwd)"

echo "[STEP 1] 计算所有因子..."
python -u scripts/get_all_factor.py > get_all_factor.log 2>&1
echo "[STEP 1 DONE]"

echo "[STEP 2] 因子分析..."
python main.py > main.log 2>&1
echo "[STEP 2 DONE]"

echo "[STEP 3] 上传结果到数据库..."
python scripts/add_data_factor_summary_ic.py > upload.log 2>&1
python scripts/add_data_factor_summary_full.py >> upload.log 2>&1
echo "[STEP 3 DONE]"

echo "[ALL DONE] 全部任务完成 ✅"


# nohup scripts/run.sh > run.log 2>&1 &
