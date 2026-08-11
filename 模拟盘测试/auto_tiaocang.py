#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动调仓脚本 - 每周三晚上23:00执行
功能：
1. 下载最新的qlib数据
2. 执行模型推理
3. 转换pkl为csv
4. 复制到指定目录
"""

import subprocess
import os
import sys
import re
import pandas as pd
from datetime import datetime, timedelta
import shutil
import logging

# 配置日志
log_dir = '/data/zhangyudong/FDU/Quant-Platform-Prod-new/模拟盘测试/logs'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'auto_tiaocang_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# 配置路径
BASE_DIR = '/data/zhangyudong/FDU/Quant-Platform-Prod-new'
MODEL_DIR = os.path.join(BASE_DIR, '模型')
SIMULATION_DIR = os.path.join(BASE_DIR, '模拟盘测试')
STRATEGY_NAME = 'STGATim'
BASH_SCRIPT = os.path.join(MODEL_DIR, 'scrpits/STGATimtiaocang.sh')
MODEL_RESULT_DIR = os.path.join(MODEL_DIR, 'data/result/STGATim')
PKL_KEYWORD = 'alpha158_all_seq30_horizon5'
LOCAL_SCORE_FILENAME = 'STGATim_30_5_Scores.csv'
OUTPUT_FILENAME = 'MTSMatrix.csv'
TRAIN_BEGIN_DATE = '2015-01-02'
TRAIN_END_DATE = '2025-01-01'
QLIB_DATA_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_data')
OUTPUT_DIR = '/data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出'


def run_command(cmd, cwd=None, shell=True):
    """执行shell命令"""
    logger.info(f"执行命令: {cmd}")
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        logger.info(f"命令执行成功: {result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e.stderr}")
        raise


def download_qlib_data(date_str):
    """下载qlib数据"""
    logger.info(f"开始下载qlib数据，日期: {date_str}")
    
    url = f"https://github.com/chenditc/investment_data/releases/download/{date_str}/qlib_bin.tar.gz"
    
    # 切换到临时目录下载
    original_dir = os.getcwd()
    temp_dir = '/tmp'
    tar_file = os.path.join(temp_dir, 'qlib_bin.tar.gz')
    
    try:
        os.chdir(temp_dir)
        
        # 下载文件
        wget_cmd = f"wget -O {tar_file} {url}"
        run_command(wget_cmd)
        
        # 解压文件到用户的.qlib目录
        logger.info(f"解压qlib数据到 {QLIB_DATA_DIR}...")
        os.makedirs(QLIB_DATA_DIR, exist_ok=True)
        tar_cmd = f"tar -zxvf {tar_file} -C {QLIB_DATA_DIR} --strip-components=1"
        run_command(tar_cmd)
        
        # 删除压缩包
        logger.info("清理下载文件...")
        if os.path.exists(tar_file):
            os.remove(tar_file)
        
        logger.info(f"qlib数据下载完成，已解压到: {QLIB_DATA_DIR}")
        
    finally:
        os.chdir(original_dir)


def run_model_inference(date_str):
    """执行模型推理"""
    logger.info(f"开始模型推理，策略: {STRATEGY_NAME}，推理日期: {date_str}")

    if not os.path.exists(BASH_SCRIPT):
        raise FileNotFoundError(f"推理脚本不存在: {BASH_SCRIPT}")
    
    # 读取bash脚本
    with open(BASH_SCRIPT, 'r', encoding='utf-8') as f:
        script_content = f.read()
    
    # 替换infer_date
    modified_content, replaced_count = re.subn(
        r'--infer_date\s+\d{4}-\d{2}-\d{2}',
        f'--infer_date {date_str}',
        script_content
    )
    if replaced_count == 0:
        raise ValueError("推理脚本中未找到 --infer_date 参数，无法替换推理日期")
    
    # 创建临时脚本
    temp_script = os.path.join(SIMULATION_DIR, f'temp_tiaocang_{os.getpid()}.sh')
    try:
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(modified_content)

        # 执行bash脚本
        logger.info("执行模型推理脚本...")
        run_command(f"bash {temp_script}", cwd=MODEL_DIR)
        logger.info("模型推理完成")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


def convert_ts_code(code):
    """转换股票代码格式（SH600000 → 600000.SH）"""
    if pd.isna(code):
        return code
    code = str(code).strip()
    # 匹配 SH/SZ + 数字 → 数字.SH/SZ
    match_sh = re.match(r'(SH)(\d+)', code)
    match_sz = re.match(r'(SZ)(\d+)', code)
    if match_sh:
        return match_sh.group(2) + '.SH'
    if match_sz:
        return match_sz.group(2) + '.SZ'
    # 如果已是标准格式（如000001.SZ），保持
    if '.' in code:
        return code
    return code  # 其他保持


def convert_pkl_to_csv(train_begin_date, train_end_date):
    """将pkl文件转换为csv"""
    logger.info(f"开始转换pkl文件为csv，策略: {STRATEGY_NAME}...")
    
    # 构建pkl文件路径
    pkl_filename = f'{PKL_KEYWORD}_train_begin_date{train_begin_date}_train_end_date{train_end_date}.pkl'
    pkl_path = os.path.join(MODEL_RESULT_DIR, pkl_filename)
    
    # 如果指定的pkl文件不存在，尝试查找最新的pkl文件
    if not os.path.exists(pkl_path):
        logger.warning(f"pkl文件不存在: {pkl_path}")
        logger.info("尝试查找最新的pkl文件...")
        
        result_dir = MODEL_RESULT_DIR
        if os.path.exists(result_dir):
            pkl_files = [f for f in os.listdir(result_dir) if f.endswith('.pkl') and PKL_KEYWORD in f]
            if pkl_files:
                # 按修改时间排序，获取最新的
                pkl_files_with_time = [(f, os.path.getmtime(os.path.join(result_dir, f))) for f in pkl_files]
                pkl_files_with_time.sort(key=lambda x: x[1], reverse=True)
                latest_pkl = pkl_files_with_time[0][0]
                pkl_path = os.path.join(result_dir, latest_pkl)
                logger.info(f"找到最新pkl文件: {latest_pkl}")
            else:
                logger.error(f"result目录中没有找到匹配的pkl文件: {result_dir}")
                raise FileNotFoundError(f"result目录中没有找到匹配的pkl文件")
        else:
            logger.error(f"result目录不存在: {result_dir}")
            raise FileNotFoundError(f"result目录不存在: {result_dir}")
    
    logger.info(f"读取pkl文件: {pkl_path}")
    df = pd.read_pickle(pkl_path)
    
    # 转换instrument格式
    df['instrument'] = df['instrument'].apply(convert_ts_code)
    
    # 清洗datetime → YYYYMMDD
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')
    
    # Pivot宽表
    dta = df.pivot(index='instrument', columns='datetime', values='score').fillna(0)
    
    # 插入ts_code列
    dta.insert(0, 'ts_code', dta.index)
    dta = dta.reset_index(drop=True)
    
    # 保存到模拟盘测试目录（不包含表头）
    csv_path = os.path.join(SIMULATION_DIR, LOCAL_SCORE_FILENAME)
    dta.to_csv(csv_path, index=False, header=False)
    
    logger.info(f"转换完成: {csv_path}")
    logger.info(f"形状: {dta.shape}")
    logger.info(f"前5行ts_code: {dta['ts_code'].head().tolist()}")
    
    return csv_path


def copy_to_output(csv_path):
    """复制csv文件到输出目录"""
    logger.info(f"复制{STRATEGY_NAME} csv文件到输出目录...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 目标文件路径
    dest_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # 复制文件
    shutil.copy2(csv_path, dest_path)
    logger.info(f"文件已复制到: {dest_path}")
    


def main():
    """主函数"""
    try:
        logger.info("=" * 80)
        logger.info("自动调仓脚本开始执行")
        logger.info("=" * 80)
        
        # 输出关键路径信息
        logger.info(f"工作目录: {BASE_DIR}")
        logger.info(f"模型目录: {MODEL_DIR}")
        logger.info(f"Qlib数据目录: {QLIB_DATA_DIR}")
        logger.info(f"输出目录: {OUTPUT_DIR}")
        logger.info(f"当前策略: {STRATEGY_NAME}")
        
        # 获取当前日期和推理日期
        today = datetime.now()
        download_date_str = today.strftime('%Y-%m-%d')  # 用于下载数据
        infer_date = today + timedelta(days=1)  # 推理日期为当天+1
        infer_date_str = infer_date.strftime('%Y-%m-%d')
        
        logger.info(f"当前日期: {download_date_str}")
        logger.info(f"推理日期: {infer_date_str} (当天+1)")
        
        # 步骤1: 下载qlib数据
        logger.info("\n[步骤1] 下载qlib数据")
        download_qlib_data(download_date_str)
        
        # 步骤2: 执行模型推理
        logger.info("\n[步骤2] 执行模型推理")
        run_model_inference(infer_date_str)
        
        # 步骤3: 转换pkl为csv
        logger.info("\n[步骤3] 转换pkl为csv")
        # 根推理脚本中的配置
        csv_path = convert_pkl_to_csv(TRAIN_BEGIN_DATE, TRAIN_END_DATE)
        
        # 步骤4: 复制文件到输出目录
        logger.info("\n[步骤4] 复制文件到输出目录")
        copy_to_output(csv_path)
        
        logger.info("=" * 80)
        logger.info("自动调仓脚本执行成功！")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

