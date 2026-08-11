#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并因子数据和标签数据生成最终的训练数据文件
"""
import pyarrow.parquet as pq
from pathlib import Path
import pandas as pd

# 配置
output_directory = 'test_data_files'

# 读取因子列表
with open('feature_columns.txt', 'r') as f:
    factor_names = [line.strip() for line in f if line.strip()]  # 过滤空行

print(f"从 feature_columns.txt 读取到 {len(factor_names)} 个因子名称")

folder_path = Path(output_directory)
parquet_files = sorted(folder_path.glob("*.parquet"))

print(f"在 {output_directory} 目录中找到 {len(parquet_files)} 个 parquet 文件")

factor_series = []

# 添加 label 到因子列表
factor_names.append("label")

print("\n开始处理文件...")
processed_count = 0
skipped_count = 0

for file in parquet_files:
    # 从文件名中提取因子名
    # 文件名格式: factor_data_XXX.parquet 或 label.parquet
    if file.stem == 'label':
        factor_name = 'label'
    elif file.stem.startswith('factor_data_'):
        factor_name = file.stem.replace('factor_data_', '')
    else:
        print(f"跳过未知格式文件: {file.name}")
        skipped_count += 1
        continue
    
    # 检查因子是否在列表中
    if factor_name not in factor_names:
        print(f"跳过不在列表中的因子: {factor_name}")
        skipped_count += 1
        continue

    print(f"[{processed_count + 1}] 处理: {factor_name}...")
    
    try:
        df = pq.read_table(str(file)).to_pandas()
        
        # 查找值列(排除 ts_code 和 trade_date)
        value_col = [col for col in df.columns if col not in ['ts_code', 'trade_date']]
        
        if len(value_col) != 1:
            print(f"  警告: {file.name} 应该有1个值列,实际有 {len(value_col)} 个: {value_col}")
            if len(value_col) == 0:
                print(f"  跳过该文件")
                skipped_count += 1
                continue
            # 如果有多个值列,使用第一个
            value_col = value_col[0]
        else:
            value_col = value_col[0]
        
        # 设置索引并提取值列
        series = df.set_index(['ts_code', 'trade_date'])[value_col]
        series.name = factor_name
        factor_series.append(series)
        processed_count += 1
        
        print(f"  成功: 形状 {series.shape}, 缺失值 {series.isna().sum()}")
        
    except Exception as e:
        print(f"  错误: 处理 {factor_name} 时出错: {e}")
        skipped_count += 1
        continue

print(f"\n处理完成:")
print(f"  成功处理: {processed_count} 个文件")
print(f"  跳过: {skipped_count} 个文件")

if len(factor_series) == 0:
    print("\n错误: 没有成功处理任何文件!")
    exit(1)

print(f"\n合并 {len(factor_series)} 个因子...")
merged = pd.concat(factor_series, axis=1)

print(f"合并后形状: {merged.shape}")
print(f"列名: {merged.columns.tolist()}")

# 重置索引并排序
merged = merged.reset_index().sort_values(by=['trade_date', 'ts_code'])

print(f"最终形状: {merged.shape}")
print(f"列数: {len(merged.columns)}")
print(f"  - ts_code: 1")
print(f"  - trade_date: 1")
print(f"  - 特征列: {len(merged.columns) - 3}")
print(f"  - label: 1")

# 保存
output_file = "test_data.parquet"
print(f"\n保存到 {output_file}...")
merged.to_parquet(output_file, engine='pyarrow', compression='snappy', index=False)

# 验证保存的文件
print(f"\n验证保存的文件...")
verify_df = pd.read_parquet(output_file)
print(f"验证成功!")
print(f"  形状: {verify_df.shape}")
print(f"  列名: {verify_df.columns.tolist()[:10]}...")  # 只显示前10列
print(f"  前5行:")
print(verify_df.head())

print(f"\n完成! 数据已保存到 {output_file}")

