from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
import pandas as pd

# 读取 Excel 文件
file_path = 'full_summary.xlsx'
df = pd.read_excel(file_path)

# 定义期望的列顺序
desired_columns = [
    'factor_name', '分析日期', 'IC均值', 'IC波动率', '年化ICIR', 
    '年化收益', '夏普比率', '最大回撤', '胜率', '换手率', '分析耗时'
]

# 检查 DataFrame 中是否包含所有期望的列
existing_columns = [col for col in desired_columns if col in df.columns]
missing_columns = [col for col in desired_columns if col not in df.columns]

if missing_columns:
    print(f"警告：以下列在 DataFrame 中不存在：{missing_columns}")

# 重新排列列顺序（只保留存在的列）
df = df[existing_columns + [col for col in df.columns if col not in existing_columns]]

# 显示 DataFrame 的基本信息（可选）
print(df.head())  # 查看前几行数据
print(df.info())  # 查看数据框的基本信息
print(df)

# 连接 DolphinDB 数据库
s = ddb.session()
s.connect("10.176.55.211", 8848, "admin", "yeslab2024")

# 创建表 appender
appender = ddb.TableAppender(dbPath='dfs://YesLab', tableName="factor_summary_full", ddbSession=s)

df["分析日期"] = pd.to_datetime(df["分析日期"])
num = appender.append(df)
