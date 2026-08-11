import os
import re
import json
import pandas as pd

# 设置input文件夹路径
input_folder = 'input'

# 用于存储结果的字典
results = {}

# 正则表达式匹配文件名中的两个数字
pattern = re.compile(r'Xgboost_(\d+)_(\d+)_predictions\.csv')

# 遍历input文件夹下的所有文件
for filename in os.listdir(input_folder):
    match = pattern.match(filename)
    if match:
        file_key = filename.replace('.csv', '')
        second_number = int(match.group(2))

        # 读取CSV文件
        file_path = os.path.join(input_folder, filename)
        try:
            df = pd.read_csv(file_path)

            # 提取所有列名中非 "ts_code" 的日期列，并排序
            date_columns = sorted(col for col in df.columns if col != 'ts_code')

            if date_columns:
                begin_date = date_columns[0]
                end_date = date_columns[-1]
            else:
                begin_date = end_date = None  # 若无日期列，设为None

            # 构造结果条目
            results[file_key] = {
                "begin_date": begin_date,
                "end_date": end_date,
                "channel_num": second_number,
                "position_cycle": second_number
            }

        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")

# 写入 config.json 到当前目录
with open('config.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)

print("JSON 文件已生成：config.json")
