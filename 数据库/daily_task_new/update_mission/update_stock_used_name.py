from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv

import tushare as ts

def update_stock_used_name():
    load_dotenv()
    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    code_list = pro.namechange(fields = 'ts_code')
    code_list = code_list.drop_duplicates(subset=['ts_code'])

    # 删除已有数据
    delete_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'stock_used_name')
        delete from t
    '''
    fields=[
        "ts_code",
        "name",
        "start_date",
        "end_date",
        "ann_date",
        "change_reason"
    ]
    df = pd.DataFrame(columns = fields)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df["ann_date"] = pd.to_datetime(df["ann_date"])
    # 拉取并写入数据
    for index, row in tqdm(code_list.iterrows(), total=len(code_list)):
        error_count = 0  # 初始化错误计数器
        max_errors = 10  # 最大允许错误次数
        while True:
            try:
                df_new = pro.namechange(**{
                    "ts_code": row[0],
                    "start_date": "",
                    "end_date": "",
                    "limit": "",
                    "offset": ""
                }, fields=fields)
                df_new["start_date"] = pd.to_datetime(df_new["start_date"])
                df_new["end_date"] = pd.to_datetime(df_new["end_date"])
                df_new["ann_date"] = pd.to_datetime(df_new["ann_date"])
                # 向数据库中写入数据
                df = pd.concat([df, df_new], ignore_index=True)
                break
            except Exception as e:
                error_count += 1  # 错误计数加1
                print(f"发生异常: {e}, 1min后重新尝试... (错误次数: {error_count}/{max_errors})")
                
                if error_count >= max_errors:
                    print(f"已达到最大错误次数 {max_errors}，停止尝试...")
                    import sys
                    sys.exit(1)
                    
                time.sleep(60)  # 休息1min

    return delete_query, df