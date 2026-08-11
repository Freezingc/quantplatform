from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv

import tushare as ts

def update_index_info_table():
    load_dotenv()
    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 删除已有数据
    delete_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'index_info_table')
        delete from t
    '''

    # 拉取并写入数据
    error_count = 0  # 初始化错误计数器
    max_errors = 10  # 最大允许错误次数

    fields=[
        "ts_code",
        "name",
        "fullname",
        "market",
        "publisher",
        "index_type",
        "category",
        "base_date",
        "base_point",
        "list_date",
        "weight_rule",
        "desc",
        "exp_date"
    ]
    df = pd.DataFrame(columns = fields)
    df["base_date"] = pd.to_datetime(df["base_date"])
    df["list_date"] = pd.to_datetime(df["list_date"])
    df["exp_date"] = pd.to_datetime(df["exp_date"])
    df['date_useless'] = pd.to_datetime('2025-01-01')

    while True:
        try:
            df_new = pro.index_basic(**{
                "ts_code": "",
                "market": "",
                "publisher": "",
                "category": "",
                "name": "",
                "limit": "",
                "offset": ""
            }, fields=fields)

            df_new["base_date"] = pd.to_datetime(df_new["base_date"])
            df_new["list_date"] = pd.to_datetime(df_new["list_date"])
            df_new["exp_date"] = pd.to_datetime(df_new["exp_date"])
            df_new['date_useless'] = pd.to_datetime('2025-01-01')
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