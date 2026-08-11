from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv

import tushare as ts

def update_stock_info_table():
    load_dotenv()
    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 删除已有数据
    delete_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'stock_info_table')
        delete from t
    '''

    fields = [
        "ts_code",
        "symbol",
        "name",
        "area",
        "industry",
        "cnspell",
        "market",
        "list_date",
        "act_name",
        "act_ent_type",
        "enname",
        "fullname",
        "curr_type",
        "exchange",
        "list_status",
        "delist_date",
        "is_hs"
    ]
    
    # 初始化df
    df = pd.DataFrame(columns = fields)
    df["list_date"] = pd.to_datetime(df["list_date"])
    df["delist_date"] = pd.to_datetime(df["delist_date"])
    # 拉取并写入数据
    error_count = 0  # 初始化错误计数器
    max_errors = 10  # 最大允许错误次数
    while True:
        try:
            df_new = pro.stock_basic(**{
                "ts_code": "",
                "name": "",
                "exchange": "",
                "market": "",
                "is_hs": "",
                "list_status": "",
                "limit": "",
                "offset": ""
            }, fields=fields)
            df_new["list_date"] = pd.to_datetime(df_new["list_date"])
            df_new["delist_date"] = pd.to_datetime(df_new["delist_date"])
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