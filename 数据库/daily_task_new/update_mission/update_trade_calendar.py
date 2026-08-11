from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv

import tushare as ts

def update_trade_calendar():
    load_dotenv()
    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 删除已有数据
    delete_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'trade_calendar')
        delete from t
    '''
    fields = [
        "exchange",
        "cal_date",
        "is_open",
        "pretrade_date"
    ]

    # 初始化df
    df = pd.DataFrame(columns = fields)
    df["cal_date"] = pd.to_datetime(df["cal_date"])
    df["pretrade_date"] = pd.to_datetime(df["pretrade_date"])

    error_count = 0  # 初始化错误计数器
    max_errors = 10  # 最大允许错误次数

    while True:
        try:
            df_new = pro.trade_cal(**{
                "exchange": "",
                "cal_date": "",
                "start_date": "",
                "end_date": "",
                "is_open": "",
                "limit": "",
                "offset": ""
            }, fields=fields)
            df_new["cal_date"] = pd.to_datetime(df_new["cal_date"])
            df_new["pretrade_date"] = pd.to_datetime(df_new["pretrade_date"])
            # 合并
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