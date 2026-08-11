from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv

import tushare as ts

def update_sw_ind_code():
    load_dotenv()
    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 删除已有数据
    delete_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'sw_ind_code')
        delete from t
    '''
    fields = [
        "index_code",
        "industry_name",
        "level",
        "industry_code",
        "is_pub",
        "parent_code",
        "src"
    ]
    # 初始化df

    df = pd.DataFrame(columns = fields)
    tmp = df["src"].str.replace(r'^sw', '', case=False, regex=True)
    df["date_useless"] = pd.to_datetime(
        tmp,
        format='%Y',          # 匹配4位年份
    )

    error_count = 0  # 初始化错误计数器
    max_errors = 10  # 最大允许错误次数
    # 拉取并写入数据

    while True:
        try:
            print(1234)
            df_new = pro.index_classify(**{
                "index_code": "",
                "level": "",
                "src": "",
                "parent_code": "",
                "limit": "",
                "offset": ""
            }, fields=fields)
            tmp = df_new["src"].str.replace(r'^sw', '', case=False, regex=True)
            # 步骤2: 转换为日期 (忽略无效格式)
            df_new["date_useless"] = pd.to_datetime(
                tmp,
                format='%Y',          # 匹配4位年份
            )
            # print(df_new["date_useless"])
            # print(type(df_new['date_useless'].iloc[0]))
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
    