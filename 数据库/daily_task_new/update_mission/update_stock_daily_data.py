from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import tushare as ts

def update_stock_daily_data():
    load_dotenv()  # 默认加载当前目录的 .env
    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 连接 Tushare API
    pro = ts.pro_api(os.getenv('TUSHARE_TOKEN'))

    # 获取数据库中已有数据的最大日期
    max_date_query = '''
        db = database("dfs://YesLab")
        t = loadTable(db,'stock_daily_data')
        select max(trade_date) from t
    '''
    max_date_result = s.run(max_date_query)

    # 如果数据库中有数据，获取最大日期并增加一天
    if not pd.isna(max_date_result['max_trade_date'].iloc[0]):
        max_date = pd.to_datetime(max_date_result['max_trade_date'][0])
        start_date = (max_date + pd.Timedelta(days=1)).strftime('%Y%m%d')  # 转换为 YYYY.MM.DD 格式
    else:
        # 如果数据库为空，设定默认的开始日期
        start_date = ''

    # 获取当前日期
    today = datetime.today()
    # 格式化为 YYYY.MM.DD 的格式
    end_date = today.strftime("%Y%m%d")


    print("更新起始日期:",start_date)
    print("更新截止日期:",end_date)

    # 获取交易日历，确保从最大日期之后的日期开始
    time_list = pro.trade_cal(exchange='', is_open=1, 
                            start_date = start_date, 
                            end_date = end_date, 
                            fields ='cal_date')
    print(time_list)
    fields=[
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount"
    ]
    df = pd.DataFrame(columns = fields)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    # 拉取并写入数据
    for date in tqdm(time_list["cal_date"]):
        print(date)
        print(type(date))
        error_count = 0  # 初始化错误计数器
        max_errors = 10  # 最大允许错误次数
        while True:
            try:
                df_new = pro.daily(**{
                    "ts_code": "",
                    "trade_date": date,
                    "start_date": "",
                    "end_date": "",
                    "offset": "",
                    "limit": ""
                }, fields=fields)
                # 将 'trade_date' 转换为 datetime 格式
                df_new["trade_date"] = pd.to_datetime(df_new["trade_date"])
                # 向数据库中追加数据
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

    return "", df