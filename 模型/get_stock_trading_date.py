import pandas as pd
import dolphindb as ddb
import numpy as np
import os
import time

def get_nth_trading_day_from_recent(n=30):
    """
    查询最近 n 个交易日，并返回其中的第一个（即 n 天前的交易日）
    """
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "liuziyang", "123456")
    query = f'''
        db = database("dfs://YesLab")
        t = loadTable(db, "factor_daily_data")
        recent_n = select distinct trade_date from t order by trade_date desc limit {n}
        select min(trade_date) as first_date from recent_n
    '''
    result = s.run(query)
    date_value = result.iloc[0, 0]
    date_dt = pd.to_datetime(date_value)
    return date_dt.strftime('%Y.%m.%d')

print(get_nth_trading_day_from_recent())