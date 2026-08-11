import dolphindb as ddb
import pandas as pd
from datetime import datetime
import sys
sys.path.append("/home/chenxutian/chenxutian_data/Quant-Platform-Prod/数据库")
from factor_calculators.base import FactorCalculator
from add_factor_narrow import update_factor, set_last_update_time

from dotenv import load_dotenv
load_dotenv()  # 默认加载当前目录的 .env
import os
username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
password = os.getenv('dolphinDB_PASSWORD')


print(11111111111111)

s = ddb.session()
s.connect("10.176.55.211", 8848, username, password)

query = '''
    db = database("dfs://YesLab")
    t = loadTable(db, "factor_metadata")
    select factor_name, store_factor,  method, lookback_days from t
'''
ts_codes = s.run(query)
print(ts_codes)
print(22222222)

# 遍历所有因子并执行更新
for index, row in ts_codes.iterrows():
    
    factor_name = row['factor_name']
    '''
    if factor_name not in ['Alpha33']:
        continue
    '''
    method = row['method']
    lookback_days = row['lookback_days']
    store_factor = row['store_factor']

    # 获取需要更新的日期范围
    query_max_date = f'''
        db = database("dfs://YesLab")
        t = loadTable(db, "{store_factor}")
        select max(trade_date) as max_date from t where factor_name = '{factor_name}'
    '''
    max_date_result = s.run(query_max_date)
    
    print(query_max_date)

    if pd.isna(max_date_result.loc[0, 'max_date']):
        start_time = "2020.01.01" #默认从2020.01.01开始
        print(1234567890123456,start_time)
    else:
        print(1111111111111111,pd.to_datetime(max_date_result.loc[0, 'max_date']))
        start_time = (pd.to_datetime(max_date_result.loc[0, 'max_date']) + pd.Timedelta(days=1)).strftime('%Y.%m.%d')
        print(start_time)

    end_time = datetime.now().strftime("%Y.%m.%d")
    print(f"更新因子: {factor_name} 从 {start_time} 至 {end_time}")

    try:
        # 动态加载因子计算类
        module = __import__(f"factor_calculators.{method}", fromlist=[method])
        FactorClass = getattr(module, method)

        # 实例化因子计算类
        calculator = FactorClass(s, lookback_days)
        
        #计算需要更新的因子值
        factor = calculator.calculate(start_time, end_time)
        query = f'''
            db = database("dfs://YesLab")
            t = loadTable(db, "factor_metadata")
            select store_factor from t where factor_name = '{factor_name}'
        '''
        table_name_result = s.run(query)
        table_name = table_name_result.loc[0, 'store_factor']
        print(f'因子：{factor_name}    存储表名: {table_name}')
        
        # 将结果写入数据库
        update_factor(s, factor_name, factor, table_name)

        # 最后在factor_metadata表中更新last_update_time
        set_last_update_time(s, factor_name)
        
        print(f"因子 {factor_name} 更新完成")
    except Exception as e:
        print(f"更新因子 {factor_name} 时出错: {e}")

print(3333333333333333333)
s.close()