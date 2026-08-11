from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
import os
from datetime import datetime, timedelta
import sys
import importlib
import inspect
import pkgutil
from pathlib import Path
from factor_calculators.base import FactorCalculator
import numpy as np
import argparse

from config import FACTOR_CONFIG

# 动态导入所有因子计算函数
def discover_factor_calculators():
    """
    动态发现并导入factor_calculators目录中的所有FactorCalculator子类
    返回：{类名: 类} 的字典
    """
    calculator_classes = {}
    
    # 获取factor_calculators包中的所有模块
    package_path = Path(__file__).parent / "factor_calculators"
    package_name = "factor_calculators"
    
    for finder, name, _ in pkgutil.iter_modules([str(package_path)]):
        # 跳过base.py模块
        if name == "base":
            continue
            
        try:
            # 动态导入模块
            module = importlib.import_module(f"{package_name}.{name}")
            
            # 遍历模块中的所有成员
            for member_name, member in inspect.getmembers(module):
                # 检查是否是FactorCalculator的子类且不是FactorCalculator本身
                if (inspect.isclass(member) and 
                    issubclass(member, FactorCalculator) and 
                    member is not FactorCalculator):
                    calculator_classes[member_name] = member
        except Exception as e:
            print(f"加载模块 {name} 时出错: {e}")
    
    return calculator_classes

def get_table_name(s):
    # 构建查询语句
    cmd = f"""
    db = database("dfs://YesLab")
    t = loadTable(db, 'factor_metadata')
    select distinct store_factor from t 
    """
    
    # 执行查询
    result = s.run(cmd)
    store_table_list = result['store_factor'].tolist()
    # print(store_table_list)
    def extract_number(s):
        return int(s.replace('factor_data', ''))
    if store_table_list == []:
        max_value = 0
    else:
        max_value = max(store_table_list, key=extract_number)

    cdm2 = f"""
    db = database("dfs://YesLab")
    t = loadTable(db, 'factor_metadata')
    select count store_factor from t 
    where store_factor = '{max_value}'
    """
    result2 = s.run(cdm2)
    count = result2.iloc[0,0]
    # print(type(count))

    if count >= 50:
        # 若该表已经存储了50个因子，则创建一个新的表
        new_value = extract_number(max_value) + 1
        cmd3 = f"""
        db = database(directory="dfs://YesLab")
tbSchema=table(1:0, `ts_code`trade_date`factor_name`value,[SYMBOL,DATE,STRING,DOUBLE])
sortKeyMappingFunction=[hashBucket{{,499}}]
db.createPartitionedTable(table=tbSchema, tableName=`factor_data{new_value}, partitionColumns=`trade_date, sortColumns=`ts_code`trade_date,sortKeyMappingFunction=sortKeyMappingFunction)
        """
        s.run(cmd3)
        return f"factor_data{new_value}"
    else:
        # print(1111)
        return f"factor_data{extract_number(max_value)}"


# 添加因子到factor_metadata表
def add2metadata(s, factor_name, creator, create_date, formula, lookback_days, method,
                        needed_factor = [],
                        category = "",
                        update_frequency = "daily",
                        description = "无"):
    appender = ddb.TableAppender(dbPath='dfs://YesLab', tableName="factor_metadata", ddbSession=s)
    query_is_exist = f'''
        db = database("dfs://YesLab")
        t = loadTable(db, 'factor_metadata')
        select * from t where factor_name = '{factor_name}'
    '''
    result = s.run(query_is_exist)
    if not result.empty:
        print(f"{factor_name}已在factor_metadata表中存在")
        return

    query = f'''
        db = database("dfs://YesLab")
        t = loadTable(db, "factor_metadata")
        select max(factor_id) from t
    '''
    result = s.run(query)
    if result is None or result.empty or pd.isna(result.iloc[0, 0]):
        new_id = 0
    else:
        max_factor_id = int(result.iloc[0, 0])  # 确保转换为整数
        new_id = max_factor_id + 1

    table_name = get_table_name(s)
    
    df = pd.DataFrame({
        "factor_id": [new_id],
        "factor_name": [factor_name],
        "creator": [creator],
        "create_date": [pd.to_datetime(create_date)],
        "is_verified": [0],
        "verifier": [None],
        "veirify_date": [pd.NaT],
        "update_frequency": [update_frequency],
        "formula": [formula],
        "description": [description],
        "lookback_days": [lookback_days],
        "last_update_time": [None],
        "method": [method],
        "needed_factor": [needed_factor],
        "store_factor": [table_name],
        "category": [category],
    })
    num = appender.append(df)
    print(f"因子 {factor_name} 已添加到factor_metadata表中")
    return table_name


# 向数据库中插入因子数据
def update_factor(s, factor_name, factor, table_name): 
    """
    更新因子数据
    """
    schema = s.run(f"schema(loadTable('dfs://YesLab', '{table_name}'))")
    all_columns = schema['colDefs'].name.tolist()
    
    # 构建需要处理的数据列表
    data_to_process = []
    for trade_date, row in factor.iterrows():
        for ts_code, value in row.items():
            data_to_process.append({
                "trade_date": trade_date,
                "ts_code": ts_code,
                "value": value
            })

    # 创建批处理容器
    insert_rows = []

    for data in tqdm(data_to_process):
        new_row = {col: None for col in all_columns}
        #print(new_row)
        new_row.update({
            "trade_date": data['trade_date'],
            "ts_code": data['ts_code'],
            "factor_name": factor_name,
            "value": data['value'],
        })
        insert_rows.append(new_row)

    # 执行批量插入
    if insert_rows:
        print(f"将插入 {len(insert_rows)} 条新数据到因子表")
        df_insert = pd.DataFrame(insert_rows)
        # 确保列顺序与数据库一致
        df_insert = df_insert[all_columns]
        df_insert = df_insert.replace({
            None: np.nan,
        })
        cmd = "tableInsert{loadTable('dfs://YesLab', '" + table_name + "')}"
        s.run(cmd, df_insert)

    print(f"因子 {factor_name} 在factor窄表中完成更新")
    return

# 在factor_metadata表中更新last_update_time
def set_last_update_time(s, factor_name):
    """
    更新因子元数据表中的last_update_time
    """
    script = f'''
        update loadTable('dfs://YesLab', 'factor_metadata')
        set last_update_time = date(`{pd.to_datetime('today').strftime('%Y.%m.%d')})
        where factor_name = "{factor_name}"
    '''
    s.run(script)
    print(f"将因子 {factor_name} 的 last_update_time 更新为当前时间")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    # 添加 --test_mode 参数
    parser.add_argument(
        '--test_mode', 
        '-m',  # 短参数形式
        action='store_true',  # 如果存在参数则为 True，否则为 False
        help='启用测试模式'
    )
    args = parser.parse_args()
    # 使用参数
    if args.test_mode:
        print("测试模式已启用，因子将不会被添加到数据库中")
        # 这里添加测试模式的逻辑
    else:
        print("正常模式，因子将被添加到数据库中，请谨慎操作")
        # 正常逻辑

    from dotenv import load_dotenv
    load_dotenv()  # 默认加载当前目录的 .env
    # 使用配置
    username = "zhangyudong"  # 从环境变量读取
    password = "123456"  # 从环境变量读取

    for factor_config in FACTOR_CONFIG:

        factor_name = factor_config["factor_name"]
        creator = factor_config["creator"]
        create_date = factor_config["create_date"]
        formula = factor_config["formula"]
        description = factor_config["description"]
        lookback_days = factor_config["lookback_days"]
        method = factor_config["method"]
        needed_factor = factor_config["needed_factor"] if "needed_factor" in factor_config else []
        category = factor_config["category"] if "category" in factor_config else ""

        #假设数据库因子全部从2020.01.01开始
        if factor_config['start_time'] == "":
            start_time = "2020.01.01"
        else:
            start_time = factor_config['start_time']
        if factor_config['end_time'] == "":
            end_time = datetime.now().strftime("%Y.%m.%d")
        else:
            end_time = factor_config['end_time']

        print(f"添加因子: {factor_name}, 方法: {method}, 起始时间: {start_time}, 结束时间: {end_time}")
        
        s = ddb.session()
        s.connect("10.176.55.211", 8848, username, password)


        calculators = discover_factor_calculators()

        if method not in calculators:
            print(calculators)
            print(f"未找到名为 {method} 的因子计算器")
        else:
            try:
                # 获取类并实例化
                calculator_class = calculators[method]
                calculator = calculator_class(s, lookback_days)
                factor = calculator.calculate(start_time, end_time)
                print(f"计算结果: {factor}")
            except Exception as e:
                print(f"实例化 {method} 时出错: {e}")
        
        if not args.test_mode:
            # 先添加因子到factor_metadata表
            judge_query = f'''
                db = database("dfs://YesLab")
                t = loadTable(db, 'factor_metadata')
                select * from t where factor_name = '{factor_name}'
            '''
            flag = s.run(judge_query)
            if flag.empty:
                table_name = add2metadata(s = s,
                                    factor_name = factor_name,
                                    creator = creator,
                                    create_date = create_date,
                                    formula = formula,
                                    description = description,
                                    lookback_days = lookback_days,
                                    method = method,
                                    needed_factor = needed_factor,
                                    category = category
                                    )
                print(f"新因子将存储在表: {table_name}")
            else:
                table_name = flag.iloc[0]['store_factor']
                print(f"该因子存储在表: {table_name}")
            # 将因子计算结果添加到数据库
            update_factor(s, factor_name, factor, table_name)
            
            # 最后在factor_metadata表中更新last_update_time
            set_last_update_time(s, factor_name)
    
    s.close()