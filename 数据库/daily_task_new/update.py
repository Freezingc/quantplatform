import tushare as ts
from tqdm import tqdm
import dolphindb as ddb
import pandas as pd
from pandas.tseries.offsets import Day
import time
from datetime import datetime, date, timedelta
import os
from dotenv import load_dotenv
import importlib
import pkgutil
from pathlib import Path
import sys

# 动态导入数据更新函数
def discover_function():
    """
    动态发现并导入update_mission目录中的所有模块
    """
    
    package_path = Path(__file__).parent / "update_mission"

    if not package_path.exists():
        raise FileNotFoundError(f"目录不存在: {package_path}")
    
    modules = {}
    
    # 遍历目录下的所有 .py 文件（排除 __init__.py）
    for module_info in pkgutil.iter_modules([str(package_path)]):
        module_name = module_info.name
        
        # 跳过 __pycache__ 或其他非 .py 文件
        if module_name.startswith("_"):
            continue
        
        try:
            # 动态导入模块
            full_module_name = f"update_mission.{module_name}"
            module = importlib.import_module(full_module_name)
            modules[module_name] = module
        except ImportError as e:
            print(f"⚠️ 无法导入模块 {module_name}: {e}")
    
    return modules


update_dict = {
    "update_stock_info_table": "stock_info_table",
    "update_sw_ind_code": "sw_ind_code",
    "update_index_info_table": "index_info_table",
    "update_balance_sheet": "balance_sheet",
    "update_cashflow_table": "cashflow_table",
    "update_company_income": "company_income_table",
    "update_index_component": "index_component_table",
    "update_index_daily_data": "index_daily_data",
    "update_stock_daily_data": "stock_daily_data",
    "update_stock_factor_daily": "stock_factor_daily",
    "update_financial_metrics_table": "financial_metrics_table",
    "update_stock_used_name": "stock_used_name",
    "update_stock_trading_suspension_info": "stock_trading_suspension_info",
    "update_trade_calendar": "trade_calendar",
}
'''
update_dict = {
    "update_financial_metrics_table": "financial_metrics_table",
}'''
if __name__ == "__main__":
    load_dotenv()  # 默认加载当前目录的 .env

    username = os.getenv('dolphinDB_USERNAME')  # 从环境变量读取
    password = os.getenv('dolphinDB_PASSWORD')

    today_date = date.today()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{today_date}更新开始时间：{start_datetime}")

    # 连接 DolphinDB 数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)

    # 动态获取更新函数
    modules = discover_function()

    mission_list = []
    
    # 遍历每个更新任务
    print("正在获取更新数据")

    # 记录循环开始时间
    start_time = datetime.now()
    max_runtime = timedelta(hours=20)  # 最大允许运行时间：20小时

    for update_mission, table_name in tqdm(update_dict.items()):
        print(f"{update_mission}正在获取数据")
        try:
            # 检查是否超时
            current_time = datetime.now()
            if current_time - start_time > max_runtime:
                raise TimeoutError("任务执行时间超过 20 小时，强制终止")

            # 创建表 appender
            appender = ddb.TableAppender(dbPath='dfs://YesLab', tableName=table_name, ddbSession=s)

            if update_mission in modules:  # 先检查字典里是否有这个模块
                module = modules[update_mission]  # 获取模块对象
                if hasattr(module, update_mission) and callable(getattr(module, update_mission)):
                    delete_query, data = getattr(module, update_mission)()  # 执行函数
                else:
                    raise AttributeError(f"模块 {update_mission} 中没有可调用的 {update_mission} 函数")
            else:
                raise AttributeError(f"模块 {update_mission} 不存在")

            mission_list.append(
                {
                    "name": update_mission,
                    "appender": appender,
                    "delete_query": delete_query,
                    "data": data,
                }
            )
        except TimeoutError as e:
            # 捕获超时异常并终止
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⏰ {update_mission} 数据获取执行超时: {e}, 结束时间：{end_datetime}")
            sys.exit(1)
        except Exception as e:
            # 捕获其他异常并终止
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"❌ {update_mission} 数据获取执行报错: {e}, 结束时间：{end_datetime}")
            sys.exit(1)
        print(f"{update_mission}数据获取完成")

    print("✅ 所有数据获取完成，未超时")

    print(mission_list)
    
    print("正在更新数据")
    for mission in tqdm(mission_list):
        print(f"{mission['name']}正在更新数据")
        if mission["data"].empty:
            print(f"    {mission['name']}数据为空")
            continue
        else:
            print(f"    {mission['name']}数据量: {len(mission['data'])}")
        if mission["delete_query"] != "":
            s.run(mission["delete_query"])
        mission["appender"].append(mission["data"])
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{today_date}更新结束, 结束时间：{end_datetime}")
