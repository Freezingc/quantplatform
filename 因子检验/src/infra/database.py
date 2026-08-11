"""
数据库连接和查询管理模块
"""

import dolphindb as ddb
import pandas as pd
from typing import Optional, List
import logging

from ..config import DATABASE_CONFIG, BENCHMARK_MAPPING

logger = logging.getLogger(__name__)



class DatabaseManager:
    """数据库连接和查询管理器"""

    def __init__(self):
        self.session = None
        self.connect()


    def connect(self):
        self.session = ddb.session()
        self.session.connect(
            DATABASE_CONFIG["host"],
            DATABASE_CONFIG["port"],
            DATABASE_CONFIG["username"],
            DATABASE_CONFIG["password"]
        )
        logger.info("成功连接DolphinDB数据库")


    @staticmethod
    def date_format_transform(date: str) -> str:
        """转换日期格式"""
        return '`' + '.'.join(date.split('-'))
    

    def get_stock_basic_info(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票基本信息"""
        query = f'''
            db = database("dfs://YesLab")
            t1 = loadTable(db, "stock_daily_data")
            t2 = loadTable(db, "stock_info_table")
            
            WITH filtered_t1 AS (
                SELECT ts_code, trade_date, low, high, open, close
                FROM t1
                WHERE trade_date BETWEEN date({self.date_format_transform(start_date)}) AND date({self.date_format_transform(end_date)})
            )
            SELECT 
                t2.name, 
                t2.fullname, 
                filtered_t1.trade_date, 
                filtered_t1.ts_code, 
                t2.exchange,
                t2.list_date, 
                filtered_t1.low, 
                filtered_t1.high, 
                filtered_t1.open, 
                filtered_t1.close
            FROM filtered_t1
            JOIN t2 ON filtered_t1.ts_code = t2.ts_code
        '''
        
        result = self.session.run(query)
        result.rename(columns={'ts_code': 'instrument', 'trade_date': 'date'}, inplace=True)
        return result.drop_duplicates(keep='first')
    

    def get_stock_industry_info(self) -> pd.DataFrame:
        """获取股票行业信息"""
        query = f'''
            db = database("dfs://YesLab")
            t = loadTable(db, "code_index")  
            SELECT l1_code, l1_name, ts_code FROM t;
        '''
        
        result = self.session.run(query)
        result.rename(columns={'ts_code': 'instrument'}, inplace=True)
        return result.drop_duplicates(keep='first')
    

    def get_market_cap_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取市值数据"""
        query = f'''
            db = database("dfs://YesLab")
            t = loadTable(db, "stock_factor_daily")
            SELECT ts_code, trade_date, amount, total_mv
            FROM t
            WHERE trade_date BETWEEN date({self.date_format_transform(start_date)}) AND date({self.date_format_transform(end_date)});
        '''
        
        result = self.session.run(query)
        result.rename(columns={'trade_date': 'date', 'ts_code': 'instrument'}, inplace=True)
        return result.drop_duplicates(keep='first')
    

    def compute_returns(self, price_data: pd.DataFrame, name: str, horizon: int = 1) -> pd.DataFrame:
        """
        基于价格数据计算指定持有期的收益率, 其列名为name, 持有期为horizon
        PS: 目前统一使用【开盘价】, 例如仅持有1天, T0的因子对应的收益率是: T+1日开盘买入,T+2开盘卖出
        """
        result = price_data.copy()
        result[name] = (
            result.groupby('ts_code')['open'].shift(-(horizon+1)) /
            result.groupby('ts_code')['open'].shift(-1) - 1
        )
        
        result = result.dropna(subset=[name])[['ts_code', 'trade_date', name]].rename(
            columns={'ts_code': 'instrument', 'trade_date': 'date'}
        )
        return result


    def get_daily_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取原始日价格数据"""
        query = f'''
            db = database("dfs://YesLab")
            stock_daily = loadTable(db, "stock_factor_daily")
            
            SELECT ts_code, trade_date, open
            FROM stock_daily
            WHERE trade_date BETWEEN date({self.date_format_transform(start_date)}) AND date({self.date_format_transform(end_date)})
        '''
        
        raw_data = self.session.run(query)
        return raw_data.drop_duplicates(keep='first').sort_values(['ts_code', 'trade_date'])


    def get_benchmark_price_data(self, benchmark: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取基准收益率数据"""
        if benchmark not in BENCHMARK_MAPPING:
            raise ValueError(f"不支持的基准: {benchmark}")
        
        bm_code = BENCHMARK_MAPPING[benchmark]

        query = f"""
            db = database("dfs://YesLab")
            t = loadTable(db, "index_daily_data")

            SELECT ts_code, trade_date, open
            FROM t
            WHERE ts_code = "{bm_code}"
                AND trade_date BETWEEN date({self.date_format_transform(start_date)}) AND date({self.date_format_transform(end_date)});
        """
        
        raw_data = self.session.run(query)
        return raw_data.drop_duplicates(keep='first').sort_values(['ts_code', 'trade_date'])
    

    def close(self):
        """关闭数据库连接"""
        if self.session and not self.session.isClosed():
            self.session.close()
            logger.info("数据库连接已关闭")

    
    def __del__(self):
        """析构函数，确保数据库连接关闭"""
        self.close()