from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha94Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close","vol","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 添加成交量数据

        # 3. 实现因子计算逻辑
        # SUM((CLOSE>DELAY(CLOSE,1)?vol:(CLOSE<DELAY(CLOSE,1)?-vol:0)),30)
        
        # 计算前一日收盘价（DELAY(CLOSE,1)）
        delay_close = close.shift(1)
        
        # 判断条件：CLOSE > DELAY(CLOSE,1)
        condition_up = close > delay_close
        
        # 判断条件：CLOSE < DELAY(CLOSE,1)  
        condition_down = close < delay_close
        
        # 计算每日的符号成交量：(CLOSE>DELAY(CLOSE,1)?vol:(CLOSE<DELAY(CLOSE,1)?-vol:0))
        signed_vol = vol.copy()
        signed_vol[condition_up] = vol[condition_up]  # 上涨时取正成交量
        signed_vol[condition_down] = -vol[condition_down]  # 下跌时取负成交量
        signed_vol[~(condition_up | condition_down)] = 0  # 平盘时取0
        
        # 计算30日滚动求和
        factor = signed_vol.rolling(window=30, min_periods=1).sum()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha94",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:(CLOSE<DELAY(CLOSE,1)?-VOLUME:0)),30)",
        "description": "30日内每日成交量带方向性的累计和",
        "lookback_days": 30,
        "method": "Alpha94Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
