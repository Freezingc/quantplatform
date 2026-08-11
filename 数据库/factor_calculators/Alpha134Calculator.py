from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha134Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要成交量数据
        
        # 3. 计算因子：(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME
        # 计算12天前的收盘价
        close_lag12 = close.shift(12)  # 相当于DELAY(CLOSE,12)
        
        # 计算价格变化率
        price_change_rate = (close - close_lag12) / close_lag12
        
        # 乘以成交量
        factor = price_change_rate * vol
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha134",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(CLOSE-DELAY(CLOSE,12))/DELAY(CLOSE,12)*VOLUME",
        "description": "12日价格涨幅率乘以当日成交量",
        "lookback_days": 11,
        "method": "Alpha134Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
