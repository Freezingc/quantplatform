from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha53Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算每日收盘价是否大于前一日收盘价
        close_up = (close > close.shift(1))
        
        # 计算过去12天内上涨天数的占比，并乘以100
        factor = close_up.rolling(window=12, min_periods=1).sum() / 12 * 100
        
        return factor
'''
# 添加因子信息
    {
        "factor_name": "Alpha53",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100",
        "description": "alpha191因子53",
        "lookback_days": 12,
        "method": "Alpha53Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
