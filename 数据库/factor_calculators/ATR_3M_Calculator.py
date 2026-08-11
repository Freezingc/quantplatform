from .base import FactorCalculator
import numpy as np
import pandas as pd

class ATR_3M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "high", "low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        low = data.xs('low', level=0)
        
        # 3. 计算ATR真实波动率
        pre_close = close.shift(1)
        ATR = (high - low).combine(abs(pre_close - high), np.maximum).combine(abs(pre_close - low), np.maximum)
        ATR = ATR.replace([np.inf, -np.inf], np.nan)
        
        # 4. 计算3个月真实波动率（60个交易日）
        ATR_3M = ATR.rolling(60).mean()
        ATR_3M = ATR_3M.replace([np.inf, -np.inf], np.nan)
        
        return ATR_3M

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "3个月真实波动率",
    "creator": "suyanning",
    "create_date": "2025.09.12",
    "formula": "SMA(ATR, 60) where ATR = MAX(high-low, ABS(pre_close-high), ABS(pre_close-low))",
    "description": "3个月真实波动率",
    "lookback_days": 60,
    "method": "ATR_3M_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
