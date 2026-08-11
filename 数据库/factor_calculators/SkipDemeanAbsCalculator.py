from .base import FactorCalculator
import numpy as np
import pandas as pd

class SkipDemeanAbsCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open_price = data.xs('open', level=0)
        close = data.xs('close', level=0)
        
        # 3. 计算前一日收盘价
        pre_close = close.shift(1)
        
        # 4. 计算隔夜跳空 = open / pre_close
        skip = open_price / pre_close
        
        # 5. 计算每行（每个交易日）的均值
        skip_mean = skip.mean(axis=1)
        
        # 6. 计算跳空减去均值的绝对值 = abs(skip - skip_mean)
        skip_demean_abs = abs(skip.sub(skip_mean, axis=0))
        
        # 7. 处理可能的无穷大值和缺失值
        skip_demean_abs = skip_demean_abs.replace([np.inf, -np.inf], np.nan)
        
        return skip_demean_abs

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "跳空去均值绝对值",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "ABS(open / pre_close - MEAN(open / pre_close, axis=1))",
    "description": "跳空去均值绝对值，计算隔夜跳空相对于当日市场平均跳空的偏离程度",
    "lookback_days": 1,
    "method": "SkipDemeanAbsCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串"",
    "category": "波动率因子"
}
'''

