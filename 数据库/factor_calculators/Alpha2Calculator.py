from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha2Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high"]  # 指定该因子需要的原始因子


    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)

        # 3. 实现因子计算逻辑
        numerator = ((close - low) - (high - close))
        denominator = (high - low)
        ratio = numerator / denominator
        delta = ratio.diff(axis=0)  # 计算1期差分，axis=1表示按列(时间)差分
        factor = -1 * delta

        factor = factor.reindex(index=close.index, columns=close.columns)

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha2",
    "creator": "chenxutian",
    "create_date": "2025.08.20",
    "formula": "(-1 * DELTA((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)), 1))",
    "description": "alpha191因子2",
    "lookback_days": 1,
    "method": "Alpha2Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2025.08.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''