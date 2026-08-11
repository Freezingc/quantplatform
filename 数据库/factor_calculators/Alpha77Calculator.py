from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha77Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        amount = data.xs('amount', level=0)
        vol = data.xs('vol', level=0)
        vwap = amount * 10 / vol
        
        # 计算第一部分: ((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH))
        part1 = ((high + low) / 2 + high) - (vwap + high)
        
        # 计算DECAYLINEAR(part1, 20)
        def decay_linear(series, window):
            weights = np.arange(window, 0, -1)
            weights = weights / weights.sum()
            return series.rolling(window).apply(lambda x: np.dot(x, weights))
        
        decay_part1 = decay_linear(part1, 20)
        
        # 计算第一部分的RANK
        rank_part1 = decay_part1.rank(axis=1, pct=True)
        
        # 计算第二部分: CORR(((HIGH + LOW) / 2), MEAN(vol,40), 3)
        hl_mean = (high + low) / 2
        vol_mean = vol.rolling(40).mean()
        
        def rolling_corr(a, b, window):
            return a.rolling(window).corr(b)
        
        corr_part = rolling_corr(hl_mean, vol_mean, 3)
        
        # 计算DECAYLINEAR(corr_part, 6)
        decay_corr = decay_linear(corr_part, 6)
        
        # 计算第二部分的RANK
        rank_part2 = decay_corr.rank(axis=1, pct=True)
        
        # 计算最终的MIN
        factor = np.minimum(rank_part1, rank_part2)
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha77",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "MIN(RANK(DECAYLINEAR(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 20)), RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(vol,40), 3), 6)))",
        "description": "alpha191因子77",
        "lookback_days": 46,
        "method": "Alpha77Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

