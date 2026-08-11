from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha87Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "close", "high", "open", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol
        
        # 4. 实现辅助函数
        def delta(series, period):
            return series.diff(period)
        
        def decay_linear(series, window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            def apply_decay(window_values):
                if len(window_values) < window:
                    return np.nan
                return np.dot(window_values, weights)
            
            return series.rolling(window).apply(apply_decay, raw=True)
        
        def ts_rank(series, window):
            def rank_func(window_values):
                if len(window_values) < window:
                    return np.nan
                return rankdata(window_values)[-1] / len(window_values)
            
            return series.rolling(window).apply(rank_func, raw=True)
        
        def cross_sectional_rank(series):
            # 横截面排名
            return series.rank(axis=1, pct=True)
        
        # 5. 计算第一部分：RANK(DECAYLINEAR(DELTA(VWAP, 4), 7))
        delta_vwap = delta(vwap, 4)
        decay_delta_vwap = decay_linear(delta_vwap, 7)
        part1 = cross_sectional_rank(decay_delta_vwap)
        
        # 6. 计算第二部分：TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)
        # 简化表达式：LOW * 0.9 + LOW * 0.1 = LOW
        numerator = low - vwap
        denominator = open - ((high + low) / 2)
        
        # 处理分母为0的情况
        denominator = denominator.replace(0, np.nan)
        ratio = numerator / denominator
        
        decay_ratio = decay_linear(ratio, 11)
        part2 = ts_rank(decay_ratio, 7)
        
        # 7. 合并两部分并取负值
        factor = (part1 + part2) * -1
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha87",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "((RANK(DECAYLINEAR(DELTA(VWAP, 4), 7)) + TSRANK(DECAYLINEAR(((((LOW * 0.9) + (LOW * 0.1)) - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)) * -1)",
        "description": "alpha191因子87",
        "lookback_days": 10,
        "method": "Alpha87Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

