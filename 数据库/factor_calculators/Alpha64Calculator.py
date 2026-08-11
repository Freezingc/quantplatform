from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha64Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)
        close = data.xs('close', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount * 10 / vol
        
        # 定义辅助函数
        def rank(df):
            return df.rank(axis=1, pct=True)
        
        def decay_linear(df, window):
            weights = np.arange(1, window+1)
            weights = weights / weights.sum()
            return df.rolling(window=window).apply(lambda x: np.dot(x, weights))
        
        def corr(df1, df2, window):
            return df1.rolling(window=window).corr(df2)
        
        def mean(df, window):
            return df.rolling(window=window,min_periods=1).mean()
        
        # 计算第一部分: RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(vol), 4), 4))
        rank_vwap = rank(vwap)
        rank_vol = rank(vol)
        corr_part = corr(rank_vwap, rank_vol, 4)
        decay_corr = decay_linear(corr_part, 4)
        part1 = rank(decay_corr)
        
        # 计算第二部分: RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(vol,60)), 4), 13), 14))
        rank_close = rank(close)
        mean_vol = mean(vol, 60)
        rank_mean_vol = rank(mean_vol)
        corr_part2 = corr(rank_close, rank_mean_vol, 4)
        max_corr = corr_part2.rolling(window=13).max()
        decay_max_corr = decay_linear(max_corr, 14)
        part2 = rank(decay_max_corr)
        
        # 计算最终因子: MAX(part1, part2) * -1
        factor = np.maximum(part1, part2) * -1
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha64",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(MAX(RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(vol), 4), 4)), RANK(DECAYLINEAR(MAX(CORR(RANK(CLOSE), RANK(MEAN(vol,60)), 4), 13), 14))) * -1)",
        "description": "alpha191因子64",
        "lookback_days": 87,
        "method": "Alpha64Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

