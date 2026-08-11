from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha61Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "vol", "low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount * 10 / vol
        
        # 计算第一部分：MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12))
        # 计算DELTA(VWAP, 1)
        delta_vwap = vwap.diff(1)
        
        # 计算DECAYLINEAR(DELTA(VWAP, 1), 12)
        def decay_linear(series, window):
            weights = np.arange(1, window+1)
            weights = weights / weights.sum()
            return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)
        
        decay_delta_vwap = delta_vwap.apply(lambda x: decay_linear(x, 12))
        
        # 计算RANK
        rank_part1 = decay_delta_vwap.rank(axis=1, pct=True)
        
        # 计算第二部分：RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(vol,80), 8)), 17))
        # 计算MEAN(vol,80)
        mean_vol = vol.rolling(window=80,min_periods=1).mean()
        
        # 计算CORR(LOW, MEAN(vol,80), 8)
        def rolling_corr(df1, df2, window):
            corr_df = pd.DataFrame(index=df1.index, columns=df1.columns)
            for col in df1.columns:
                corr_df[col] = df1[col].rolling(window).corr(df2[col])
            return corr_df
        
        corr_low_vol = rolling_corr(low, mean_vol, 8)
        
        # 计算RANK(CORR)
        rank_corr = corr_low_vol.rank(axis=1, pct=True)
        
        # 计算DECAYLINEAR(RANK(CORR), 17)
        decay_rank_corr = rank_corr.apply(lambda x: decay_linear(x, 17))
        
        # 计算RANK
        rank_part2 = decay_rank_corr.rank(axis=1, pct=True)
        
        # 计算MAX(part1, part2)
        max_rank = pd.DataFrame(np.maximum(rank_part1.values, rank_part2.values), 
                            index=rank_part1.index, 
                            columns=rank_part1.columns)
        
        # 最终因子 = MAX * -1
        factor = max_rank * -1
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha61",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(MAX(RANK(DECAYLINEAR(DELTA(VWAP, 1), 12)), RANK(DECAYLINEAR(RANK(CORR((LOW),MEAN(vol,80), 8)), 17))) * -1)",
        "description": "alpha191因子61",
        "lookback_days": 102,
        "method": "Alpha61Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

