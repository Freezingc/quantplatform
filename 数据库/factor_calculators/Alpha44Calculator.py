from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha44Calculator(FactorCalculator):
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

        # 实现因子计算逻辑
        # 第一部分: TSRANK(DECAYLINEAR(CORR(LOW, MEAN(vol,10), 7), 6), 4)
        
        # 计算MEAN(vol, 10)
        mean_vol_10 = vol.rolling(window=10).mean()
        
        # 计算CORR(LOW, MEAN(vol,10), 7)
        corr_low_meanvol = low.rolling(window=7).corr(mean_vol_10)
        
        # 计算DECAYLINEAR(CORR(...), 6)
        def decay_linear(series, window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            def apply_decay(window_values):
                if len(window_values) < window:
                    return np.nan
                return np.dot(window_values, weights)
            
            return series.rolling(window=window).apply(apply_decay, raw=True)
        
        decay_corr = decay_linear(corr_low_meanvol, 6)
        
        # 计算TSRANK(DECAYLINEAR(...), 4)
        def ts_rank(series, window):
            def calculate_rank(window_values):
                if len(window_values) < window or np.isnan(window_values).any():
                    return np.nan
                return pd.Series(window_values).rank(pct=True).iloc[-1]
            
            return series.rolling(window=window).apply(calculate_rank, raw=True)
        
        tsrank_part1 = ts_rank(decay_corr, 4)
        
        # 第二部分: TSRANK(DECAYLINEAR(DELTA(VWAP, 3), 10), 15)
        
        # 计算DELTA(VWAP, 3)
        delta_vwap = vwap.diff(3)
        
        # 计算DECAYLINEAR(DELTA(...), 10)
        decay_delta = decay_linear(delta_vwap, 10)
        
        # 计算TSRANK(DECAYLINEAR(...), 15)
        tsrank_part2 = ts_rank(decay_delta, 15)
        
        # 合并两部分
        factor = tsrank_part1 + tsrank_part2
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha44",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "(TSRANK(DECAYLINEAR(CORR(((LOW )), MEAN(vol,10), 7), 6),4) + TSRANK(DECAYLINEAR(DELTA((VWAP), 3), 10), 15))",
    "description": "alpha191因子44",
    "lookback_days": 26,
    "method": "Alpha44Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''