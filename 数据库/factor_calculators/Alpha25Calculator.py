from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha25Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算RET (收益率)
        ret = close.pct_change()
        
        # 4. 实现DECAYLINEAR函数
        def decay_linear(series, window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            
            def apply_decay(window_values):
                if len(window_values) < window:
                    return np.nan
                return np.dot(window_values, weights)
            
            return series.rolling(window).apply(apply_decay, raw=True)
        
        # 5. 计算因子各个部分
        # 5.1 计算MEAN(vol,20)
        mean_vol_20 = vol.rolling(20).mean()
        
        # 5.2 计算vol / MEAN(vol,20)
        vol_ratio = vol / mean_vol_20
        
        # 5.3 计算DECAYLINEAR((vol / MEAN(vol,20)), 9)
        decay_vol = vol_ratio.apply(lambda col: decay_linear(col, 9), axis=0)
        
        # 5.4 计算RANK(DECAYLINEAR(...))
        rank_decay = decay_vol.rank(axis=1, pct=True)
        
        # 5.5 计算DELTA(CLOSE, 7)
        delta_close_7 = close.diff(7)
        
        # 5.6 计算DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR(...)))
        part1 = delta_close_7 * (1 - rank_decay)
        
        # 5.7 计算RANK(part1)
        rank_part1 = part1.rank(axis=1, pct=True)
        
        # 5.8 计算SUM(RET, 250)
        sum_ret_250 = ret.rolling(250).sum()
        
        # 5.9 计算RANK(SUM(RET, 250))
        rank_sum_ret = sum_ret_250.rank(axis=1, pct=True)
        
        # 6. 组合最终因子
        factor = (-1 * rank_part1) * (1 + rank_sum_ret)
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha25",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "((-1 * RANK((DELTA(CLOSE, 7) * (1 - RANK(DECAYLINEAR((vol / MEAN(vol,20)), 9)))))) * (1 + RANK(SUM(RET, 250)))))",
    "description": "alpha191因子25​",
    "lookback_days": 250,
    "method": "Alpha25Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''