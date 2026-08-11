from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha73Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount * 10 / vol
        
        # 定义辅助函数
        def decay_linear(series, window):
            weights = np.arange(1, window+1)
            weights = weights / weights.sum()
            return series.rolling(window).apply(lambda x: np.sum(x * weights))
        
        def tsrank(series, window):
            return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # 计算第一部分：TSRANK(DECAYLINEAR(DECAYLINEAR(CORR(CLOSE, vol, 10), 16), 4), 5)
        # 1.1 计算CORR(CLOSE, vol, 10)
        corr_close_vol_10 = close.rolling(10).corr(vol)
        
        # 1.2 第一次DECAYLINEAR(窗口=16)
        decay1 = decay_linear(corr_close_vol_10, 16)
        
        # 1.3 第二次DECAYLINEAR(窗口=4)
        decay2 = decay_linear(decay1, 4)
        
        # 1.4 TSRANK(窗口=5)
        part1 = tsrank(decay2, 5)
        
        # 计算第二部分：RANK(DECAYLINEAR(CORR(VWAP, MEAN(vol,30), 4),3))
        # 2.1 计算MEAN(vol,30)
        mean_vol_30 = vol.rolling(window=30,min_periods=1).mean()
        
        # 2.2 计算CORR(VWAP, MEAN(vol,30), 4)
        corr_vwap_meanvol = vwap.rolling(window=4,min_periods=1).corr(mean_vol_30)
        
        # 2.3 DECAYLINEAR(窗口=3)
        decay_part2 = decay_linear(corr_vwap_meanvol, 3)
        
        # 2.4 RANK (横截面排名)
        def cross_sectional_rank(df):
            return df.rank(axis=1, pct=True)
        
        part2 = cross_sectional_rank(decay_part2)
        
        # 最终因子计算
        factor = (part1 - part2) * -1
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha73",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "((TSRANK(DECAYLINEAR(DECAYLINEAR(CORR((CLOSE), vol, 10), 16), 4), 5) - RANK(DECAYLINEAR(CORR(VWAP, MEAN(vol,30), 4),3))) * -1)",
        "description": "alpha191因子73",
        "lookback_days": 34,
        "method": "Alpha73Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

