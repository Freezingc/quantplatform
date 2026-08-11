from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha130Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount","high",'low']  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount / vol * 10
        
        # 3. 优化版本 - 使用向量化计算
        def vectorized_decay_linear(df, window):
            """向量化实现的线性衰减"""
            result = pd.DataFrame(index=df.index, columns=df.columns)
            weights = np.arange(1, window + 1)[::-1]  # 倒序权重
            weights = weights / weights.sum()
            
            for i in range(window - 1, len(df)):
                window_data = df.iloc[i-window+1:i+1]
                result.iloc[i] = (window_data.values.T * weights).sum(axis=1)
            
            return result
        
        # 第一部分计算
        hl_avg = (high + low) / 2
        mean_vol_40 = vol.rolling(40).mean()
        
        # 计算滚动相关系数
        corr1 = pd.DataFrame(index=hl_avg.index, columns=hl_avg.columns)
        for i in range(8, len(hl_avg)):
            window_hl = hl_avg.iloc[i-8:i+1]
            window_vol = mean_vol_40.iloc[i-8:i+1]
            corr1.iloc[i] = window_hl.corrwith(window_vol, axis=0)
        
        decay_corr1 = vectorized_decay_linear(corr1, 10)
        
        # 第二部分计算
        rank_vwap = vwap.rank(axis=1, pct=True)
        rank_vol = vol.rank(axis=1, pct=True)
        
        corr2 = pd.DataFrame(index=rank_vwap.index, columns=rank_vwap.columns)
        for i in range(6, len(rank_vwap)):
            window_vwap = rank_vwap.iloc[i-6:i+1]
            window_vol = rank_vol.iloc[i-6:i+1]
            corr2.iloc[i] = window_vwap.corrwith(window_vol, axis=0)
        
        decay_corr2 = vectorized_decay_linear(corr2, 3)
        
        # 最终因子计算
        factor1 = decay_corr1.rank(axis=1, pct=True)
        factor2 = decay_corr2.rank(axis=1, pct=True)
        
        # 避免除零
        factor2 = factor2.replace(0, np.nan)
        factor = factor1 / factor2
        
        # 移除NaN
        min_periods = 49  # 40 + 9
        factor = factor.iloc[min_periods:]
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha130",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(DECAYLINEAR(CORR(((HIGH + LOW) / 2), MEAN(VOLUME,40), 9), 10)) / RANK(DECAYLINEAR(CORR(RANK(VWAP), RANK(VOLUME), 7),3)))",
        "description": "通过比较价格中枢与成交量的长期相关性排名、与价格成交量排名的短期相关性排名的比值",
        "lookback_days": 56,
        "method": "Alpha130Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
