from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha119Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","open","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount / vol * 10
        
        # 3. 预定义计算函数
        def cross_section_rank(df):
            """横截面排名"""
            return df.rank(axis=1, pct=True)
        
        def ts_rank_vectorized(series, window):
            """向量化的时间序列排名"""
            if isinstance(series, pd.Series):
                series = pd.DataFrame(series)
            
            result = pd.DataFrame(index=series.index, columns=series.columns)
            for i in range(window-1, len(series)):
                if i >= window-1:
                    window_data = series.iloc[i-window+1:i+1]
                    ranks = window_data.rank(axis=0, pct=True)
                    result.iloc[i] = ranks.iloc[-1]
            return result
        
        def decay_linear_vectorized(df, window):
            """向量化的线性衰减加权平均"""
            weights = np.arange(1, window + 1)[::-1]  # 递减权重
            weights = weights / weights.sum()
            
            result = pd.DataFrame(index=df.index, columns=df.columns)
            for i in range(window-1, len(df)):
                if i >= window-1:
                    window_data = df.iloc[i-window+1:i+1]
                    result.iloc[i] = (window_data * weights.reshape(-1, 1)).sum(axis=0)
            return result
        
        def rolling_corr_vectorized(df1, df2, window):
            """向量化的滚动相关系数"""
            result = pd.DataFrame(index=df1.index, columns=df1.columns)
            
            for i in range(window-1, len(df1)):
                if i >= window-1:
                    window1 = df1.iloc[i-window+1:i+1]
                    window2 = df2.iloc[i-window+1:i+1]
                    
                    # 计算相关系数
                    for col in df1.columns:
                        if not (window1[col].isna().any() or window2[col].isna().any()):
                            corr = np.corrcoef(window1[col], window2[col])[0, 1]
                            result.iat[i, df1.columns.get_loc(col)] = corr
            return result
        
        # 4. 计算第一部分
        # MEAN(VOLUME, 5)
        mean_vol_5 = vol.rolling(5, min_periods=5).mean()
        
        # SUM(MEAN(VOLUME,5), 26)
        sum_mean_vol_5_26 = mean_vol_5.rolling(26, min_periods=26).sum()
        
        # CORR(VWAP, SUM(...), 5)
        corr_part1 = rolling_corr_vectorized(vwap, sum_mean_vol_5_26, 5)
        
        # DECAYLINEAR(CORR(...), 7)
        decay_part1 = decay_linear_vectorized(corr_part1, 7)
        
        # RANK(...)
        rank_part1 = cross_section_rank(decay_part1)
        
        # 5. 计算第二部分
        # RANK(OPEN)
        rank_open = cross_section_rank(open)
        
        # MEAN(VOLUME, 15) 和 RANK(MEAN(...))
        mean_vol_15 = vol.rolling(15, min_periods=15).mean()
        rank_mean_vol_15 = cross_section_rank(mean_vol_15)
        
        # CORR(RANK(OPEN), RANK(...), 21)
        corr_part2 = rolling_corr_vectorized(rank_open, rank_mean_vol_15, 21)
        
        # MIN(CORR(...), 9)
        min_corr_part2 = corr_part2.rolling(9, min_periods=9).min()
        
        # TSRANK(MIN(...), 7)
        tsrank_part2 = ts_rank_vectorized(min_corr_part2, 7)
        
        # DECAYLINEAR(TSRANK(...), 8)
        decay_part2 = decay_linear_vectorized(tsrank_part2, 8)
        
        # RANK(...)
        rank_part2 = cross_section_rank(decay_part2)
        
        # 6. 计算最终因子
        factor = rank_part1 - rank_part2
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha119",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(DECAYLINEAR(CORR(VWAP, SUM(MEAN(VOLUME,5), 26), 5), 7)) - RANK(DECAYLINEAR(TSRANK(MIN(CORR(RANK(OPEN), RANK(MEAN(VOLUME,15)), 21), 9), 7), 8)))",
        "description": "通过比较“量价相关性的趋势强度”与“开盘价-成交量排名的相关性的时序稳定性”的横截面相对强弱，捕捉量价关系中的动量与反转效应差异",
        "lookback_days": 55,
        "method": "Alpha119Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
