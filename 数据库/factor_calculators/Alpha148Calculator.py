from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha148Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","open","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        
        # 计算 MEAN(vol, 60)
        mean_vol_60 = vol.rolling(window=60, min_periods=1).mean()
        
        # 计算 SUM(MEAN(vol, 60), 9)
        sum_mean_vol_9 = mean_vol_60.rolling(window=9, min_periods=1).sum()
        
        # 计算 CORR(OPEN, SUM(MEAN(vol,60), 9), 6)
        def rolling_corr(a, b, window):
            """计算两个DataFrame的滚动相关系数"""
            n = len(a)
            result = pd.DataFrame(np.nan, index=a.index, columns=a.columns)
            
            for i in range(window-1, n):
                start_idx = i - window + 1
                window_a = a.iloc[start_idx:i+1]
                window_b = b.iloc[start_idx:i+1]
                
                for col in a.columns:
                    if not (window_a[col].isna().any() or window_b[col].isna().any()):
                        result.iloc[i, result.columns.get_loc(col)] = window_a[col].corr(window_b[col])
            
            return result
        
        corr_6 = rolling_corr(open, sum_mean_vol_9, 6)
        
        # 计算横截面排名函数
        def cross_sectional_rank(df):
            return df.rank(axis=1, pct=True)
        
        # 计算 RANK(CORR(...))
        rank_corr = cross_sectional_rank(corr_6)
        
        # 计算 TSMIN(OPEN, 14)
        tsmin_open_14 = open.rolling(window=14, min_periods=1).min()
        
        # 计算 (OPEN - TSMIN(OPEN, 14)) 的排名
        open_diff = open - tsmin_open_14
        rank_open_diff = cross_sectional_rank(open_diff)
        
        # 计算最终因子
        factor = (rank_corr < rank_open_diff).astype(int) * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha148",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "STD(vol,20)",
        "description": "成交量20日标准差",
        "lookback_days": 9,
        "method": "Alpha148Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
