from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha115Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量列名为'vol'
        
        # 3. 定义辅助函数
        def calculate_tsrank(series, window):
            """计算时间序列排名"""
            ranks = series.rolling(window=window, min_periods=window)
            return ranks.apply(lambda x: pd.Series(x).rank().iloc[-1] / window)
        
        def calculate_cross_sectional_rank(df):
            """计算横截面排名"""
            return df.rank(axis=1, pct=True, na_option='keep')
        
        # 4. 计算因子
        # 4.1 计算第一个相关系数
        part1 = high * 0.9 + close * 0.1
        mean_vol_30 = vol.rolling(window=30, min_periods=30).mean()
        corr1 = part1.rolling(window=10, min_periods=10).corr(mean_vol_30)
        
        # 4.2 计算第二个相关系数
        hl_avg = (high + low) / 2
        tsrank_hl = calculate_tsrank(hl_avg, 4)
        tsrank_vol = calculate_tsrank(vol, 10)
        corr2 = tsrank_hl.rolling(window=7, min_periods=7).corr(tsrank_vol)
        
        # 4.3 计算横截面排名
        rank_corr1 = calculate_cross_sectional_rank(corr1)
        rank_corr2 = calculate_cross_sectional_rank(corr2)
        
        # 4.4 计算最终因子
        # 处理可能出现的负数底数问题
        factor = rank_corr1.clip(lower=0.0001) ** rank_corr2
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha115",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(CORR(((HIGH * 0.9) + (CLOSE * 0.1)), MEAN(VOLUME,30), 10))^RANK(CORR(TSRANK(((HIGH + LOW) / 2), 4), TSRANK(VOLUME, 10), 7)))",
        "description": "通过计算股价加权序列与成交量均值的相关性排名，再以股价均值与成交量的时间序列排名的相关性排名为指数，构建了一个结合价格动量、成交量信息及它们之间非线性关系的复合技术指标",
        "lookback_days": 38,
        "method": "Alpha115Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
