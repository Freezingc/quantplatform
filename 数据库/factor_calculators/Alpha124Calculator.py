from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha124Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算 VWAP (成交量加权平均价)
        # 由于没有volume数据，使用典型价格代替：TP = (high + low + close) / 3
        # 假设用典型价格作为近似
        typical_price = (high + low + close) / 3
        
        # 4. 计算 CLOSE - VWAP
        close_minus_vwap = close - typical_price
        
        # 5. 计算 TSMAX(CLOSE, 30) - 过去30天的最大值
        tsmax_close_30 = close.rolling(window=30, min_periods=1).max()
        
        # 6. 计算 RANK - 横截面排名
        def cross_sectional_rank(df):
            # 对每个时间点，计算横截面排名
            return df.rank(axis=1, pct=True)
        
        rank_tsmax = cross_sectional_rank(tsmax_close_30)
        

        # 使用rolling apply进行向量化计算
        def decay_linear(df, period):
            """向量化的DECAYLINEAR计算"""
            weights = np.arange(1, period + 1)
            weights = weights / weights.sum()
            
            def apply_decay(series):
                if len(series) >= period:
                    return np.dot(series[-period:].values, weights)
                else:
                    available_weights = np.arange(1, len(series) + 1)
                    available_weights = available_weights / available_weights.sum()
                    return np.dot(series.values, available_weights)
            
            return df.rolling(window=period, min_periods=1).apply(apply_decay, raw=False)
        
        # 对每个股票分别计算 DECAYLINEAR
        decay_result = rank_tsmax.copy()
        for col in decay_result.columns:
            decay_result[col] = decay_linear(rank_tsmax[col], 2)
        
        # 8. 计算最终因子
        # 避免除零，添加小的偏移量
        factor = close_minus_vwap / (decay_result + 1e-8)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha124",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(CLOSE - VWAP) / DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)",
        "description": "这个因子衡量当前收盘价相对典型价格的偏离程度，经过过去30日最高收盘价的横截面排名并进行近期加权衰减调整后的标准化度量",
        "lookback_days": 30,
        "method": "Alpha124Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
