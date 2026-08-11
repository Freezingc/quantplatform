from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha158Calculator(FactorCalculator):
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
        
        # 3. 计算SMA(CLOSE, 15, 2)
        # SMA(CLOSE, 15, 2) 通常表示15日的简单移动平均，权重为2
        # 这里假设是15天的加权移动平均，权重为2
        sma_close_15_2 = close.rolling(window=15, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*13)/15 if len(x) == 15 else x.mean()
        )
        # 4. 计算分子部分
        # (HIGH - SMA(CLOSE, 15, 2)) - (LOW - SMA(CLOSE, 15, 2))
        # 可以简化为 HIGH - LOW
        high_minus_sma = high - sma_close_15_2
        low_minus_sma = low - sma_close_15_2
        numerator = high_minus_sma - low_minus_sma
        
        # 5. 计算因子值
        factor = numerator / close
        
        # 6. 确保factor的格式与输入数据一致
        # 保持相同的索引和列结构
        factor = factor.reindex_like(close)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha158",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((HIGH-SMA(CLOSE,15,2))-(LOW-SMA(CLOSE,15,2)))/CLOSE",
        "description": "衡量股价当日最高最低价差的相对强度",
        "lookback_days": 14,
        "method": "Alpha158Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
