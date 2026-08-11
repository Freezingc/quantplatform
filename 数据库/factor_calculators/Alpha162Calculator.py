from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha162Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 计算因子逻辑
        # 计算价格变化
        price_change = close - close.shift(1)
        
        # 计算正变化和绝对变化
        positive_change = np.maximum(price_change, 0)
        abs_change = np.abs(price_change)
        
        
        # 计算SMA
        sma_positive = positive_change.rolling(window=12, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/12 if len(x) == 12 else x.mean()
        )
        sma_abs = abs_change.rolling(window=12, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/12 if len(x) == 12 else x.mean()
        )
        
        # 避免除以零
        sma_abs = sma_abs.replace(0, np.nan)
        
        # 计算RSI-like指标
        rsi_like = (sma_positive / sma_abs) * 100
        
        # 计算过去12期的最小值和最大值
        min_rsi = rsi_like.rolling(window=12, min_periods=1).min()
        max_rsi = rsi_like.rolling(window=12, min_periods=1).max()
        
        # 计算最终的因子值
        numerator = rsi_like - min_rsi
        denominator = max_rsi - min_rsi
        
        # 避免分母为零
        denominator = denominator.replace(0, np.nan)
        
        factor = numerator / denominator
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha162",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*162-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*162,12))/(MAX(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*162,12)-MIN(SMA(MAX(CLOSE-DELAY(CLOSE,1),0),12, 1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),12,1)*162,12))",
        "description": "基于RSI动量指标的标准化版本，表示当前RSI值在过去12天RSI值范围中的相对位置",
        "lookback_days": 12,
        "method": "Alpha162Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
