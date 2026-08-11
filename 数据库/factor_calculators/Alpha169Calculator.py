from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha169Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # 步骤1: 计算收盘价差值
        close_diff = close - close.shift(1)
        
        # 步骤2: 计算SMA(差值, 9, 1) - 使用pandas的ewm方法模拟SMA
        # SMA(price, n, m) 通常可以通过 α = m/n 的指数加权平均来近似
        sma_9_1 = close_diff.rolling(window=9, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*8)/9 if len(x) == 9 else x.mean()
        )
        
        # 步骤3: 延迟1天
        delay_sma = sma_9_1.shift(1)
        
        # 步骤4: 计算12日和26日平均值
        mean_12 = delay_sma.rolling(window=12, min_periods=1).mean()
        mean_26 = delay_sma.rolling(window=26, min_periods=1).mean()
        
        # 步骤5: 计算差值
        diff = mean_12 - mean_26
        
        # 步骤6: 计算最终的SMA(差值, 10, 1)
        factor = diff.ewm(span=10, adjust=False).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha169",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1),12)-MEAN(DELAY(SMA(CLOSE-DELAY(CLOSE,1),9,1),1), 26),10,1)",
        "description": "基于收盘价动量变化率的双重平滑MACD型动量因子",
        "lookback_days": 44,
        "method": "Alpha169Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
