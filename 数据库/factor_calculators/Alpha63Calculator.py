from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha63Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算价格变化
        delta = close - close.shift(1)
        
        # 计算正变化和绝对变化
        positive_delta = delta.clip(lower=0)
        abs_delta = delta.abs()
        
        
        # 计算分子部分：SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)
        numerator = positive_delta.rolling(window=6, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*5)/6 if len(x) == 6 else x.mean()
        )
        # 计算分母部分：SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)
        denominator = abs_delta.rolling(window=6, min_periods=1).mean()
        
        # 计算因子值
        factor = numerator / denominator * 100
        
        # 4. 返回因子
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha63",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "SMA(MAX(CLOSE-DELAY(CLOSE,1),0),6,1)/SMA(ABS(CLOSE-DELAY(CLOSE,1)),6,1)*100",
        "description": "alpha191因子63",
        "lookback_days": 6d,
        "method": "Alpha63Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

