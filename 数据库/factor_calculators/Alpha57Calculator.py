from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha57Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)

        # 3. 实现因子计算逻辑
        # 计算9日最低价
        min_low_9 = low.rolling(window=9).min()
        
        # 计算9日最高价
        max_high_9 = high.rolling(window=9).max()
        
        # 计算(CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100
        numerator = close - min_low_9
        denominator = max_high_9 - min_low_9
        # 避免除以0的情况
        raw_factor = numerator / denominator.replace(0, 1) * 100
        
        # 计算3日简单移动平均(SMA)
        factor = raw_factor.rolling(window=3, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/3 if len(x) == 3 else x.mean()
        )
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha57",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)",
        "description": "alpha191因子57",
        "lookback_days": 69,
        "method": "Alpha57Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

