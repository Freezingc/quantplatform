from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha71Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算24日均值
        mean_close_24 = close.rolling(window=24,min_periods=1).mean()
        
        # 计算因子：(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100
        factor = (close - mean_close_24) / mean_close_24 * 100
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha71",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(CLOSE-MEAN(CLOSE,24))/MEAN(CLOSE,24)*100",
        "description": "alpha191因子71",
        "lookback_days": 23,
        "method": "Alpha71Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

