from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha126Calculator(FactorCalculator):
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
        
        # 3. 实现因子计算逻辑
        # 计算 (CLOSE + HIGH + LOW) / 3
        factor = (close + high + low) / 3
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha126",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(CLOSE+HIGH+LOW)/3",
        "description": "典型价格，它代表了每个交易日内股票价格的平均水平",
        "lookback_days": 0,
        "method": "Alpha126Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
