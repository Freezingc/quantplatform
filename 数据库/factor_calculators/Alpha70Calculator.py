from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha70Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        amount = data.xs('amount', level=0)  # 提取amount数据

        # 3. 实现因子计算逻辑 - STD(AMOUNT,6)
        # 计算每个标的的6日滚动标准差
        factor = amount.rolling(window=6,min_periods=1).std()
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha70",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "STD(AMOUNT,6)",
        "description": "alpha191因子70",
        "lookback_days": 5,
        "method": "Alpha70Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

