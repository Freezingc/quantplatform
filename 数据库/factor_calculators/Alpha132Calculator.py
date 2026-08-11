from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha132Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        amount = data.xs('amount', level=0)  # 添加成交额数据

        # 3. 实现因子计算逻辑 - MEAN(AMOUNT,20) 20日移动平均
        # 使用rolling窗口计算移动平均
        factor = amount.rolling(window=20, min_periods=1).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha132",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN(AMOUNT,20)",
        "description": "计算每只股票成交额的20日移动平均",
        "lookback_days": 19,
        "method": "Alpha132Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "流动性因子"
    }
'''
