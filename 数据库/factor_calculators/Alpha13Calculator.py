from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha13Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "low", "amount", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        vol = data.xs('vol', level=0)       # 添加成交量数据，假设字段名为'vol'
        amount = data.xs('amount', level=0) # 添加成交金额数据，假设字段名为'amount'
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算几何平均 (HIGH * LOW)^0.5
        geometric_mean = (high * low) ** 0.5


        # 3. 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol
        
        # 计算最终因子：几何平均 - VWAP
        factor = geometric_mean - vwap

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha13",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "(((HIGH * LOW)^0.5) - VWAP)",
    "description": "alpha191因子13",
    "lookback_days": 0,
    "method": "Alpha13Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''