from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha81Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 获取成交量数据
        
        factor = vol.rolling(window=21, min_periods=1).apply(
            lambda x: (x[-1]*2 + x[:-1].mean()*19)/21 if len(x) == 21 else x.mean()
        )
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha81",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "SMA(VOLUME,21,2)",
        "description": "alpha191因子81",
        "lookback_days": 20,
        "method": "Alpha81Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

