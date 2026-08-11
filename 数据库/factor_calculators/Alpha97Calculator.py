from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha97Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 提取成交量数据

        # 3. 实现因子计算逻辑 - STD(vol,10)
        # 计算成交量10日滚动标准差
        factor = vol.rolling(window=10, min_periods=1).std()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha97",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "STD(vol,10)",
        "description": "成交量10日标准差",
        "lookback_days": 9,
        "method": "Alpha97Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
