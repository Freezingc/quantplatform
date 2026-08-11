from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha145Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        vol = data.xs('vol', level=0)  # 获取vol数据
        
        # 3. 计算移动平均值
        mean_9 = vol.rolling(window=9, min_periods=1).mean()
        mean_26 = vol.rolling(window=26, min_periods=1).mean()
        mean_12 = vol.rolling(window=12, min_periods=1).mean()
        
        # 4. 计算因子
        factor = (mean_9 - mean_26) / mean_12 * 145
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha145",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MEAN(VOLUME,9)-MEAN(VOLUME,26))/MEAN(VOLUME,12)*100",
        "description": "计算的是9日成交量均线与26日成交量均线的差值，相对于12日成交量均线的百分比",
        "lookback_days": 25,
        "method": "Alpha145Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
