from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha133Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算过去20日的最高价位置因子
        highday_20_factor = ((20 - high.rolling(window=20).apply(
            lambda x: 20 - (x.argmax() + 1) if not x.isnull().all() else np.nan
        )) / 20) * 100
        
        # 计算过去20日的最低价位置因子
        lowday_20_factor = ((20 - low.rolling(window=20).apply(
            lambda x: 20 - (x.argmin() + 1) if not x.isnull().all() else np.nan
        )) / 20) * 100
        
        # 计算最终因子：高低位置因子差值
        factor = highday_20_factor - lowday_20_factor
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha133",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((20-HIGHDAY(HIGH,20))/20)*100-((20-LOWDAY(LOW,20))/20)*100",
        "description": "过去20日中最高价与最低价出现位置的相对距离差异",
        "lookback_days": 19,
        "method": "Alpha133Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
