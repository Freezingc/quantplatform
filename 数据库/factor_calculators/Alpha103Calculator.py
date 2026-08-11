from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha103Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        
        # 3. 实现因子计算逻辑
        # LOWDAY(LOW,20) 计算20日内最低价出现的天数（距离当前的天数）
        def lowday_20(series):
            # 找到最近20日内的最低价位置
            if len(series) < 20:
                return np.nan
            # 使用argmin找到最小值的位置（0表示当前日，1表示前一日，以此类推）
            min_idx = series.iloc[-20:].argmin()  # 在最近20个数据中找最小值位置
            return min_idx  # 返回的是相对位置（0-19）
        
        # 计算每个标的的20日最低价位置
        lowday_values = low.rolling(window=20, min_periods=20).apply(lowday_20, raw=False)
        
        # 计算因子：((20 - LOWDAY(LOW,20)) / 20) * 100
        factor = ((20 - lowday_values) / 20) * 100
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha103",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "STD(vol,20)",
        "description": "成交量20日标准差",
        "lookback_days": 9,
        "method": "Alpha103Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
