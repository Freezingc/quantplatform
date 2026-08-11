from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha46Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算不同周期的移动平均
        mean_3 = close.rolling(window=3,min_periods=1).mean()
        mean_6 = close.rolling(window=6,min_periods=1).mean()
        mean_12 = close.rolling(window=12,min_periods=1).mean()
        mean_24 = close.rolling(window=24,min_periods=1).mean()
        
        # 计算因子值
        factor = (mean_3 + mean_6 + mean_12 + mean_24) / (4 * close)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha46",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/(4*CLOSE)",
    "description": "alpha191因子46",
    "lookback_days": 23,
    "method": "Alpha46Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''