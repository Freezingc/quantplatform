from .base import FactorCalculator
import numpy as np
import pandas as pd


class Vol_HighLow_Std_1M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high_hfq", "low_hfq"]  # 需要最高价和最低价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        high_hfq = data.xs('high_hfq', level=0)
        low_hfq = data.xs('low_hfq', level=0)

        # 3. 计算日内振幅并取标准差
        N = 21  # 1个月交易日数
        daily_range = high_hfq / low_hfq

        factor = daily_range.rolling(N, min_periods=2).std()

        return factor

'''
FACTOR_CONFIG = {
    "factor_name": "1个月日内振幅标准差",
    "creator": "jiangyushi",
    "create_date": "2026.02.05",
    "formula": "STD(high / low, 21)",
    "description": "过去1个月（最高价/最低价）的标准差",
    "lookback_days": 21,
    "method": "Vol_HighLow_Std_1M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''

