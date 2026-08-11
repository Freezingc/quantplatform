from .base import FactorCalculator
import numpy as np
import pandas as pd


class Vol_UpShadow_Avg_3M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open_hfq", "high_hfq", "close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        open_hfq = data.xs('open_hfq', level=0)
        high_hfq = data.xs('high_hfq', level=0)
        close_hfq = data.xs('close_hfq', level=0)

        # 3. 计算标准化上影线
        upper_body = np.maximum(open_hfq, close_hfq)
        up_shadow = (high_hfq - upper_body) / high_hfq

        # 4. 过去 3 个月取均值
        N = 21 * 3
        factor = up_shadow.rolling(N).mean()

        return factor
    

'''
FACTOR_CONFIG = {
    "factor_name": "3个月上影线均值",
    "creator": "jiangyushi",
    "create_date": "2026.02.05",
    "formula": "MEAN((high - max(open, close)) / high, 63)",
    "description": "标准化上影线因子过去3个月的均值；其中，标准化上影线=（最高价-max(开盘价，收盘价））/最高价",
    "lookback_days": 63,
    "method": "Vol_UpShadow_Avg_3M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''
