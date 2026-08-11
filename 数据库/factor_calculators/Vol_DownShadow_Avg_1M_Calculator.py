from .base import FactorCalculator
import numpy as np
import pandas as pd


class Vol_DownShadow_Avg_1M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open_hfq", "low_hfq", "close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        open_hfq = data.xs('open_hfq', level=0)
        low_hfq = data.xs('low_hfq', level=0)
        close_hfq = data.xs('close_hfq', level=0)

        # 3. 计算标准化下影线
        down_body = np.minimum(open_hfq, close_hfq)
        down_shadow = (down_body - low_hfq) / low_hfq

        # 4. 过去 1 个月取均值
        N = 21
        factor = down_shadow.rolling(N).mean()

        return factor
    

'''
FACTOR_CONFIG = {
    "factor_name": "1个月下影线均值",
    "creator": "jiangyushi",
    "create_date": "2026.02.24",
    "formula": "MEAN((min(open, close) - low) / low, 21)",
    "description": "标准化下影线因子过去1个月的均值；其中，标准化下影线=（min(开盘价，收盘价）-最低价）/最低价",
    "lookback_days": 21,
    "method": "Vol_DownShadow_Avg_1M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''
