from .base import FactorCalculator
import numpy as np
import pandas as pd


class Vol_UpShadow_Std_6M_Calculator(FactorCalculator):
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

        # 4. 过去 6 个月取标准差
        N = 21 * 6
        factor = up_shadow.rolling(N).std()

        return factor
    

'''
FACTOR_CONFIG = {
    "factor_name": "6个月上影线标准差",
    "creator": "jiangyushi",
    "create_date": "2026.02.24",
    "formula": "STD((high - max(open, close)) / high, 126)",
    "description": "标准化上影线因子过去6个月的标准差；其中，标准化上影线=（最高价-max(开盘价，收盘价））/最高价",
    "lookback_days": 126,
    "method": "Vol_UpShadow_Std_6M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''
