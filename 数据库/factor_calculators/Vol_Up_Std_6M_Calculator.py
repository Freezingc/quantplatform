from .base import FactorCalculator
import numpy as np
import pandas as pd

class Vol_Up_Std_6M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 需要收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)

        # 3. 计算上行波动率
        N = 21 * 6
        ret = close_hfq.pct_change()

        # 仅保留上行收益率（下跌日设为 NaN）
        up_ret = ret.where(ret > 0)

        # 计算过去 6 个月上行收益率的标准差
        factor = up_ret.rolling(N, min_periods=2).std()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "6个月上行波动率",
    "creator": "jiangyushi",
    "create_date": "2026.01.29",
    "formula": "STD(ret | ret > 0, 126)",
    "description": "个股过去6个月内上行（日收益率>0）收益率序列的标准差",
    "lookback_days": 126,
    "method": "Vol_Up_Std_6M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''