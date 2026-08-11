from .base import FactorCalculator
import numpy as np
import pandas as pd

class Vol_Down_Std_1M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 需要收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)

        # 3. 计算下行波动率
        N = 21
        ret = close_hfq.pct_change()

        # 仅保留下行收益率（上涨日设为 NaN）
        down_ret = ret.where(ret < 0)

        # 计算过去 1 个月下行收益率的标准差
        factor = down_ret.rolling(N, min_periods=2).std()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月下行波动率",
    "creator": "jiangyushi",
    "create_date": "2026.01.22",
    "formula": "STD(ret | ret < 0, 21)",
    "description": "个股过去1个月内下行（日收益率<0）收益率序列的标准差",
    "lookback_days": 21,
    "method": "Vol_Down_Std_1M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''