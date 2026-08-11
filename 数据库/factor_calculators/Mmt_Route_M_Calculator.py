from .base import FactorCalculator
import numpy as np
import pandas as pd


class Mmt_Route_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_qfq"]  # 需要前复权收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取相关字段
        close_hfq = data.xs("close_qfq", level=0)

        # 3. 计算日收益率
        daily_ret = close_hfq / close_hfq.shift(1) - 1

        # 4. 计算1个月路径调整动量
        N_month = 20

        sum_ret = daily_ret.rolling(N_month).sum()
        sum_abs_ret = daily_ret.abs().rolling(N_month).sum()

        factor = sum_ret / sum_abs_ret

        return factor
'''
FACTOR_CONFIG = {
    "factor_name": "1个月路径调整动量",
    "creator": "zhangyudong",
    "create_date": "2025.12.24",
    "formula": "SUM(ret, 20) / SUM(ABS(ret), 20)",
    "description": "过去1个月累计收益率除以过去1个月日度涨跌幅绝对值之和，用于衡量趋势的平滑程度",
    "lookback_days": 20,
    "method": "Mmt_Route_M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量因子"
}
'''
