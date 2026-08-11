from .base import FactorCalculator
import numpy as np
import pandas as pd


class Mmt_Discrete_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_qfq"]  # 需要前复权收盘价（qfq）

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取相关字段
        close_qfq = data.xs("close_qfq", level=0)

        # 3. 计算日收益率
        daily_ret = close_qfq / close_qfq.shift(1) - 1

        # 4. 计算1个月信息离散度动量
        N_month = 20

        up_cnt = (daily_ret > 0).rolling(N_month).sum()
        down_cnt = (daily_ret < 0).rolling(N_month).sum()

        factor = (up_cnt - down_cnt) / N_month

        return factor
'''
FACTOR_CONFIG = {
    "factor_name": "1个月信息离散度动量",
    "creator": "zhangyudong",
    "create_date": "2025.12.24",
    "formula": "(COUNT(ret>0,20) - COUNT(ret<0,20)) / 20",
    "description": "过去1个月内上涨天数占比减去下跌天数占比，用于衡量收益方向的一致性",
    "lookback_days": 20,
    "method": "Mmt_Discrete_M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量因子"
}
'''
