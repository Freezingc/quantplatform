from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Overnight_A_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open_hfq", "close_hfq"]  # 需要前复权开盘与收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取相关字段
        open_hfq = data.xs("open_hfq", level=0)
        close_hfq = data.xs("close_hfq", level=0)

        # 3. 计算隔夜收益率（当日开盘 / 昨日收盘 - 1）
        overnight_ret = open_hfq / close_hfq.shift(1) - 1

        # 4. 计算滚动窗口求和
        N_year = 252
        N_month = 20

        sum_year = overnight_ret.rolling(N_year).sum()
        sum_month = overnight_ret.rolling(N_month).sum()

        factor = sum_year - sum_month

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1年隔夜动量",
    "creator": "jiangyushi",
    "create_date": "2025.12.08",
    "formula": "SUM(overnight_ret, 252) - SUM(overnight_ret, 20)",
    "description": "过去1年的隔夜涨跌幅之和 - 过去1个月的隔夜涨跌幅之和",
    "lookback_days": 252,
    "method": "Mmt_Overnight_A_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量反转因子"
}
'''
