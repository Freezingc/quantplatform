from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Off_Limit_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 只需要收盘价

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需字段
        close_hfq = data.xs("close_hfq", level=0)

        # 3. 计算收盘涨跌幅
        ret = close_hfq.pct_change()

        # 4. 去掉涨跌停收益（涨幅或跌幅 >= 10%）
        limit_mask = ret.abs() >= 0.1
        ret_filtered = ret.mask(limit_mask)

        # 5. 过去 20 日求和
        N_month = 20
        factor = ret_filtered.rolling(N_month).sum()

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月去涨跌停动量",
    "creator": "jiangyushi",
    "create_date": "2025.12.08",
    "formula": "SUM(|ret|<0.1, 20)",
    "description": "除去涨跌停日后，过去1个月的收盘涨跌幅",
    "lookback_days": 20,
    "method": "Mmt_Off_Limit_M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量反转因子"
}
'''
