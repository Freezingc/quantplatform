from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Time_Rank_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        close_hfq = data.xs("close_hfq", level=0)

        # 定义窗口
        T_RANK = 252  # 时序排名的窗口（1年通常取252个交易日）
        N_MEAN = 20   # 滚动平均的窗口（1个月）

        # 2. 计算每日价格在过去1年内的时序排名 (Time-series Rank)
        # pct=True 将排名归一化到 [0, 1]
        time_rank = close_hfq.rolling(window=T_RANK, min_periods=120).rank(pct=True)

        # 3. 计算过去 20 个交易日排名均值 (Moving Average of Rank)
        # 这一步是为了平滑排名，观察个股是否稳定处于历史高位/低位
        factor = time_rank.rolling(window=N_MEAN, min_periods=10).mean()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月时序rank动量",
    "creator": "jiangyushi",
    "create_date": "2026.01.15",
    "formula": "Mean(TS_Rank(close, 252D), 20D)",
    "description": "每日计算个股价格在时序（1年内）的排名，取过去20个交易日排名均值",
    "lookback_days": 252,
    "method": "Mmt_Time_Rank_M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量反转因子"
}
'''