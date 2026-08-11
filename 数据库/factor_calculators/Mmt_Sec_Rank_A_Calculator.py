from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Sec_Rank_A_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        close_hfq = data.xs("close_hfq", level=0)

        # 2. 计算日收益率
        ret = close_hfq.pct_change()

        # 3. 计算每日横截面排名 (Cross-sectional Rank)
        # axis=1 表示在每一行（日期）上，对所有列（股票）进行排名
        # pct=True 将排名映射到 [0, 1] 区间，消除股票数量变化的影响
        daily_rank = ret.rank(axis=1, pct=True)

        # 4. 计算过去 252 个交易日的排名均值 (Time-series Mean)
        N = 252
        factor = daily_rank.rolling(window=N, min_periods=10).mean()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1年横截面rank动量",
    "creator": "jiangyushi",
    "create_date": "2026.01.15",
    "formula": "Mean(Rank(ret_daily, cross_section), 252D)",
    "description": "每日计算个股日收益在横截面的排名，取过去1年排名均值",
    "lookback_days": 252,
    "method": "Mmt_Sec_Rank_A_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量反转因子"
}
'''