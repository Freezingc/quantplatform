from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Range_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high_hfq", "low_hfq", "close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        high_hfq = data.xs("high_hfq", level=0)
        low_hfq = data.xs("low_hfq", level=0)
        close_hfq = data.xs("close_hfq", level=0)

        # 2. 基础指标
        daily_range = (high_hfq - low_hfq) / close_hfq
        ret = close_hfq.pct_change()

        N = 20
        q = 0.2

        # 3. 计算滚动秩 (Rolling Rank) 
        # 对每一列（每个股票）计算过去 N 天振幅的百分比排名
        # pct=True 会将排名归一化到 [0, 1]
        range_rank = daily_range.rolling(N, min_periods=N).rank(pct=True)

        # 4. 构造高振幅和低振幅的收益掩码
        # 如果当前振幅在过去 N 天属于前 20%，保留其收益率，否则设为 NaN
        ret_high_group = ret.where(range_rank >= (1 - q))
        ret_low_group = ret.where(range_rank <= q)

        # 5. 计算滚动平均
        # 它会自动忽略 NaN，从而只对符合条件的收益率求平均
        avg_ret_high = ret_high_group.rolling(N, min_periods=4).mean()
        avg_ret_low = ret_low_group.rolling(N, min_periods=4).mean()

        # 6. 计算差值
        factor = avg_ret_high - avg_ret_low

        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月振幅调整动量",
    "creator": "jiangyushi",
    "create_date": "2026.01.15",
    "formula": "Mean(ret | rank(range) >= 0.8, 20D) - Mean(ret | rank(range) <= 0.2, 20D)",
    "description": "过去1个月内，振幅大的前20%的收盘收益率 - 振幅小的后20%的收盘收益率",
    "lookback_days": 20,
    "method": "Mmt_Range_M_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "动量反转因子"
}
'''
