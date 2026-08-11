from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_082_Calculator(FactorCalculator):
    """遗传规划因子 #82，Rank-IC=0.05062
    表达式: add(ts_std_5(ts_rank_60(ts_median_30(ts_delta_ratio_10(open)))), amount)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        amount   = data.xs('amount',        level=0)
        open = open_
        _v1 = (open - open.shift(10)) / open.shift(10).replace(0, np.nan)
        _v2 = _v1.rolling(30).median()
        _v3 = _v2.rolling(60).rank(pct=True)
        _v4 = _v3.rolling(5).std()
        _v5 = _v4 + amount
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
