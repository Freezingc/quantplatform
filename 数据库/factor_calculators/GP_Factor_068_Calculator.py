from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_068_Calculator(FactorCalculator):
    """遗传规划因子 #68，Rank-IC=0.050814
    表达式: mul(amount, ts_rank_60(neg(neg(amount))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        _v1 = -amount
        _v2 = -_v1
        _v3 = _v2.rolling(60).rank(pct=True)
        _v4 = amount * _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
