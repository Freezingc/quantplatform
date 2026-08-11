from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_085_Calculator(FactorCalculator):
    """遗传规划因子 #85，Rank-IC=0.050593
    表达式: max(amount, ts_rank_20(ts_delta_60(cs_rank(close))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rank(axis=1, pct=True)
        _v2 = _v1 - _v1.shift(60)
        _v3 = _v2.rolling(20).rank(pct=True)
        _v4 = amount.where(amount >= _v3, _v3)
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
