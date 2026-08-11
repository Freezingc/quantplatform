from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_071_Calculator(FactorCalculator):
    """遗传规划因子 #71，Rank-IC=0.050774
    表达式: add(cs_rank(close), mul(ts_rank_5(high), amount))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "high", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        high     = data.xs('high',          level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rank(axis=1, pct=True)
        _v2 = high.rolling(5).rank(pct=True)
        _v3 = _v2 * amount
        _v4 = _v1 + _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
