from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_038_Calculator(FactorCalculator):
    """遗传规划因子 #38，Rank-IC=0.051997
    表达式: mul(ts_rank_10(close), cs_rank(amount))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = close.rolling(10).rank(pct=True)
        _v2 = amount.rank(axis=1, pct=True)
        _v3 = _v1 * _v2
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
