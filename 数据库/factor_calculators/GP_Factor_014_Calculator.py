from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_014_Calculator(FactorCalculator):
    """遗传规划因子 #14，Rank-IC=0.055833
    表达式: ts_max_5(mul(pct_chg, cs_rank(amount)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = amount.rank(axis=1, pct=True)
        _v2 = pct_chg * _v1
        _v3 = _v2.rolling(5).max()
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
