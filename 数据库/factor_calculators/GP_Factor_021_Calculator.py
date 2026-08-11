from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_021_Calculator(FactorCalculator):
    """遗传规划因子 #21，Rank-IC=0.054441
    表达式: add(pow(pct_chg, cs_rank(vol)), ts_delta_ratio_60(amount))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = vol.rank(axis=1, pct=True)
        _v2 = (pct_chg ** _v1).replace([np.inf, -np.inf], np.nan)
        _v3 = (amount - amount.shift(60)) / amount.shift(60).replace(0, np.nan)
        _v4 = _v2 + _v3
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
