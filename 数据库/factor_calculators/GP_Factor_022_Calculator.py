from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_022_Calculator(FactorCalculator):
    """遗传规划因子 #22，Rank-IC=0.054289
    表达式: mul(ts_max_60(pct_chg), amount)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = pct_chg.rolling(60).max()
        _v2 = _v1 * amount
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
