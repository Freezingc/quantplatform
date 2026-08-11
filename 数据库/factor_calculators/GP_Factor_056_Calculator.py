from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_056_Calculator(FactorCalculator):
    """遗传规划因子 #56，Rank-IC=0.051119
    表达式: max(ts_rank_30(cs_rank(close)), cs_rank(ts_std_10(pct_chg)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = close.rank(axis=1, pct=True)
        _v2 = _v1.rolling(30).rank(pct=True)
        _v3 = pct_chg.rolling(10).std()
        _v4 = _v3.rank(axis=1, pct=True)
        _v5 = _v2.where(_v2 >= _v4, _v4)
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
