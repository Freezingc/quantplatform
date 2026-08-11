from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_024_Calculator(FactorCalculator):
    """遗传规划因子 #24，Rank-IC=0.053887
    表达式: ts_std_10(cs_rank(pct_chg))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = pct_chg.rank(axis=1, pct=True)
        _v2 = _v1.rolling(10).std()
        factor = _v2.replace([np.inf, -np.inf], np.nan)
        return factor
