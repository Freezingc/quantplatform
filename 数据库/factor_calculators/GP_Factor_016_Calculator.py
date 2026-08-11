from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_016_Calculator(FactorCalculator):
    """遗传规划因子 #16，Rank-IC=0.055667
    表达式: ts_max_5(mul(cs_zscore(pct_chg), turnover))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(pct_chg.std(axis=1) + 1e-10, axis=0)
        _v2 = _v1 * turnover
        _v3 = _v2.rolling(5).max()
        factor = _v3.replace([np.inf, -np.inf], np.nan)
        return factor
