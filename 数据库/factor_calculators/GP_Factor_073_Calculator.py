from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_073_Calculator(FactorCalculator):
    """遗传规划因子 #73，Rank-IC=0.050722
    表达式: ts_min_5(mul(cs_zscore(pct_chg), ts_delta_60(sub(-0.526315789473685, pct_chg))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(pct_chg.std(axis=1) + 1e-10, axis=0)
        _v2 = -0.526315789473685 - pct_chg
        _v3 = _v2 - _v2.shift(60)
        _v4 = _v1 * _v3
        _v5 = _v4.rolling(5).min()
        factor = _v5.replace([np.inf, -np.inf], np.nan)
        return factor
