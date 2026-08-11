from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_047_Calculator(FactorCalculator):
    """遗传规划因子 #47，Rank-IC=0.05168
    表达式: mul(turnover, cs_mean(ts_regression_5(high, cs_zscore(pct_chg))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(pct_chg.std(axis=1) + 1e-10, axis=0)
        _v2 = high.rolling(5).cov(_v1) / high.rolling(5).var().replace(0, np.nan)
        _v3 = _v2.mean(axis=1)
        _v4 = turnover.mul(_v3, axis=0)
        factor = _v4.replace([np.inf, -np.inf], np.nan)
        return factor
