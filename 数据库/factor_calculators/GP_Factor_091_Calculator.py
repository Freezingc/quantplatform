from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_091_Calculator(FactorCalculator):
    """遗传规划因子 #91，Rank-IC=0.050502
    表达式: mul(amount, cs_mean(ts_regression_5(ts_min_5(ts_rank_20(turnover)), cs_zscore(close))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = turnover.rolling(20).rank(pct=True)
        _v2 = _v1.rolling(5).min()
        _v3 = close.sub(close.mean(axis=1), axis=0).div(close.std(axis=1) + 1e-10, axis=0)
        _v4 = _v2.rolling(5).cov(_v3) / _v2.rolling(5).var().replace(0, np.nan)
        _v5 = _v4.mean(axis=1)
        _v6 = amount.mul(_v5, axis=0)
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
