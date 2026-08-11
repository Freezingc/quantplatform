from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_076_Calculator(FactorCalculator):
    """遗传规划因子 #76，Rank-IC=0.0507
    表达式: add(amount, ts_mean_30(ts_regression_30(ts_median_5(low), ts_skewness_30(ts_rank_10(close)))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "amount"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        low      = data.xs('low',           level=0)
        amount   = data.xs('amount',        level=0)
        _v1 = low.rolling(5).median()
        _v2 = close.rolling(10).rank(pct=True)
        _v3 = _v2.rolling(30).skew()
        _v4 = _v1.rolling(30).cov(_v3) / _v1.rolling(30).var().replace(0, np.nan)
        _v5 = _v4.rolling(30).mean()
        _v6 = amount + _v5
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
