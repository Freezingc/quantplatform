from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_070_Calculator(FactorCalculator):
    """遗传规划因子 #70，Rank-IC=0.05079
    表达式: mul(turnover, cs_rank(mul(cs_zscore(amount), ts_rank_20(ts_median_5(close)))))
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
        _v1 = amount.sub(amount.mean(axis=1), axis=0).div(amount.std(axis=1) + 1e-10, axis=0)
        _v2 = close.rolling(5).median()
        _v3 = _v2.rolling(20).rank(pct=True)
        _v4 = _v1 * _v3
        _v5 = _v4.rank(axis=1, pct=True)
        _v6 = turnover * _v5
        factor = _v6.replace([np.inf, -np.inf], np.nan)
        return factor
