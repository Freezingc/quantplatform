from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_046_Calculator(FactorCalculator):
    """遗传规划因子 #46，Rank-IC=0.051712
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_20(ts_rank_20(high))), ts_sum_10(turnover))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v1 = high.rolling(20).rank(pct=True)
        _v2 = _v1 - _v1.shift(20)
        _v3 = np.tanh(_v2)
        _v4 = turnover.rolling(10).sum()
        _v5 = (_v3 ** _v4).replace([np.inf, -np.inf], np.nan)
        _v6 = _v5.rank(axis=1, pct=True)
        _v7 = turnover * _v6
        factor = _v7.replace([np.inf, -np.inf], np.nan)
        return factor
