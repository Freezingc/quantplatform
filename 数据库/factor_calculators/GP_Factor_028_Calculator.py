from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_028_Calculator(FactorCalculator):
    """遗传规划因子 #28，Rank-IC=0.053094
    表达式: add(ts_regression_residual_30(shift_60(square(tanh(high))), open), turnover)
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "high", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        high     = data.xs('high',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = np.tanh(high)
        _v2 = _v1 ** 2
        _v3 = _v2.shift(60)
        _v5 = _v3.rolling(30).mean()
        _v6 = open.rolling(30).mean()
        _v7 = _v3.rolling(30).cov(open) / _v3.rolling(30).var().replace(0, np.nan)
        _v4 = (open - _v6) - _v7 * (_v3 - _v5)
        _v8 = _v4 + turnover
        factor = _v8.replace([np.inf, -np.inf], np.nan)
        return factor
