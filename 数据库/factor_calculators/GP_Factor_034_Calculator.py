from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_034_Calculator(FactorCalculator):
    """遗传规划因子 #34，Rank-IC=0.052363
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(square(ts_delta_60(amount)))), ts_sum_10(cs_zscore(...
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
        _v1 = amount - amount.shift(60)
        _v2 = _v1 ** 2
        _v3 = (_v2 - _v2.shift(20)) / _v2.shift(20).replace(0, np.nan)
        _v4 = np.tanh(_v3)
        _v5 = close.rolling(5).cov(5.78947368421052) / close.rolling(5).var().replace(0, np.nan)
        _v6 = _v5.rank(axis=1, pct=True)
        _v7 = _v6.rolling(20).std()
        _v8 = _v7.rolling(10).rank(pct=True)
        _v9 = _v8.rolling(20).sum()
        _v10 = np.tanh(_v9)
        _v11 = _v10.rolling(20).min()
        _v12 = _v11.sub(_v11.mean(axis=1), axis=0).div(_v11.std(axis=1) + 1e-10, axis=0)
        _v13 = _v12.rolling(10).sum()
        _v14 = (_v4 ** _v13).replace([np.inf, -np.inf], np.nan)
        _v15 = _v14.rank(axis=1, pct=True)
        _v16 = turnover * _v15
        factor = _v16.replace([np.inf, -np.inf], np.nan)
        return factor
