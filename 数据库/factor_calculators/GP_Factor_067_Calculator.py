from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_067_Calculator(FactorCalculator):
    """遗传规划因子 #67，Rank-IC=0.050834
    表达式: mul(turnover, cs_mean(ts_regression_5(ts_rank_30(ts_rank_20(amount)), cs_zscore(ts_regression_residu...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        open_    = data.xs('open',          level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = amount.rolling(20).rank(pct=True)
        _v2 = _v1.rolling(30).rank(pct=True)
        _v3 = close.rank(axis=1, pct=True)
        _v4 = _v3.rolling(10).std()
        _v5 = open.rolling(5).min()
        _v6 = _v5.rolling(30).sum()
        _v7 = _v6 ** 2
        _v8 = _v7 - _v7.shift(5)
        _v10 = _v4.rolling(5).mean()
        _v11 = _v8.rolling(5).mean()
        _v12 = _v4.rolling(5).cov(_v8) / _v4.rolling(5).var().replace(0, np.nan)
        _v9 = (_v8 - _v11) - _v12 * (_v4 - _v10)
        _v13 = _v9 - _v9.shift(5)
        _v15 = _v13.rolling(60).mean()
        _v16 = close.rolling(60).mean()
        _v17 = _v13.rolling(60).cov(close) / _v13.rolling(60).var().replace(0, np.nan)
        _v14 = (close - _v16) - _v17 * (_v13 - _v15)
        _v18 = _v14.sub(_v14.mean(axis=1), axis=0).div(_v14.std(axis=1) + 1e-10, axis=0)
        _v19 = _v2.rolling(5).cov(_v18) / _v2.rolling(5).var().replace(0, np.nan)
        _v20 = _v19.mean(axis=1)
        _v21 = turnover.mul(_v20, axis=0)
        factor = _v21.replace([np.inf, -np.inf], np.nan)
        return factor
