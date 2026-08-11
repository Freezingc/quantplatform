from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_057_Calculator(FactorCalculator):
    """遗传规划因子 #57，Rank-IC=0.051096
    表达式: mul(turnover, cs_mean(ts_regression_5(ts_rank_30(amount), cs_zscore(ts_regression_residual_60(ts_del...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "vol", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        open_    = data.xs('open',          level=0)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = amount.rolling(30).rank(pct=True)
        _v2 = close.rolling(5).sum()
        _v4 = vol.rolling(5).mean()
        _v5 = _v2.rolling(5).mean()
        _v6 = vol.rolling(5).cov(_v2) / vol.rolling(5).var().replace(0, np.nan)
        _v3 = (_v2 - _v5) - _v6 * (vol - _v4)
        _v7 = _v3.rolling(10).std()
        _v8 = open.rolling(5).min()
        _v9 = _v8.rolling(30).sum()
        _v10 = _v9 ** 2
        _v11 = _v10 - _v10.shift(5)
        _v13 = _v7.rolling(5).mean()
        _v14 = _v11.rolling(5).mean()
        _v15 = _v7.rolling(5).cov(_v11) / _v7.rolling(5).var().replace(0, np.nan)
        _v12 = (_v11 - _v14) - _v15 * (_v7 - _v13)
        _v16 = _v12 - _v12.shift(5)
        _v18 = _v16.rolling(60).mean()
        _v19 = close.rolling(60).mean()
        _v20 = _v16.rolling(60).cov(close) / _v16.rolling(60).var().replace(0, np.nan)
        _v17 = (close - _v19) - _v20 * (_v16 - _v18)
        _v21 = _v17.sub(_v17.mean(axis=1), axis=0).div(_v17.std(axis=1) + 1e-10, axis=0)
        _v22 = _v1.rolling(5).cov(_v21) / _v1.rolling(5).var().replace(0, np.nan)
        _v23 = _v22.mean(axis=1)
        _v24 = turnover.mul(_v23, axis=0)
        factor = _v24.replace([np.inf, -np.inf], np.nan)
        return factor
