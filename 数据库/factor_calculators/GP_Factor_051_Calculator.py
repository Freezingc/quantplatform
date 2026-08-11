from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_051_Calculator(FactorCalculator):
    """遗传规划因子 #51，Rank-IC=0.05139
    表达式: add(pow(ts_std_10(pct_chg), cs_rank(vol)), ts_max_60(ts_rank_60(ts_regression_residual_30(ts_rank_60...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "vol", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        high     = data.xs('high',          level=0)
        vol      = data.xs('vol',           level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = pct_chg.rolling(10).std()
        _v2 = vol.rank(axis=1, pct=True)
        _v3 = (_v1 ** _v2).replace([np.inf, -np.inf], np.nan)
        _v4 = high - high.shift(10)
        _v5 = _v4.rolling(60).cov(vol) / _v4.rolling(60).var().replace(0, np.nan)
        _v6 = _v5.rolling(60).rank(pct=True)
        _v7 = vol.rolling(60).skew()
        _v8 = 5.789473684210526 * -0.5
        _v9 = _v7 - _v8
        _v10 = _v9.rolling(30).mean()
        _v12 = _v6.rolling(30).mean()
        _v13 = _v10.rolling(30).mean()
        _v14 = _v6.rolling(30).cov(_v10) / _v6.rolling(30).var().replace(0, np.nan)
        _v11 = (_v10 - _v13) - _v14 * (_v6 - _v12)
        _v15 = _v11.rolling(60).rank(pct=True)
        _v16 = _v15.rolling(60).max()
        _v17 = _v3 + _v16
        factor = _v17.replace([np.inf, -np.inf], np.nan)
        return factor
