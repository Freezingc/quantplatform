from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_010_Calculator(FactorCalculator):
    """遗传规划因子 #10，Rank-IC=0.057004
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(vol)), ts_sum_10(cs_zscore(ts_min_20(tanh(ts_sum_20...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = (vol - vol.shift(20)) / vol.shift(20).replace(0, np.nan)
        _v2 = np.tanh(_v1)
        _v3 = open.rank(axis=1, pct=True)
        _v4 = _v3.rolling(20).std()
        _v5 = _v4.rolling(10).rank(pct=True)
        _v6 = _v5.rolling(20).sum()
        _v7 = np.tanh(_v6)
        _v8 = _v7.rolling(20).min()
        _v9 = _v8.sub(_v8.mean(axis=1), axis=0).div(_v8.std(axis=1) + 1e-10, axis=0)
        _v10 = _v9.rolling(10).sum()
        _v11 = (_v2 ** _v10).replace([np.inf, -np.inf], np.nan)
        _v12 = _v11.rank(axis=1, pct=True)
        _v13 = turnover * _v12
        factor = _v13.replace([np.inf, -np.inf], np.nan)
        return factor
