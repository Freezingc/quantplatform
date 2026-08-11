from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_092_Calculator(FactorCalculator):
    """遗传规划因子 #92，Rank-IC=0.050499
    表达式: mul(pow(abs(high), turnover), abs(add(tanh(ts_delta_ratio_20(vol)), ts_sum_10(cs_zscore(ts_min_20(ts...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "high", "vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        high     = data.xs('high',          level=0)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)
        open = open_
        _v1 = high.abs()
        _v2 = (_v1 ** turnover).replace([np.inf, -np.inf], np.nan)
        _v3 = (vol - vol.shift(20)) / vol.shift(20).replace(0, np.nan)
        _v4 = np.tanh(_v3)
        _v5 = open - open.shift(10)
        _v6 = _v5.rolling(60).min()
        _v7 = _v6.rolling(20).min()
        _v8 = _v7.sub(_v7.mean(axis=1), axis=0).div(_v7.std(axis=1) + 1e-10, axis=0)
        _v9 = _v8.rolling(10).sum()
        _v10 = _v4 + _v9
        _v11 = _v10.abs()
        _v12 = _v2 * _v11
        factor = _v12.replace([np.inf, -np.inf], np.nan)
        return factor
