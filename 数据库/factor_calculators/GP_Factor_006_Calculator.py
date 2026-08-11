from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_006_Calculator(FactorCalculator):
    """遗传规划因子 #6，Rank-IC=0.058023
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(vol)),
               ts_sum_10(cs_zscore(ts_min_20(tanh(ts_sum_20(ts_rank_10(
               ts_std_20(relu(inv(ts_mean_30(turnover)))))))))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)

        _v1  = (vol - vol.shift(20)) / vol.shift(20).replace(0, np.nan)
        _v2  = np.tanh(_v1)
        _v3  = turnover.rolling(30).mean()
        _v4  = (1 / _v3.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        _v5  = _v4.clip(lower=0)
        _v6  = _v5.rolling(20).std()
        _v7  = _v6.rolling(10).rank(pct=True)
        _v8  = _v7.rolling(20).sum()
        _v9  = np.tanh(_v8)
        _v10 = _v9.rolling(20).min()
        _v11 = _v10.sub(_v10.mean(axis=1), axis=0).div(_v10.std(axis=1) + 1e-10, axis=0)
        _v12 = _v11.rolling(10).sum()
        _v13 = (_v2 ** _v12).replace([np.inf, -np.inf], np.nan)
        _v14 = _v13.rank(axis=1, pct=True)
        factor = (turnover * _v14).replace([np.inf, -np.inf], np.nan)
        return factor
