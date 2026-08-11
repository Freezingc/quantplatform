from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_005_Calculator(FactorCalculator):
    """遗传规划因子 #5，Rank-IC=0.058498
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(vol)),
               ts_sum_10(cs_zscore(ts_min_20(tanh(ts_sum_20(ts_rank_10(
               ts_std_20(cs_rank(ts_regression_5(close, 5.789473684210526))))))))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        vol      = data.xs('vol',           level=0)
        turnover = data.xs('turnover_rate', level=0)

        _v1  = (vol - vol.shift(20)) / vol.shift(20).replace(0, np.nan)
        _v2  = np.tanh(_v1)
        _v3  = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        _v4  = _v3.rank(axis=1, pct=True)
        _v5  = _v4.rolling(20).std()
        _v6  = _v5.rolling(10).rank(pct=True)
        _v7  = _v6.rolling(20).sum()
        _v8  = np.tanh(_v7)
        _v9  = _v8.rolling(20).min()
        _v10 = _v9.sub(_v9.mean(axis=1), axis=0).div(_v9.std(axis=1) + 1e-10, axis=0)
        _v11 = _v10.rolling(10).sum()
        _v12 = (_v2 ** _v11).replace([np.inf, -np.inf], np.nan)
        _v13 = _v12.rank(axis=1, pct=True)
        factor = (turnover * _v13).replace([np.inf, -np.inf], np.nan)
        return factor
