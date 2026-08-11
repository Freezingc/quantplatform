from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_053_Calculator(FactorCalculator):
    """遗传规划因子 #53，Rank-IC=0.051285
    表达式: mul(turnover, cs_mean(ts_regression_5(pct_chg, cs_zscore(ts_regression_residual_60(ts_regression_res...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)
        _v2 = pct_chg.rolling(5).mean()
        _v3 = close.rolling(5).mean()
        _v4 = pct_chg.rolling(5).cov(close) / pct_chg.rolling(5).var().replace(0, np.nan)
        _v1 = (close - _v3) - _v4 * (pct_chg - _v2)
        _v6 = _v1.rolling(60).mean()
        _v7 = close.rolling(60).mean()
        _v8 = _v1.rolling(60).cov(close) / _v1.rolling(60).var().replace(0, np.nan)
        _v5 = (close - _v7) - _v8 * (_v1 - _v6)
        _v9 = _v5.sub(_v5.mean(axis=1), axis=0).div(_v5.std(axis=1) + 1e-10, axis=0)
        _v10 = pct_chg.rolling(5).cov(_v9) / pct_chg.rolling(5).var().replace(0, np.nan)
        _v11 = _v10.mean(axis=1)
        _v12 = turnover.mul(_v11, axis=0)
        factor = _v12.replace([np.inf, -np.inf], np.nan)
        return factor
