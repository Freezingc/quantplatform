from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_064_Calculator(FactorCalculator):
    """遗传规划因子 #64，Rank-IC=0.050879
    表达式: add(pow(exp(sign(ts_regression_residual_5(close, ts_mean_60(ts_std_10(sub(amount, 7.894736842105264)...
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        _v1 = amount - 7.894736842105264
        _v2 = _v1.rolling(10).std()
        _v3 = _v2.rolling(60).mean()
        _v5 = close.rolling(5).mean()
        _v6 = _v3.rolling(5).mean()
        _v7 = close.rolling(5).cov(_v3) / close.rolling(5).var().replace(0, np.nan)
        _v4 = (_v3 - _v6) - _v7 * (close - _v5)
        _v8 = np.sign(_v4)
        _v9 = np.exp(_v8).replace([np.inf, -np.inf], np.nan)
        _v10 = pct_chg.rolling(10).sum()
        _v11 = (_v9 ** _v10).replace([np.inf, -np.inf], np.nan)
        _v12 = _v11 + amount
        factor = _v12.replace([np.inf, -np.inf], np.nan)
        return factor
