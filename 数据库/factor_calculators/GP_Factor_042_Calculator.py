from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_042_Calculator(FactorCalculator):
    """遗传规划因子 #42，Rank-IC=0.051798
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(turnover)), ts_sum_10(cs_zscore(ts_regression_residual_60(relu(ts_std_30(turnover)), vol))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        vol      = data.xs('vol',          level=0)
        turnover = data.xs('turnover_rate', level=0)
        # tanh(ts_delta_ratio_20(turnover))
        _v1 = (turnover - turnover.shift(20)) / turnover.shift(20).replace(0, np.nan)
        _v2 = np.tanh(_v1)
        # ts_regression_residual_60(relu(ts_std_30(turnover)), vol)
        _x  = turnover.rolling(30).std().clip(lower=0)          # relu(ts_std_30(turnover))
        _mx = _x.rolling(60).mean()
        _my = vol.rolling(60).mean()
        _beta = _x.rolling(60).cov(vol) / _x.rolling(60).var().replace(0, np.nan)
        _v3 = (vol - _my) - _beta * (_x - _mx)                  # residual
        # cs_zscore → ts_sum_10
        _v4 = _v3.sub(_v3.mean(axis=1), axis=0).div(_v3.std(axis=1) + 1e-10, axis=0)
        _v5 = _v4.rolling(10).sum()
        # pow → cs_rank → mul
        _v6 = (_v2 ** _v5).replace([np.inf, -np.inf], np.nan)
        _v7 = _v6.rank(axis=1, pct=True)
        factor = (turnover * _v7).replace([np.inf, -np.inf], np.nan)
        return factor
