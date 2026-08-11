from .base import FactorCalculator
import numpy as np
import pandas as pd

def _slope(arr):
    if not np.all(np.isfinite(arr)) or len(arr) < 2:
        return np.nan
    t = np.arange(1, len(arr) + 1, dtype=float)
    cov = np.mean(t * arr) - np.mean(t) * np.mean(arr)
    var_t = np.var(t)
    return np.nan if var_t <= 0 else cov / var_t


class GP_Factor_081_Calculator(FactorCalculator):
    """遗传规划因子 #81，Rank-IC=0.050621
    表达式: mul(turnover, ts_rank_60(cs_zscore(cs_max(ts_corr_60(cs_zscore(close),
               sigmoid(ts_kurtosis_30(ts_rank_60(ts_regression_30(ts_delta_20(ts_regression_slope_60(sin(ts_median_30(pct_chg)))), turnover)))))))))
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

        # cs_zscore(close) → DF
        _v1 = close.sub(close.mean(axis=1), axis=0).div(close.std(axis=1) + 1e-10, axis=0)

        # ts_regression_slope_60(sin(ts_median_30(pct_chg))) → DF
        _v2 = pct_chg.rolling(30).median()
        _v3 = np.sin(_v2)
        _v4 = _v3.rolling(60).apply(_slope, raw=True)

        # ts_delta_20 → DF；ts_regression_30(DF, turnover_DF) → DF
        _v5 = _v4 - _v4.shift(20)
        _v6 = _v5.rolling(30).cov(turnover) / _v5.rolling(30).var().replace(0, np.nan)

        # ts_rank_60 → DF；ts_kurtosis_30 → DF；sigmoid → DF
        _v7 = _v6.rolling(60).rank(pct=True)
        _v8 = _v7.rolling(30).kurt()
        _v9 = 1 / (1 + np.exp(-_v8))

        # ts_corr_60(DF_v1, DF_v9) → DF
        _v10 = _v1.rolling(60).corr(_v9)

        # cs_max(DF) → Series
        _v11 = _v10.max(axis=1)

        # cs_zscore(Series) → 时间维度标准化（与原始算子行为一致）
        _v12 = (_v11 - _v11.mean()) / (_v11.std() + 1e-10)

        # ts_rank_60(Series) → Series；mul(turnover_DF, Series) → DF.mul(axis=0)
        _v13 = _v12.rolling(60).rank(pct=True)
        factor = turnover.mul(_v13, axis=0).replace([np.inf, -np.inf], np.nan)
        return factor
