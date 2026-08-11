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


class GP_Factor_035_Calculator(FactorCalculator):
    """遗传规划因子 #35，Rank-IC=0.052225
    表达式: ts_min_5(mul(amount, ts_delta_60(sub(ts_sum_10(ts_rank_10(cs_rank(
               ts_regression_5(ts_kurtosis_60(ts_regression_slope_20(
               cs_max(ts_delta_20(ts_mean_20(amount))))), ts_sum_10(pct_chg))))),
               pct_chg))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        amount  = data.xs('amount',  level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # ts_mean_20 → ts_delta_20 → cs_max → ts_regression_slope_20 → ts_kurtosis_60
        # 结果均为 Series（cs_max 降维）
        _v1 = amount.rolling(20).mean()
        _v2 = _v1 - _v1.shift(20)                          # ts_delta_20(DF) → DF
        _v3 = _v2.max(axis=1)                               # cs_max → Series
        _v4 = _v3.rolling(20).apply(_slope, raw=True)       # ts_regression_slope_20(Series) → Series
        _v5 = _v4.rolling(60).kurt()                        # ts_kurtosis_60(Series) → Series

        # ts_sum_10(pct_chg) → DF
        _v6 = pct_chg.rolling(10).sum()

        # ts_regression_5(x=Series _v5, y=DF _v6) → 逐列循环，结果为 DF
        _mx   = _v5.rolling(5).mean()
        _varx = _v5.rolling(5).var().replace(0, np.nan)
        _v7   = pd.DataFrame(index=_v6.index, columns=_v6.columns, dtype=float)
        for _c in _v6.columns:
            _cov     = _v5.rolling(5).cov(_v6[_c])
            _v7[_c]  = (_cov / _varx).replace([np.inf, -np.inf], np.nan)

        # cs_rank(DF) → DF；ts_rank_10(DF) → DF；ts_sum_10(DF) → DF
        _v8  = _v7.rank(axis=1, pct=True)
        _v9  = _v8.rolling(10).rank(pct=True)
        _v10 = _v9.rolling(10).sum()

        # sub(DF, pct_chg) → ts_delta_60 → mul(amount, DF) → ts_min_5
        _v11 = _v10 - pct_chg
        _v12 = _v11 - _v11.shift(60)
        _v13 = amount * _v12
        factor = _v13.rolling(5).min().replace([np.inf, -np.inf], np.nan)
        return factor
