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


class GP_Factor_033_Calculator(FactorCalculator):
    """遗传规划因子 #33，Rank-IC=0.052515
    表达式: ts_max_5(mul(amount, sub(turnover, ts_regression_residual_60(
               ts_sum_5(ts_regression_residual_10(ts_mean_5(ts_mean_5(cs_max(
               ts_regression_30(ts_regression_slope_5(pct_chg),
               ts_corr_5(ts_corr_5(close, cs_min(vol)), ts_rank_5(ts_sum_20(ts_rank_60(turnover)))))))),
               pct_chg)), low))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "low", "vol", "amount", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        low      = data.xs('low',           level=0)
        vol      = data.xs('vol',           level=0)
        amount   = data.xs('amount',        level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)

        # ts_regression_slope_5(pct_chg) → DF
        _v1 = pct_chg.rolling(5).apply(_slope, raw=True)

        # cs_min(vol) → Series；ts_corr_5(close_DF, Series) → DF
        _v2 = vol.min(axis=1)
        _v3 = close.rolling(5).corr(_v2)

        # ts_sum_20(ts_rank_60(turnover)) → DF；ts_rank_5(DF) → DF
        _v4 = turnover.rolling(60).rank(pct=True)
        _v5 = _v4.rolling(20).sum()
        _v6 = _v5.rolling(5).rank(pct=True)

        # ts_corr_5(DF, DF) → DF；ts_regression_30(DF_v1, DF_v7) → DF
        _v7 = _v3.rolling(5).corr(_v6)
        _v8 = _v1.rolling(30).cov(_v7) / _v1.rolling(30).var().replace(0, np.nan)

        # cs_max(DF) → Series；ts_mean_5 × 2 → Series
        _v9  = _v8.max(axis=1)
        _v10 = _v9.rolling(5).mean()
        _v11 = _v10.rolling(5).mean()           # x = Series

        # ts_regression_residual_10(x=Series _v11, y=DF pct_chg) → 逐列循环
        _mx   = _v11.rolling(10).mean()
        _varx = _v11.rolling(10).var().replace(0, np.nan)
        _v12  = pd.DataFrame(index=pct_chg.index, columns=pct_chg.columns, dtype=float)
        for _c in pct_chg.columns:
            _my      = pct_chg[_c].rolling(10).mean()
            _cov     = _v11.rolling(10).cov(pct_chg[_c])
            _beta    = _cov / _varx
            _v12[_c] = ((pct_chg[_c] - _my) - _beta * (_v11 - _mx)).replace([np.inf, -np.inf], np.nan)

        # ts_sum_5(DF) → DF；ts_regression_residual_60(DF, DF_low) → DF（两者列名相同）
        _v16 = _v12.rolling(5).sum()
        _mx2  = _v16.rolling(60).mean()
        _my2  = low.rolling(60).mean()
        _varx2 = _v16.rolling(60).var().replace(0, np.nan)
        _beta2 = _v16.rolling(60).cov(low) / _varx2
        _v17  = (low - _my2) - _beta2 * (_v16 - _mx2)

        # sub(turnover, DF) → DF；mul(amount, DF) → DF；ts_max_5 → DF
        _v21 = turnover - _v17
        _v22 = amount * _v21
        factor = _v22.rolling(5).max().replace([np.inf, -np.inf], np.nan)
        return factor
