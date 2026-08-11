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


class GP_Factor_095_Calculator(FactorCalculator):
    """遗传规划因子 #95，Rank-IC=0.050458
    表达式: ts_max_5(ts_regression_residual_10(
               ts_mean_5(ts_mean_5(cs_max(ts_kurtosis_60(ts_regression_slope_20(
                   cs_rank(ts_mean_5(gt(lt(ts_skewness_20(ts_regression_slope_5(
                       mul(amount, pct_chg))),
                       ts_rank_20(ts_median_5(pct_chg))),
                       ts_rank_60(cs_zscore(close)))))))))),
               cs_zscore(pct_chg)))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount", "pct_chg"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close   = data.xs('close',   level=0)
        amount  = data.xs('amount',  level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # 逐步计算 x（最终为 Series）
        _v1  = amount * pct_chg
        _v2  = _v1.rolling(5).apply(_slope, raw=True)       # ts_regression_slope_5 → DF
        _v3  = _v2.rolling(20).skew()                        # ts_skewness_20 → DF
        _v4  = pct_chg.rolling(5).median()                   # ts_median_5 → DF
        _v5  = _v4.rolling(20).rank(pct=True)                # ts_rank_20 → DF
        _v6  = (_v3 < _v5).astype(float)                    # lt → DF
        _v7  = close.sub(close.mean(axis=1), axis=0).div(close.std(axis=1) + 1e-10, axis=0)  # cs_zscore
        _v8  = _v7.rolling(60).rank(pct=True)               # ts_rank_60 → DF
        _v9  = (_v6 > _v8).astype(float)                    # gt → DF
        _v10 = _v9.rolling(5).mean()                         # ts_mean_5 → DF
        _v11 = _v10.rank(axis=1, pct=True)                  # cs_rank → DF
        _v12 = _v11.rolling(20).apply(_slope, raw=True)     # ts_regression_slope_20 → DF
        _v13 = _v12.rolling(60).kurt()                       # ts_kurtosis_60 → DF
        _v14 = _v13.max(axis=1)                              # cs_max → Series
        _v15 = _v14.rolling(5).mean()                        # ts_mean_5 → Series
        _v16 = _v15.rolling(5).mean()                        # ts_mean_5 → Series  (x)

        # y：cs_zscore(pct_chg) → DF
        _v17 = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(pct_chg.std(axis=1) + 1e-10, axis=0)

        # ts_regression_residual_10(x=Series _v16, y=DF _v17) → 逐列循环
        _mx   = _v16.rolling(10).mean()
        _varx = _v16.rolling(10).var().replace(0, np.nan)
        _v18  = pd.DataFrame(index=_v17.index, columns=_v17.columns, dtype=float)
        for _c in _v17.columns:
            _my      = _v17[_c].rolling(10).mean()
            _cov     = _v16.rolling(10).cov(_v17[_c])
            _beta    = _cov / _varx
            _v18[_c] = ((_v17[_c] - _my) - _beta * (_v16 - _mx)).replace([np.inf, -np.inf], np.nan)

        factor = _v18.rolling(5).max().replace([np.inf, -np.inf], np.nan)
        return factor
