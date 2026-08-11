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


class GP_Factor_059_Calculator(FactorCalculator):
    """遗传规划因子 #59，Rank-IC=0.051038
    表达式: max(sub(turnover, cs_zscore(shift_5(ts_regression_slope_5(ts_mean_5(cs_max(ts_corr_60(
               ts_min_5(cs_sum(ts_delta_30(ts_std_10(ts_sum_5(ts_median_60(pct_chg)))))),
               ts_corr_30(open, ts_rank_60(sign(ts_max_10(ts_median_60(pct_chg)))))))))))),
           ts_sum_5(pct_chg))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "pct_chg", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        open_    = data.xs('open',          level=0)
        pct_chg  = data.xs('pct_chg',       level=0)
        turnover = data.xs('turnover_rate', level=0)

        # 右侧 ts_corr_60 的两个参数
        # 参数1: ts_min_5(cs_sum(ts_delta_30(ts_std_10(ts_sum_5(ts_median_60(pct_chg))))))
        _a1 = pct_chg.rolling(60).median()
        _a2 = _a1.rolling(5).sum()
        _a3 = _a2.rolling(10).std()
        _a4 = _a3 - _a3.shift(30)
        _a5 = _a4.sum(axis=1)          # cs_sum → Series
        _a6 = _a5.rolling(5).min()     # ts_min_5(Series) → Series

        # 参数2: ts_corr_30(open, ts_rank_60(sign(ts_max_10(ts_median_60(pct_chg)))))
        _b1 = pct_chg.rolling(60).median()
        _b2 = _b1.rolling(10).max()
        _b3 = np.sign(_b2)
        _b4 = _b3.rolling(60).rank(pct=True)
        _b5 = open_.rolling(30).corr(_b4)   # DF.rolling.corr(DF) → DF

        # ts_corr_60(Series _a6, DF _b5) → 逐列
        _v1 = pd.DataFrame(index=_b5.index, columns=_b5.columns, dtype=float)
        for _c in _b5.columns:
            _v1[_c] = _a6.rolling(60).corr(_b5[_c])

        # cs_max(DF) → Series；ts_mean_5 → Series；ts_regression_slope_5 → Series
        _v2 = _v1.max(axis=1)
        _v3 = _v2.rolling(5).mean()
        _v4 = _v3.rolling(5).apply(_slope, raw=True)
        _v5 = _v4.shift(5)

        # cs_zscore(Series) → 在时间维度上标准化（与原始算子一致）
        _v6 = (_v5 - _v5.mean()) / (_v5.std() + 1e-10)

        # sub(turnover_DF, Series) → DF.sub(axis=0)
        _v7 = turnover.sub(_v6, axis=0)

        # max(DF, DF)
        _v8 = pct_chg.rolling(5).sum()
        factor = _v7.where(_v7 >= _v8, _v8).replace([np.inf, -np.inf], np.nan)
        return factor
