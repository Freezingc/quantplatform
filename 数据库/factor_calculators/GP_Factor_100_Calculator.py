from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_100_Calculator(FactorCalculator):
    """遗传规划因子 #100，Rank-IC=0.050410
    表达式: mul(turnover, cs_rank(pow(tanh(ts_delta_ratio_20(square(ts_delta_60(amount)))),
               ts_sum_10(cs_zscore(ts_delta_5(tanh(log(ts_rank_10(ts_std_20(cs_rank(ts_regression_5(close, 5.79)))))))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount", "turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close    = data.xs('close',         level=0)
        amount   = data.xs('amount',        level=0)
        turnover = data.xs('turnover_rate', level=0)

        # tanh(ts_delta_ratio_20(square(ts_delta_60(amount))))
        _v1 = amount - amount.shift(60)
        _v2 = _v1 ** 2
        _v3 = (_v2 - _v2.shift(20)) / _v2.shift(20).replace(0, np.nan)
        _v4 = np.tanh(_v3)

        # ts_regression_5(close, 5.79) → y=常数，cov(x,const)=0，beta=0 → 结果为 0
        _v5 = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        # cs_rank(zeros) → 0.5 (等值排名)；ts_std_20 → DF；ts_rank_10 → DF
        _v6 = _v5.rank(axis=1, pct=True)
        _v7 = _v6.rolling(20).std()
        _v8 = _v7.rolling(10).rank(pct=True)

        # log → DF；tanh → DF；ts_delta_5 → DF
        _v9  = np.log(_v8.where(_v8 > 0, np.nan))
        _v10 = np.tanh(_v9)
        _v11 = _v10 - _v10.shift(5)

        # cs_zscore(DF) → DF；ts_sum_10 → DF
        _v12 = _v11.sub(_v11.mean(axis=1), axis=0).div(_v11.std(axis=1) + 1e-10, axis=0)
        _v13 = _v12.rolling(10).sum()

        # pow → cs_rank → mul
        _v14 = (_v4 ** _v13).replace([np.inf, -np.inf], np.nan)
        _v15 = _v14.rank(axis=1, pct=True)
        factor = (turnover * _v15).replace([np.inf, -np.inf], np.nan)
        return factor
