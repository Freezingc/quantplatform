from .base import FactorCalculator
import numpy as np
import pandas as pd


class GP_Factor_080_Calculator(FactorCalculator):
    """遗传规划因子 #80，Rank-IC=0.050621
    表达式: mul(turnover, cs_rank(shift_1(ts_mean_5(cs_max(ts_delta_30(turnover))))))
    来源: 20260215_163844.csv
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        turnover = data.xs('turnover_rate', level=0)

        _v1 = turnover - turnover.shift(30)      # ts_delta_30 → DF
        _v2 = _v1.max(axis=1)                    # cs_max → Series
        _v3 = _v2.rolling(5).mean()              # ts_mean_5(Series) → Series
        _v4 = _v3.shift(1)                       # shift_1(Series) → Series
        # cs_rank(Series) → 在时间维度上排名（与原始算子 cs_rank(Series) 行为一致）
        _v5 = _v4.rank(pct=True)
        factor = (turnover.mul(_v5, axis=0)).replace([np.inf, -np.inf], np.nan)
        return factor
