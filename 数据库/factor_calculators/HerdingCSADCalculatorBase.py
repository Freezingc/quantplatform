"""CSAD 羊群效应计算器的数据库适配与进程内结果复用。"""

from __future__ import annotations

import pandas as pd

from .base import FactorCalculator
from .HerdingCSADCore import compute_herding_csad_factors


_RESULT_CACHE = {}


class HerdingCSADCalculatorBase(FactorCalculator):
    output_name = None

    def __init__(self, s, lookback_days=120):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]

    def calculate(self, start_time, end_time):
        """预取120个收益观测所需的121个收盘价。"""
        original_lookback = self.lookback_days
        self.lookback_days = original_lookback + 1
        try:
            processed_start = self.preprocess_time_range(start_time, end_time)
        finally:
            self.lookback_days = original_lookback
        factor = self._calculate(processed_start, end_time)
        return factor[
            (factor.index >= pd.to_datetime(start_time))
            & (factor.index <= pd.to_datetime(end_time))
        ]

    def _calculate(self, start_time, end_time):
        if self.output_name is None:
            raise NotImplementedError("子类必须指定 output_name")

        cache_key = (start_time, end_time, int(self.lookback_days))
        cached = _RESULT_CACHE.get(cache_key)
        if cached is None:
            data = self.fetch_data(start_time, end_time)
            close = data.xs("close_hfq", level=0).sort_index()
            returns = close.pct_change(fill_method=None)
            cached = compute_herding_csad_factors(
                returns,
                long_window=self.lookback_days,
                short_window=20,
                csad_std_window=21,
                group_size=10,
                min_observations=20,
                progress=True,
            )
            # 四个因子在配置和日更任务中连续执行。只保留最近一次区间，既复用
            # 昂贵的滚动相关矩阵，又避免长期任务不断累积历史结果。
            _RESULT_CACHE.clear()
            _RESULT_CACHE[cache_key] = cached

        return cached[self.output_name]
