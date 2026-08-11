"""Corwin-Schultz 价差因子的 FactorCalculator 适配层。"""

from __future__ import annotations

from .base import FactorCalculator
from .CorwinSchultzSpread import corwin_schultz_factors


class CorwinSchultzCalculatorBase(FactorCalculator):
    output_name = None

    def __init__(self, s, lookback_days=21):
        super().__init__(s, lookback_days)
        self.required_factors = ["high_hfq", "low_hfq"]

    def _calculate(self, start_time, end_time):
        if self.output_name is None:
            raise NotImplementedError("子类必须指定 output_name")

        data = self.fetch_data(start_time, end_time)
        high = data.xs("high_hfq", level=0).sort_index()
        low = data.xs("low_hfq", level=0).reindex_like(high)

        result = corwin_schultz_factors(
            high,
            low,
            composite_window=21,
            # 严格复现文章代码；不额外执行 alpha >= 0 截断。
            clip_negative=False,
        )
        return getattr(result, self.output_name)

