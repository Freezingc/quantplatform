from .base import FactorCalculator
import numpy as np
import pandas as pd

class TerriDecayReturnCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "pct_chg"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)

        # 2. 提取所需因子
        close = data.xs('close', level=0)
        pct_chg = data.xs('pct_chg', level=0)

        # 3. 计算市场平均收益率
        market = close.mean(axis=1).pct_change() * 100

        # 4. 计算惊恐度 Terri
        bias = (pct_chg.sub(market, axis=0)).abs()
        bench = pct_chg.abs().add(market.abs(), axis=0) + 0.1
        terri = bias / bench

        # 5. 注意力衰减
        terri_decay = terri - terri.rolling(2).mean().shift(1)
        terri_decay[terri_decay < 0] = 0

        # 6. 注意力衰减-惊恐收益（滚动均值）
        terri_decay_ret = (terri_decay * pct_chg).rolling(20).mean().iloc[20:]

        return terri_decay_ret

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "注意力衰减-惊恐收益",
    "creator": "suyanning",
    "create_date": "2025.10.16",
    "formula": "market = SMA_axis1(pct_change(mean(close)),1)*100; bias = ABS(pct_chg - market); bench = ABS(pct_chg) + ABS(market) + 0.1; terri = bias/bench; terri_decay = terri - DELAY(SMA(terri,2),1); terri_decay = MAX(terri_decay,0); SMA(terri_decay*pct_chg, N)",
    "description": "注意力衰减-惊恐收益：对惊恐度做衰减后的收益滚动均值",
    "lookback_days": 20,
    "method": "TerriDecayReturnCalculator",
    "start_time": "",
    "end_time": ""
}
'''


