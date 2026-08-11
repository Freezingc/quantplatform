from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha41Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 需要成交量来计算VWAP
        amount = data.xs('amount', level=0)

        # 3. 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol

        # 4. 计算DELTA(VWAP, 3) - VWAP的3期差分
        delta_vwap = vwap.diff(3)

        # 5. 计算MAX(DELTA(VWAP, 3), 5) - 5期窗口内的最大值
        max_delta = delta_vwap.rolling(window=5).max()

        # 6. 计算RANK(MAX(...)) - 横截面的排名
        # 使用rank(axis=1)进行横截面排名，pct=True表示使用百分比排名
        rank_max = max_delta.rank(axis=1, pct=True)

        # 7. 乘以-1
        factor = rank_max * -1

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha41",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "(RANK(MAX(DELTA((VWAP), 3), 5))* -1)",
    "description": "alpha191因子41",
    "lookback_days": 7,
    "method": "Alpha41Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''