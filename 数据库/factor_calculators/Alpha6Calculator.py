from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha6Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        high = data.xs('high', level=0)
        
        # 3. 计算因子
        # 计算 (OPEN * 0.85) + (HIGH * 0.15)
        weighted_sum = open * 0.85 + high * 0.15
        
        # 计算 DELTA(weighted_sum, 4) - 4天差分
        delta = weighted_sum.diff(4, axis=0)

        # 计算 SIGN(DELTA)
        sign = delta.apply(np.sign)
        
        # 计算 RANK(SIGN)
        rank = sign.rank(axis=1, method='min')
        
        # 乘以-1
        factor = rank * -1
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha6",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "(RANK(SIGN(DELTA((((OPEN * 0.85) + (HIGH * 0.15))), 4)))* -1)",
    "description": "alpha191因子6。该因子先计算sign，再算rank，意义不明",
    "lookback_days": 4,
    "method": "Alpha6Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
