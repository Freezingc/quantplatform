from .base import FactorCalculator
import numpy as np
import pandas as pd
from factor_calculators.DayReversalVolatilityCalculator import DayReversalVolatilityCalculator
from factor_calculators.DayReversalTurnoverCalculator import DayReversalTurnoverCalculator

class ModifiedDayReversalCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        factor1 = DayReversalVolatilityCalculator(self.s, 20).calculate(start_time, end_time)
        factor2 = DayReversalTurnoverCalculator(self.s, 20).calculate(start_time, end_time)
        
        # 修正日间反转 = factor1 + factor2
        factor = factor1 + factor2
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "修正日间反转",
    "creator": "suyanning",
    "create_date": "2025.09.25",
    "formula": "factor1 = DayReversalVolatilityCalculator ; factor2 = DayReversalTurnoverCalculator; factor = factor1 + factor2",
    "description": "修正日间反转因子，结合波动率和换手率的反转效应",
    "lookback_days": 20,
    "method": "ModifiedDayReversalCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
