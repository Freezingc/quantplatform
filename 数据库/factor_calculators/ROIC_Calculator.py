from .base import FactorCalculator
import numpy as np
import pandas as pd

class ROIC_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_financial_factors = ["roic"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_financial_data(start_time, end_time)
        # 2. 提取所需因子
        roic = data.xs('roic', level=0)
        
        roic = roic.replace([np.inf, -np.inf], np.nan)
        
        return roic

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "投资资本回报率",
    "creator": "zhangyudong",
    "create_date": "2025.12.10",
    "formula": "roic",
    "description": "投资资本回报率",
    "lookback_days": 1,
    "method": "ROIC_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
