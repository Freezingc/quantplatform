from .base import FactorCalculator
import numpy as np
import pandas as pd

class dividend_yield_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["dv_ratio"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        dv_ratio = data.xs('dv_ratio', level=0)
        
        # 3. 股息率 = dv_ratio
        dividend_yield = dv_ratio
        
        # 4. 转换索引类型为字符串
        # dividend_yield.index = dividend_yield.index.astype(str)
        
        return dividend_yield

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "股息率",
    "creator": "suyanning",
    "create_date": "2025.09.12",
    "formula": "dv_ratio",
    "description": "股息率",
    "lookback_days": 1,
    "method": "dividend_yield_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
