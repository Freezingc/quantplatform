from .base import FactorCalculator
import numpy as np
import pandas as pd

class DV_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["dv_ttm"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        dv = data.xs('dv_ttm', level=0)
        
        return dv

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "dv_ttm股息率",
    "creator": "jiangyushi",
    "create_date": "2025.12.02",
    "formula": "dv",
    "description": "DP_TTM（现金股利 / 股票市值）",
    "lookback_days": 1,
    "method": "DV_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
