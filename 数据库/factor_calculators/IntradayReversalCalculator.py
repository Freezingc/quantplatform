from .base import FactorCalculator
import numpy as np
import pandas as pd

class IntradayReversalCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)
        
        # 3. 计算日内反转因子
        # 日内收益率 = 收盘价 / 开盘价 - 1
        factor = close / open - 1
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日内反转",
    "creator": "suyanning",
    "create_date": "2025.09.25",
    "formula": "close / open - 1",
    "description": "日内反转因子，计算当日收盘价相对于开盘价的涨跌幅",
    "lookback_days": 1,
    "method": "IntradayReversalCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
