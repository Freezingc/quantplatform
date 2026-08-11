from .base import FactorCalculator
import numpy as np
import pandas as pd

class FlyOpenCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open_price = data.xs('open', level=0)
        turnover = data.xs('turnover_rate', level=0)
        
        # 3. 计算过去20天的开盘价和换手率的相关系数
        window = 20
        factor = open_price.rolling(window=window).corr(turnover)
        
        # 4. 处理可能的无穷大值和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "量价齐飞-开盘价",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "CORR(open, turnover_rate, 20)",
    "description": "量价齐飞-开盘价因子，计算过去20天开盘价与换手率的滚动相关系数",
    "lookback_days": 20,
    "method": "FlyOpenCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "量价相关性因子"
}
'''

