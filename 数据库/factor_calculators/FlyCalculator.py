from .base import FactorCalculator
import numpy as np
import pandas as pd

class FlyCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        turnover = data.xs('turnover_rate', level=0)
        
        # 3. 计算过去20天的close和turnover的相关系数
        # 对每一列（每只股票）计算close和turnover的滚动相关系数
        window = 20
        factor = close.rolling(window=window).corr(turnover)
        
        # 4. 处理可能的无穷大值和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "量价齐飞",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "CORR(close, turnover_rate, 20)",
    "description": "量价齐飞因子，计算过去20天收盘价与换手率的滚动相关系数",
    "lookback_days": 20,
    "method": "FlyCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "量价相关性因子"
}
'''

