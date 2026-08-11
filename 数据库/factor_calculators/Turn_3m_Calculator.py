from .base import FactorCalculator
import numpy as np
import pandas as pd

class Turn_3m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        turnover_rate = data.xs('turnover_rate', level=0)
        
        # 3. 计算
        N = 3 * 21
        factor = turnover_rate.rolling(N).mean()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日均换手率（3个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.11.05",
    "formula": "MEAN(turnover_rate, 63)",
    "description": "个股最近 3 个月内日均换手率",
    "lookback_days": 63,
    "method": "Turn_3m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "流动性因子"
}
'''
