from .base import FactorCalculator
import numpy as np
import pandas as pd

class Bias_Turn_6m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        turnover_rate = data.xs('turnover_rate', level=0)
        
        # 3. 计算
        N = 6 * 21
        LN = 21 * 12 * 2  # 2年 = 504个交易日
        mean_turn = turnover_rate.rolling(N).mean()
        long_mean = turnover_rate.rolling(LN).mean()

        factor = mean_turn / long_mean - 1

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日均换手率偏离（6个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.11.05",
    "formula": "MEAN(turnover_rate, 126) / MEAN(turnover_rate, 504) - 1",
    "description": "个股最近 6 个月内日均换手率除以最近 2 年内日均换手率，再减去 1",
    "lookback_days": 504,
    "method": "Bias_Turn_6m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "流动性因子"
}
'''
