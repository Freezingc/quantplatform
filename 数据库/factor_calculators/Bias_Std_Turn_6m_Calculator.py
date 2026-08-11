from .base import FactorCalculator
import numpy as np
import pandas as pd

class Bias_Std_Turn_6m_Calculator(FactorCalculator):
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
        std_turn = turnover_rate.rolling(N).std()
        long_std = turnover_rate.rolling(LN).std()

        factor = std_turn / long_std - 1

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日换手率波动偏离（6个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.11.09",
    "formula": "STD(turnover_rate, 126) / STD(turnover_rate, 504) - 1",
    "description": "个股最近 6 个月内日换手率序列的标准差除以最近 2 年内日换手率序列的标准差，再减去 1",
    "lookback_days": 504,
    "method": "Bias_Std_Turn_6m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "流动性因子"
}
'''
