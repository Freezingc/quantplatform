from .base import FactorCalculator
import numpy as np
import pandas as pd

class Low_R_Std_4m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq", "low_hfq"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        low_hfq = data.xs('low_hfq', level=0)
        
        # 3. 计算
        N = 4 * 21
        prev_close = close_hfq.shift(1)
        low_return_series = (low_hfq / prev_close) - 1
        factor = low_return_series.rolling(window=N).std()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日内最大跌幅波动率（4个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.11.19",
    "formula": "STD(LOW_HFQ / SHIFT(CLOSE_HFQ, 1) - 1, 84)",
    "description": "在过去 4 个月内，用个股每日最低价除以前一日收盘价计算日内最大跌幅，此处有复权处理，再计算最大跌幅序列的标准差",
    "lookback_days": 84,
    "method": "Low_R_Std_4m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "波动率因子"
}
'''
