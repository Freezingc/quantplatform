from .base import FactorCalculator
import numpy as np
import pandas as pd

class Std_4m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        
        # 3. 计算
        N = 4 * 21
        ret = close_hfq.pct_change()
        factor = ret.rolling(N).std()

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日收益率波动（4个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.11.19",
    "formula": "STD(daily_ret, 84)",
    "description": "个股最近 4 个月内日收益率序列的标准差",
    "lookback_days": 84,
    "method": "Std_4m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "波动率因子"
}
'''
