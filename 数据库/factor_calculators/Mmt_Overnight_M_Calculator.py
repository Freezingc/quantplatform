from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Overnight_M_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq", "open_hfq"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        open_hfq = data.xs('open_hfq', level=0)
        
        # 3. 计算
        N = 1 * 21
        overnight_return = (open_hfq / close_hfq.shift(1)) - 1
        factor = overnight_return.rolling(N).sum()
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月隔夜动量",
    "creator": "jiangyushi",
    "create_date": "2025.12.02",
    "formula": "SUM(OPEN / SHIFT(CLOSE,1), 21)",
    "description": "过去1个月内，隔夜涨跌幅之和",
    "lookback_days": 21,
    "method": "Mmt_Overnight_M_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''
