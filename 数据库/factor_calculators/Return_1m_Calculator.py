from .base import FactorCalculator
import numpy as np
import pandas as pd

class Return_1m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        
        # 3. 计算
        N = 1 * 21
        factor = close_hfq.pct_change(N)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "1个月收益率",
    "creator": "jiangyushi",
    "create_date": "2025.10.23",
    "formula": "(price_t / price_{t-21}) - 1",
    "description": "个股最近1个月收益率",
    "lookback_days": 21,
    "method": "Return_1m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''
