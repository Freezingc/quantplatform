from .base import FactorCalculator
import numpy as np
import pandas as pd

class Mmt_Highest_Days_A_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq"]

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        
        # 3. 计算逻辑
        N = 252 
        factor = close_hfq.rolling(window=N, min_periods=N//2).apply(
            lambda x: (len(x) - 1 - x.argmax()), 
            raw=True
        )

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "近1年最高价日期距今的天数",
    "creator": "jiangyushi",
    "create_date": "2026.01.22",
    "formula": "DAYS_SINCE_MAX(close_hfq, 252)",
    "description": "过去1年最高价出现的日期距离当前日期的天数",
    "lookback_days": 252,
    "method": "Mmt_Highest_Days_A_Calculator",
    "start_time": "", 
    "end_time": "",  
    "category": "动量反转因子"
}
'''