from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha30Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        #计算逻辑暂未实现
        
        #return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha30",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "WMA((REGRESI(CLOSE/DELAY(CLOSE)-1,MKT,SMB,HML，60))^2,20)",
    "description": "alpha191因子30​",
    "lookback_days": 78,
    "method": "Alpha30Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''