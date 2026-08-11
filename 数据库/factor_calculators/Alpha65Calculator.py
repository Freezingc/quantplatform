from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha65Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算6日均线
        mean_close_6 = close.rolling(window=6,min_periods=1).mean()
        
        # 计算因子 MEAN(CLOSE,6)/CLOSE
        factor = mean_close_6 / close
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha65",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "MEAN(CLOSE,6)/CLOSE",
        "description": "alpha191因子65",
        "lookback_days": 6,
        "method": "Alpha65Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

