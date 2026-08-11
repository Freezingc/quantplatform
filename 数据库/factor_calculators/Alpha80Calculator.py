from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha80Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 获取成交量数据
        
        # 实现因子计算逻辑
        # DELAY(vol,5) 表示5天前的成交量
        delay_vol = vol.shift(5)
        
        # 计算因子：(vol-DELAY(vol,5))/DELAY(vol,5)*100
        factor = (vol - delay_vol) / delay_vol * 100
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha80",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(vol-DELAY(vol,5))/DELAY(vol,5)*100",
        "description": "alpha191因子80",
        "lookback_days": 4,
        "method": "Alpha80Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

