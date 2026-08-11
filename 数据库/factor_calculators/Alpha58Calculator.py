from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha58Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算每日收盘价是否大于前一日收盘价
        close_shifted = close.shift(1)  # DELAY(CLOSE,1)
        up_days = (close > close_shifted).astype(int)  # CLOSE>DELAY(CLOSE,1)
        
        # 计算20天内上涨天数的总和
        sum_up_days = up_days.rolling(20).sum()  # COUNT(...,20)
        
        # 计算因子值
        factor = sum_up_days / 20 * 100
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha58",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "COUNT(CLOSE>DELAY(CLOSE,1),20)/20*100",
        "description": "alpha191因子58",
        "lookback_days": 20,
        "method": "Alpha58Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

