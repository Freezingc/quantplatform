from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha88Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算20日前的收盘价（使用shift(20)实现DELAY功能）
        close_delay_20 = close.shift(20)
        
        # 计算因子：(当前收盘价 - 20日前收盘价) / 20日前收盘价 * 100
        factor = (close - close_delay_20) / close_delay_20 * 100
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha88",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "(CLOSE-DELAY(CLOSE,20))/DELAY(CLOSE,20)*100",
        "description": "alpha191因子88",
        "lookback_days": 20,
        "method": "Alpha88Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
