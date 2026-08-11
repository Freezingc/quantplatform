from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha106Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # CLOSE - DELAY(CLOSE, 20) 表示当前收盘价减去20天前的收盘价
        # 使用shift(20)获取20天前的数据
        close_20_days_ago = close.shift(20)
        
        # 计算因子值
        factor = close - close_20_days_ago
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha106",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "CLOSE-DELAY(CLOSE,20)",
        "description": "股票当前收盘价相对于20天前收盘价的价格变动幅度。",
        "lookback_days": 19,
        "method": "Alpha106Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
