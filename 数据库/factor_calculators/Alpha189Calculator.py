from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha189Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算6日移动平均
        close_mean_6 = close.rolling(window=6, min_periods=6).mean()
        
        # 计算收盘价与6日平均的绝对差值
        abs_diff = (close - close_mean_6).abs()
        
        # 计算绝对差值的6日移动平均
        factor = abs_diff.rolling(window=6, min_periods=6).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha189",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN(ABS(CLOSE-MEAN(CLOSE,6)),6)",
        "description": "衡量了股价相对其短期趋势（6日均线）的绝对偏离程度在过去6天的平均波动幅度",
        "lookback_days": 9,
        "method": "Alpha189Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
