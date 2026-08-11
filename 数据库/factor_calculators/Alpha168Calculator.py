from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha168Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 计算vol的20日移动平均
        vol_mean_20 = vol.rolling(window=20).mean()
        
        # 计算因子 (-1 * vol / MEAN(vol, 20))
        factor = -1 * vol / vol_mean_20
        
        # 重命名索引以符合返回格式
        factor.index.name = 'trade_date'
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha168",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1*vol/MEAN(vol,20))",
        "description": "当日成交量相对于过去20日平均成交量的偏离程度",
        "lookback_days": 19,
        "method": "Alpha168Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
