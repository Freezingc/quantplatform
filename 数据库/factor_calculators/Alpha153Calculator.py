from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha153Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算不同周期的移动平均
        ma3 = close.rolling(window=3, min_periods=1).mean()
        ma6 = close.rolling(window=6, min_periods=1).mean()
        ma12 = close.rolling(window=12, min_periods=1).mean()
        ma24 = close.rolling(window=24, min_periods=1).mean()
        
        # 计算四个移动平均的算术平均
        factor = (ma3 + ma6 + ma12 + ma24) / 4
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha153",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MEAN(CLOSE,3)+MEAN(CLOSE,6)+MEAN(CLOSE,12)+MEAN(CLOSE,24))/4",
        "description": "收盘价在3日、6日、12日和24日四个时间尺度移动平均的加权综合趋势指标",
        "lookback_days": 23,
        "method": "Alpha153Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
