from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha155Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        vol = data.xs('vol', level=0)
        
        # 4. 计算因子
        sma_vol_13_2 = vol.rolling(window=13, min_periods=2).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        sma_vol_27_2 =  vol.rolling(window=27, min_periods=2).apply(
            lambda x: (x[-1] + x[:-1].mean()*25)/27 if len(x) == 27 else x.mean()
        )
        
        # 第一项: SMA(vol,13,2)-SMA(vol,27,2)
        term1 = sma_vol_13_2 - sma_vol_27_2
        
        # 第二项: SMA(term1, 10, 2)
        term2 = term1.rolling(window=10, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*9)/10 if len(x) == 10 else x.mean()
        )
        
        # 最终因子: term1 - term2
        factor = term1 - term2
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha155",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(vol,13,2)-SMA(vol,27,2)-SMA(SMA(vol,13,2)-SMA(vol,27,2),10,2)",
        "description": "基于成交量的双重平滑差值动量指标，通过比较13日与27日成交量平滑均值的差值，再减去该差值的10日平滑均值，捕捉成交量变化趋势的加速度",
        "lookback_days": 46,
        "method": "Alpha155Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
