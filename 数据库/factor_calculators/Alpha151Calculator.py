from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha151Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 实现因子计算逻辑
        # SMA(CLOSE-DELAY(CLOSE,20),20,1)
        
        # 计算 CLOSE - DELAY(CLOSE, 20)
        close_diff = close - close.shift(20)
        
        # 计算 SMA(..., 20, 1)
        # SMA(X, N, M) 表示对X求N周期的加权移动平均，M为权重
        # SMA公式: SMA = M/N * X + (N-M)/N * 上一周期的SMA
        # 对于SMA(..., 20, 1)，权重为1/20
        
        factor = close_diff.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
    
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha151",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(CLOSE-DELAY(CLOSE,20),20,1)",
        "description": "当前收盘价相对于20天前收盘价的差值，然后对这个差值序列进行20天的加权移动平均",
        "lookback_days": 40,
        "method": "Alpha151Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
