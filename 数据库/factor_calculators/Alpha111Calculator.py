from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha111Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high","vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算核心值
        # 避免除零错误
        high_low_diff = high - low
        high_low_diff = high_low_diff.replace(0, np.nan)
        
        # 计算核心值
        core = vol * ((close - low) - (high - close)) / high_low_diff
        
        
        # 5. 计算两个SMA并相减
        sma_11 = core.rolling(window=11, min_periods=2).apply(
            lambda x: (x[-1] + x[:-1].mean()*9)/11 if len(x) == 11 else x.mean()
        )
        sma_4 = core.rolling(window=4, min_periods=2).apply(
            lambda x: (x[-1] + x[:-1].mean()*2)/4 if len(x) == 4 else x.mean()
        )
        # 6. 计算最终因子
        factor = sma_11 - sma_4
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha111",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),11,2)-SMA(VOL*((CLOSE-LOW)-(HIGH-CLOSE))/(HIGH-LOW),4,2)",
        "description": "基于成交量加权的价格位置相对强度在两种不同平滑周期下的差异",
        "lookback_days": 10,
        "method": "Alpha111Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
