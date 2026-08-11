from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha102Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 需要添加vol数据

        # 3. 实现因子计算逻辑
        # DELAY(vol, 1) - 成交量滞后1期
        vol_delay = vol.shift(1)
        
        # vol - DELAY(vol, 1)
        vol_diff = vol - vol_delay
        
        # MAX(vol-DELAY(vol,1), 0)
        max_diff = vol_diff.clip(lower=0)
        
        # ABS(vol-DELAY(vol,1))
        abs_diff = vol_diff.abs()

        
        # SMA(MAX(vol-DELAY(vol,1),0),6,1)
        numerator = max_diff.rolling(window=6, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*5)/6 if len(x) == 6 else x.mean()
        )
        # SMA(ABS(vol-DELAY(vol,1)),6,1)
        denominator = abs_diff.rolling(window=6, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*5)/6 if len(x) == 6 else x.mean()
        )
        # 最终因子计算
        factor = numerator / denominator * 100
        
        # 确保factor的格式与输入数据一致
        factor = factor.reindex_like(vol)  # 保持相同的索引和列结构
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha102",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA(MAX(vol-DELAY(vol,1),0),6,1)/SMA(ABS(vol-DELAY(vol,1)),6,1)*100",
        "description": "衡量的是成交量变化的非对称性强度",
        "lookback_days": 6,
        "method": "Alpha102Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
