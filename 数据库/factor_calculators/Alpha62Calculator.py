from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha62Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["high", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 计算RANK(vol)
        rank_vol = vol.rank(axis=1, pct=True)
        
        # 计算5日CORR(HIGH, RANK(vol))
        # 使用rolling计算相关性
        corr_df = high.rolling(window=5).corr(other=rank_vol)
        
        # 计算-1 * CORR
        factor = -1 * corr_df
        return factor
        
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha62",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(-1 * CORR(HIGH, RANK(VOLUME), 5))",
        "description": "alpha191因子62",
        "lookback_days": 4,
        "method": "Alpha62Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

