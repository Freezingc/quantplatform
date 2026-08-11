from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha74Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        vwap = amount * 10 / vol  # 计算VWAP
        
        # 计算第一部分因子
        # SUM(((LOW * 0.35) + (VWAP * 0.65)), 20)
        part1 = (low * 0.35 + vwap * 0.65).rolling(20).sum()
        
        # SUM(MEAN(vol,40), 20)
        part2 = vol.rolling(window=40,min_periods=1).mean().rolling(window=20).sum()
        
        # CORR(part1, part2, 7)
        corr1 = part1.rolling(window=7,min_periods=1).corr(part2)
        
        # RANK(CORR)
        rank1 = corr1.rank(axis=1, pct=True)
        
        # 计算第二部分因子
        # RANK(VWAP)
        rank_vwap = vwap.rank(axis=1, pct=True)
        
        # RANK(vol)
        rank_vol = vol.rank(axis=1, pct=True)
        
        # CORR(RANK(VWAP), RANK(vol), 6)
        corr2 = rank_vwap.rolling(window=6,min_periods=1).corr(rank_vol)
        
        # RANK(CORR)
        rank2 = corr2.rank(axis=1, pct=True)
        
        # 合并两部分因子
        factor = rank1 + rank2
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha74",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(RANK(CORR(SUM(((LOW * 0.35) + (VWAP * 0.65)), 20), SUM(MEAN(vol,40), 20), 7)) + RANK(CORR(RANK(VWAP), RANK(vol), 6)))",
        "description": "alpha191因子74",
        "lookback_days": 64,
        "method": "Alpha74Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

