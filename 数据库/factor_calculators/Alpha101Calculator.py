from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha101Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        amount = data.xs('amount', level=0)

        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol

        # 3. 因子计算逻辑
        # 第一部分：CORR(CLOSE, SUM(MEAN(vol,30), 37), 15)
        mean_vol_30 = vol.rolling(window=30).mean()
        sum_mean_vol_37 = mean_vol_30.rolling(window=37).sum()
        corr_part1 = close.rolling(window=15).corr(sum_mean_vol_37)
        rank_corr_part1 = corr_part1.rank(axis=1, pct=True)
        
        # 第二部分：CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(vol), 11)
        weighted_high_vwap = (high * 0.1) + (vwap * 0.9)
        rank_weighted = weighted_high_vwap.rank(axis=1, pct=True)
        rank_vol = vol.rank(axis=1, pct=True)
        
        # 计算相关性
        corr_part2 = rank_weighted.rolling(window=11).corr(rank_vol)
        rank_corr_part2 = corr_part2.rank(axis=1, pct=True)
        
        # 最终因子计算
        condition = rank_corr_part1 < rank_corr_part2
        factor = condition.astype(int) * -1
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha101",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK(CORR(CLOSE, SUM(MEAN(vol,30), 37), 15)) < RANK(CORR(RANK(((HIGH * 0.1) + (VWAP * 0.9))), RANK(vol), 11))) * -1)",
        "description": "比较两种不同的量价关系模式的有效性:长期量价趋势关系 vs 短期量价排名关系,当短期排名关系的量价协同性优于长期趋势关系时，因子取值为-1",
        "lookback_days": 79,
        "method": "Alpha101Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
