from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha56Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "open", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        open_ = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        
        # 计算第一部分: RANK((OPEN - TSMIN(OPEN, 12)))
        # 计算TSMIN(OPEN, 12)
        tmin_open = open_.rolling(window=12).min()
        part1 = open_ - tmin_open
        # 对每个时间点进行横截面排名
        rank_part1 = part1.rank(axis=1, pct=True)
        
        # 计算第二部分: RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(vol,40), 19), 13))^5))
        # 计算(HIGH + LOW)/2
        hl_avg = (high + low) / 2
        # 计算SUM(((HIGH + LOW)/2), 19)
        sum_hl = hl_avg.rolling(window=19).sum()
        
        # 计算MEAN(vol, 40)
        mean_vol = vol.rolling(window=40).mean()
        # 计算SUM(MEAN(vol,40), 19)
        sum_mean_vol = mean_vol.rolling(window=19).sum()
        
        # 计算CORR(SUM_HL, SUM_MEAN_VOL, 13)
        corr_values = sum_hl.rolling(window=13).corr(sum_mean_vol)
        
        # 对CORR结果进行横截面排名
        rank_corr = corr_values.rank(axis=1, pct=True)
        
        # 对排名结果进行5次方运算
        rank_corr_pow = rank_corr ** 5
        
        # 对最终结果再进行一次排名
        rank_part2 = rank_corr_pow.rank(axis=1, pct=True)
        
        # 比较两部分排名结果
        factor = (rank_part1 < rank_part2).astype(int)

        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha56",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "(RANK((OPEN - TSMIN(OPEN, 12))) < RANK((RANK(CORR(SUM(((HIGH + LOW) / 2), 19), SUM(MEAN(vol,40), 19), 13))^5)))",
        "description": "alpha191因子56",
        "lookback_days": 69,
        "method": "Alpha56Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

