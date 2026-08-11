from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha105Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open","vol","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 实现因子计算逻辑
        # (-1 * CORR(RANK(OPEN), RANK(vol), 10))
        
        # 对每日的OPEN价格进行横截面排名
        rank_open = open.rank(axis=1, pct=True, method='dense')
        
        # 对每日的vol进行横截面排名
        rank_vol = vol.rank(axis=1, pct=True, method='dense')
        
        # 计算滚动10天的相关性
        # 注意：这里使用rolling(window=10).corr()计算相关性
        factor = rank_open.rolling(window=10).corr(rank_vol)
        
        # 乘以-1
        factor = -1 * factor
        
        return factor
'''
# 添加因子信息
    {
        "factor_name": "Alpha105",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1 * CORR(RANK(OPEN), RANK(vol), 10))",
        "description": "衡量过去10天开盘价与成交量在横截面上的排名相关性，并取其负值",
        "lookback_days": 9,
        "method": "Alpha105Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
