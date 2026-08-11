from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha7Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "amount", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 假设vol数据存在
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol
        
        # 计算VWAP - CLOSE
        vwap_close_diff = vwap - close
        
        # 计算MAX((VWAP - CLOSE), 3) - 3日最大值
        max_diff = vwap_close_diff.rolling(window=3).max()
        
        # 计算MIN((VWAP - CLOSE), 3) - 3日最小值
        min_diff = vwap_close_diff.rolling(window=3).min()
        
        # 计算DELTA(vol, 3) - 3日变化量
        delta_vol = vol.diff(periods=3)
        
        # 计算各项的排名
        rank_max = max_diff.rank(axis=1, pct=True)  # 横向排名
        rank_min = min_diff.rank(axis=1, pct=True)  # 横向排名
        rank_vol = delta_vol.rank(axis=1, pct=True)  # 横向排名
        
        # 组合因子
        factor = (rank_max + rank_min) * rank_vol
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha7",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "((RANK(MAX((VWAP - CLOSE), 3)) + RANK(MIN((VWAP - CLOSE), 3))) * RANK(DELTA(vol, 3)))",
    "description": "alpha191因子7",
    "lookback_days": 3,
    "method": "Alpha7Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
