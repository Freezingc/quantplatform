from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha42Calculator(FactorCalculator):
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
        # 计算HIGH的10日标准差
        std_high_10 = high.rolling(window=10).std()
        
        # 对标准差进行排名（横截面排名）
        rank_std = std_high_10.rank(axis=1, pct=True)  # pct=True返回百分比排名
        
        # 计算-1 * RANK(STD(HIGH, 10))
        neg_rank_std = -1 * rank_std
        
        # 计算HIGH和vol的10日相关系数
        corr_high_vol = high.rolling(window=10).corr(vol)
        
        # 计算最终因子：(-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, vol, 10)
        factor = neg_rank_std * corr_high_vol
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha42",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "((-1 * RANK(STD(HIGH, 10))) * CORR(HIGH, vol, 10))",
    "description": "alpha191因子42",
    "lookback_days": 9,
    "method": "Alpha42Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''