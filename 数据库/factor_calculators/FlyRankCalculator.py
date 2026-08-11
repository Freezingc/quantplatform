from .base import FactorCalculator
import numpy as np
import pandas as pd

class FlyRankCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        turnover = data.xs('turnover_rate', level=0)
        
        # 3. 计算收盘价和换手率的排名（按行排名，即每个交易日对所有股票排名）
        close_rank = close.rank(axis=1)
        turnover_rank = turnover.rank(axis=1)
        
        # 4. 计算过去20天的排名相关系数
        window = 20
        factor = close_rank.rolling(window=window).corr(turnover_rank)
        
        # 5. 处理可能的无穷大值和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "量价齐飞-排名",
    "creator": "suyanning",
    "create_date": "2025.01.01",
    "formula": "CORR(RANK(close, axis=1), RANK(turnover_rate, axis=1), 20)",
    "description": "量价齐飞-排名因子，计算过去20天收盘价排名与换手率排名的滚动相关系数",
    "lookback_days": 20,
    "method": "FlyRankCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "量价相关性因子"
}
'''

