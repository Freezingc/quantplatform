from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha8Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "amount", "high", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol
        
        # 4. 计算中间表达式 (((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)
        part1 = ((high + low) / 2) * 0.2
        part2 = vwap * 0.8
        combined = part1 + part2
        
        # 5. 计算DELTA(..., 4) - 4天差值
        delta = combined.diff(4)
        
        # 6. 乘以-1
        neg_delta = delta * -1
        
        # 7. 计算RANK (横截面排序)
        # 对每个交易日的数据进行排序并计算rank
        factor = neg_delta.rank(pct=True, axis=1)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha8",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "RANK(DELTA(((((HIGH + LOW) / 2) * 0.2) + (VWAP * 0.8)), 4) * -1)",
    "description": "alpha191因子8",
    "lookback_days": 4,
    "method": "Alpha8Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
