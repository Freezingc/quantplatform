from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha12Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "amount", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)  # 添加open数据
        vol = data.xs('vol', level=0)       # 添加成交量数据，假设字段名为'vol'
        amount = data.xs('amount', level=0) # 添加成交金额数据，假设字段名为'amount'

        # 3. 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol
        
        # 4. 计算 SUM(VWAP, 10) / 10
        sum_vwap_10 = vwap.rolling(window=10).sum()
        avg_vwap_10 = sum_vwap_10 / 10
        
        # 5. 计算第一部分：RANK((OPEN - (SUM(VWAP, 10) / 10)))
        part1 = open - avg_vwap_10
        # 对每个时间点横截面排名
        rank_part1 = part1.rank(axis=1, pct=True)
        
        # 6. 计算第二部分：-1 * RANK(ABS((CLOSE - VWAP)))
        part2 = abs(close - vwap)
        rank_part2 = part2.rank(axis=1, pct=True)
        negative_rank_part2 = -1 * rank_part2
        
        # 7. 计算最终因子
        factor = rank_part1 * negative_rank_part2
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha12",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "(RANK((OPEN - (SUM(VWAP, 10) / 10)))) * (-1 * (RANK(ABS((CLOSE - VWAP)))))",
    "description": "alpha191因子12",
    "lookback_days": 18,
    "method": "Alpha12Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''