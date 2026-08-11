from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha33Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "low", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算收益率
        ret = close.pct_change()
        
        # 4. 计算各子部分
        # TSMIN(LOW, 5)
        tsmin_low_5 = low.rolling(window=5).min()
        
        # DELAY(TSMIN(LOW, 5), 5)
        delay_tsmin = tsmin_low_5.shift(5)
        
        # (-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)
        part1 = (-1 * tsmin_low_5) + delay_tsmin
        
        # SUM(RET, 240) - SUM(RET, 20)
        sum_ret_240 = ret.rolling(window=240).sum()
        sum_ret_20 = ret.rolling(window=20).sum()

        # ((SUM(RET, 240) - SUM(RET, 20)) / 220)
        part2 = (sum_ret_240 - sum_ret_20) / 220
        

        # RANK(part2) - 横截面排名
        rank_part2 = part2.rank(axis=1, pct=True)
        
        # TSRANK(vol, 5) - 时间序列排名
        tsrank_vol_5 = vol.rolling(window=5).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], 
            raw=False
        )
        
        # 5. 组合所有部分
        factor = part1 * rank_part2 * tsrank_vol_5
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha33",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "((((-1 * TSMIN(LOW, 5)) + DELAY(TSMIN(LOW, 5), 5)) * RANK(((SUM(RET, 240) - SUM(RET, 20)) / 220))) * TSRANK(vol, 5))",
    "description": "alpha191因子33",
    "lookback_days": 239,
    "method": "Alpha33Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''