from .base import FactorCalculator
import numpy as np
import pandas as pd

class TsRank_EP2_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pe_ttm"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        ep = 1 / data.xs('pe_ttm', level=0)
        ep = ep.replace([np.inf, -np.inf], np.nan)
        
        # 3. 计算
        N = 120
        factor = ep.rolling(N).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "EP历史分位数（2季度）",
    "creator": "jiangyushi",
    "create_date": "2025.12.02",
    "formula": "ts_rank(ep, 120)",
    "description": "净利润(TTM)/总市值在过去 2 个季度里的分位数",
    "lookback_days": 120,
    "method": "TsRank_EP2_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
