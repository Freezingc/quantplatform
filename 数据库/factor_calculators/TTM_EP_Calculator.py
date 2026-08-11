from .base import FactorCalculator
import numpy as np
import pandas as pd

class TTM_EP_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pe_ttm"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pe_ttm = data.xs('pe_ttm', level=0)
        
        # 3. 计算滚动市盈率倒数 = 1/pe_ttm
        ttm_ep = 1 / pe_ttm
        
        # 4. 处理无穷大值
        ttm_ep = ttm_ep.replace([np.inf, -np.inf], np.nan)
        
        return ttm_ep

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "TTM_EP滚动市盈率倒数",
    "creator": "suyanning",
    "create_date": "2025.09.01",
    "formula": "1/pe_ttm",
    "description": "TTM_EP滚动市盈率倒数",
    "lookback_days": 1,
    "method": "TTM_EP_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
