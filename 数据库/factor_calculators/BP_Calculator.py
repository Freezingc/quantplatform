from .base import FactorCalculator
import numpy as np
import pandas as pd

class BP_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pb"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pb = data.xs('pb', level=0)
        
        # 3. 计算BP账面市值比 = 1/pb
        bp = 1 / pb
        
        # 4. 处理无穷大值
        bp = bp.replace([np.inf, -np.inf], np.nan)
        
        return bp

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "BP账面市值比",
    "creator": "suyanning",
    "create_date": "2025.09.01",
    "formula": "1/pb",
    "description": "BP账面市值比",
    "lookback_days": 1,
    "method": "BP_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
