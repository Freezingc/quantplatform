from .base import FactorCalculator
import numpy as np
import pandas as pd

class SP_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["ps"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        ps = data.xs('ps', level=0)
        
        # 3. 计算市销率倒数 = 1/ps
        sp = 1 / ps
        
        # 4. 处理无穷大值
        sp = sp.replace([np.inf, -np.inf], np.nan)
        
        return sp

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "SP市销率倒数",
    "creator": "suyanning",
    "create_date": "2025.09.12",
    "formula": "1/ps",
    "description": "SP市销率倒数",
    "lookback_days": 1,
    "method": "SP_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
