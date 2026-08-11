from .base import FactorCalculator
import numpy as np
import pandas as pd

class Quart_EP_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pe"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pe = data.xs('pe', level=0)
        
        # 3. 计算单季度市盈率倒数 = 1/pe
        quart_ep = 1 / pe
        
        # 4. 处理无穷大值
        quart_ep = quart_ep.replace([np.inf, -np.inf], np.nan)
        
        return quart_ep

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Quart_EP单季度市盈率倒数",
    "creator": "suyanning",
    "create_date": "2025.09.01",
    "formula": "1/pe",
    "description": "Quart_EP单季度市盈率倒数",
    "lookback_days": 1,
    "method": "Quart_EP_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
