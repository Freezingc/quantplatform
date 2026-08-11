from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha150Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 计算典型价格 (Typical Price)
        typical_price = (close + high + low) / 3
        
        # 计算因子值 = 典型价格 * 成交量
        factor = typical_price * vol
        
        # 确保factor的格式与输入数据一致
        # 对齐列名，只保留所有数据都有的股票代码
        common_columns = close.columns.intersection(vol.columns)
        factor = factor[common_columns]
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha150",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(CLOSE+HIGH+LOW)/3*VOLUME",
        "description": "典型价格与成交量的乘积",
        "lookback_days": 0,
        "method": "Alpha150Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
