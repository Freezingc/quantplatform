from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha171Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high","open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)  # 注意：因子公式中包含OPEN，需要提取

        # 3. 实现因子计算逻辑
        # 避免除零错误，添加一个极小值
        epsilon = 1e-10
        
        # 计算分子部分: (-1 * ((LOW - CLOSE) * (OPEN^5)))
        numerator = -1 * (low - close) * (open ** 5)
        
        # 计算分母部分: ((CLOSE - HIGH) * (CLOSE^5))
        denominator = (close - high) * (close ** 5)
        
        # 避免除零，添加极小值epsilon
        denominator = denominator.replace(0, epsilon)
        
        # 计算最终因子
        factor = numerator / denominator
        
        # 4. 处理可能的异常值
        # 设置上下限，避免极端值
        factor = factor.clip(lower=-1e6, upper=1e6)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha171",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((-1 * ((LOW - CLOSE) * (OPEN^5))) / ((CLOSE - HIGH) * (CLOSE^5)))",
        "description": "通过比较价格位置（最高、最低、收盘、开盘）的五次方放大效应，衡量价格在日内相对位置中的异常波动强度",
        "lookback_days": 0,
        "method": "Alpha171Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
