from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha185Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子（需要添加 open 数据）
        open = data.xs('open', level=0)  # 添加 open 数据
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算公式: RANK((-1 * ((1 - (OPEN / CLOSE))^2)))
        
        # 计算 OPEN/CLOSE
        ratio = open / close
        
        # 计算 (1 - (OPEN/CLOSE))
        diff = 1 - ratio
        
        # 计算 ((1 - (OPEN/CLOSE))^2)
        diff_squared = diff ** 2
        
        # 计算 -1 * ((1 - (OPEN/CLOSE))^2)
        negative_squared = -1 * diff_squared
        
        # 实现 RANK() 函数 - 对负值的平方进行排序
        # rank 函数：值越小，排名越低（从1开始）
        factor = negative_squared.rank(axis=1, method='average')
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha185",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "RANK((-1 * ((1 - (OPEN / CLOSE))^2)))",
        "description": "通过计算当日开盘价相对收盘价的变化幅度的平方的负向排名，来识别那些开盘后价格走势与常规方向（高开高走或低开低走）反向运行的股票",
        "lookback_days": 0,
        "method": "Alpha185Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
