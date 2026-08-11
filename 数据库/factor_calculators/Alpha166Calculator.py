from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha166Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 实现因子计算逻辑
        n = 20  # 窗口大小
        
        # 计算收益率: CLOSE/DELAY(CLOSE,1)-1
        returns = close / close.shift(1) - 1
        
        # 计算MEAN(CLOSE/DELAY(CLOSE,1)-1,20)，即20日平均收益率
        mean_returns = returns.rolling(window=n).mean()
        
        # 计算CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20)
        diff_returns = returns - mean_returns
        
        # 计算SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)
        sum_diff_returns = diff_returns.rolling(window=n).sum()
        
        # 计算(CLOSE/DELAY(CLOSE,1),20)^2
        # 注意：原公式可能有误，应该是(CLOSE/DELAY(CLOSE,1)-1)^2
        returns_squared = (close / close.shift(1) - 1) ** 2
        
        # 计算SUM((CLOSE/DELAY(CLOSE,1),20)^2,20)
        sum_returns_squared = returns_squared.rolling(window=n).sum()
        
        # 计算分母：(20-1)*(20-2)*(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5
        # 注意：原公式中(20-2)后面缺少*号，应该是(20-2)*
        denominator = (n-1) * (n-2) * (sum_returns_squared ** 1.5)
        
        # 计算分子：-20*(20-1)^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)
        numerator = -20 * ((n-1) ** 1.5) * sum_diff_returns
        
        # 计算最终因子
        factor = numerator / denominator
        
        # 4. 返回因子（确保格式一致）
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha166",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "-20* ( 20-1 ) ^1.5*SUM(CLOSE/DELAY(CLOSE,1)-1-MEAN(CLOSE/DELAY(CLOSE,1)-1,20),20)/((20-1)*(20-2)(SUM((CLOSE/DELAY(CLOSE,1),20)^2,20))^1.5)",
        "description": "基于收益率偏度的20日标准化高阶矩因子，它衡量了股票收益率分布相对于正态分布的偏离程度",
        "lookback_days": 20,
        "method": "Alpha166Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
