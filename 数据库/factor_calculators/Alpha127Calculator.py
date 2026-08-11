from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha127Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)

        # 计算过去12日的收盘价最大值
        max_close_12 = close.rolling(window=12, min_periods=1).max()
        
        # 避免除零，将0替换为NaN
        max_close_12 = max_close_12.replace(0, np.nan)
        
        # 计算中间值: 100*(CLOSE-MAX(CLOSE,12))/MAX(CLOSE,12)
        # 这里需要注意处理除零的情况
        intermediate = 100 * (close - max_close_12) / max_close_12
        
        # 计算平方
        squared = intermediate ** 2
        
        # 计算均值（假设是过去N日的均值，这里假设N=20，如果没有指定使用与MAX相同的窗口12）
        # 如果需要使用特定窗口，请明确指定
        window_size = 20  # 你可以根据需要调整这个窗口大小
        mean_squared = squared.rolling(window=window_size, min_periods=1).mean()
        
        # 计算平方根得到最终因子
        factor = mean_squared ** (1/2)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha127",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(MEAN((100*(CLOSE-MAX(CLOSE,12))/(MAX(CLOSE,12)))^2))^(1/2)",
        "description": "收盘价相对于其近期最高点的偏离度的波动率指标，衡量价格从近期高点回撤的稳定性",
        "lookback_days": 11,
        "method": "Alpha127Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
