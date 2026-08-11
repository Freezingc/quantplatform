from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha167Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)

        # 实现因子计算逻辑
        # 计算每日涨跌：当天收盘价 - 前一天的收盘价
        close_diff = close - close.shift(1)
        
        # 只保留上涨的部分：如果涨跌大于0，则为涨跌值，否则为0
        positive_diff = close_diff.where(close_diff > 0, 0)
        
        # 计算过去12个交易日的上涨幅度总和
        # 使用rolling窗口，min_periods=1确保即使数据不足也能计算
        factor = positive_diff.rolling(window=12, min_periods=1).sum()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha167",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)",
        "description": "过去12个交易日内，每日上涨幅度的累计和",
        "lookback_days": 12,
        "method": "Alpha167Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
