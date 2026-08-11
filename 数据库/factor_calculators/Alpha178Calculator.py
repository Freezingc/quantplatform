from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha178Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要提取vol数据
        
        # 实现因子计算逻辑
        # CLOSE-DELAY(CLOSE,1) 表示今日收盘价减去昨日收盘价
        close_diff = close - close.shift(1)  # shift(1) 相当于 DELAY(CLOSE,1)
        
        # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1) 计算收益率
        returns = close_diff / close.shift(1)
        
        # (CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*vol
        factor = returns * vol
        
        # 处理可能的NaN值（第一天没有前一天的close，所以是NaN）
        factor = factor.fillna(0)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha178",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*VOLUME",
        "description": "价格变化率乘以成交量，它衡量了价格变动幅度所带动的交易活跃度",
        "lookback_days": 1,
        "method": "Alpha178Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
