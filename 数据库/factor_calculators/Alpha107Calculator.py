from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha107Calculator(FactorCalculator):
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
        open = data.xs('open', level=0)  # 注意：open是Python关键字，所以加了下划线
        
        # 3. 实现因子计算逻辑
        
        # 计算各个延迟序列
        high_delay1 = high.shift(1)  # DELAY(HIGH, 1)
        close_delay1 = close.shift(1)  # DELAY(CLOSE, 1)
        low_delay1 = low.shift(1)  # DELAY(LOW, 1)
        
        # 计算三个子表达式
        expr1 = open - high_delay1
        expr2 = open - close_delay1
        expr3 = open - low_delay1
        
        # 计算每个表达式的横截面排名
        # 使用rank(axis=1)对每行（每个交易日）进行横截面排名
        # pct=True 将排名转换为百分比排名（0到1之间），然后乘以2再减1，将范围映射到-1到1
        rank1 = expr1.rank(axis=1, pct=True) * 2 - 1
        rank2 = expr2.rank(axis=1, pct=True) * 2 - 1
        rank3 = expr3.rank(axis=1, pct=True) * 2 - 1
        
        # 计算最终因子
        # (-1 * RANK((OPEN - DELAY(HIGH, 1)))) 
        # * RANK((OPEN - DELAY(CLOSE, 1))) 
        # * RANK((OPEN - DELAY(LOW, 1)))
        factor = (-1 * rank1) * rank2 * rank3
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha107",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(((-1 * RANK((OPEN - DELAY(HIGH, 1)))) * RANK((OPEN - DELAY(CLOSE, 1)))) * RANK((OPEN - DELAY(LOW, 1))))",
        "description": "通过比较当日开盘价与前一日最高、收盘、最低价的相对位置，在横截面上识别那些开盘表现异常弱势的股票",
        "lookback_days": 1,
        "method": "Alpha107Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
