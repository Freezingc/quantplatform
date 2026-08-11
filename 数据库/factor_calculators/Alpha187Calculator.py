from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha187Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","open","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)
        
        # 计算前一日的开盘价
        delay_open = open.shift(1)
        
        # 计算两个差值
        high_minus_open = high - open
        open_minus_delay = open - delay_open
        
        # 计算最大值
        max_values = pd.DataFrame(np.maximum(high_minus_open.values, open_minus_delay.values),
                                index=high_minus_open.index,
                                columns=high_minus_open.columns)
        
        # 应用条件判断
        condition = open <= delay_open
        result = max_values.where(~condition, 0)
        
        # 计算20日滚动和
        factor = result.rolling(window=20, min_periods=1).sum()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha187",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM((OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1)))),20)",
        "description": "过去20天内，当日开盘价高于前一日开盘价时，取最高价与开盘价的涨幅、以及当日开盘价相对前一日开盘价的涨幅两者中的较大值，否则为0的累积和",
        "lookback_days": 20,
        "method": "Alpha187Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
