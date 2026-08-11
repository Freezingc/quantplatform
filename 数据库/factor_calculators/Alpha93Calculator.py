from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha93Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","open","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        open = data.xs('open', level=0)  # 添加open价格

        # 实现因子计算逻辑
        # DELAY(OPEN,1)：前一天的open价格
        delay_open = open.shift(1)
        
        # 计算两个可能的值：(OPEN-LOW) 和 (OPEN-DELAY(OPEN,1))
        open_minus_low = open - low
        open_minus_delay_open = open - delay_open
        
        # 使用np.maximum计算逐元素最大值，避免循环
        max_value = np.maximum(open_minus_low, open_minus_delay_open)
        
        # 修正条件判断：使用逐元素比较
        # 如果OPEN>=DELAY(OPEN,1)则为0，否则为MAX值
        condition = open >= delay_open
        inner_value = pd.DataFrame(np.where(condition, 0, max_value), 
                                index=open.index, 
                                columns=open.columns)
        
        # SUM(..., 20)：20日求和
        factor = inner_value.rolling(window=20, min_periods=1).sum()
        
        return factor

'''
# 添加因子信息
    {
        "factor_name": "Alpha93",
        "creator": "chenxutian",
        "create_date": "2025.10.15",
        "formula": "SUM((OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1)))),20)",
        "description": "在开盘价低于昨日开盘价的日子里，开盘价相对于最低价和前日开盘价的最大跌幅的20日总和",
        "lookback_days": 20,
        "method": "Alpha93Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子​"
    }
'''
