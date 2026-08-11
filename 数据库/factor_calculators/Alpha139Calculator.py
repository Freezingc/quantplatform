from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha139Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","open","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        open = data.xs('open', level=0)  # 添加open数据
        vol = data.xs('vol', level=0)  # 添加vol数据
        
        # 3. 实现因子计算逻辑：(-1 * CORR(OPEN, VOLUME, 10))
        # 计算10天的滚动相关系数
        corr_window = 10
        corr_series = open.rolling(window=corr_window).corr(vol)
        
        # 乘以-1得到最终因子
        factor = -1 * corr_series
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha139",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1 * CORR(OPEN, VOLUME, 10))",
        "description": "衡量的是过去10个交易日内开盘价与成交量之间的负相关性强度，当开盘价和成交量走势呈现明显负相关时因子值会变大",
        "lookback_days": 9,
        "method": "Alpha139Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
