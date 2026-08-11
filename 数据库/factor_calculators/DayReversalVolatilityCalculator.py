from .base import FactorCalculator
import numpy as np
import pandas as pd

class DayReversalVolatilityCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pct_chg = data.xs('pct_chg', level=0)
        
        # 3. 计算日间反转-波动翻转因子
        # 计算收益率均值和标准差
        return_mean = pct_chg.rolling(20).mean()
        return_std = pct_chg.rolling(20).std()
        
        # 创建低波动率掩码
        low_std_mask = -1 * (return_std.apply(lambda x: x < x.mean(), axis=1))
        low_std_mask.replace(0, 1, inplace=True)
        
        # 计算因子值
        factor = low_std_mask * return_mean
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日间反转-波动翻转",
    "creator": "suyanning",
    "create_date": "2025.09.25",
    "formula": "return_mean = pct_chg.rolling(20).mean(); return_std = pct_chg.rolling(20).std(); low_std_mask = -1 * (return_std.apply(lambda x: x < x.mean(), axis=1)); low_std_mask.replace(0, 1, inplace=True); low_std_mask * return_mean",
    "description": "日间反转-波动翻转因子，在低波动率时反转收益率均值",
    "lookback_days": 20,
    "method": "DayReversalVolatilityCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
