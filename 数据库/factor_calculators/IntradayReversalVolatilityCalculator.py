from .base import FactorCalculator
import numpy as np
import pandas as pd

class IntradayReversalVolatilityCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)
        
        # 3. 计算日内反转-波动翻转因子
        # 计算日内收益率
        inday_return = close / open - 1
        
        # 计算日内收益率均值和标准差
        inday_return_mean = inday_return.rolling(20).mean()
        inday_return_std = inday_return.rolling(20).std()
        
        # 创建低波动率掩码
        low_std_mask = -1 * (inday_return_std.apply(lambda x: x < x.mean(), axis=1))
        low_std_mask.replace(0, 1, inplace=True)
        
        # 计算因子值
        factor = low_std_mask * inday_return_mean
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日内反转-波动翻转",
    "creator": "suyanning",
    "create_date": "2025.09.25",
    "formula": "inday_return = close / open - 1; inday_return_mean = inday_return.rolling(20).mean(); inday_return_std = inday_return.rolling(20).std(); low_std_mask = -1 * (inday_return_std.apply(lambda x: x < x.mean(), axis=1)); low_std_mask.replace(0, 1, inplace=True); low_std_mask * inday_return_mean",
    "description": "日内反转-波动翻转因子，在低波动率时反转日内收益率均值",
    "lookback_days": 20,
    "method": "IntradayReversalVolatilityCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
