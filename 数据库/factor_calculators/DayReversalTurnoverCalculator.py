from .base import FactorCalculator
import numpy as np
import pandas as pd

class DayReversalTurnoverCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        pct_chg = data.xs('pct_chg', level=0)
        turnover = data.xs('turnover_rate', level=0)
        
        # 3. 计算日间反转-换手翻转因子
        # 计算换手率变化
        turnover_diff = turnover.diff()
        
        # 创建低换手率掩码
        low_turnover_mask = -1 * (turnover_diff.apply(lambda x: x < x.mean(), axis=1))
        low_turnover_mask.replace(0, 1, inplace=True)
        
        # 计算因子值
        factor = (low_turnover_mask * pct_chg).rolling(20).mean()
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "日间反转-换手翻转",
    "creator": "suyanning",
    "create_date": "2025.09.25",
    "formula": "turnover_diff = turnover.diff(); low_turnover_mask = -1*(turnover_diff < turnover_diff.mean()); low_turnover_mask.replace(0,1); SMA(low_turnover_mask*pct_chg,20)",
    "description": "日间反转-换手翻转因子，在低换手率时反转收益率",
    "lookback_days": 20,
    "method": "DayReversalTurnoverCalculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": ""  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
