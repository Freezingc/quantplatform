from .base import FactorCalculator
import numpy as np
import pandas as pd

class Wgt_Return_6m_Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close_hfq", "turnover_rate"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close_hfq = data.xs('close_hfq', level=0)
        turnover_rate = data.xs('turnover_rate', level=0)
        
        # 3. 计算
        N = 6 * 21
        daily_ret = close_hfq.pct_change()
        num = (daily_ret * turnover_rate).rolling(N).sum()
        den = turnover_rate.rolling(N).sum()

        factor = num / den
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "换手率加权收益率（6个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.10.15",
    "formula": "SUM(daily_ret * turnover_rate, 126) / SUM(turnover_rate, 126)",
    "description": "个股最近 6 个月内以每日换手率作为权重对每日收益率求算术平均值",
    "lookback_days": 126,
    "method": "Wgt_Return_6m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''
