from .base import FactorCalculator
import numpy as np
import pandas as pd

class Exp_Wgt_Return_1m_Calculator(FactorCalculator):
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
        N = 1 * 21
        decay = np.exp(-np.arange(N)[::-1] / (N * 4))  # 指数衰减系数, 最近一天的指数最大

        returns = close_hfq.pct_change()
        weighted_returns = returns * turnover_rate

        def ewma_func(x):
            return np.sum(x * decay[-len(x):]) # 指数加权移动平均（求和）
        
        num = weighted_returns.rolling(N).apply(ewma_func, raw=True)
        den = turnover_rate.rolling(N).apply(ewma_func, raw=True)

        exp_wgt_return = num / den
        
        return exp_wgt_return

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "指数衰减换手率加权收益率（1个月内）",
    "creator": "jiangyushi",
    "create_date": "2025.10.29",
    "formula": "SUM(daily_ret * turnover_rate * exp(-x_i/(N*4)), 21) / SUM(turnover_rate * exp(-x_i/(N*4)), 21)",
    "description": "个股最近 1 个月内以每日换手率乘以函数 exp(-x_i/N/4)作为权重, 对每日收益率求算术平均值。x_i 为该日距离截面日的天数（排除非交易日）",
    "lookback_days": 21,
    "method": "Exp_Wgt_Return_1m_Calculator",
    "start_time": "", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    "category": "动量反转因子"
}
'''
