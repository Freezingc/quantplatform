from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha114Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)

        vwap = amount / vol * 10

        # 3.1 计算 (HIGH - LOW) / (SUM(CLOSE, 5) / 5)
        high_low_diff = high - low
        close_5ma = close.rolling(window=5).sum() / 5
        hl_div_ma5 = high_low_diff / close_5ma
        
        # 3.2 计算 DELAY(hl_div_ma5, 2) 的 RANK
        delay_hl_div_ma5 = hl_div_ma5.shift(2)  # DELAY 2天
        rank_delay = delay_hl_div_ma5.rank(axis=1, pct=True)  # 横截面排名
        
        # 3.3 计算 RANK(RANK(VOLUME))
        rank_vol = vol.rank(axis=1, pct=True)  # VOLUME的RANK
        rank_rank_vol = rank_vol.rank(axis=1, pct=True)  # RANK的RANK
        
        # 3.4 计算分子: RANK(DELAY(...)) * RANK(RANK(VOLUME))
        numerator = rank_delay * rank_rank_vol
        
        # 3.5 计算分母: (hl_div_ma5) / (VWAP - CLOSE)
        vwap_close_diff = vwap - close
        denominator = hl_div_ma5 / vwap_close_diff
        
        # 3.6 计算最终因子
        factor = numerator / denominator
        
        # 4. 处理可能的异常值
        # 4.1 处理无穷大值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        # 4.2 处理分母为0的情况（如果VWAP-CLOSE为0）
        factor = factor.fillna(method='ffill').fillna(method='bfill')
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha114",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK(DELAY(((HIGH - LOW) / (SUM(CLOSE, 5) / 5)), 2)) * RANK(RANK(VOLUME))) / (((HIGH - LOW) / (SUM(CLOSE, 5) / 5)) / (VWAP - CLOSE)))",
        "description": "衡量股价波动相对强度、成交活跃度的时序变化与价格偏离程度的复合关系，来识别短期价格动量与流动性协同效应的多空信号。",
        "lookback_days": 5,
        "method": "Alpha114Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
