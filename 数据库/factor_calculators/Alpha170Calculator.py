from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha170Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量字段是'vol'
        amount = data.xs('amount', level=0)  # 假设成交额字段是'amount'

        vwap = amount / vol * 10 # 计算VWAP
        
        # 3. 计算因子各个部分
        
        # 3.1 计算 RANK((1 / CLOSE))
        rank_inv_close = 1 / close
        rank_inv_close = rank_inv_close.rank(axis=1, pct=True)  # 横截面排名
        
        # 3.2 计算 MEAN(vol,20)
        mean_vol_20 = vol.rolling(window=20).mean()
        
        # 3.3 计算第一部分: (RANK((1 / CLOSE)) * vol) / MEAN(vol,20)
        part1 = (rank_inv_close * vol) / mean_vol_20
        
        # 3.4 计算 RANK((HIGH - CLOSE))
        high_minus_close = high - close
        rank_high_minus_close = high_minus_close.rank(axis=1, pct=True)
        
        # 3.5 计算 SUM(HIGH, 5) / 5
        sum_high_5 = high.rolling(window=5).sum()
        avg_high_5 = sum_high_5 / 5
        
        # 3.6 计算第二部分: (HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5)
        part2 = (high * rank_high_minus_close) / avg_high_5
        
        # 3.7 计算第三部分: RANK((VWAP - DELAY(VWAP, 5)))
        vwap_diff = vwap - vwap.shift(5)  # DELAY(VWAP, 5) 用 shift(5) 实现
        rank_vwap_diff = vwap_diff.rank(axis=1, pct=True)
        
        # 3.8 计算最终因子
        factor = (part1 * part2) - rank_vwap_diff
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha170",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((((RANK((1 / CLOSE)) * vol) / MEAN(vol,20)) * ((HIGH * RANK((HIGH - CLOSE))) / (SUM(HIGH, 5) / 5))) - RANK((VWAP - DELAY(VWAP, 5))))",
        "description": "结合了价格反转、成交量异常放大和高位抛压信号，通过比较（价格低位时的成交量放大程度）与（高位时的相对抛压强度），再减去价格动量反转信号，综合识别短期可能反弹的股票",
        "lookback_days": 9,
        "method": "Alpha170Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
