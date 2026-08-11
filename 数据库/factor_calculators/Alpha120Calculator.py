from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha120Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量字段名为'vol'
        amount = data.xs('amount', level=0)  # 假设成交金额字段名为'amount'
        
        # 3. 计算VWAP（成交量加权平均价）
        # VWAP = (high + low + close) / 3
        vwap = amount / vol * 10
        
        # 4. 计算因子
        # 计算 (VWAP - CLOSE) 和 (VWAP + CLOSE)
        numerator = vwap - close
        denominator = vwap + close
        
        # 对每个时间点横截面进行排名
        def rank_series(series):
            # 对每一行（每个交易日）进行排名
            # 使用method='dense'确保相同的值获得相同的排名
            return series.rank(axis=1, method='dense', pct=True)
        
        # 计算分子和分母的排名
        rank_numerator = rank_series(numerator)
        rank_denominator = rank_series(denominator)
        
        # 5. 计算最终因子：RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE))
        factor = rank_numerator / rank_denominator
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha120",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK((VWAP - CLOSE)) / RANK((VWAP + CLOSE)))",
        "description": "每个股票在VWAP相对收盘价偏离度的横截面排名与VWAP相对收盘价中心位置排名的比值",
        "lookback_days": 0,
        "method": "Alpha120Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
