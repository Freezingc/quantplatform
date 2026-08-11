from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha184Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["open","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)  # 注意：公式中需要OPEN数据
        
        # 3. 计算 DELAY((OPEN - CLOSE), 1)
        # 先计算OPEN - CLOSE
        open_minus_close = open - close
        
        # 然后计算延迟1天
        # 注意：这里假设数据是按时间升序排列的
        delayed_open_minus_close = open_minus_close.shift(1)
        
        # 4. 计算 CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)
        # 200日的滚动相关性
        corr_series = delayed_open_minus_close.rolling(window=200).corr(close)
        
        # 5. 计算第一个排名 RANK(CORR(...))
        # 横截面排名：在每个时间点上，对所有股票进行排名
        rank_corr = corr_series.rank(axis=1, pct=True)  # pct=True返回百分比排名(0-1)
        
        # 6. 计算第二个排名 RANK((OPEN - CLOSE))
        # 计算OPEN - CLOSE
        open_close_diff = open - close
        
        # 横截面排名
        rank_diff = open_close_diff.rank(axis=1, pct=True)
        
        # 7. 合并两个排名
        factor = rank_corr + rank_diff
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha184",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(CORR(DELAY((OPEN - CLOSE), 1), CLOSE, 200)) + RANK((OPEN - CLOSE)))",
        "description": "通过计算昨天开盘收盘价差与今日收盘价的200日相关性排名加上今日开盘收盘价差排名挖掘那些价格动量与开盘效应具有稳定相关性的股票",
        "lookback_days": 200,
        "method": "Alpha184Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
