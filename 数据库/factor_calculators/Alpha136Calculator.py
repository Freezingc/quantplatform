from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha136Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","open"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        open = data.xs('open', level=0)  # 需要open数据
        vol = data.xs('vol', level=0)  # 需要vol数据，假设字段名是'vol'
        
        # 3. 计算收益率 (RET)
        # 通常使用收盘价计算日收益率: (close_t - close_{t-1}) / close_{t-1}
        ret = close.pct_change()
        
        # 4. 计算 DELTA(RET, 3)
        # DELTA(RET, 3) = RET - RET.shift(3)
        delta_ret = ret - ret.shift(3)
        
        # 5. 计算 RANK(DELTA(RET, 3)) - 横截面排名
        # 对每一行（每个交易日）进行横截面排名
        rank_delta = delta_ret.rank(axis=1, pct=True)  # pct=True得到百分比排名
        
        # 6. 计算 CORR(OPEN, VOLUME, 10) - 10期滚动相关系数
        # 使用滚动窗口计算相关系数
        corr_open_vol = open.rolling(window=10).corr(vol)
        
        # 7. 计算因子值: (-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10)
        factor = (-1 * rank_delta) * corr_open_vol
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha136",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((-1 * RANK(DELTA(RET, 3))) * CORR(OPEN, VOLUME, 10))",
        "description": "基于收益率反转效应和量价相关性的复合因子，它寻找前期收益率下降但开盘价与成交量正相关性较强的股票",
        "lookback_days": 10,
        "method": "Alpha136Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
