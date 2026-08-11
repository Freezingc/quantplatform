from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha104Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 需要vol数据
        
        # 3. 实现因子计算逻辑
        # 步骤1: 计算CORR(HIGH, vol, 5) - 5日相关系数
        corr_5 = high.rolling(window=5).corr(vol)
        
        # 步骤2: 计算DELTA(CORR(HIGH, vol, 5), 5) - 5日相关系数的5日差值
        delta_corr_5 = corr_5.diff(5)
        
        # 步骤3: 计算STD(CLOSE, 20) - 20日收盘价标准差
        std_close_20 = close.rolling(window=20).std()
        
        # 步骤4: 计算RANK(STD(CLOSE, 20)) - 横截面排名
        # 在每个时间点，对所有股票的标准差进行排名
        rank_std = std_close_20.rank(axis=1, pct=False)  # axis=1表示按行计算
        
        # 步骤5: 计算乘积 DELTA(CORR(...), 5) * RANK(STD(...))
        product = delta_corr_5 * rank_std
        
        # 步骤6: 乘以-1得到最终因子
        factor = -1 * product
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha104",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1 * (DELTA(CORR(HIGH, VOLUME, 5), 5) * RANK(STD(CLOSE, 20))))",
        "description": "这个因子通过衡量高波动股票的量价背离加剧程度，来识别可能的价格反转信号",
        "lookback_days": 19,
        "method": "Alpha104Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
