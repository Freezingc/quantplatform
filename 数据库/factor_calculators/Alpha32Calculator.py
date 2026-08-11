from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha32Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 需要vol数据

        # 3. 实现因子计算逻辑
        # 第一步：对HIGH和vol分别进行秩排序
        rank_high = high.rank(axis=1, pct=True)  # 按行排序，使用百分比排名
        rank_vol = vol.rank(axis=1, pct=True)
        
        # 第二步：计算3期相关系数
        corr_window = 3
        # 使用rolling计算相关系数，min_periods=2确保至少有2个数据点
        corr_result = rank_high.rolling(window=corr_window).corr(rank_vol)
        
        # 第三步：对相关系数进行秩排序
        rank_corr = corr_result.rank(axis=1, pct=True)
        
        # 第四步：计算3期求和
        sum_window = 3
        sum_result = rank_corr.rolling(window=sum_window).sum()
        
        # 第五步：乘以-1得到最终因子
        factor = -1 * sum_result
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha32",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "(-1 * SUM(RANK(CORR(RANK(HIGH), RANK(vol), 3)), 3))",
    "description": "alpha191因子32",
    "lookback_days": 4,
    "method": "Alpha32Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''