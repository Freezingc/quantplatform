from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha48Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要成交量数据

        # 3. 实现因子计算逻辑
        # 计算每日价格变化方向
        sign1 = np.sign(close - close.shift(1))  # SIGN((CLOSE - DELAY(CLOSE, 1)))
        sign2 = np.sign(close.shift(1) - close.shift(2))  # SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))
        sign3 = np.sign(close.shift(2) - close.shift(3))  # SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))
        
        # 求和三个符号值
        sign_sum = sign1 + sign2 + sign3
        
        # 计算排名（横截面排名）
        # 使用rank(axis=1)进行横向排名，pct=True得到百分比排名
        rank_values = sign_sum.rank(axis=1, pct=True)
        
        # 计算成交量相关部分
        sum_vol_5 = vol.rolling(window=5).sum()  # SUM(vol, 5)
        sum_vol_20 = vol.rolling(window=20).sum()  # SUM(vol, 20)
        
        # 组合所有部分：-1 * 排名 * 5日成交量总和 / 20日成交量总和
        factor = -1 * rank_values * sum_vol_5 / sum_vol_20
        
        # 处理可能的除零错误和缺失值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor
'''
# 添加因子信息
{
    "factor_name": "Alpha48",
    "creator": "chenxutian",
    "create_date": "2025.08.27",
    "formula": "(-1*((RANK(((SIGN((CLOSE - DELAY(CLOSE, 1))) + SIGN((DELAY(CLOSE, 1) - DELAY(CLOSE, 2)))) + SIGN((DELAY(CLOSE, 2) - DELAY(CLOSE, 3)))))) * SUM(vol, 5)) / SUM(vol, 20))",
    "description": "alpha191因子48",
    "lookback_days": 19,
    "method": "Alpha48Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''