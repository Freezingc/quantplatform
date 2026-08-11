from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha113Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要vol数据
        
        # 3. 实现因子计算逻辑
        
        # 计算 SUM(DELAY(CLOSE, 5), 20)
        # DELAY(CLOSE, 5)：收盘价滞后5天
        close_delay5 = close.shift(5)
        # SUM(DELAY(CLOSE, 5), 20)：对滞后5天的收盘价计算20天滚动和
        sum_delay_close_20 = close_delay5.rolling(window=20, min_periods=1).sum()
        
        # 计算 (SUM(DELAY(CLOSE, 5), 20) / 20)
        avg_delay_close = sum_delay_close_20 / 20
        
        # 计算 RANK((SUM(DELAY(CLOSE, 5), 20) / 20))
        # 横截面排名（每行/每天对各股票进行排名）
        rank_avg_delay = avg_delay_close.rank(axis=1, pct=True)
        
        # 计算 CORR(CLOSE, VOLUME, 2)
        # 计算收盘价和成交量的2天滚动相关系数
        corr_close_vol = close.rolling(window=2, min_periods=1).corr(vol)
        
        # 计算 SUM(CLOSE, 5) 和 SUM(CLOSE, 20)
        sum_close_5 = close.rolling(window=5, min_periods=1).sum()
        sum_close_20 = close.rolling(window=20, min_periods=1).sum()
        
        # 计算 CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2)
        # 计算两个滚动和序列的2天滚动相关系数
        corr_sum_close = sum_close_5.rolling(window=2, min_periods=1).corr(sum_close_20)
        
        # 计算 RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))
        rank_corr_sum = corr_sum_close.rank(axis=1, pct=True)
        
        # 计算整个因子表达式
        # (RANK(avg_delay) * CORR(close, vol, 2)) * RANK(CORR(sum_close_5, sum_close_20, 2))
        factor_part1 = rank_avg_delay * corr_close_vol
        factor_part2 = factor_part1 * rank_corr_sum
        
        # 乘以-1
        factor = -1 * factor_part2
        
        # 4. 返回因子，保持与输入相同的格式
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha113",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1 * ((RANK((SUM(DELAY(CLOSE, 5), 20) / 20)) * CORR(CLOSE, VOLUME, 2)) * RANK(CORR(SUM(CLOSE, 5), SUM(CLOSE, 20), 2))))",
        "description": "结合价格延迟平均排名、价量短期相关性、以及不同周期价格和的相关性排名",
        "lookback_days": 24,
        "method": "Alpha113Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
