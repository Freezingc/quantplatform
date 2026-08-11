from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha179Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        vol = data.xs('vol', level=0)  # 注意：确认成交量字段名
        amount = data.xs('amount', level=0)  # 注意：确认成交额字段名

        # 3. 计算辅助指标
        vwap = amount * 10 / vol  # 计算VWAP
        mean_vol_50 = vol.rolling(50).mean()
        
        # 4. 计算截面排名函数
        def cross_section_rank(df):
            # 对每个交易日，在所有股票间进行排名
            # 使用pandas的rank方法，确保与原始数据形状一致
            return df.rank(axis=1, method='dense', pct=True)
        
        # 5. 计算第一个相关性：VWAP和vol的4天相关性
        corr1 = vwap.rolling(4).corr(vol)
        
        # 6. 计算第二个相关性
        # 6.1 计算LOW的截面排名
        rank_low = cross_section_rank(low)
        
        # 6.2 计算50日均成交量的截面排名
        rank_mean_vol = cross_section_rank(mean_vol_50)
        
        # 6.3 计算12天相关性
        corr2 = rank_low.rolling(12).corr(rank_mean_vol)
        
        # 7. 对相关性进行截面排名
        rank_corr1 = cross_section_rank(corr1)
        rank_corr2 = cross_section_rank(corr2)
        
        # 8. 计算最终因子
        factor = rank_corr1 * rank_corr2
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha179",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(RANK(CORR(VWAP, vol, 4)) *RANK(CORR(RANK(LOW), RANK(MEAN(vol,50)), 12)))",
        "description": "股价与成交量的短期（4日）相关性排名与低价股与长期（50日）均量排名的中期（12日）相关性排名的乘积效应",
        "lookback_days": 62,
        "method": "Alpha179Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
