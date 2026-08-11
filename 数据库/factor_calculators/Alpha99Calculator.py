from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha99Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)

        # 3. 使用更高效的实现方式
        def calculate_factor(close, vol, window=5):
            # 横截面排名函数
            def cross_section_rank(df):
                return df.rank(axis=1, pct=True)
            
            # 对close和vol进行排名
            rank_close = cross_section_rank(close)
            rank_vol = cross_section_rank(vol)
            
            # 使用rolling协方差计算
            covariance = rank_close.rolling(window=window).cov(rank_vol)
            
            # 对协方差结果进行横截面排名并乘以-1
            rank_cov = cross_section_rank(covariance)
            factor = -1 * rank_cov
            
            return factor
        
        factor = calculate_factor(close, vol, window=5)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha99",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(-1 * RANK(COVIANCE(RANK(CLOSE), RANK(VOLUME), 5)))",
        "description": "衡量股价与成交量在短期内的协同变化关系的反向排名",
        "lookback_days": 4,
        "method": "Alpha99Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
