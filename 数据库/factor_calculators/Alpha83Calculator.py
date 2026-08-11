from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha83Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)

        # 3. 实现因子计算逻辑
        # 对HIGH和vol进行排名
        rank_high = high.rank(axis=1, pct=True)
        rank_vol = vol.rank(axis=1, pct=True)
        
        # 计算滚动协方差
        def calculate_covariance(window):
            """计算窗口内两个序列的协方差"""
            x, y = window
            return np.cov(x, y)[0, 1] if len(x) > 1 else np.nan
        
        # 对每个时间点计算协方差
        covariance_result = pd.DataFrame(index=rank_high.index, columns=rank_high.columns)
        
        for stock in rank_high.columns:
            for i in range(4, len(rank_high)):
                if i >= 4:  # 确保有足够的数据点
                    window_high = rank_high[stock].iloc[i-4:i+1]
                    window_vol = rank_vol[stock].iloc[i-4:i+1]
                    covariance_result.iloc[i, covariance_result.columns.get_loc(stock)] = np.cov(
                        window_high, window_vol
                    )[0, 1] if len(window_high) > 1 else np.nan
        
        # 对协方差进行排名
        rank_covariance = covariance_result.rank(axis=1, pct=True)
        
        # 乘以-1得到最终因子
        factor = -1 * rank_covariance
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha83",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "(-1 * RANK(COVIANCE(RANK(HIGH), RANK(vol), 5)))",
        "description": "alpha191因子83",
        "lookback_days": 4,
        "method": "Alpha83Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

