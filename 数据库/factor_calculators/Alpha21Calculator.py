from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha21Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算6日均价
        mean_close_6 = close.rolling(window=6).mean()
        
        # 创建序列 [1,2,3,4,5,6] 作为自变量
        sequence = np.array([1, 2, 3, 4, 5, 6])
        
        # 计算每个时间点、每个股票的回归beta
        def calculate_beta(series):
            if len(series) < 6 or series.isna().any():
                return np.nan
            # 计算线性回归的beta系数 (cov(x,y)/var(x))
            x = sequence
            y = series.values
            cov_xy = np.cov(x, y)[0, 1]
            var_x = np.var(x)
            return cov_xy / var_x
        
        # 对每个股票计算滚动beta
        factor = mean_close_6.rolling(window=6).apply(calculate_beta, raw=False)
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha21",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "REGBETA(MEAN(CLOSE,6),SEQUENCE(6))",
    "description": "alpha191因子21​。表示计算6日均价与序列[1,2,3,4,5,6]的线性回归beta系数",
    "lookback_days": 5,
    "method": "Alpha21Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''