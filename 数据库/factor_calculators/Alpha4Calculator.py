from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha4Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子


    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算所需中间变量
        # 8日收盘价均值
        sum_close_8 = close.rolling(window=8, min_periods=8).sum()
        mean_close_8 = sum_close_8 / 8
        
        # 8日收盘价标准差
        std_close_8 = close.rolling(window=8, min_periods=8).std()
        
        # 2日收盘价均值
        sum_close_2 = close.rolling(window=2, min_periods=2).sum()
        mean_close_2 = sum_close_2 / 2
        
        # 20日成交量均值
        mean_vol_20 = vol.rolling(window=20, min_periods=20).mean()
        
        # 4. 实现因子计算逻辑
        # 计算条件1: (mean_close_8 + std_close_8) < mean_close_2
        condition1 = (mean_close_8 + std_close_8) < mean_close_2
        
        # 计算条件2: mean_close_2 < (mean_close_8 - std_close_8)
        condition2 = mean_close_2 < (mean_close_8 - std_close_8)
        
        # 计算条件3: (1 < (vol / mean_vol_20)) | ((vol / mean_vol_20) == 1)
        vol_ratio = vol / mean_vol_20
        condition3 = (1 < vol_ratio) | (vol_ratio == 1)
        
        # 实现三元运算符逻辑
        factor = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        
        # 第一层条件
        factor[condition1] = -1
        
        # 第二层条件
        mask = ~condition1
        factor[mask & condition2] = 1
        
        # 第三层条件
        remaining_mask = mask & ~condition2
        factor[remaining_mask & condition3] = 1
        factor[remaining_mask & ~condition3] = -1
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha4",
    "creator": "chenxutian",
    "create_date": "2025.08.21",
    "formula": "((((SUM(CLOSE, 8) / 8) + STD(CLOSE, 8)) < (SUM(CLOSE, 2) / 2)) ? (-1 * 1) : (((SUM(CLOSE, 2) / 2) < ((SUM(CLOSE, 8) / 8) - STD(CLOSE, 8))) ? 1 : (((1 < (vol / MEAN(vol,20))) || ((vol / MEAN(vol,20)) == 1)) ? 1 : (-1 * 1))))",
    "description": "alpha191因子4。if 短期均线上穿波动带上轨：卖出信号 else if 短期均线下穿波动带下轨：买入信号 else if 当日成交量大于或等于20日均量： 买入信号 else 卖出信号",
    "lookback_days": 19,
    "method": "Alpha4Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''
