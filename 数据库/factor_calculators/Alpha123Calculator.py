from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha123Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","low","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 计算中间变量
        # 3.1 计算(HIGH + LOW) / 2
        hl_mean = (high + low) / 2
        
        # 3.2 计算SUM(((HIGH + LOW) / 2), 20)
        sum_hl_mean_20 = hl_mean.rolling(window=20, min_periods=1).sum()
        
        # 3.3 计算MEAN(VOLUME, 60)
        mean_vol_60 = vol.rolling(window=60, min_periods=1).mean()
        
        # 3.4 计算SUM(MEAN(VOLUME,60), 20)
        sum_mean_vol_20 = mean_vol_60.rolling(window=20, min_periods=1).sum()
        
        # 4. 计算第一个CORR: CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)
        corr1 = sum_hl_mean_20.rolling(window=9, min_periods=1).corr(sum_mean_vol_20)
        
        # 5. 计算第二个CORR: CORR(LOW, VOLUME, 6)
        corr2 = low.rolling(window=6, min_periods=1).corr(vol)
        
        # 6. 计算RANK
        # 6.1 计算corr1的横截面rank
        rank_corr1 = corr1.rank(axis=1, pct=True, method='dense')
        
        # 6.2 计算corr2的横截面rank
        rank_corr2 = corr2.rank(axis=1, pct=True, method='dense')
        
        # 7. 比较并乘以-1
        factor = (rank_corr1 < rank_corr2) * -1
        
        # 8. 将布尔值转换为数值
        factor = factor.astype(float)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha123",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((RANK(CORR(SUM(((HIGH + LOW) / 2), 20), SUM(MEAN(VOLUME,60), 20), 9)) < RANK(CORR(LOW, VOLUME, 6))) * -1)",
        "description": "通过比较20日均价和20日成交量均值的9日相关性排名是否低于最低价和成交量的6日相关性排名，来识别当前价格动量与成交量关系反转的信号，当条件满足时给出-1的看空信号",
        "lookback_days": 78,
        "method": "Alpha123Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
