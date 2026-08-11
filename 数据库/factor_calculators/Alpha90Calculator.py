from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha90Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交金额/成交量)
        vwap = amount * 10 / vol
        
        # 4. 实现因子计算逻辑
        # RANK(CORR(RANK(VWAP), RANK(vol), 5)) * -1
        
        # 第一步：对VWAP和vol分别进行横截面排名
        rank_vwap = vwap.rank(axis=1, pct=True)
        rank_vol = vol.rank(axis=1, pct=True)
        
        # 第二步：计算5期滚动相关系数
        # 使用更高效的方法计算滚动相关系数
        corr_result = pd.DataFrame(index=vwap.index, columns=vwap.columns)
        
        # 对每个股票单独计算滚动相关系数
        for col in vwap.columns:
            # 创建临时Series用于计算
            vwap_series = rank_vwap[col]
            vol_series = rank_vol[col]
            
            # 计算滚动相关系数
            rolling_corr = vwap_series.rolling(window=5).corr(vol_series)
            corr_result[col] = rolling_corr
        
        # 第三步：对相关系数进行横截面排名
        rank_corr = corr_result.rank(axis=1, pct=True)
        
        # 第四步：乘以-1
        factor = rank_corr * -1

        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha90",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "(RANK(CORR(RANK(VWAP), RANK(vol), 5)) * -1)",
        "description": "alpha191因子90",
        "lookback_days": 4,
        "method": "Alpha90Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
