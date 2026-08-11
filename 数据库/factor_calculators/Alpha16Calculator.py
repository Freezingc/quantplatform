from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha16Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)  # 需要成交量数据
        amount = data.xs('amount', level=0)  # 需要成交金额数据
        vwap = amount * 10 / vol  # 计算VWAP（成交量

        # 实现因子计算逻辑
        # 步骤分解：
        # 1. RANK(vol) - 对成交量进行排名
        # 2. RANK(VWAP) - 对VWAP进行排名
        # 3. CORR(rank_vol, rank_vwap, 5) - 计算5期相关系数
        # 4. RANK(corr_result) - 对相关系数进行排名
        # 5. TSMAX(rank_corr, 5) - 计算5期最大值
        # 6. -1 * tsmax_result - 取负值
        
        # 对vol和VWAP进行横截面排名
        rank_vol = vol.rank(axis=1, pct=True)
        rank_vwap = vwap.rank(axis=1, pct=True)
        
        # 使用apply按列计算，避免内存碎片
        corr_window = 5
        
        # 一次性计算所有列的相关系数
        corr_list = []
        for col in rank_vol.columns:
            col_corr = rank_vol[col].rolling(window=corr_window).corr(rank_vwap[col])
            corr_list.append(col_corr)
        
        # 使用concat一次性合并所有列
        corr_result = pd.concat(corr_list, axis=1)
        corr_result.columns = rank_vol.columns
        
        # 对相关系数进行横截面排名
        rank_corr = corr_result.rank(axis=1, pct=True, na_option='keep')
        
        # 计算时间序列最大值
        tsmax_window = 5
        tsmax_result = rank_corr.rolling(window=tsmax_window, min_periods=1).max()
        
        # 取负值
        factor = -1 * tsmax_result
        
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha16",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "(-1 * TSMAX(RANK(CORR(RANK(vol), RANK(VWAP), 5)), 5))",
    "description": "alpha191因子16​​",
    "lookback_days": 8,
    "method": "Alpha16Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''