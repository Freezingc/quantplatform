from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha17Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","amount","low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)  

        # 3. 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol

        # 4. 计算 MAX(VWAP, 15) - 过去15天的VWAP最大值
        max_vwap_15 = vwap.rolling(window=15, min_periods=1).max()

        # 5. 计算 VWAP - MAX(VWAP, 15)
        vwap_diff = vwap - max_vwap_15
        # 6. 计算 DELTA(CLOSE, 5) - 5天前收盘价的变化
        delta_close_5 = close.diff(5)
        # 7. 计算 RANK((VWAP - MAX(VWAP, 15)))
        # 对每个时间截面的横截面数据进行排名
        def cross_section_rank(series):
            return series.rank(pct=True)
        
        ranked_vwap_diff = vwap_diff.apply(cross_section_rank, axis=1)

        # 8. 计算最终因子: RANK(...)^DELTA(...)
        factor = ranked_vwap_diff.pow(delta_close_5)
        
        # 9. 处理可能的NaN值
        factor = factor.fillna(0)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha17",
    "creator": "chenxutian",
    "create_date": "2025.08.22",
    "formula": "RANK((VWAP - MAX(VWAP, 15)))^DELTA(CLOSE, 5)",
    "description": "alpha191因子17​",
    "lookback_days": 14,
    "method": "Alpha17Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''