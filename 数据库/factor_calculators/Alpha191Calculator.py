from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha191Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)
        
        # 3. 使用向量化计算
        # 3.1 计算20日均成交量
        vol_ma20 = vol.rolling(window=20, min_periods=1).mean()
        
        # 3.2 计算5日滚动相关性（向量化版本）
        # 创建一个空的DataFrame用于存储相关性
        corr_values = pd.DataFrame(index=vol_ma20.index, columns=vol_ma20.columns)
        
        # 使用apply函数逐列计算
        def calculate_corr(col_vol, col_low):
            return col_vol.rolling(window=5, min_periods=1).corr(col_low)
        
        for stock in vol_ma20.columns:
            corr_values[stock] = calculate_corr(vol_ma20[stock], low[stock])
        
        # 3.3 计算中间价
        mid_price = (high + low) / 2
        
        # 3.4 计算最终因子
        factor = corr_values + mid_price - close
        
        # 处理可能的缺失值
        factor = factor.fillna(0)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha191",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((CORR(MEAN(vol,20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)",
        "description": "衡量成交量与股价的短期相关性修正后的盘中价格中位数相对于收盘价的溢价程度",
        "lookback_days": 24,
        "method": "Alpha191Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
