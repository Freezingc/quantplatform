from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha110Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        # 计算前一日收盘价
        close_prev = close.shift(1)
        
        # 计算上涨部分：max(0, HIGH - DELAY(CLOSE,1))
        up_part = (high - close_prev).clip(lower=0)
        
        # 计算下跌部分：max(0, DELAY(CLOSE,1) - LOW)
        down_part = (close_prev - low).clip(lower=0)
        
        # 计算20日上涨总和
        sum_up_20 = up_part.rolling(window=20, min_periods=1).sum()
        
        # 计算20日下跌总和
        sum_down_20 = down_part.rolling(window=20, min_periods=1).sum()
        
        # 计算RSI因子，避免除零错误
        # 当分母为0时，如果分子也为0，则结果为0；否则为100
        with pd.option_context('mode.use_inf_as_na', True):
            # 先计算原始值
            factor_raw = sum_up_20 / sum_down_20 * 100
            
            # 处理分母为0的情况
            factor = factor_raw.where(sum_down_20 != 0, 0)
            factor = factor.where(~(sum_up_20 > 0) | ~(sum_down_20 == 0), 100)
            
            # 替换无穷大和NaN值为0
            factor = factor.fillna(0)
            factor = factor.replace([np.inf, -np.inf], 0)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha110",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUM(MAX(0,HIGH-DELAY(CLOSE,1)),20)/SUM(MAX(0,DELAY(CLOSE,1)-LOW),20)*100",
        "description": "通过比较20日内上涨幅度总和与下跌幅度总和的比例来衡量价格上涨的强度",
        "lookback_days": 20,
        "method": "Alpha110Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
