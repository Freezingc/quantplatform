from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha159Calculator(FactorCalculator):
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
        
        # 计算延迟的收盘价
        delay_close = close.shift(1)
        
        # 计算 MIN(LOW, DELAY(CLOSE, 1))
        min_low_delay_close = pd.DataFrame(
            np.minimum(low.values, delay_close.values), 
            index=low.index, 
            columns=low.columns
        )
        
        # 计算 MAX(HIGH, DELAY(CLOSE, 1))，注意原公式中是"HGIH"，应该是"HIGH"的拼写错误
        max_high_delay_close = pd.DataFrame(
            np.maximum(high.values, delay_close.values), 
            index=high.index, 
            columns=high.columns
        )
        
        # 计算分子部分：CLOSE - SUM(MIN(LOW, DELAY(CLOSE, 1)), N)
        # 计算分母部分：SUM(MAX(HIGH, DELAY(CLOSE, 1)) - MIN(LOW, DELAY(CLOSE, 1)), N)
        
        # 6日周期
        sum_min_6 = min_low_delay_close.rolling(window=6, min_periods=1).sum()
        sum_max_min_6 = (max_high_delay_close - min_low_delay_close).rolling(window=6, min_periods=1).sum()
        part_6 = (close - sum_min_6) / sum_max_min_6.replace(0, np.nan) * 12 * 24
        
        # 12日周期
        sum_min_12 = min_low_delay_close.rolling(window=12, min_periods=1).sum()
        sum_max_min_12 = (max_high_delay_close - min_low_delay_close).rolling(window=12, min_periods=1).sum()
        part_12 = (close - sum_min_12) / sum_max_min_12.replace(0, np.nan) * 6 * 24
        
        # 24日周期
        sum_min_24 = min_low_delay_close.rolling(window=24, min_periods=1).sum()
        sum_max_min_24 = (max_high_delay_close - min_low_delay_close).rolling(window=24, min_periods=1).sum()
        part_24 = (close - sum_min_24) / sum_max_min_24.replace(0, np.nan) * 6 * 24
        
        # 组合三个部分
        numerator = part_6 + part_12 + part_24
        denominator = 6 * 12 + 6 * 24 + 12 * 24  # 72 + 144 + 288 = 504
        
        # 计算最终因子
        factor = numerator * 100 / denominator
        
        # 清理无效值
        factor = factor.replace([np.inf, -np.inf], np.nan)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha159",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),6))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),6) *12*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),12))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),12)*6*24+(CLOSE-SUM(MIN(LOW,DELAY(CLOSE,1)),24))/SUM(MAX(HGIH,DELAY(CLOSE,1))-MIN(LOW,DELAY(CLOSE,1)),24)*6*24)*100/(6*12+6*24+12*24)",
        "description": "基于股价在特定周期内相对位置的多周期加权动量因子，衡量收盘价相对于过去N日内价格区间（考虑前一日收盘价调整后）的偏离程度",
        "lookback_days": 25,
        "method": "Alpha159Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
