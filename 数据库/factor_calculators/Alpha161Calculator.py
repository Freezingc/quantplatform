from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha161Calculator(FactorCalculator):
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
        
        # 3. 计算因子
        delay_close = close.shift(1)  # DELAY(CLOSE, 1)
        
        hl_diff = high - low
        abs_delay_high = np.abs(delay_close - high)
        abs_delay_low = np.abs(delay_close - low)
        
        # 使用np.maximum进行向量化比较
        temp_max = np.maximum(hl_diff, abs_delay_high)
        max_values = np.maximum(temp_max, abs_delay_low)
        
        # 转换为DataFrame
        max_df = pd.DataFrame(
            max_values,
            index=hl_diff.index,
            columns=hl_diff.columns
        )
        
        # 计算12日移动平均
        factor = max_df.rolling(window=12, min_periods=1).mean()
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha161",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "MEAN(MAX(MAX((HIGH-LOW),ABS(DELAY(CLOSE,1)-HIGH)),ABS(DELAY(CLOSE,1)-LOW)),12)",
        "description": "每日真实波幅（取最高-最低、昨日收盘到当日最高、昨日收盘到当日最低三者中的最大值）的12日移动平均",
        "lookback_days": 12,
        "method": "Alpha161Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
