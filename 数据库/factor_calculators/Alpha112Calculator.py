from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha112Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        
        # 3. 计算价格变化
        price_diff = close - close.shift(1)
        
        # 4. 实现原公式中的条件逻辑
        # SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)
        up_series = pd.DataFrame(
            np.where(price_diff > 0, price_diff, 0),
            index=price_diff.index,
            columns=price_diff.columns
        )
        sum_up = up_series.rolling(window=12, min_periods=1).sum()
        
        # SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12)
        down_series = pd.DataFrame(
            np.where(price_diff < 0, price_diff.abs(), 0),
            index=price_diff.index,
            columns=price_diff.columns
        )
        sum_down = down_series.rolling(window=12, min_periods=1).sum()
        
        # 5. 计算最终因子
        # 避免除零错误
        denominator = sum_up + sum_down
        denominator[denominator == 0] = 1e-10
        
        factor = (sum_up - sum_down) / denominator * 100
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha112",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)-SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))/(SUM((CLOSE-DELAY(CLOSE,1)>0?CLOSE-DELAY(CLOSE,1):0),12)+SUM((CLOSE-DELAY(CLOSE,1)<0?ABS(CLOSE-DELAY(CLOSE,1)):0),12))*100",
        "description": "12日RSI相对强弱指标",
        "lookback_days": 11,
        "method": "Alpha112Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
