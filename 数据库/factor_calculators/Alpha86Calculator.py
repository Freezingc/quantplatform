from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha86Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "close"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)

        # 3. 实现因子计算逻辑
        # 计算各个延迟值
        delay_close_1 = close.shift(1)  # DELAY(CLOSE, 1)
        delay_close_10 = close.shift(10)  # DELAY(CLOSE, 10)
        delay_close_20 = close.shift(20)  # DELAY(CLOSE, 20)
        
        # 计算中间表达式
        expr1 = (delay_close_20 - delay_close_10) / 10
        expr2 = (delay_close_10 - close) / 10
        diff_expr = expr1 - expr2
        
        # 计算条件判断部分
        condition1 = (0.25 < diff_expr)  # 0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))
        condition2 = (diff_expr < 0)  # (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0
        
        # 计算close的变化量
        close_diff = close - delay_close_1
        
        # 根据条件构建因子值
        factor_values = pd.DataFrame(index=close.index, columns=close.columns)
        
        # 应用条件逻辑
        # condition1为真时返回-1
        factor_values[condition1] = -1
        
        # condition1为假且condition2为真时返回1
        factor_values[(~condition1) & condition2] = 1
        
        # 其他情况返回(-1 * (CLOSE - DELAY(CLOSE, 1)))
        other_condition = (~condition1) & (~condition2)
        factor_values[other_condition] = (-1 * close_diff)[other_condition]
        
        # 应用外层函数：计算13日移动平均，然后5日移动平均，再15日移动平均，最后乘以-1
        factor = factor_values.rolling(window=13).mean()  # 13日移动平均
        factor = factor.rolling(window=5).mean()  # 5日移动平均
        factor = factor.rolling(window=15).mean()  # 15日移动平均
        factor = factor * -1  # 乘以-1
        
        # 处理NaN值（由于移动平均产生的）
        factor = factor.fillna(method='ffill').fillna(method='bfill')
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha86",
        "creator": "chenxutian",
        "create_date": "2025.10.05",
        "formula": "((0.25 < (((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10))) ? (-1 * 1) : (((((DELAY(CLOSE, 20) - DELAY(CLOSE, 10)) / 10) - ((DELAY(CLOSE, 10) - CLOSE) / 10)) < 0) ? 1 : ((-1 * 1) * (CLOSE - DELAY(CLOSE, 1)))))",
        "description": "alpha191因子86",
        "lookback_days": 19,
        "method": "Alpha86Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.09"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

