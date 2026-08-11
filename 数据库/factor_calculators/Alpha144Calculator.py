from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha144Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["amount","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算前一日收盘价
        delay_close = close.shift(1)
        
        # 4. 计算基础指标
        price_change_ratio = (close / delay_close - 1).abs()
        
        # 避免除零错误
        amount_safe = amount.copy()
        amount_safe[amount_safe <= 0] = np.nan
        
        # 计算每个时间点的值
        value = price_change_ratio / amount_safe
        
        # 5. 创建条件矩阵
        condition = close < delay_close
        
        # 6. 使用rolling apply计算
        window_size = 20
        
        def custom_sumif(arr):
            """自定义函数计算每个窗口的SUMIF"""
            values = arr[0]  # 数值部分
            conditions = arr[1]  # 条件部分
            
            if len(values) < 1:
                return np.nan
                
            # 计算满足条件的数值之和
            mask = conditions.astype(bool)
            sum_if = np.nansum(np.where(mask, values, 0))
            
            # 计算满足条件的数量
            count_if = np.sum(mask)
            
            # 避免除零
            if count_if == 0:
                return np.nan
                
            return sum_if / count_if
        
        # 为每列单独计算
        factor = pd.DataFrame(index=close.index, columns=close.columns)
        
        for col in close.columns:
            # 准备数据
            values = value[col].values
            conditions = condition[col].values
            
            # 滚动计算
            for i in range(len(values)):
                if i >= window_size - 1:
                    window_start = i - window_size + 1
                    window_values = values[window_start:i+1]
                    window_conditions = conditions[window_start:i+1]
                    
                    # 计算
                    mask = window_conditions.astype(bool)
                    sum_if = np.nansum(np.where(mask, window_values, 0))
                    count_if = np.sum(mask)
                    
                    if count_if > 0:
                        factor.iloc[i, factor.columns.get_loc(col)] = sum_if / count_if
                    else:
                        factor.iloc[i, factor.columns.get_loc(col)] = np.nan
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha144",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SUMIF(ABS(CLOSE/DELAY(CLOSE,1)-1)/AMOUNT,20,CLOSE<DELAY(CLOSE,1))/COUNT(CLOSE<DELAY(CLOSE, 1),20)",
        "description": "计算的是过去20天内，在股价下跌的日子中，每日涨跌幅绝对值的换手率倒数的均值。",
        "lookback_days": 20,
        "method": "Alpha144Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
