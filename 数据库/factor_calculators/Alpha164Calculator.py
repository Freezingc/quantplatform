from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha164Calculator(FactorCalculator):
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
        
        # 3. 计算 DELAY(CLOSE,1)
        delay_close = close.shift(1)
        
        # 4. 计算条件部分：((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)
        condition = close > delay_close
        temp1 = pd.DataFrame(np.where(condition, 
                                    1/(close - delay_close), 
                                    1),
                            index=close.index,
                            columns=close.columns)
        
        # 5. 计算 MIN(temp1, 12)
        min_temp1 = temp1.rolling(window=12).min()
        
        # 6. 计算分子：(temp1 - min_temp1)
        numerator = temp1 - min_temp1
        
        # 7. 计算分母：(HIGH-LOW)
        denominator = high - low
        
        # 8. 计算临时结果：numerator/denominator * 100
        # 避免除以0
        temp_result = numerator / denominator.replace(0, np.nan) * 100
        
        # 9. 计算 SMA(..., 13, 2)
        # SMA = EMA(EMA(x, N), M)，这里N=13, M=2
        # 先计算13日的EMA
        
        factor = temp_result.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha164",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA((((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-DELAY(CLOSE,1)):1)-MIN(((CLOSE>DELAY(CLOSE,1))?1/(CLOSE-D ELAY(CLOSE,1)):1),12))/(HIGH-LOW)*100,13,2)",
        "description": "基于价格变化与价格区间比率的13日平滑移动平均",
        "lookback_days": 13,
        "method": "Alpha164Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
