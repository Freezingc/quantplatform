from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha173Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        
        # 3. 实现因子计算逻辑
        
        # 第一步：计算SMA(CLOSE,13,2)
        sma_close_13_2 = close.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        # 第二步：计算SMA(SMA(CLOSE,13,2),13,2)
        sma_sma_close = sma_close_13_2.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        # 第三步：计算LOG(CLOSE)
        log_close = np.log(close)
        
        # 第四步：计算SMA(LOG(CLOSE),13,2)
        sma_log_close = log_close.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        # 第五步：计算SMA(SMA(LOG(CLOSE),13,2),13,2)
        sma_sma_log_close = sma_log_close.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        # 第六步：计算SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
        sma_sma_sma_log_close = sma_sma_log_close.rolling(window=13, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*11)/13 if len(x) == 13 else x.mean()
        )
        
        # 第七步：组合最终因子
        # 3*SMA(CLOSE,13,2) - 2*SMA(SMA(CLOSE,13,2),13,2) + SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)
        factor = 3 * sma_close_13_2 - 2 * sma_sma_close + sma_sma_sma_log_close
        
        # 确保factor的格式与输入数据一致
        factor = pd.DataFrame(factor, index=close.index, columns=close.columns)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha173",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "3*SMA(CLOSE,13,2)-2*SMA(SMA(CLOSE,13,2),13,2)+SMA(SMA(SMA(LOG(CLOSE),13,2),13,2),13,2)",
        "description": "通过嵌套的指数加权移动平均捕捉价格变化的高阶动态特征",
        "lookback_days": 38,
        "method": "Alpha173Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
