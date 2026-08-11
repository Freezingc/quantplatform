from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha174Calculator(FactorCalculator):
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
        
        # 计算前一日收盘价
        delay_close = close.shift(1)
        
        # 计算20日收盘价标准差
        std_close_20 = close.rolling(window=20, min_periods=1).std()
        
        # 条件判断：如果今日收盘 > 昨日收盘，则为20日标准差，否则为0
        condition = close > delay_close
        intermediate_value = np.where(condition, std_close_20, 0)
        
        # 将中间值转换为DataFrame
        intermediate_df = pd.DataFrame(intermediate_value, 
                                      index=close.index, 
                                      columns=close.columns)
        
        # 使用rolling计算20日简单移动平均
        # 注意：这里假设SMA(20,1)是20日简单移动平均
        factor = intermediate_df.rolling(window=20, min_periods=1).apply(
            lambda x: (x[-1] + x[:-1].mean()*19)/20 if len(x) == 20 else x.mean()
        )
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha174",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)",
        "description": "当股票上涨时，其收盘价近期波动率的20日移动平均，反映了上涨期间的股价波动特征",
        "lookback_days": 39,
        "method": "Alpha174Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "波动率因子"
    }
'''
