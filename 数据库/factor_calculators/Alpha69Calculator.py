from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha69Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["low", "high", "open"]  # 指定该因子需要的原始因子
 
    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)
        
        # 计算DTM
        # DTM: (OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))
        delay_open = open.shift(1)
        condition_dtm = open > delay_open
        high_minus_open = high - open
        open_minus_delay = open - delay_open
        dtm = pd.DataFrame(np.where(condition_dtm, np.maximum(high_minus_open, open_minus_delay), 0),
                        index=open.index, columns=open.columns)
        
        # 计算DBM
        # DBM = iif(open >= mfirst(open, 2), 0, max(open - low, open - mfirst(open, 2)))
        mfirst_open = open.rolling(window=2,min_periods=1).mean()
        condition_dbm = open < mfirst_open
        open_minus_low = open - low
        open_minus_mfirst = open - mfirst_open
        dbm = pd.DataFrame(np.where(condition_dbm, np.maximum(open_minus_low, open_minus_mfirst), 0),
                        index=open.index, columns=open.columns)
        
        # 计算SUM(DTM,20)和SUM(DBM,20)
        sum_dtm_20 = dtm.rolling(20).sum()
        sum_dbm_20 = dbm.rolling(20).sum()
        
        # 计算最终因子
        # (SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20) : 
        # (SUM(DTM,20)=SUM(DBM,20) ? 0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))
        condition1 = sum_dtm_20 > sum_dbm_20
        condition2 = sum_dtm_20 == sum_dbm_20
        
        numerator = sum_dtm_20 - sum_dbm_20
        
        part1 = numerator / sum_dtm_20
        part2 = pd.DataFrame(0, index=sum_dtm_20.index, columns=sum_dtm_20.columns)
        part3 = numerator / sum_dbm_20
        
        factor = pd.DataFrame(np.where(condition1, part1, np.where(condition2, part2, part3)),
                            index=sum_dtm_20.index, columns=sum_dtm_20.columns)
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha69",
        "creator": "chenxutian",
        "create_date": "2025.09.07",
        "formula": "(SUM(DTM,20)>SUM(DBM,20) ? (SUM(DTM,20)-SUM(DBM,20))/SUM(DTM,20) : (SUM(DTM,20)=SUM(DBM,20) ? 0:(SUM(DTM,20)-SUM(DBM,20))/SUM(DBM,20)))",
        "description": "alpha191因子69",
        "lookback_days": 20,
        "method": "Alpha69Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''

