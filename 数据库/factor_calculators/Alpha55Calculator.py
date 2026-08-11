from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha55Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "low", "high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        open = data.xs('open', level=0)
        
        # 3. 计算各项中间结果
        delay_close = close.shift(1)
        delay_open = open.shift(1)
        delay_low = low.shift(1)
        
        # 分子部分
        numerator = 16 * (close - delay_close + (close - open) / 2 + delay_close - delay_open)
        
        # 分母的第一部分条件判断
        cond1 = (abs(high - delay_close) > abs(low - delay_close)) & \
                (abs(high - delay_close) > abs(high - delay_low))
        
        cond2 = (abs(low - delay_close) > abs(high - delay_low)) & \
                (abs(low - delay_close) > abs(high - delay_close))
        
        part1 = np.where(cond1, 
                        abs(high - delay_close) + abs(low - delay_close) / 2 + abs(delay_close - delay_open) / 4,
                        np.where(cond2,
                                abs(low - delay_close) + abs(high - delay_close) / 2 + abs(delay_close - delay_open) / 4,
                                abs(high - delay_low) + abs(delay_close - delay_open) / 4))
        
        # 分母的第二部分
        part2 = np.maximum(abs(high - delay_close), abs(low - delay_close))
        
        # 完整分母
        denominator = part1 * part2
        
        # 避免除以0
        denominator = denominator.replace(0, np.nan)
        
        # 计算每日的因子值
        daily_factor = numerator / denominator
        
        # 计算20日求和
        factor = daily_factor.rolling(20).sum()
        
        return factor
    
'''
# 添加因子信息
    {
        "factor_name": "Alpha55",
        "creator": "chenxutian",
        "create_date": "2025.09.01",
        "formula": "SUM(16*(CLOSE-DELAY(CLOSE,1)+(CLOSE-OPEN)/2+DELAY(CLOSE,1)-DELAY(OPEN,1))/((ABS(HIGH-DELAY(CLOSE,1))>ABS(LOW-DELAY(CLOSE,1)) & ABS(HIGH-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1))?ABS(HIGH-DELAY(CLOSE,1))+ABS(LOW-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:(ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(LOW,1)) & ABS(LOW-DELAY(CLOSE,1))>ABS(HIGH-DELAY(CLOSE,1))?ABS(LOW-DELAY(CLOSE,1))+ABS(HIGH-DELAY(CLOSE,1))/2+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4:ABS(HIGH-DELAY(LOW,1))+ABS(DELAY(CLOSE,1)-DELAY(OPEN,1))/4)))*MAX(ABS(HIGH-DELAY(CLOSE,1)),ABS(LOW-DELAY(CLOSE,1))),20)",
        "description": "alpha191因子55",
        "lookback_days": 20,
        "method": "Alpha55Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
    }
'''
