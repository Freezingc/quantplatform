from .base import FactorCalculator
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class Alpha40Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要获取成交量数据

        # 3. 实现因子计算逻辑
        # 计算前一天收盘价（延迟一天）
        delay_close = close.shift(1)
        
        # 判断当天收盘价是否大于前一天收盘价
        close_gt_delay = close > delay_close
        
        # 计算分子：上涨日的成交量之和（26天滚动求和）
        numerator = close_gt_delay.astype(float).mul(vol).rolling(window=26).sum()
        
        # 计算分母：下跌日的成交量之和（26天滚动求和）
        denominator = (~close_gt_delay).astype(float).mul(vol).rolling(window=26).sum()
        
        # 避免除零错误，将分母中的0替换为NaN
        denominator = denominator.replace(0, float('nan'))
        
        # 计算因子值：分子/分母 * 100
        factor = numerator / denominator * 100
        
        # 处理可能的NaN值（如分母为0的情况）
        factor = factor.fillna(0)  # 或者使用其他适当的填充值

        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha40",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),26)/SUM((CLOSE<=DELAY(CLOSE,1)?VOLUME:0),26)*100",
    "description": "alpha191因子40",
    "lookback_days": 27,
    "method": "Alpha40Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''