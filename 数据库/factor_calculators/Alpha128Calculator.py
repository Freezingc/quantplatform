from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha128Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high","low"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        low = data.xs('low', level=0)  # 注意：xs的axis参数
        close = data.xs('close', level=0)
        high = data.xs('high', level=0)
        vol = data.xs('vol', level=0)  # 假设成交量列名为'vol'
        
        # 3. 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 4. 计算延迟一期的典型价格
        # 先确保数据按时间排序
        typical_price = typical_price.sort_index()
        typical_price_shift = typical_price.shift(1)
        
        # 5. 计算条件表达式
        # 条件1: 典型价格 > 延迟典型价格
        condition_up = typical_price > typical_price_shift
        # 条件2: 典型价格 < 延迟典型价格
        condition_down = typical_price < typical_price_shift
        
        # 6. 计算加权值
        weighted_value = typical_price * vol
        
        # 7. 计算条件加权值
        weighted_up = weighted_value.where(condition_up, 0)
        weighted_down = weighted_value.where(condition_down, 0)
        
        # 8. 计算14日滚动和
        # 使用rolling窗口计算
        sum_up = weighted_up.rolling(window=14, min_periods=1).sum()
        sum_down = weighted_down.rolling(window=14, min_periods=1).sum()
        
        # 9. 计算因子
        # 处理除零情况
        ratio = sum_up / sum_down.replace(0, np.nan)  # 将0替换为nan避免除零错误
        # 计算最终因子
        factor_value = 100 - (100 / (1 + ratio))
        
        # 10. 处理可能存在的NaN值
        factor = factor_value.fillna(50)  # 当ratio为NaN时，因子值为50
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha128",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "100-(100/(1+SUM(((HIGH+LOW+CLOSE)/3>DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0),14)/SUM(((HIGH+LOW+CLOSE)/3<DELAY((HIGH+LOW+CLOSE)/3,1)?(HIGH+LOW+CLOSE)/3*VOLUME:0), 14)))",
        "description": "通过比较典型价格变化方向的成交量加权和来计算的一个动量指标",
        "lookback_days": 14,
        "method": "Alpha128Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
