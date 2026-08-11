from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha26Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)

        # 3. 计算VWAP (成交量加权平均价)
        vwap = amount * 10 / vol
        
        # 4. 计算第一部分：(((SUM(CLOSE, 7) / 7) - CLOSE))
        # 计算7日收盘价移动平均
        close_ma_7 = close.rolling(window=7).mean()
        part1 = close_ma_7 - close
        
        # 5. 计算第二部分：CORR(VWAP, DELAY(CLOSE, 5), 230)
        # 延迟5天的收盘价
        delay_close_5 = close.shift(5)
        

        # 计算230天的相关系数
        def rolling_corr(df1, df2, window):
            """计算两个DataFrame的滚动相关系数"""
            result = pd.DataFrame(index=df1.index, columns=df1.columns)
            
            for col in df1.columns:
                # 对每列分别计算滚动相关系数
                s1 = df1[col]
                s2 = df2[col]
                result[col] = s1.rolling(window=window).corr(s2)
            
            return result
        
        part2 = rolling_corr(vwap, delay_close_5, 230)

        # 6. 合并两部分得到最终因子
        factor = part1 + part2
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha26",
    "creator": "chenxutian",
    "create_date": "2025.08.25",
    "formula": "((((SUM(CLOSE, 7) / 7) - CLOSE)) + ((CORR(VWAP, DELAY(CLOSE, 5), 230))))",
    "description": "alpha191因子26​",
    "lookback_days": 234,
    "method": "Alpha26Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''