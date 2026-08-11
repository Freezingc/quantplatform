from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha143Calculator(FactorCalculator):
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
        # 计算前一日收盘价（延迟操作）
        shifted_close = close.shift(1)
        
        # 计算日收益率：(当日收盘价 - 前一日收盘价) / 前一日收盘价
        ret = (close - shifted_close) / shifted_close
        
        # 初始化因子DataFrame，所有值设为1.0（首日因子值）
        factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        
        # 对每支股票（列）单独计算因子序列
        for col in close.columns:
            close_series = close[col]
            ret_series = ret[col]
            # 判断条件：当日收盘价是否大于前一日收盘价
            cond_series = close_series > close_series.shift(1)
            
            # 初始化因子值列表，首日因子为1.0
            factor_values = [1.0]
            
            # 从第二个交易日开始迭代计算
            for i in range(1, len(close_series)):
                prev_factor = factor_values[-1]  # 前一日因子值
                if cond_series.iloc[i]:  # 条件成立
                    current_ret = ret_series.iloc[i]
                    # 若收益率为NaN（如前一日收盘价为0），则保持因子不变
                    if pd.isna(current_ret):
                        new_factor = prev_factor
                    else:
                        new_factor = prev_factor * current_ret  # 乘以收益率
                else:  # 条件不成立，因子保持不变
                    new_factor = prev_factor
                factor_values.append(new_factor)
            
            # 将该股票的因子序列赋给factor DataFrame
            factor[col] = factor_values
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha143",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "CLOSE>DELAY(CLOSE,1)?(CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1)*SELF:SELF",
        "description": "这是一个基于价格趋势的递归复合因子，当价格上涨时因子值按涨幅比例递增，下跌时保持前值不变，本质是价格上升区间的累积复利增长因子",
        "lookback_days": 1,
        "method": "Alpha143Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因"
    }
'''
