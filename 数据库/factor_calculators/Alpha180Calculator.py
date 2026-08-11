from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha180Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需数据
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 需要成交量数据
        
        # 3. 计算中间变量
        # 20日成交量均值
        mean_vol_20 = vol.rolling(window=20, min_periods=1).mean()
        
        # 计算条件：MEAN(vol,20) < vol
        condition = mean_vol_20 < vol
        
        # 计算ELSE部分：-1 * vol
        else_part = -1 * vol
        
        # 计算THEN部分：
        # 7日收盘价差分
        delta_close_7 = close.diff(7)
        
        # ABS(DELTA(CLOSE, 7))
        abs_delta = delta_close_7.abs()
        
        # TSRANK(ABS(DELTA(CLOSE, 7)), 60)
        # 时间序列排名：计算过去60天内的排名（0-1之间）
        def ts_rank(series):
            return series.rolling(window=60, min_periods=1).apply(
                lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5
            )
        
        tsrank_abs = abs_delta.rolling(window=60, min_periods=1).apply(
            lambda x: (x.rank().iloc[-1] - 1) / (len(x) - 1) if len(x) > 1 else 0.5
        )
        
        # SIGN(DELTA(CLOSE, 7))
        sign_delta = delta_close_7.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        
        # 计算THEN部分：(-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7))
        then_part = (-1 * tsrank_abs) * sign_delta
        
        # 4. 根据条件选择结果
        factor = pd.DataFrame(index=condition.index, columns=condition.columns)
        
        # 使用条件判断赋值
        for col in condition.columns:
            for idx in condition.index:
                if condition.loc[idx, col]:
                    factor.loc[idx, col] = then_part.loc[idx, col]
                else:
                    factor.loc[idx, col] = else_part.loc[idx, col]
        
        # 或者更高效的方式（避免循环）：
        # factor = then_part.where(condition, else_part)
        
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha180",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "((MEAN(vol,20) < vol) ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7)) : (-1 * vol)))",
        "description": "基于成交量放大信号进行条件判断：当成交量突破20日均值时，返回基于收盘价7日变化的时序排名与方向符号的组合因子；否则返回成交量的负值",
        "lookback_days": 66,
        "method": "Alpha180Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "量价相关性因子"
    }
'''
