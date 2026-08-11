from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha142Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol","close","high"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)  # 提取成交量数据

        # 3. 定义辅助函数
        def ts_rank_series(series, window):
            """计算时间序列排名：返回当前值在过去window天内的排名分位数"""
            return series.rolling(window).apply(
                lambda x: x.rank(pct=True).iloc[-1] if not x.isnull().all() else np.nan,
                raw=False
            )

        def rank_cross_section(df):
            """横截面排名：对每个交易日，所有股票进行排名，返回分位数"""
            return df.rank(axis=1, pct=True)

        def delta(series, period):
            """计算差分"""
            return series.diff(period)

        # 4. 计算因子各部分
        # 第一部分：-1 * RANK(TSRANK(CLOSE, 10))
        tsrank_close = pd.DataFrame(index=close.index, columns=close.columns)
        for col in close.columns:
            tsrank_close[col] = ts_rank_series(close[col], 10)
        rank_tsrank_close = rank_cross_section(tsrank_close)
        part1 = -1 * rank_tsrank_close

        # 第二部分：RANK(DELTA(DELTA(CLOSE, 1), 1))
        delta1 = delta(close, 1)  # 一阶差分
        delta2 = delta(delta1, 1)  # 二阶差分
        rank_delta2 = rank_cross_section(delta2)
        part2 = rank_delta2

        # 第三部分：RANK(TSRANK((vol / MEAN(vol,20)), 5))
        vol_mean_20 = vol.rolling(20).mean()  # 20日平均成交量
        vol_ratio = vol / vol_mean_20
        tsrank_vol_ratio = pd.DataFrame(index=vol_ratio.index, columns=vol_ratio.columns)
        for col in vol_ratio.columns:
            tsrank_vol_ratio[col] = ts_rank_series(vol_ratio[col], 5)
        rank_tsrank_vol_ratio = rank_cross_section(tsrank_vol_ratio)
        part3 = rank_tsrank_vol_ratio

        # 5. 组合为最终因子
        factor = part1 * part2 * part3
        return factor


'''
# 添加因子信息
    {
        "factor_name": "Alpha142",
        "creator": "chenxutian",
        "create_date": "2025.12.04",
        "formula": "(((-1 * RANK(TSRANK(CLOSE, 10))) * RANK(DELTA(DELTA(CLOSE, 1), 1))) * RANK(TSRANK((vol /MEAN(vol,20)), 5)))",
        "description": "结合了价格动量反转、价格加速度反转和成交量异常反转三个维度，通过负向加权价格动量、正向加权价格加速度和成交量异常来识别短期反转机会",
        "lookback_days": 23,
        "method": "Alpha142Calculator",
        "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
        "end_time": "2020.01.20",  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
        "category": "动量反转因子"
    }
'''
