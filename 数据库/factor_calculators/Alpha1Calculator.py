from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha1Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close", "open", "vol"]  # 指定该因子需要的原始因子


    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        open = data.xs('open', level=0)
        close = data.xs('close', level=0)
        vol = data.xs('vol', level=0)
        # 3. 计算alpha1因子（向量化实现）
        # 计算delta = log(vol) - 2天前的log(vol)
        log_vol = np.log(vol)
        delta = log_vol - log_vol.shift(1)
        # 计算收益率 (close - open)/open
        returns = (close - open) / open
        
        # 计算百分比排名（整个DataFrame一次性计算）
        delta_rank = delta.rank(axis=1, pct=True)  # 截面排名
        return_rank = returns.rank(axis=1, pct=True)
        merged = pd.concat([delta_rank.stack(), return_rank.stack()], axis=1)
        merged.columns = ['delta', 'return']

        # 计算6天滚动窗口的秩相关系数
        def rolling_rank_corr(x):
            return x.iloc[:, 0].rolling(window=6,min_periods=1).corr(x.iloc[:, 1])
        
        rolling_corr = merged.groupby(level=1).apply(rolling_rank_corr)

        rolling_corr.reset_index(level=0, drop=True, inplace=True)
        # 8. 取负值
        rolling_corr = -rolling_corr
        rolling_corr = rolling_corr.to_frame(name='corr_value')  # 转为 DataFrame
        rolling_corr = rolling_corr.reset_index()  # 恢复索引为列
        rolling_corr.columns = ['trade_date', 'ts_code', 'corr_value']

        # 重新对齐原始索引和列顺序
        factor = rolling_corr.reset_index().pivot(
            index='ts_code', 
            columns='trade_date', 
            values='corr_value'
        )
        factor = factor.T
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha1",
    "creator": "chenxutian",
    "create_date": "2025.08.20",
    "formula": "(-1 * CORR(RANK(DELTA(LOG(VOLUME), 1)), RANK(((CLOSE - OPEN) / OPEN)), 6))",
    "description": "alpha191因子1",
    "lookback_days": 6,
    "method": "Alpha1Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2025.08.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''