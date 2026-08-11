from .base import FactorCalculator
import numpy as np
import pandas as pd

class Alpha36Calculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["vol", "amount"]  # 指定该因子需要的原始因子

    def _calculate(self, start_time, end_time):
        # 1. 获取数据
        data = self.fetch_data(start_time, end_time)
        
        # 2. 提取所需因子
        vol = data.xs('vol', level=0)
        amount = data.xs('amount', level=0)
        
        # 3. 计算VWAP (成交量加权平均价)
        # VWAP = (high + low + close) / 3 * vol / vol.sum()
        # 但通常VWAP是日内成交量加权平均，这里用(high+low+close)/3作为近似
        vwap = amount * 10 / vol
        
        # 4. 计算RANK(vol) - 横截面排名
        def rank_dataframe(df):
            # 对每一行进行横截面排名 (0-1之间)
            return df.rank(axis=1, pct=True)
        
        rank_vol = rank_dataframe(vol)
        rank_vwap = rank_dataframe(vwap)
        
        # 5. 计算CORR(RANK(vol), RANK(VWAP), 6) - 6期相关系数
        def rolling_correlation(x, y, window):
            # 计算两个DataFrame的滚动相关系数
            result = pd.DataFrame(index=x.index, columns=x.columns)
            
            for col in x.columns:
                # 对每一列计算滚动相关系数
                x_series = x[col]
                y_series = y[col]
                result[col] = x_series.rolling(window=window).corr(y_series)
            
            return result
        
        corr_6 = rolling_correlation(rank_vol, rank_vwap, 6)
        
        # 6. 计算SUM(CORR, 2) - 2期求和
        sum_2 = corr_6.rolling(window=2).sum()
        
        # 7. 计算最终的RANK - 横截面排名
        factor = rank_dataframe(sum_2)
        
        return factor

'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "Alpha36",
    "creator": "chenxutian",
    "create_date": "2025.08.26",
    "formula": "RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))",
    "description": "alpha191因子36",
    "lookback_days": 6,
    "method": "Alpha36Calculator",
    "start_time": "2020.01.01", # 可以为空，默认从2020.01.01开始，如果为空，请填写空字符串""
    "end_time": "2020.01.10"  # 可以为空，默认为当日日期，如果为空，请填写空字符串""
}
'''