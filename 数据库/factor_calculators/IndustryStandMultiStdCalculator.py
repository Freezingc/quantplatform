from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryStandMultiStdCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["close"]

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)
        close = data.xs('close', level=0)

        industry_info = self.fetcher.get_stock_industry_info()
        industry_info = industry_info.dropna(subset=['l1_name'])
        industry_map = industry_info.set_index('ts_code')['l1_name']
        industry_map = industry_map[~industry_map.index.duplicated()]

        common_stocks = close.columns.intersection(industry_map.index)
        close = close[common_stocks]
        industry_map = industry_map.loc[common_stocks]

        stock_ret = close / close.shift(1) - 1
        industry_ret_standard = stock_ret.T.groupby(industry_map).transform(
            lambda x: (x - x.mean()) / x.std()
        ).T
        industry_std = stock_ret.T.groupby(industry_map).transform('std').T

        factor = industry_std * industry_ret_standard
        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业标准化乘行业波动率",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "industry_std * industry_z_score",
    "description": "行业内收益z-score乘以行业收益截面标准差，放大高波动行业的偏离信号",
    "lookback_days": 1,
    "method": "IndustryStandMultiStdCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
