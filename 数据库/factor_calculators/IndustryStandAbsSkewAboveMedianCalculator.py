from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryStandAbsSkewAboveMedianCalculator(FactorCalculator):
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
        industry_skew = stock_ret.T.groupby(industry_map).transform('skew').T.rolling(5).mean()

        skew_median = industry_skew.quantile(0.5, axis=1)
        factor = industry_ret_standard.abs() * (industry_skew >= skew_median)
        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业标准化绝对值偏度高于截面中位数过滤",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "ABS(industry_z_score) * (industry_skew >= CROSS_SECTION_MEDIAN(industry_skew))",
    "description": "行业内收益z-score绝对值，仅在行业偏度高于当日截面中位数时保留",
    "lookback_days": 5,
    "method": "IndustryStandAbsSkewAboveMedianCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
