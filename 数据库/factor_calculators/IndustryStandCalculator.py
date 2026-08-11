from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryStandCalculator(FactorCalculator):
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
        factor = stock_ret.T.groupby(industry_map).transform(
            lambda x: (x - x.mean()) / x.std()
        ).T

        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业内收益标准化",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "(daily_ret - industry_mean_ret) / industry_std_ret",
    "description": "个股日收益在所属行业内做截面标准化（z-score），反映个股相对行业的偏离程度",
    "lookback_days": 1,
    "method": "IndustryStandCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
