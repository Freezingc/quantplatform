from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryCatchup20dCalculator(FactorCalculator):
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

        window = 20
        stock_ret_20 = close / close.shift(window) - 1
        industry_ret_20 = stock_ret_20.T.groupby(industry_map).transform('mean').T

        factor = industry_ret_20 / stock_ret_20
        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业补涨_除法_20日",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "industry_ret_20d / stock_ret_20d",
    "description": "行业20日涨跌幅 / 个股20日涨跌幅，行业价格为行业内等权均价，衡量个股相对行业的补涨潜力",
    "lookback_days": 20,
    "method": "IndustryCatchup20dCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
