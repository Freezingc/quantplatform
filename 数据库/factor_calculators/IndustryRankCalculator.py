from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryRankCalculator(FactorCalculator):
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
        factor = stock_ret.T.groupby(industry_map).transform(lambda x: x.rank(pct=True)).T

        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业内收益排名",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "RANK_PCT(daily_ret) within industry",
    "description": "个股日收益在所属行业内的百分位排名（0~1），越高说明当日表现越强",
    "lookback_days": 1,
    "method": "IndustryRankCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
