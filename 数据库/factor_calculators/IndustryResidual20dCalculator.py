from .base import FactorCalculator
import numpy as np
import pandas as pd


class IndustryResidual20dCalculator(FactorCalculator):
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
        industry_ret = stock_ret.T.groupby(industry_map).transform('mean').T

        window = 20
        x = industry_ret
        y = stock_ret
        beta = y.rolling(window).cov(x) / x.rolling(window).var()
        alpha = y.rolling(window).mean() - beta * x.rolling(window).mean()
        factor = y - beta * x - alpha

        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


'''
# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "行业残差_20日滚动",
    "creator": "suyanning",
    "create_date": "2026.03.23",
    "formula": "daily_ret - beta * industry_ret - alpha (20日滚动OLS残差)",
    "description": "个股日收益对行业日收益做20日滚动OLS回归的残差，剔除行业共同涨跌后的个股特质收益",
    "lookback_days": 20,
    "method": "IndustryResidual20dCalculator",
    "start_time": "",
    "end_time": "",
    "category": "行业补涨因子"
}
'''
