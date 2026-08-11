from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class EP_TTM_Calculator(FactorCalculator):
    """
    市盈率倒数（EP_TTM）因子：
    EP_TTM = 净利润 TTM / 总市值
    等价于 EP = 1 / PE

    计算逻辑：
    1) 净利润（利润表）为累计值 → 转单季度
    2) 单季度净利润滚动4期求和 → NetProfit TTM
    3) EP_TTM = NetProfit_TTM / total_mv

    字段说明：
    - n_income : 净利润（累计值，利润表）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 利润表：净利润（累计）
        self.required_income_factors = [
            "n_income"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        income_data = self.fetch_income_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        net_profit_cum = income_data.xs("n_income", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度净利润
        net_profit_single = cumulative_to_single_quarter(net_profit_cum)

        # 3) 净利润 TTM
        net_profit_ttm = rolling_sum_quarters(net_profit_single, n_quarters=4)

        # 4) EP（市盈率倒数）
        ep_ttm = net_profit_ttm / total_mv.replace(0, np.nan)

        ep_ttm = ep_ttm.replace([np.inf, -np.inf], np.nan)
        return ep_ttm


FACTOR_CONFIG = {
    "factor_name": "EP_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.02.05",
    "formula": "净利润TTM / 总市值",
    "description": "市盈率倒数（EP），衡量盈利收益率，TTM口径",
    "lookback_days": 750,
    "method": "EP_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
