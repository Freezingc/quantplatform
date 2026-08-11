from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class DP_Calculator(FactorCalculator):
    """
    股息率（DP）因子：
    DP = 分红 TTM / 总市值

    计算逻辑：
    1) 现金分红（现金流量表）为累计值 → 转单季度
    2) 单季度分红滚动4期求和 → Dividend TTM
    3) DP = Dividend_TTM / total_mv

    字段说明：
    - incl_dvd_profit_paid_sc_ms : 现金分红（含少数股东，现金流量表累计值）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 现金流量表：现金分红（累计）
        self.required_cashflow_factors = [
            "incl_dvd_profit_paid_sc_ms"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        dividend_cum = cashflow_data.xs("incl_dvd_profit_paid_sc_ms", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度
        dividend_single = cumulative_to_single_quarter(dividend_cum)

        # 3) 分红 TTM
        dividend_ttm = rolling_sum_quarters(dividend_single, n_quarters=4)

        # 4) 股息率
        dp = dividend_ttm / total_mv.replace(0, np.nan)

        dp = dp.replace([np.inf, -np.inf], np.nan)
        return dp


FACTOR_CONFIG = {
    "factor_name": "DP",
    "creator": "zhangyudong",
    "create_date": "2026.02.05",
    "formula": "分红TTM / 总市值",
    "description": "股息率（Dividend Yield），衡量分红回报水平，TTM口径",
    "lookback_days": 750,
    "method": "DP_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
