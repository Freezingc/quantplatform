from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class SP_TTM_Calculator(FactorCalculator):
    """
    市销率倒数（SP_TTM）因子：
    SP_TTM = 营业收入 TTM / 总市值
    等价于 SP = 1 / PS

    计算逻辑：
    1) 营业收入（利润表累计值）→ 转单季度
    2) 单季度收入滚动4期求和 → Revenue TTM
    3) SP_TTM = Revenue_TTM / total_mv

    字段说明：
    - revenue : 营业收入（累计值，利润表）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 利润表：营业收入（累计）
        self.required_income_factors = [
            "revenue"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        income_data = self.fetch_income_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        revenue_cum = income_data.xs("revenue", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度收入
        revenue_single = cumulative_to_single_quarter(revenue_cum)

        # 3) 收入 TTM
        revenue_ttm = rolling_sum_quarters(revenue_single, n_quarters=4)

        # 4) 市销率倒数
        sp_ttm = revenue_ttm / total_mv.replace(0, np.nan)

        sp_ttm = sp_ttm.replace([np.inf, -np.inf], np.nan)
        return sp_ttm


FACTOR_CONFIG = {
    "factor_name": "SP_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "营业收入TTM / 总市值",
    "description": "市销率倒数（SP），衡量收入收益率相对估值水平",
    "lookback_days": 750,
    "method": "SP_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}