from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class FCFP_TTM_Calculator(FactorCalculator):
    """
    自由现金流市值比（FCFP_TTM）因子：
    FCFP_TTM = 自由现金流 TTM / 总市值
    等价于 FCF Yield（自由现金流收益率）

    计算逻辑：
    1) 自由现金流（现金流量表）为累计值 → 转单季度
    2) 单季度自由现金流滚动4期求和 → FCF TTM
    3) FCFP_TTM = FCF_TTM / total_mv

    字段说明：
    - free_cashflow : 自由现金流（累计值，现金流量表）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 现金流量表：自由现金流（累计）
        self.required_cashflow_factors = [
            "free_cashflow"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        fcf_cum = cashflow_data.xs("free_cashflow", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度
        fcf_single = cumulative_to_single_quarter(fcf_cum)

        # 3) FCF TTM
        fcf_ttm = rolling_sum_quarters(fcf_single, n_quarters=4)

        # 4) 自由现金流收益率
        fcfp_ttm = fcf_ttm / total_mv.replace(0, np.nan)

        fcfp_ttm = fcfp_ttm.replace([np.inf, -np.inf], np.nan)
        return fcfp_ttm


FACTOR_CONFIG = {
    "factor_name": "FCFP_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.02.05",
    "formula": "自由现金流TTM / 总市值",
    "description": "自由现金流收益率（FCF Yield），衡量现金创造能力相对估值水平",
    "lookback_days": 750,
    "method": "FCFP_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
