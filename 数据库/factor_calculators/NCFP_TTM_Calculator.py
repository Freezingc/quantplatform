from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class NCFP_TTM_Calculator(FactorCalculator):
    """
    净现金流市值比（NCFP_TTM）因子：
    NCFP_TTM = 净现金流 TTM / 总市值

    计算逻辑：
    1) 净现金流（现金流量表，累计值）→ 转单季度
    2) 单季度净现金流滚动4期求和 → Net Cashflow TTM
    3) NCFP_TTM = NetCashflow_TTM / total_mv

    字段说明：
    - n_incr_cash_cash_equ : 净现金流（现金及现金等价物净增加额，累计值）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 现金流量表：净现金流（累计）
        self.required_cashflow_factors = [
            "n_incr_cash_cash_equ"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        net_cf_cum = cashflow_data.xs("n_incr_cash_cash_equ", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度
        net_cf_single = cumulative_to_single_quarter(net_cf_cum)

        # 3) 净现金流 TTM
        net_cf_ttm = rolling_sum_quarters(net_cf_single, n_quarters=4)

        # 4) 净现金流市值比
        ncfp_ttm = net_cf_ttm / total_mv.replace(0, np.nan)

        ncfp_ttm = ncfp_ttm.replace([np.inf, -np.inf], np.nan)
        return ncfp_ttm


FACTOR_CONFIG = {
    "factor_name": "NCFP_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.02.05",
    "formula": "净现金流TTM / 总市值",
    "description": "净现金流收益率，衡量整体现金净流入相对于估值水平的强弱",
    "lookback_days": 750,
    "method": "NCFP_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
