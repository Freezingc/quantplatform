from .base import (
    FactorCalculator,
    cumulative_to_single_quarter,
    rolling_sum_quarters
)
import numpy as np


class OCFP_TTM_Calculator(FactorCalculator):
    """
    经营性现金流市值比（OCFP_TTM）因子：
    OCFP_TTM = 经营性现金流 TTM / 总市值
    等价于 Operating Cashflow Yield

    计算逻辑：
    1) 经营性现金流（现金流量表累计值）→ 转单季度
    2) 单季度经营性现金流滚动4期求和 → OCF TTM
    3) OCFP_TTM = OCF_TTM / total_mv

    字段说明：
    - c_inf_fr_operate_a : 经营活动产生的现金流量净额（累计值）
    - total_mv : 总市值（日频行情因子）
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 现金流量表：经营性现金流（累计）
        self.required_cashflow_factors = [
            "c_inf_fr_operate_a"
        ]

        # 行情：总市值
        self.required_factors = [
            "total_mv"
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        daily_data = self.fetch_data(start_time, end_time)

        ocf_cum = cashflow_data.xs("c_inf_fr_operate_a", level=0)
        total_mv = daily_data.xs("total_mv", level=0)

        # 2) 累计 → 单季度
        ocf_single = cumulative_to_single_quarter(ocf_cum)

        # 3) OCF TTM
        ocf_ttm = rolling_sum_quarters(ocf_single, n_quarters=4)

        # 4) 经营性现金流收益率
        ocfp_ttm = ocf_ttm / total_mv.replace(0, np.nan)

        ocfp_ttm = ocfp_ttm.replace([np.inf, -np.inf], np.nan)
        return ocfp_ttm


FACTOR_CONFIG = {
    "factor_name": "OCFP_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.02.05",
    "formula": "经营性现金流TTM / 总市值",
    "description": "经营现金流收益率，衡量主营业务现金创造能力相对估值水平",
    "lookback_days": 750,
    "method": "OCFP_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "估值因子",
}
