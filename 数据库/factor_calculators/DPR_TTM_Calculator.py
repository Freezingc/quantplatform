from .base import FactorCalculator, rolling_sum_quarters
import numpy as np

class DPR_TTM_Calculator(FactorCalculator):
    """
    股利支付率（DPR_TTM）因子：
    DPR_TTM = 现金分红总额 / 净利润
    其中：股利支付率是通过计算每个季度的现金分红总额与净利润的比率来求得，
    然后使用TTM（Trailing Twelve Months，过去12个月的滚动数据）来计算。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["c_dividends"]  # 现金分红总额
        self.required_income_factors = ["n_income"]  # 净利润

    def _calculate(self, start_time, end_time):
        # 1) 获取现金分红总额和净利润数据
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        income_data = self.fetch_income_data(start_time, end_time)

        cash_dividends = cashflow_data.xs("c_dividends", level=0)  # 现金分红总额
        net_profit = income_data.xs("n_income", level=0)  # 净利润

        # 2) 计算每个季度的股利支付率
        dpr = cash_dividends / net_profit.replace(0, np.nan)  # 防止除以零

        # 3) 计算股利支付率的TTM（过去四个季度滚动求和）
        dpr_ttm = rolling_sum_quarters(dpr, n_quarters=4)

        dpr_ttm = dpr_ttm.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return dpr_ttm

# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "DPR_TTM",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "现金分红总额 / 净利润",
    "description": "股利支付率，衡量企业利润分配的比例，使用过去12个月数据（TTM）",
    "lookback_days": 750,
    "method": "DPR_TTM_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}
