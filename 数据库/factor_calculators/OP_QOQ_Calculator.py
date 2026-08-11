from .base import FactorCalculator, lag_by_quarters
import numpy as np


class OP_QOQ_Calculator(FactorCalculator):
    """
    营业利润增速（单季度环比，OP_QOQ）因子：
    OP_QOQ = (本季度营业利润 - 上季度营业利润) / 上季度营业利润
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)

        # 所需因子：营业利润
        self.required_income_factors = [
            "operate_profit"  # 营业利润（利润表中的数据）
        ]

    def _calculate(self, start_time, end_time):
        # 1) 取数：获取营业利润数据
        income_data = self.fetch_income_data(start_time, end_time)

        operate_profit = income_data.xs("operate_profit", level=0)  # 本季度营业利润

        # 2) 计算上季度营业利润
        operate_profit_lag = lag_by_quarters(operate_profit, 1)  # 上季度营业利润

        # 3) 计算营业利润增速（环比）
        op_qoq = (operate_profit - operate_profit_lag) / operate_profit_lag.replace(0, np.nan)

        op_qoq = op_qoq.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return op_qoq


FACTOR_CONFIG = {
    "factor_name": "OP_QOQ",
    "creator": "zhangyudong",
    "create_date": "2026.02.25",
    "formula": "(本季度营业利润 - 上季度营业利润) / 上季度营业利润",
    "description": "营业利润增速（单季度环比），衡量公司短期盈利增长情况",
    "lookback_days": 750,
    "method": "OP_QOQ_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "盈利因子",
}