from .base import FactorCalculator, cumulative_to_single_quarter, rolling_sum_quarters, lag_by_quarters
import numpy as np
import pandas as pd



class CFOAD_Calculator(FactorCalculator):
    """
    全部资产现金回收率变动（CFOAD）因子：
    CFOAD = 当期全部资产现金回收率 TTM - 上期全部资产现金回收率 TTM
    其中：
    - CFOA = 经营活动现金流量 / 总资产
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_cashflow_factors = ["c_inf_fr_operate_a"]  # 经营活动现金流量
        self.required_balance_sheet_factors = ["total_assets"]  # 总资产

    def _calculate(self, start_time, end_time):
        # 1) 取数：经营活动现金流量和总资产
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        cashflow = cashflow_data.xs("c_inf_fr_operate_a", level=0)
        total_assets = balance_data.xs("total_assets", level=0)

        # 2) 计算现金回收率（CFOA）
        cfoa = cashflow / total_assets.replace(0, np.nan)  # 防止除以零

        # 3) 计算CFOA TTM（过去四个季度滚动求和）
        cfoa_ttm = rolling_sum_quarters(cfoa, n_quarters=4)

        # 4) 计算CFOAD（当期与上期的CFOA TTM差异）
        cfoa_ttm_lag = lag_by_quarters(cfoa_ttm, 1)
        cfoad = cfoa_ttm - cfoa_ttm_lag

        cfoad = cfoad.replace([np.inf, -np.inf], np.nan)  # 替换无效值
        return cfoad

# 添加因子信息
FACTOR_CONFIG = {
    "factor_name": "CFOAD",
    "creator": "zhangyudong",
    "create_date": "2026.01.29",
    "formula": "当期全部资产现金回收率 TTM - 上期全部资产现金回收率 TTM",
    "description": "全部资产现金回收率变动，衡量现金流回收效率的边际改善或恶化",
    "lookback_days": 750,
    "method": "CFOAD_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子",
}
