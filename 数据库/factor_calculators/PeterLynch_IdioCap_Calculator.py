from .base import FactorCalculator
import numpy as np
import pandas as pd


class PeterLynch_IdioCap_Calculator(FactorCalculator):
    """
    彼得林奇特质市值因子（21日滚动标准差版本）

    思路源自微信公众号文章"用彼得林奇的思想改进特异市值，IC居然真的能提升？"

    核心逻辑：
    1. 每日截面回归：log(总市值) ~ 彼得林奇偏好基本面变量 + 行业哑变量
       - 解释变量包括：净现金流(正负拆分的对数)、营业收入(正负拆分的对数)、
         非流动性负债的对数、资产负债率、存货的对数
    2. 取回归残差作为"特质市值"日度值
    3. 对残差取21日滚动标准差作为最终因子值
    """

    def __init__(self, s, lookback_days=21):
        super().__init__(s, lookback_days)
        # 日频因子：市值
        self.required_factors = ["total_mv"]
        # 现金流量表：经营/投资/筹资净现金流
        self.required_cashflow_factors = [
            "n_cashflow_act",
            "n_cashflow_inv_act",
        ]
        # 利润表：营业收入
        self.required_income_factors = ["total_revenue", "revenue"]
        # 资产负债表
        self.required_balance_sheet_factors = [
            "total_liab",
            "total_cur_liab",
            "total_assets",
            "inventories",
        ]

    def _log_split_positive_negative(self, arr):
        """对正/负值分别取对数，返回 (log_positive, log_abs_negative)"""
        arr = np.asarray(arr, dtype=np.float64)
        valid = np.isfinite(arr)

        log_pos = np.full(arr.shape, np.nan, dtype=np.float64)
        log_neg = np.full(arr.shape, np.nan, dtype=np.float64)

        pos_mask = valid & (arr > 0)
        neg_mask = valid & (arr < 0)
        zero_mask = valid & (arr == 0)

        # 正负拆分采用“激活侧取log，非激活侧记0”的编码，避免完整样本被NaN过滤掉
        log_pos[pos_mask] = np.log(np.maximum(arr[pos_mask], 1e-10))
        log_pos[neg_mask | zero_mask] = 0.0

        log_neg[neg_mask] = np.log(np.maximum(-arr[neg_mask], 1e-10))
        log_neg[pos_mask | zero_mask] = 0.0

        return log_pos, log_neg

    def _safe_log(self, arr):
        """安全取对数，非正值返回NaN"""
        arr = np.asarray(arr, dtype=np.float64)
        result = np.where(arr > 0, np.log(np.maximum(arr, 1e-10)), np.nan)
        return result

    def _cross_sectional_residual(self, y, X):
        """
        对单日截面运行OLS回归，返回残差向量。
        y: (n_stocks,) 对数市值
        X: (n_stocks, n_features) 解释变量（可能含NaN）

        返回: (n_stocks,) 残差，NaN位置保持NaN
        """
        mask = ~np.isnan(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mask = mask & ~np.any(np.isnan(X), axis=1)

        if mask.sum() < max(10, X.shape[1] + 5):
            return np.full_like(y, np.nan, dtype=np.float64)

        y_clean = y[mask]
        X_clean = X[mask]

        # 标准化X以提高数值稳定性
        X_mean = np.nanmean(X_clean, axis=0)
        X_std = np.nanstd(X_clean, axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_clean - X_mean) / X_std

        try:
            X_design = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
            beta, _, _, _ = np.linalg.lstsq(X_design, y_clean, rcond=None)
            y_pred = X_design @ beta
            residuals = y_clean - y_pred
        except Exception:
            return np.full_like(y, np.nan, dtype=np.float64)

        result = np.full_like(y, np.nan, dtype=np.float64)
        result[mask] = residuals
        return result

    def _calculate(self, start_time, end_time):
        # 获取日频市值数据
        mv_data = self.fetch_data(start_time, end_time)
        total_mv = mv_data.xs("total_mv", level=0)

        # 获取财报数据（按日对齐，自动前向填充）
        cashflow_data = self.fetch_cashflow_data(start_time, end_time)
        income_data = self.fetch_income_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)

        # 净现金流 = 经营净现金流 + 投资净现金流（近似总净现金流）
        n_cashflow_act = cashflow_data.xs("n_cashflow_act", level=0)
        n_cashflow_inv = cashflow_data.xs("n_cashflow_inv_act", level=0)
        net_cashflow = n_cashflow_act + n_cashflow_inv

        # 营业收入优先使用 total_revenue，否则用 revenue
        try:
            revenue = income_data.xs("total_revenue", level=0)
        except KeyError:
            revenue = income_data.xs("revenue", level=0)

        # 资产负债表
        total_liab = balance_data.xs("total_liab", level=0)
        total_cur_liab = balance_data.xs("total_cur_liab", level=0)
        total_assets = balance_data.xs("total_assets", level=0)
        non_cur_liab = total_liab - total_cur_liab  # 非流动负债近似
        inventories = balance_data.xs("inventories", level=0)
        debt_to_assets = total_liab / total_assets  # 资产负债率

        # 对齐所有数据的日期和股票（取交集）
        all_indices = [total_mv, net_cashflow, revenue, non_cur_liab, inventories, debt_to_assets]
        common_index = total_mv.index
        for df in all_indices[1:]:
            common_index = common_index.intersection(df.index)
        common_columns = total_mv.columns
        for df in all_indices[1:]:
            common_columns = common_columns.intersection(df.columns)

        total_mv = total_mv.loc[common_index, common_columns]
        net_cashflow = net_cashflow.loc[common_index, common_columns]
        revenue = revenue.loc[common_index, common_columns]
        non_cur_liab = non_cur_liab.loc[common_index, common_columns]
        inventories = inventories.loc[common_index, common_columns]
        debt_to_assets = debt_to_assets.loc[common_index, common_columns]

        # 预计算面板上的解释变量
        log_mv = np.log(total_mv.values.astype(np.float64))

        log_cf_pos, log_cf_neg = self._log_split_positive_negative(net_cashflow.values)
        log_rev_pos, log_rev_neg = self._log_split_positive_negative(revenue.values)
        log_non_cur_liab = self._safe_log(non_cur_liab.values)
        lev = debt_to_assets.values.astype(np.float64)
        log_inv = self._safe_log(inventories.values)

        # 逐日截面回归，计算残差
        n_days = len(common_index)
        n_stocks = len(common_columns)
        residuals = np.full((n_days, n_stocks), np.nan, dtype=np.float64)

        for d in range(n_days):
            X_features = np.column_stack([
                log_cf_pos[d],
                log_cf_neg[d],
                log_rev_pos[d],
                log_rev_neg[d],
                log_non_cur_liab[d],
                lev[d],
                log_inv[d],
            ])
            residuals[d] = self._cross_sectional_residual(log_mv[d], X_features)

        resi_df = pd.DataFrame(residuals, index=common_index, columns=common_columns)

        # 对残差取21日滚动标准差
        rolling_std = resi_df.rolling(window=self.lookback_days, min_periods=max(5, self.lookback_days // 2)).std(ddof=1)

        rolling_std = rolling_std.replace([np.inf, -np.inf], np.nan)

        return rolling_std


'''
FACTOR_CONFIG = {
    "factor_name": "彼得林奇特质市值_21日滚动标准差",
    "creator": "zhangyudong",
    "create_date": "2026.05.24",
    "formula": "ts_std(residual_of_cross_sectional_regression, 21)",
    "description": "基于彼得林奇偏好基本面变量（净现金流、营业收入、非流动负债、资产负债率、存货）的截面回归残差，取21日滚动标准差。",
    "lookback_days": 21,
    "method": "PeterLynch_IdioCap_Calculator",
    "start_time": "",
    "end_time": "",
    "category": "基本面因子"
}
'''
