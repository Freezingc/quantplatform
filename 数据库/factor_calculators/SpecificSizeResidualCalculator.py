from .base import FactorCalculator
import numpy as np
import pandas as pd


class SpecificSizeResidualCalculator(FactorCalculator):
    """
    特异市值因子（截面回归残差）：
    以 ln(总市值) 为被解释变量，解释变量包括：
    - 行业哑变量
    - ln(净资产)
    - 资产负债率
    - ln(正净利润)
    - ln(负净利润绝对值)

    因子取值：每日截面回归残差（actual - fitted）。
    """

    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["total_mv"]
        self.required_balance_sheet_factors = [
            "total_hldr_eqy_inc_min_int",
            "total_liab",
            "total_cur_assets",
            "total_nca",
        ]
        self.required_income_factors = ["n_income"]
        self._industry_map = None
        self.min_obs = 30

    def _get_industry_map(self):
        if self._industry_map is not None:
            return self._industry_map

        query = '''
            db = database("dfs://YesLab")
            t = loadTable(db, "stock_info_table")
            select ts_code, industry from t
        '''
        info = self.s.run(query)

        if info is None or info.empty:
            self._industry_map = pd.Series(dtype=object)
            return self._industry_map

        info = info.dropna(subset=["ts_code", "industry"]).copy()
        info["ts_code"] = info["ts_code"].astype(str)
        info["industry"] = info["industry"].astype(str)
        info = info[info["industry"].str.len() > 0]
        info = info.drop_duplicates(subset=["ts_code"], keep="last")

        self._industry_map = info.set_index("ts_code")["industry"]
        return self._industry_map

    def _calculate(self, start_time, end_time):
        daily_data = self.fetch_data(start_time, end_time)
        balance_data = self.fetch_balance_sheet_data(start_time, end_time)
        income_data = self.fetch_income_data(start_time, end_time)

        total_mv = daily_data.xs("total_mv", level=0).sort_index()

        equity = balance_data.xs("total_hldr_eqy_inc_min_int", level=0).reindex(
            index=total_mv.index,
            columns=total_mv.columns,
        )

        total_liab = balance_data.xs("total_liab", level=0).reindex(
            index=total_mv.index,
            columns=total_mv.columns,
        )
        total_assets = (
            balance_data.xs("total_cur_assets", level=0)
            + balance_data.xs("total_nca", level=0)
        ).reindex(
            index=total_mv.index,
            columns=total_mv.columns,
        )

        net_profit = income_data.xs("n_income", level=0).reindex(
            index=total_mv.index,
            columns=total_mv.columns,
        )

        ln_mv = np.log(total_mv.where(total_mv > 0))
        ln_equity = np.log(equity.where(equity > 0))
        debt_asset = total_liab / total_assets.replace(0, np.nan)

        ln_pos_np = np.log(net_profit.where(net_profit > 0))
        ln_abs_neg_np = np.log((-net_profit).where(net_profit < 0))

        industry_map = self._get_industry_map()
        industry_for_cols = pd.Series(total_mv.columns, index=total_mv.columns).map(industry_map)

        factor = pd.DataFrame(np.nan, index=total_mv.index, columns=total_mv.columns, dtype=float)

        total_dates = len(total_mv.index)
        for i, dt in enumerate(total_mv.index, start=1):
            y = ln_mv.loc[dt]
            row_df = pd.DataFrame(
                {
                    "y": y,
                    "ln_equity": ln_equity.loc[dt],
                    "debt_asset": debt_asset.loc[dt],
                    "ln_pos_np": ln_pos_np.loc[dt],
                    "ln_abs_neg_np": ln_abs_neg_np.loc[dt],
                    "industry": industry_for_cols,
                }
            )

            row_df = row_df.replace([np.inf, -np.inf], np.nan)
            row_df = row_df.dropna(subset=["y", "ln_equity", "debt_asset", "industry"])

            if row_df.empty:
                continue

            row_df = row_df[
                row_df["ln_pos_np"].notna() | row_df["ln_abs_neg_np"].notna()
            ]
            if len(row_df) < self.min_obs:
                continue

            x_num = row_df[["ln_equity", "debt_asset", "ln_pos_np", "ln_abs_neg_np"]].fillna(0.0)
            x_ind = pd.get_dummies(row_df["industry"], prefix="ind", drop_first=True)
            x = pd.concat([x_num, x_ind], axis=1)
            x.insert(0, "const", 1.0)

            x_mat = x.to_numpy(dtype=float)
            y_vec = row_df["y"].to_numpy(dtype=float)

            try:
                beta = np.linalg.lstsq(x_mat, y_vec, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue

            fitted = x_mat @ beta
            resid = y_vec - fitted
            factor.loc[dt, row_df.index] = resid

            if i % 50 == 0 or i == total_dates:
                print(f"SpecificSizeResidual 进度: {i}/{total_dates}")

        factor = factor.replace([np.inf, -np.inf], np.nan)
        return factor


FACTOR_CONFIG = {
    "factor_name": "SpecificSizeResidual",
    "creator": "zhangyudong",
    "create_date": "2026.05.06",
    "formula": "CS OLS: ln(total_mv) ~ industry_dummies + ln(equity) + debt_asset + ln(pos_net_profit) + ln(abs(neg_net_profit)); factor=residual",
    "description": "特异市值因子：用行业和基本面解释对数市值，取每日截面回归残差",
    "lookback_days": 1,
    "method": "SpecificSizeResidualCalculator",
    "start_time": "",
    "end_time": "",
    "category": "规模因子",
}
