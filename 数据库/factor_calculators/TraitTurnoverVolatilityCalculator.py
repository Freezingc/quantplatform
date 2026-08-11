from .base import FactorCalculator
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class TraitTurnoverVolatilityCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["turnover_rate", "total_mv", "pb", "close_hfq"]

        self.explanatory_lag = 0
        self.abnormal_window = 20
        self.min_abnormal_periods = 10
        self.y_lags = (1, 5)
        self.ridge_eps = 1e-8

    @staticmethod
    def _cross_section_spread(turnover_df, sort_key_df, mode):
        out = pd.Series(np.nan, index=turnover_df.index, dtype=float)

        for dt in turnover_df.index:
            turn_row = turnover_df.loc[dt]
            key_row = sort_key_df.loc[dt]
            valid = turn_row.notna() & key_row.notna()

            if valid.sum() < 3:
                continue

            key_valid = key_row[valid]
            turn_valid = turn_row[valid]

            ordered = key_valid.sort_values(kind="mergesort").index
            n = len(ordered)
            k = n // 3

            if k < 1:
                continue

            front_idx = ordered[:k]
            back_idx = ordered[-k:]

            front_mean = turn_valid.loc[front_idx].mean()
            back_mean = turn_valid.loc[back_idx].mean()

            if mode == "back_minus_front":
                out.at[dt] = back_mean - front_mean
            elif mode == "front_minus_back":
                out.at[dt] = front_mean - back_mean
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        return out

    def _calc_abnormal_turnover(self, turnover_rate):
        rolling_mean = turnover_rate.rolling(
            self.abnormal_window,
            min_periods=self.min_abnormal_periods
        ).mean()

        rolling_std = turnover_rate.rolling(
            self.abnormal_window,
            min_periods=self.min_abnormal_periods
        ).std()

        abnormal_turnover = (turnover_rate - rolling_mean) / rolling_std
        abnormal_turnover = abnormal_turnover.replace([np.inf, -np.inf], np.nan)

        return abnormal_turnover

    def _build_explanatory_factors(self, abnormal_turnover, total_mv, pb, close_hfq):
        market_turn = abnormal_turnover.mean(axis=1, skipna=True)

        smb_turn = self._cross_section_spread(
            abnormal_turnover,
            total_mv,
            "back_minus_front"
        )

        hml_turn = self._cross_section_spread(
            abnormal_turnover,
            pb,
            "front_minus_back"
        )

        mom_score = close_hfq.div(close_hfq.shift(21)).sub(1.0)

        mom_turn = self._cross_section_spread(
            abnormal_turnover,
            mom_score,
            "front_minus_back"
        )

        x = pd.concat(
            [market_turn, smb_turn, hml_turn, mom_turn],
            axis=1
        )

        x.columns = [
            "market_abturn",
            "smb_abturn",
            "hml_abturn",
            "mom_abturn"
        ]

        return x

    def _rolling_residual_vol(self, y, x, window):
        out = pd.Series(np.nan, index=y.index, dtype=float)

        merged = pd.concat([y.rename("y"), x], axis=1, join="inner")

        for lag in self.y_lags:
            merged[f"y_lag_{lag}"] = merged["y"].shift(lag)

        merged = merged.replace([np.inf, -np.inf], np.nan)

        y_arr = merged["y"].to_numpy(dtype=float)

        x_cols = [col for col in merged.columns if col != "y"]
        x_arr = merged[x_cols].to_numpy(dtype=float)

        valid = np.isfinite(y_arr) & np.isfinite(x_arr).all(axis=1)
        valid_idx = np.flatnonzero(valid)

        if valid_idx.size < window:
            return out

        yv = y_arr[valid]
        xv = x_arr[valid]

        y_win = sliding_window_view(yv, window_shape=window)
        x_win = sliding_window_view(xv, window_shape=window, axis=0)

        if x_win.shape[1] != window:
            x_win = np.swapaxes(x_win, 1, 2)

        ones = np.ones((x_win.shape[0], window, 1), dtype=float)
        design = np.concatenate([ones, x_win], axis=2)

        xtx = np.einsum("nwi,nwj->nij", design, design)
        xty = np.einsum("nwi,nw->ni", design, y_win)

        eye = np.eye(xtx.shape[1], dtype=float)

        try:
            beta = np.linalg.solve(
                xtx + self.ridge_eps * eye[None, :, :],
                xty[..., None]
            ).squeeze(-1)

        except np.linalg.LinAlgError:
            beta = np.full(
                (xtx.shape[0], design.shape[2]),
                np.nan,
                dtype=float
            )

            for i in range(xtx.shape[0]):
                try:
                    beta[i] = np.linalg.solve(
                        xtx[i] + self.ridge_eps * eye,
                        xty[i]
                    )
                except np.linalg.LinAlgError:
                    continue

        y_hat = np.einsum("nwi,ni->nw", design, beta)
        resid = y_win - y_hat

        rmse = np.sqrt(np.nanmean(resid ** 2, axis=1))

        end_positions = valid_idx[window - 1:]
        out.iloc[end_positions] = rmse

        return out

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)

        turnover_rate = data.xs("turnover_rate", level=0).sort_index()

        total_mv = data.xs("total_mv", level=0).reindex(turnover_rate.index)
        pb = data.xs("pb", level=0).reindex(turnover_rate.index)
        close_hfq = data.xs("close_hfq", level=0).reindex(turnover_rate.index)

        abnormal_turnover = self._calc_abnormal_turnover(turnover_rate)

        x = self._build_explanatory_factors(
            abnormal_turnover,
            total_mv,
            pb,
            close_hfq
        )

        if self.explanatory_lag != 0:
            x = x.shift(self.explanatory_lag)

        window = max(int(self.lookback_days), 21)

        factor = pd.DataFrame(
            np.nan,
            index=turnover_rate.index,
            columns=turnover_rate.columns,
            dtype=float
        )

        total_stock_num = len(turnover_rate.columns)

        for i, ts_code in enumerate(turnover_rate.columns, start=1):
            factor[ts_code] = self._rolling_residual_vol(
                abnormal_turnover[ts_code],
                x,
                window
            )

            if i % 200 == 0 or i == total_stock_num:
                print(f"TraitTurnoverVolatility 进度: {i}/{total_stock_num}")

        return factor


'''
FACTOR_CONFIG = {
    "factor_name": "特质异常换手波动率",
    "creator": "zhangyudong",
    "create_date": "2026.04.29",
    "formula": "abturn = zscore(turnover_rate); Rolling OLS: abturn ~ market_abturn + smb_abturn + hml_abturn + mom_abturn + own_lag_abturn; factor = residual RMSE",
    "description": "特质异常换手波动率：先对个股换手率做滚动标准化，再控制市场、规模、估值、动量换手风格以及个股自身换手惯性，取残差波动率",
    "lookback_days": 60,
    "method": "TraitTurnoverVolatilityCalculator",
    "start_time": "",
    "end_time": "",
    "category": "波动率因子"
}
'''