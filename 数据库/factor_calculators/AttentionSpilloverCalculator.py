from .base import FactorCalculator
import numpy as np
import pandas as pd


class AttentionSpilloverCalculator(FactorCalculator):
    def __init__(self, s, lookback_days):
        super().__init__(s, lookback_days)
        self.required_factors = ["pct_chg", "total_mv"]
        self._industry_map = None

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

    @staticmethod
    def _assign_size_bucket(sorted_mv):
        n = len(sorted_mv)
        pct = (np.arange(1, n + 1, dtype=float) - 0.5) / n
        bucket = np.where(pct <= 0.3, "small", np.where(pct > 0.7, "big", "mid"))
        return pd.Series(bucket, index=sorted_mv.index)

    @staticmethod
    def _leave_one_out_spillover(attention_series):
        m = len(attention_series)
        if m <= 1:
            return pd.Series(np.nan, index=attention_series.index, dtype=float)

        s = attention_series.sum()
        peer_mean = (s - attention_series) / (m - 1)
        return peer_mean - attention_series

    def _calculate(self, start_time, end_time):
        data = self.fetch_data(start_time, end_time)

        ret = data.xs("pct_chg", level=0).sort_index()
        total_mv = data.xs("total_mv", level=0).reindex(ret.index)

        cross_mean_ret = ret.mean(axis=1, skipna=True)
        abnormal_sq = ret.sub(cross_mean_ret, axis=0).pow(2)
        attention = abnormal_sq.rolling(window=21, min_periods=21).mean()

        industry_map = self._get_industry_map()
        industry_for_cols = pd.Series(ret.columns, index=ret.columns).map(industry_map)

        out = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns, dtype=float)
        col_pos = {c: i for i, c in enumerate(ret.columns)}
        out_arr = out.to_numpy()

        total_dates = len(ret.index)
        for di, dt in enumerate(ret.index, start=1):
            att_row = attention.loc[dt]
            mv_row = total_mv.loc[dt]

            row_df = pd.DataFrame({
                "attention": att_row,
                "mv": mv_row,
                "industry": industry_for_cols,
            })
            row_df = row_df.dropna(subset=["attention", "mv", "industry"])
            if row_df.empty:
                continue

            row_spill = pd.Series(np.nan, index=row_df.index, dtype=float)

            for _, ind_df in row_df.groupby("industry", sort=False):
                if ind_df.empty:
                    continue

                ind_df = ind_df.sort_values("mv").copy()
                ind_df["bucket"] = self._assign_size_bucket(ind_df["mv"])

                for _, bucket_df in ind_df.groupby("bucket", sort=False):
                    spill_bucket = self._leave_one_out_spillover(bucket_df["attention"])
                    row_spill.loc[spill_bucket.index] = spill_bucket

            if not row_spill.empty:
                cols = [col_pos[c] for c in row_spill.index]
                out_arr[di - 1, cols] = row_spill.to_numpy(dtype=float)

            if di % 200 == 0 or di == total_dates:
                print(f"AttentionSpillover 进度: {di}/{total_dates}")

        return out


'''
FACTOR_CONFIG = {
    "factor_name": "AttentionSpillover",
    "creator": "zhangyudong",
    "create_date": "2026.05.06",
    "formula": "attention=MEAN_21((ret-cross_mean(ret))^2); spillover=peer_mean(attention|same_industry&size_bucket)-attention",
    "description": "基于日收益注意力代理的溢出因子：行业内按市值30/40/30分层，取同层相近股票注意力均值与个股注意力之差",
    "lookback_days": 21,
    "method": "AttentionSpilloverCalculator",
    "start_time": "",
    "end_time": "",
    "category": "注意力因子"
}
'''
