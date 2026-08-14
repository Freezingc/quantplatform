# factor_calculators/base.py
import dolphindb as ddb
from pathlib import Path
import sys
from get_data import DataFetcher
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


QUARTER_DAYS_APPROX = 60


def shift_df(df, periods, axis=0):
    """
    使用 NumPy 实现的 DataFrame shift，减少碎片化与性能告警。
    仅对 axis=1 优化，其他情况回退到 pandas 原生 shift。
    """
    if axis != 0:
        return df.shift(periods, axis=axis)
    values = df.to_numpy(copy=True)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if periods == 0:
        out = values.copy()
    elif periods > 0:
        if periods < values.shape[0]:
            out[periods:, :] = values[:-periods, :]
    else:
        p = -periods
        if p < values.shape[0]:
            out[:-p, :] = values[p:, :]
    return pd.DataFrame(out, index=df.index, columns=df.columns)


def lag_by_quarters(df, quarters, quarter_days=QUARTER_DAYS_APPROX):
    return shift_df(df, quarter_days * quarters, axis=0)


def cumulative_to_single_quarter(cum):
    diff = cum.diff(axis=0)
    is_new = diff.ne(0) & diff.notna()
    single = diff.where(is_new, np.nan)
    single = single.ffill(axis=0).copy()
    if single.shape[0] > 0:
        single.iloc[0, :] = cum.iloc[0, :].where(single.iloc[0, :].isna())
    return single


def rolling_sum_quarters(single, n_quarters=4, quarter_days=QUARTER_DAYS_APPROX):
    window = quarter_days * n_quarters
    return single.rolling(window=window, min_periods=window).sum()


def rolling_mean_std_quarters(series, window_quarters, min_quarters=4, quarter_days=QUARTER_DAYS_APPROX):
    window = quarter_days * window_quarters
    min_periods = quarter_days * min_quarters
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return rolling_mean, rolling_std

class FactorCalculator:
    def __init__(self, s, lookback_days):
        self.s = s
        self.fetcher = DataFetcher(s = s)
        self.required_factors = []  # 子类需要指定所需的因子名称
        self.lookback_days = lookback_days  # 滚动窗口大小
        self.required_other_factors = []  # 其他可能需要的因子名称
        self.required_financial_factors = []  # 所需的财报因子名称
        self.required_balance_sheet_factors = []  # 所需的资产负债表因子名称
        self.required_cashflow_factors = []  # 所需的现金流量表因子名称
        self.required_income_factors = []  # 所需的利润表因子名称
        # Optional date-aware industry source used by canonical rematerialization.
        # None preserves the historical static stock_info_table behavior.
        self.industry_provider = None

    def get_industry_labels(self, dates, codes):
        """Return an industry-label matrix indexed by date and security code."""

        date_values = tuple(pd.Timestamp(value).normalize() for value in dates)
        code_values = tuple(dict.fromkeys(map(str, codes)))
        provider = self.industry_provider
        if provider is not None:
            labels = provider.get_industry_labels(date_values, code_values)
            if not isinstance(labels, pd.DataFrame):
                raise TypeError("industry provider must return a DataFrame")
            return labels.reindex(index=date_values, columns=code_values)

        info = self.fetcher.get_stock_industry_info()
        info = info.dropna(subset=["ts_code", "l1_name"]).copy()
        info["ts_code"] = info["ts_code"].astype(str)
        info = info.drop_duplicates(subset=["ts_code"], keep="last")
        mapping = info.set_index("ts_code")["l1_name"]
        values = mapping.reindex(code_values).to_numpy(dtype=object)
        return pd.DataFrame(
            [values.copy() for _ in date_values],
            index=date_values,
            columns=code_values,
            dtype=object,
        )

    @staticmethod
    def industry_zscore_and_skew(stock_ret, labels):
        """Compute date-aware industry z-scores and daily industry skew."""

        if not stock_ret.index.equals(labels.index) or not stock_ret.columns.equals(labels.columns):
            raise ValueError("industry returns and labels must have identical axes")
        zscore = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)
        skew = pd.DataFrame(np.nan, index=stock_ret.index, columns=stock_ret.columns)
        for trade_date in stock_ret.index:
            values = stock_ret.loc[trade_date]
            groups = labels.loc[trade_date]
            valid = values.notna() & groups.notna()
            if not valid.any():
                continue
            grouped = values.loc[valid].groupby(groups.loc[valid], sort=False)
            means = grouped.transform("mean")
            stds = grouped.transform("std")
            daily_zscore = (values.loc[valid] - means) / stds
            daily_skew = grouped.transform("skew")
            valid_codes = values.index[valid]
            zscore.loc[trade_date, valid_codes] = daily_zscore.to_numpy(dtype=float)
            skew.loc[trade_date, valid_codes] = daily_skew.to_numpy(dtype=float)
        return zscore, skew
    
    # 因为存在lookback_days，所以需要预处理时间范围，提前start_time以便获得足够的历史数据
    def preprocess_time_range(self, start_time, end_time):
        pstart_time = (datetime.strptime(start_time, "%Y.%m.%d").date() - timedelta(days=10 * self.lookback_days)).strftime("%Y.%m.%d")
        date_list = self.fetcher.get_trade_calendar(start_time = pstart_time, end_time = end_time)
        columns = []         # 转换为列表
        col_data_list = []
        for column, col_data in date_list.items():
            columns.append(column)
            col_data_list.append(col_data.iloc[0])
        
        start_idx = -1
        for i in range(len(columns)):
            if columns[i] == pd.Timestamp(start_time):
                start_idx = i
                break   
        # 向前查找n个交易日
        count = 0
        current_idx = start_idx - 1
        target_idx = -1

        while count < self.lookback_days and current_idx >= 0:
            if col_data_list[current_idx] == 1:
                count += 1
            current_idx -= 1
            target_idx = current_idx
        target_idx += 1  # 因为current_idx在循环结束时已经减了一次

        return columns[target_idx].strftime("%Y.%m.%d")
    
    def fetch_data(self, start_time, end_time):
        """获取数据并转置，由子类继承"""
        data = self.fetcher.get_stock_daily_matrix(
            factor_names=self.required_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def fetch_other_data(self, start_time, end_time):
        """获取其他数据并转置，由子类继承"""
        if not self.required_other_factors:
            return None
        data = self.fetcher.get_factor_matrix_new(
            factor_names=self.required_other_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def fetch_financial_data(self, start_time, end_time):
        """获取财报数据并转置，由子类继承"""
        if not self.required_financial_factors:
            return None
        data = self.fetcher.get_financial_metrics_matrix(
            factor_names=self.required_financial_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def fetch_balance_sheet_data(self, start_time, end_time):
        """获取资产负债表数据并转置，由子类继承"""
        if not self.required_balance_sheet_factors:
            return None
        data = self.fetcher.get_balance_sheet_matrix(
            factor_names=self.required_balance_sheet_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def fetch_cashflow_data(self, start_time, end_time):
        """获取现金流量表数据并转置，由子类继承"""
        if not self.required_cashflow_factors:
            return None
        data = self.fetcher.get_cashflow_matrix(
            factor_names=self.required_cashflow_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def fetch_income_data(self, start_time, end_time):
        """获取利润表数据并转置，由子类继承"""
        if not self.required_income_factors:
            return None
        data = self.fetcher.get_income_matrix(
            factor_names=self.required_income_factors,
            start_time=start_time,
            end_time=end_time
        )
        return data.T
    
    def calculate(self, start_time, end_time):
        """
        对外暴露的计算方法
        1. 预处理时间范围
        2. 执行实际计算
        """
        processed_start = self.preprocess_time_range(start_time, end_time)
        print(f"将开始日期从{start_time}预处理为{processed_start}")
        processed_start = (datetime.strptime(processed_start, "%Y.%m.%d").date() - timedelta(days=30)).strftime("%Y.%m.%d")
        factor = self._calculate(processed_start, end_time)
        #根据所需日期做一次过滤
        filtered_factor = factor[(factor.index >= pd.to_datetime(start_time)) & (factor.index <= pd.to_datetime(end_time))]
        return filtered_factor
    
    def _calculate(self, start_time, end_time):
        """
        实际计算逻辑，子类必须实现
        这个方法接收的是已经预处理过的时间范围
        """
        raise NotImplementedError("子类必须实现_calculate方法")
