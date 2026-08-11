import os
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import *
import torch
import time

class SDataset(Dataset):
    """
    股票数据集类，支持EMA-FFill（指数平滑前向填充）处理缺失值
    
    EMA-FFill思想（参考GRU-D）:
    - 如果 t 时刻缺失，用 x̃_t = γ·x_{t-1} + (1-γ)·μ
    - 其中 μ 是稳健均值（截断均值或中位数）
    - γ = exp(-Δt/τ) 随空窗增大而衰减
    """
     
    def _compute_robust_mean(self, series, trim_ratio=0.05):
        """
        计算稳健均值（截断均值），去除极端值的影响
        
        Args:
            series: 数据序列
            trim_ratio: 截断比例，默认去除两端各5%
        
        Returns:
            稳健均值
        """
        valid_data = series.dropna()
        if len(valid_data) == 0:
            return 0.0
        
        # 使用截断均值
        n = len(valid_data)
        trim_count = int(n * trim_ratio)
        if trim_count > 0 and n > 2 * trim_count:
            sorted_data = valid_data.sort_values()
            trimmed_data = sorted_data.iloc[trim_count:-trim_count]
            return trimmed_data.mean()
        else:
            return valid_data.median()  # 数据量不足时用中位数

    def _compute_robust_means(self, df):
        """为每个特征列计算稳健均值。"""
        robust_means = {}
        for col in df.columns:
            robust_means[col] = self._compute_robust_mean(df[col])
        return robust_means
    
    def _ema_ffill(self, df, tau=5.0, robust_means=None):
        """
        EMA-FFill: 指数平滑前向填充
        
        对每只股票的每个特征，使用指数衰减的前向填充:
        x̃_t = γ·x_{t-1} + (1-γ)·μ
        其中 γ = exp(-Δt/τ)，Δt是距离上次有效值的时间间隔
        
        Args:
            df: 特征DataFrame，MultiIndex为(trade_date, ts_code)
            tau: 衰减时间常数（单位：交易日），默认5天
                 tau越大，对历史值的依赖越强
                 tau越小，越快衰减到均值
        
        Returns:
            填充后的DataFrame
        """
        print(f"开始EMA-FFill填充 (tau={tau})...")
        
        feature_cols = df.columns.tolist()
        if robust_means is None:
            robust_means = self._compute_robust_means(df)
        
        # 按股票分组进行EMA-FFill
        def ema_ffill_stock(stock_df):
            """对单只股票的所有特征进行EMA-FFill"""
            result = stock_df.copy() 
            
            for col in feature_cols:
                series = result[col].values
                mu = robust_means[col]
                
                # 记录上次有效值和距离
                last_valid = mu  # 初始用稳健均值
                gap = 0  # 距离上次有效值的间隔
                
                for i in range(len(series)):
                    if pd.isna(series[i]):
                        # 缺失值：使用EMA-FFill
                        gap += 1
                        gamma = np.exp(-gap / tau)
                        series[i] = gamma * last_valid + (1 - gamma) * mu
                    else:
                        # 有效值：更新记录
                        last_valid = series[i]
                        gap = 0
                
                result[col] = series
            
            return result
        
        # 按股票分组处理
        filled_df = df.groupby(level='ts_code', group_keys=False).apply(ema_ffill_stock)
        
        print(f"EMA-FFill填充完成")
        return filled_df

    def _fast_ffill(self, df, robust_means=None):
        """
        快速填充（向量化）：
        1) 按股票前向填充
        2) 用稳健均值兜底仍缺失值
        """
        print("开始快速FFill填充（向量化）...")
        if robust_means is None:
            robust_means = self._compute_robust_means(df)
        filled_df = df.groupby(level='ts_code').ffill()
        filled_df = filled_df.fillna(value=robust_means)
        print("快速FFill填充完成")
        return filled_df
    
    def get_preprocessed_bundle(self):
        """导出预处理后的共享数据，供其他flag复用。"""
        return {
            'data': self.data,
            'train_df': self.train_df,
            'valid_df': self.valid_df,
            'test_df': self.test_df,
        }

    def _normalize_stock_order(self, stock_order):
        normalized = []
        for item in stock_order:
            if isinstance(item, tuple):
                normalized.append(item[-1])
            else:
                normalized.append(item)
        return normalized

    def _build_window_tensor(self, df, feature_cols, stock_order):
        """将窗口数据安全转换为 (B, T, F)，确保股票顺序与标签一致。"""
        stock_order = self._normalize_stock_order(stock_order)
        wide_df = df[feature_cols].unstack(level='ts_code')
        ordered_cols = pd.MultiIndex.from_product([feature_cols, stock_order])
        wide_df = wide_df.reindex(columns=ordered_cols)
        T = wide_df.shape[0]
        F = len(feature_cols)
        B = len(stock_order)
        x_np = wide_df.values.reshape(T, F, B).transpose(2, 0, 1)
        return torch.from_numpy(x_np).float()

    def __init__(self, args, flag='train', preprocessed_bundle=None, inference=False, fast_inference=1, infer_date=''):
        self.flag = flag
        self.args = args
        self.inference = bool(inference)
        self.fast_inference = int(fast_inference) == 1
        self.infer_date = infer_date or ''

        # 复用已预处理数据，避免重复做三次fill/zscore/reindex
        if preprocessed_bundle is not None:
            self.data = preprocessed_bundle['data']
            self.train_df = preprocessed_bundle['train_df']
            self.valid_df = preprocessed_bundle['valid_df']
            self.test_df = preprocessed_bundle['test_df']
            return

        strat_time = time.time()
        _, ext = os.path.splitext(args.data_path)
        ext = ext.lower()
        if ext == '.csv':
            self.data = pd.read_csv(args.data_path)
        elif ext == '.parquet':
            self.data = pd.read_parquet(args.data_path)
        
        self.data = self.data.sort_values(by=['trade_date', 'ts_code'])
        self.has_label = 'label' in self.data.columns
        if not self.has_label and not self.inference:
            raise ValueError("本地数据训练/评估模式要求数据包含 label 列；如仅做预测请加 --inference 1。")
        
        # 清理原始数据中的Inf值，替换为NaN
        feature_cols_raw = [col for col in self.data.columns if col not in ['ts_code', 'trade_date', 'label']]
        self.data[feature_cols_raw] = self.data[feature_cols_raw].replace([np.inf, -np.inf], np.nan)
        
        all_dates = self.data['trade_date'].unique()
        all_ts_codes = self.data[~self.data['ts_code'].str.endswith('BJ')]['ts_code'].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_ts_codes],
                                                names=['trade_date', 'ts_code'])
        current_data = self.data.set_index(['trade_date', 'ts_code'])
        reindexed_df = current_data.reindex(full_index)
        feature_cols = [col for col in reindexed_df.columns if col != 'label']

        date_index = reindexed_df.index.get_level_values('trade_date')
        train_fit_mask = (date_index >= args.train_begin_date) & (date_index < args.valid_begin_date)
        fit_df = reindexed_df.loc[train_fit_mask, feature_cols]
        if fit_df.empty:
            fit_df = reindexed_df[feature_cols]
        robust_means = self._compute_robust_means(fit_df)
        
        # ========================================
        # 缺失值填充
        # fill_method:
        # - ema: 慢但更贴近原始EMA-FFill
        # - fast: 向量化快速填充（推荐大数据）
        # ========================================
        fill_method = getattr(args, 'fill_method', 'ema')
        fill_tau = float(getattr(args, 'fill_tau', 5.0))
        if fill_method == 'fast':
            filled_features = self._fast_ffill(reindexed_df[feature_cols], robust_means=robust_means)
        else:
            filled_features = self._ema_ffill(reindexed_df[feature_cols], tau=fill_tau, robust_means=robust_means)
        
        # zscore标准化（按日期分组）
        def zscore(x):
            if len(x) == 0 or x.isna().all():
                return pd.Series(0, index=x.index)
            mean = x.mean()
            std = x.std(ddof=0)
            if pd.isna(std) or std < 1e-6:
                result = x - mean 
            else:
                result = (x - mean) / std
            # Clip极端值防止极端异常
            result = result.clip(-10, 10)
            return result
        
        filled_features = filled_features.groupby(level='trade_date').transform(zscore)
        
        # 最终安全检查：处理zscore后可能产生的NaN/Inf
        filled_features = filled_features.replace([np.inf, -np.inf], np.nan)
        # 对于zscore后仍存在的少量NaN，使用当日截面均值填充
        daily_mean_final = filled_features.groupby(level='trade_date').transform('mean')
        filled_features = filled_features.fillna(daily_mean_final)
        filled_features = filled_features.fillna(0)  # 最后兜底
        
        if self.has_label:
            filled_non_features = reindexed_df['label']
            self.data = pd.concat([filled_features, filled_non_features], axis=1)
        else:
            self.data = filled_features

        unique_dates = self.data.index.get_level_values('trade_date').unique().sort_values()
        if self.inference and self.fast_inference and self.infer_date:
            infer_dt = pd.to_datetime(self.infer_date)
            if infer_dt not in unique_dates:
                candidates = unique_dates[unique_dates <= infer_dt]
                if len(candidates) == 0:
                    raise ValueError(f"No available dates <= {self.infer_date} in dataset.")
                infer_dt = candidates.max()
            idx_end = unique_dates.get_loc(infer_dt)
            idx_start = max(0, idx_end - self.args.seq_len + 1)
            start_dt = unique_dates[idx_start]
            end_dt = unique_dates[idx_end]
            if idx_end - idx_start + 1 < self.args.seq_len:
                raise ValueError(
                    f"[Inference] Not enough history: need seq_len={self.args.seq_len}, got {idx_end - idx_start + 1}."
                )
            self.train_df = None
            self.valid_df = None
            self.test_df = self.data[(self.data.index.get_level_values('trade_date') >= start_dt) &
                        (self.data.index.get_level_values('trade_date') <= end_dt)]
        elif self.inference:
            test_idx = max(unique_dates.searchsorted(args.test_begin_date) - args.seq_len + 1, 0)
            test_begin_date = unique_dates[test_idx]
            self.train_df = None
            self.valid_df = None
            self.test_df = self.data[(self.data.index.get_level_values('trade_date') >= test_begin_date) &
                        (self.data.index.get_level_values('trade_date') <= args.test_end_date)]
        else:
            valid_idx = max(unique_dates.searchsorted(args.valid_begin_date) - args.seq_len + 1, 0)
            test_idx = max(unique_dates.searchsorted(args.test_begin_date) - args.seq_len + 1, 0)
            valid_begin_date = unique_dates[valid_idx]
            test_begin_date = unique_dates[test_idx]
            self.train_df = self.data[(self.data.index.get_level_values('trade_date') >= args.train_begin_date) &
                        (self.data.index.get_level_values('trade_date') < args.valid_begin_date)]
            self.valid_df = self.data[(self.data.index.get_level_values('trade_date') >= valid_begin_date) &
                        (self.data.index.get_level_values('trade_date') < args.test_begin_date)]
            self.test_df = self.data[(self.data.index.get_level_values('trade_date') >= test_begin_date) &
                        (self.data.index.get_level_values('trade_date') <= args.test_end_date)]
        end_time = time.time()
        print("数据预处理耗时", end_time-strat_time)

    def __len__(self):
        if self.flag == 'train':
            if self.train_df is None:
                return 0
            return max(0, len(self.train_df.index.get_level_values('trade_date').unique()) - self.args.seq_len + 1)
        if self.flag == 'valid':
            if self.valid_df is None:
                return 0
            return max(0, len(self.valid_df.index.get_level_values('trade_date').unique()) - self.args.seq_len + 1)
        if self.flag == 'test':
            if self.test_df is None:
                return 0
            if self.inference and self.fast_inference and self.infer_date:
                return 1
            return max(0, len(self.test_df.index.get_level_values('trade_date').unique()) - self.args.seq_len + 1)

    def __getitem__(self, index):
        if self.flag == 'train':
            unique_dates = self.train_df.index.get_level_values('trade_date').unique().sort_values()
            start_time = unique_dates[index]
            end_time = unique_dates[index+self.args.seq_len-1]   
            end_day_data = self.train_df.loc[end_time]
            finite_label_mask = end_day_data['label'].notna() & np.isfinite(end_day_data['label'].values)
            filtered_stocks = end_day_data[finite_label_mask].index.unique()
            # 先按时间筛选，再按股票筛选
            time_mask = (self.train_df.index.get_level_values('trade_date') >= start_time) & \
                        (self.train_df.index.get_level_values('trade_date') <= end_time)
            stock_mask = self.train_df.index.get_level_values('ts_code').isin(filtered_stocks)
            df = self.train_df[time_mask & stock_mask]
            feature_cols = [col for col in df.columns if col != 'label']
            y_df = df[df.index.get_level_values('trade_date') == end_time]['label']
            stock_order = y_df.index.get_level_values('ts_code').tolist()
            X = self._build_window_tensor(df, feature_cols, stock_order)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = torch.tensor(y_df.values, dtype=torch.float32)  # shape: (B, )
            Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            return X, Y.squeeze()
        if self.flag == 'valid':
            unique_dates = self.valid_df.index.get_level_values('trade_date').unique().sort_values()
            start_time = unique_dates[index]
            end_time = unique_dates[index+self.args.seq_len-1]   
            end_day_data = self.valid_df.loc[end_time]
            finite_label_mask = end_day_data['label'].notna() & np.isfinite(end_day_data['label'].values)
            filtered_stocks = end_day_data[finite_label_mask].index.unique()
            # 先按时间筛选，再按股票筛选
            time_mask = (self.valid_df.index.get_level_values('trade_date') >= start_time) & \
                        (self.valid_df.index.get_level_values('trade_date') <= end_time)
            stock_mask = self.valid_df.index.get_level_values('ts_code').isin(filtered_stocks)
            df = self.valid_df[time_mask & stock_mask]
            feature_cols = [col for col in df.columns if col != 'label']
            y_df = df[df.index.get_level_values('trade_date') == end_time]['label']
            stock_order = y_df.index.get_level_values('ts_code').tolist()
            X = self._build_window_tensor(df, feature_cols, stock_order)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = torch.tensor(y_df.values, dtype=torch.float32)  # shape: (B, )
            Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            return X, Y.squeeze()
        if self.flag == 'test':
            unique_dates = self.test_df.index.get_level_values('trade_date').unique().sort_values()
            if self.inference and self.fast_inference and self.infer_date:
                start_time = unique_dates[0]
                end_time = unique_dates[-1]
                df = self.test_df.loc[start_time:end_time]
                feature_cols = [col for col in df.columns if col != 'label']
                end_day_features = df[df.index.get_level_values('trade_date') == end_time][feature_cols]
                if len(end_day_features) == 0:
                    return torch.zeros(0, 0, len(feature_cols)), [], ""
                valid_feature_mask = ~end_day_features.isna().all(axis=1)
                filtered_stocks = end_day_features[valid_feature_mask].index.get_level_values('ts_code').unique().tolist()
                stock_mask = df.index.get_level_values('ts_code').isin(filtered_stocks)
                df = df[stock_mask]
                X = self._build_window_tensor(df, feature_cols, filtered_stocks)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                datetime_str = end_time.strftime('%Y-%m-%d')
                return X, [str(idx) for idx in filtered_stocks], datetime_str
            start_time = unique_dates[index]
            # 边界检查:确保不会超出日期范围
            end_idx = min(index + self.args.seq_len - 1, len(unique_dates) - 1)
            end_time = unique_dates[end_idx]
            
            # 如果时间窗口不足,返回空数据(会在test函数中被跳过)
            if end_idx - index + 1 < self.args.seq_len:
                feature_dim = len([col for col in self.test_df.columns if col != 'label'])
                if self.inference:
                    return torch.zeros(0, 0, feature_dim), [], ""
                return torch.zeros(0, 0, feature_dim), torch.zeros(0), [], ""
            
            end_day_data = self.test_df.loc[end_time]
            # 先按时间筛选，再按股票筛选
            time_mask = (self.test_df.index.get_level_values('trade_date') >= start_time) & \
                        (self.test_df.index.get_level_values('trade_date') <= end_time)
            df = self.test_df[time_mask]
            feature_cols = [col for col in df.columns if col != 'label']
            if self.inference:
                end_day_features = df[df.index.get_level_values('trade_date') == end_time][feature_cols]
                valid_feature_mask = ~end_day_features.isna().all(axis=1)
                filtered_stocks = end_day_features[valid_feature_mask].index.get_level_values('ts_code').unique().tolist()
                stock_mask = df.index.get_level_values('ts_code').isin(filtered_stocks)
                df = df[stock_mask]
                X = self._build_window_tensor(df, feature_cols, filtered_stocks)
                X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                datetime_str = end_time.strftime('%Y-%m-%d')
                return X, [str(idx) for idx in filtered_stocks], datetime_str
            y_df = df[df.index.get_level_values('trade_date') == end_time]['label']
            finite_label_mask = y_df.notna() & np.isfinite(y_df.values)
            filtered_stocks = y_df[finite_label_mask].index.get_level_values('ts_code').unique().tolist()
            stock_mask = df.index.get_level_values('ts_code').isin(filtered_stocks)
            df = df[stock_mask]
            y_df = y_df[finite_label_mask]
            stock_order = y_df.index.get_level_values('ts_code').tolist()
            X = self._build_window_tensor(df, feature_cols, stock_order)
            X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            Y = torch.tensor(y_df.values, dtype=torch.float32)  # shape: (B, )
            Y = torch.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            # 将MultiIndex转换为字符串列表,避免Timestamp对象
            stocklist = [str(idx[1]) if isinstance(idx, tuple) else str(idx) for idx in y_df.index]
            datetime_str = end_time.strftime('%Y-%m-%d')
            return X, Y.squeeze(), stocklist, datetime_str