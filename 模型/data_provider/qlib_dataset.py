import os
from torch.utils.data import Dataset
import qlib
import pandas as pd
import numpy as np
import torch

# 你原来的 handler（保持不变）
from .handler import Alpha158, Alpha360


class QDataset(Dataset):
    """
    统一支持三种模式：
    - 训练/验证/评估（inference=False）：使用 handler._data 构造特征，再在取样阶段按 label 过滤，构造 (X, y, stocklist, datetime)
    - 纯推理（inference=True）：使用 handler._data（不 DropnaLabel），构造 (X, stocklist, datetime)
      且 fast_inference=True 时仅加载 infer_date 前后最小窗口，显著加速
    """

    def __init__(
        self,
        args,
        origin: bool = False,
        flag: str = "train",
        inference: bool = False,          # 纯推理：不需要 t+h 的真实标签
        fast_inference: int = 1,          # 快速推理：只加载最小窗口
        infer_date: str = "",             # 推理目标日（如 2025-11-25）
    ):
        # ----------- qlib 初始化（按你的路径/region）-----------
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

        self.args = args
        self.flag = flag
        self.inference = bool(inference)
        self.fast_inference = int(fast_inference) == 1
        # infer_date 优先取显式参数，否则使用 test_begin_date
        self.infer_date = infer_date or getattr(args, "test_begin_date", "")

        # ===== 1) 计算 handler 时间范围：推理-缩小窗口；常规-按配置 =====
        start_time_cfg = getattr(args, "train_begin_date", None)
        end_time_cfg = getattr(args, "test_end_date", None)

        # 在推理+快速模式下，把 handler 查询范围尽量缩小（仅为减少 IO/初始化耗时）
        if self.inference and self.fast_inference and self.infer_date:
            infer_dt = pd.to_datetime(self.infer_date)
            # 回溯交易日数，可用命令行参数控制；兜底保证 >= seq_len+10
            back_bdays = max(getattr(self.args, "infer_back_bdays", 80), getattr(self.args, "seq_len", 30) + 10)
            start_time_cfg = (infer_dt - pd.tseries.offsets.BDay(back_bdays)).strftime("%Y-%m-%d")
            end_time_cfg = infer_dt.strftime("%Y-%m-%d")

        # ===== 2) 构建 handler =====
        fit_start_time = getattr(args, "fit_start_time", None) or getattr(args, "train_begin_date", None)
        fit_end_time = getattr(args, "fit_end_time", None)
        if fit_end_time is None:
            train_end = getattr(args, "train_end_date", None)
            valid_begin = getattr(args, "valid_begin_date", None)
            test_begin = getattr(args, "test_begin_date", None)
            if train_end is not None:
                fit_end_time = train_end
            elif valid_begin is not None:
                fit_end_time = (pd.to_datetime(valid_begin) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
            elif test_begin is not None:
                fit_end_time = (pd.to_datetime(test_begin) - pd.tseries.offsets.BDay(1)).strftime("%Y-%m-%d")
            else:
                fit_end_time = end_time_cfg

        def _norm_date_str(dt):
            if dt is None or str(dt) == "":
                return None
            return pd.to_datetime(dt).strftime("%Y-%m-%d")

        fit_start_time = _norm_date_str(fit_start_time)
        fit_end_time = _norm_date_str(fit_end_time)
        if fit_start_time is not None and fit_end_time is not None:
            if pd.to_datetime(fit_start_time) > pd.to_datetime(fit_end_time):
                fit_end_time = fit_start_time

        if args.alpha == 158:
            data_handler_config = {
                "start_time": start_time_cfg,
                "end_time": end_time_cfg,
                "instruments": args.instruments,
                "horizon": args.horizon,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
            }
            h = Alpha158(**data_handler_config)
        elif args.alpha == 360:
            data_handler_config = {
                "start_time": start_time_cfg,
                "end_time": end_time_cfg,
                "instruments": args.instruments,
                "horizon": args.horizon,
                "fit_start_time": fit_start_time,
                "fit_end_time": fit_end_time,
            }
            h = Alpha360(**data_handler_config)
        else:
            raise ValueError("Unsupported alpha, only 158/360 are implemented.")

        # 原始数据（供外部调试可用）
        if origin:
            self.data = h._data
            return

        # ===== 3) 统一使用 _data：避免先 DropnaLabel 再做特征填充造成标签可得性泄露 =====
        # label 的可得性过滤放到 __getitem__ 的 end_time 位置执行
        raw_df = h._data

        # ===== 4) 提取特征/标签，统一成宽表结构 =====
        # 支持 MultiIndex 列（('feature', 'xxx'), ('label','LABEL0')）或已扁平列（feature_xxx / label_LABEL0）
        if isinstance(raw_df.columns, pd.MultiIndex):
            # 典型的 qlib handler 结构：顶层 'feature' / 'label'
            if self.inference:
                feat = raw_df["feature"].copy()
                feat.columns = [str(c) for c in feat.columns]
                df = feat
            else:
                feat = raw_df["feature"].copy()
                feat.columns = [str(c) for c in feat.columns]
                lab = raw_df["label"].copy()
                # 兼容 LABEL0 命名
                if "LABEL0" in lab.columns:
                    lab = lab[["LABEL0"]].rename(columns={"LABEL0": "label"})
                else:
                    # 如无 LABEL0，取第一列为 label
                    lab = lab.rename(columns={lab.columns[0]: "label"})
                df = pd.concat([feat, lab], axis=1)
        else:
            # 已扁平列名：feature_*, label_LABEL0
            if self.inference:
                feat = raw_df.filter(like="feature_").copy()
                feat.columns = feat.columns.str.replace("feature_", "", regex=False)
                df = feat
            else:
                feat = raw_df.filter(like="feature_").copy()
                feat.columns = feat.columns.str.replace("feature_", "", regex=False)
                lab = raw_df.filter(like="LABEL0").copy()
                if "label_LABEL0" in lab.columns:
                    lab = lab.rename(columns={"label_LABEL0": "label"})
                elif "LABEL0" in lab.columns:
                    lab = lab.rename(columns={"LABEL0": "label"})
                else:
                    # 找不到 LABEL0 时，退化：不带标签（仅推理可用）
                    if not self.inference:
                        raise ValueError("LABEL0 not found in _data for evaluation mode.")
                    lab = None
                df = pd.concat([feat, lab], axis=1) if lab is not None else feat

        # reset index 到 (datetime, instrument)，并对齐完整矩阵
        df = df.reset_index()
        all_dates = df["datetime"].unique()
        all_inst = df["instrument"].unique()
        full_index = pd.MultiIndex.from_product([all_dates, all_inst], names=["datetime", "instrument"])
        curr = df.set_index(["datetime", "instrument"]).sort_index()

        if self.inference:
            # 纯推理：全是特征列
            feature_cols = list(curr.columns)
            reindexed = curr.reindex(full_index)
            # 用当日横截面均值填充特征空值
            daily_mean = reindexed[feature_cols].groupby(level="datetime").transform("mean")
            filled = reindexed[feature_cols].fillna(daily_mean)
            self.data = filled
        else:
            reindexed = curr.reindex(full_index)
            feature_cols = [c for c in reindexed.columns if c != "label"]
            daily_mean = reindexed[feature_cols].groupby(level="datetime").transform("mean")
            filled_feat = reindexed[feature_cols].fillna(daily_mean)
            self.data = pd.concat([filled_feat, reindexed["label"]], axis=1)

        # ===== 5) 时间切分：推理只保留 [infer_date - (seq_len-1), infer_date] 最小窗口 =====
        unique_dates = self.data.index.get_level_values("datetime").unique().sort_values()

        if self.inference and self.fast_inference and self.infer_date:
            infer_dt = pd.to_datetime(self.infer_date)
            if infer_dt not in unique_dates:
                # 取 <= infer_dt 的最近交易日
                candidates = unique_dates[unique_dates <= infer_dt]
                if len(candidates) == 0:
                    raise ValueError(f"No available dates <= {self.infer_date} in dataset.")
                infer_dt = candidates.max()

            idx_end = unique_dates.get_loc(infer_dt)
            idx_start = max(0, idx_end - (self.args.seq_len - 1))
            start_dt = unique_dates[idx_start]
            end_dt = unique_dates[idx_end]

            # 强制窗口长度检查（必须等于 seq_len，避免 MTSMatrix 维度错）
            T = idx_end - idx_start + 1
            if T < self.args.seq_len:
                raise ValueError(
                    f"[Inference] Not enough history: need seq_len={self.args.seq_len}, got {T}. "
                    f"Consider increasing --infer_back_bdays or using a later infer_date. "
                    f"Window: [{start_dt} ~ {end_dt}]"
                )

            self.train_df = None
            self.valid_df = None
            self.test_df = self.data[
                (self.data.index.get_level_values("datetime") >= start_dt)
                & (self.data.index.get_level_values("datetime") <= end_dt)
            ]
        else:
            # 常规三段：以 begin_date 对齐窗口起点（seq_len-1 天的窗口对齐）
            def _align_begin_date(dates, begin_date: str, seq_len: int):
                # 将 begin_date 往前推 (seq_len-1) 的索引作为起点，避免第一批不满窗口
                if begin_date is None:
                    return dates[0]
                if begin_date not in dates:
                    # 插值到最近的 >= begin_date 的交易日
                    pos = dates.searchsorted(pd.to_datetime(begin_date))
                    if pos >= len(dates):
                        pos = len(dates) - 1
                else:
                    pos = dates.get_loc(pd.to_datetime(begin_date))
                pos = max(0, pos - (seq_len - 1))
                return dates[pos]

            train_begin = getattr(args, "train_begin_date", None)
            valid_begin = getattr(args, "valid_begin_date", None)
            test_begin = getattr(args, "test_begin_date", None)
            test_end = getattr(args, "test_end_date", None)

            valid_begin_aligned = _align_begin_date(unique_dates, valid_begin, self.args.seq_len)
            test_begin_aligned = _align_begin_date(unique_dates, test_begin, self.args.seq_len)

            self.train_df = self.data[
                (self.data.index.get_level_values("datetime") >= train_begin)
                & (self.data.index.get_level_values("datetime") < valid_begin)
            ]
            self.valid_df = self.data[
                (self.data.index.get_level_values("datetime") >= valid_begin_aligned)
                & (self.data.index.get_level_values("datetime") < test_begin)
            ]
            self.test_df = self.data[
                (self.data.index.get_level_values("datetime") >= test_begin_aligned)
                & (self.data.index.get_level_values("datetime") <= test_end)
            ]

    # ===================== 基本长度 =====================
    def __len__(self):
        if self.inference and self.fast_inference:
            return 1  # 纯推理：单窗口
        if self.flag == "train":
            dts = self.train_df.index.get_level_values("datetime").unique().sort_values()
            return max(0, len(dts) - self.args.seq_len + 1)
        if self.flag == "valid":
            dts = self.valid_df.index.get_level_values("datetime").unique().sort_values()
            return max(0, len(dts) - self.args.seq_len + 1)
        if self.flag == "test":
            dts = self.test_df.index.get_level_values("datetime").unique().sort_values()
            return max(0, len(dts) - self.args.seq_len + 1)
        return 0

    # ===================== 取样本 =====================
    def __getitem__(self, index):
        if self.flag == "train":
            unique_dates = self.train_df.index.get_level_values("datetime").unique().sort_values()
            start_time = unique_dates[index]
            end_time = unique_dates[index + self.args.seq_len - 1]
            end_day = self.train_df.loc[end_time]
            # 训练/验证/评估：用 label.notna() 过滤股票
            filtered_stocks = end_day[end_day["label"].notna()].index.unique()

            time_mask = (self.train_df.index.get_level_values("datetime") >= start_time) & (
                self.train_df.index.get_level_values("datetime") <= end_time
            )
            stock_mask = self.train_df.index.get_level_values("instrument").isin(filtered_stocks)
            df = self.train_df[time_mask & stock_mask]

            feat_cols = [c for c in df.columns if c != "label"]
            wide = df[feat_cols].unstack(level="instrument")  # (T, B*F)
            T, BF = wide.shape
            B = len(filtered_stocks)
            F = len(feat_cols)
            X = wide.values.reshape(T, B, F).transpose(1, 0, 2)  # (B, T, F)

            y_df = df.loc[end_time]["label"]
            Y = torch.tensor(y_df.values)
            X = torch.from_numpy(X).float()
            return X, Y.squeeze()

        if self.flag == "valid":
            unique_dates = self.valid_df.index.get_level_values("datetime").unique().sort_values()
            start_time = unique_dates[index]
            end_time = unique_dates[index + self.args.seq_len - 1]
            end_day = self.valid_df.loc[end_time]
            filtered_stocks = end_day[end_day["label"].notna()].index.unique()

            time_mask = (self.valid_df.index.get_level_values("datetime") >= start_time) & (
                self.valid_df.index.get_level_values("datetime") <= end_time
            )
            stock_mask = self.valid_df.index.get_level_values("instrument").isin(filtered_stocks)
            df = self.valid_df[time_mask & stock_mask]

            feat_cols = [c for c in df.columns if c != "label"]
            wide = df[feat_cols].unstack(level="instrument")
            T, BF = wide.shape
            B = len(filtered_stocks)
            F = len(feat_cols)
            X = wide.values.reshape(T, B, F).transpose(1, 0, 2)

            y_df = df.loc[end_time]["label"]
            Y = torch.tensor(y_df.values)
            X = torch.from_numpy(X).float()
            return X, Y.squeeze()

        if self.flag == "test":
            unique_dates = self.test_df.index.get_level_values("datetime").unique().sort_values()

            # -------- 纯推理：不看 label、不因缺失标签过滤股票 --------
            if self.inference and self.fast_inference:
                start_time = unique_dates[0]
                end_time = unique_dates[-1]

                df = self.test_df.loc[start_time:end_time]
                feat_cols = list(df.columns)  # 全是特征

                # 在推理日筛掉“当日特征全空”的股票
                end_day = df.loc[end_time]
                mask_nonallnan = ~end_day[feat_cols].isna().all(axis=1)
                filtered_stocks = end_day[mask_nonallnan].index.unique()

                df = df[df.index.get_level_values("instrument").isin(filtered_stocks)]
                wide = df[feat_cols].unstack(level="instrument")  # (T, B*F)
                T, BF = wide.shape
                B = len(filtered_stocks)
                F = len(feat_cols)
                X = wide.values.reshape(T, B, F).transpose(1, 0, 2)  # (B, T, F)
                X = torch.from_numpy(X).float()

                # 返回：(X, 股票列表, 推理日)
                return X, filtered_stocks.tolist(), end_time.strftime("%Y-%m-%d")

            # -------- 常规测试（可评估）：按 label.notna() 过滤股票 --------
            start_time = unique_dates[index]
            end_time = unique_dates[index + self.args.seq_len - 1]
            end_day = self.test_df.loc[end_time]
            filtered_stocks = end_day[end_day["label"].notna()].index.unique()

            time_mask = (self.test_df.index.get_level_values("datetime") >= start_time) & (
                self.test_df.index.get_level_values("datetime") <= end_time
            )
            stock_mask = self.test_df.index.get_level_values("instrument").isin(filtered_stocks)
            df = self.test_df[time_mask & stock_mask]

            feat_cols = [c for c in df.columns if c != "label"]
            wide = df[feat_cols].unstack(level="instrument")
            T, BF = wide.shape
            B = len(filtered_stocks)
            F = len(feat_cols)
            X = wide.values.reshape(T, B, F).transpose(1, 0, 2)
            y_df = df.loc[end_time]["label"]
            Y = torch.tensor(y_df.values)
            X = torch.from_numpy(X).float()

            # 为了兼容 exp_qlib.test() 的签名，返回 (X, Y, stocklist, datetime)
            return X, Y.squeeze(), y_df.index.tolist(), end_time.strftime("%Y-%m-%d")

        # 兜底
        raise ValueError(f"Unsupported flag: {self.flag}")
