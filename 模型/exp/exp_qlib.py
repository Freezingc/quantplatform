from utils.tools import EarlyStopping, adjust_learning_rate
from torch.nn.parallel import DistributedDataParallel as DDP
from models import LSTM, GRU, ALSTM, TCN, Transformer, SFM, MTSMatrix, MTSMixer, GAT, GAT_improve, STGAT, STGATim, STGAT_VMD
import torch.nn.functional as F
from data_provider.qlib_dataset import QDataset
from torch.utils.data import DataLoader

import os
import time
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import pickle
from scipy.stats import pearsonr
import logging
import pandas as pd
import qlib

class QExp():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LSTM': LSTM,
            'GRU': GRU,
            'ALSTM': ALSTM,
            'TCN': TCN,
            'Transformer': Transformer,
            'SFM': SFM,
            'MTSMatrix': MTSMatrix,
            'MTSMixer': MTSMixer,
            'GAT': GAT,
            'GAT_improve': GAT_improve,
            'STGAT': STGAT,
            'STGATim': STGATim,
            'STGAT_VMD': STGAT_VMD
        }
        # 根据参数选择设备
        if args.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu}')
        else:
            self.device = torch.device('cpu')
        
        # 初始化起始epoch（必须在_build_model之前）
        self.start_epoch = 0
        
        self._build_model()

        self.is_infer = getattr(self.args, 'inference', 0) == 1
        num_workers = getattr(self.args, 'num_workers', 8)

        if self.is_infer:
            # 纯推理：只需要 test/infer 数据；不加载 train/valid
            # 关键：给 QDataset 传 inference / fast_inference / infer_date
            self.test_dataset = QDataset(args, origin=False, flag='test',
                                        inference=True,
                                        fast_inference=getattr(self.args, 'fast_inference', 1),
                                        infer_date=getattr(self.args, 'infer_date', ''))
            # 测试时使用单进程，避免 multiprocessing 错误
            self.test_loader = DataLoader(self.test_dataset,
                                          batch_size=self.args.batch_size,
                                          num_workers=0)
        else:
            # 原训练/验证/测试路径
            self.train_dataset = QDataset(args, origin=False, flag='train')
            self.valid_dataset = QDataset(args, origin=False, flag='valid')
            self.test_dataset  = QDataset(args, origin=False, flag='test')
            self.train_loader  = DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=num_workers)
            self.valid_loader  = DataLoader(self.valid_dataset, batch_size=self.args.batch_size, num_workers=num_workers)
            # 测试时使用单进程，避免 multiprocessing 错误（特别是 nohup 后台运行时）
            self.test_loader   = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=0)

        self.test_df = QDataset(args, origin=True, flag='test')
        # 简化日志格式，只输出消息内容
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            filename=self._get_log(),
            filemode='a'  # 改为追加模式，支持断点续训
        )

    def _load_state_dict_compat(self, state_dict):
        """兼容单卡/多卡(DataParallel)权重前缀差异。"""
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        if has_module_prefix:
            stripped = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            self.model.load_state_dict(stripped)
        else:
            wrapped = {f'module.{k}': v for k, v in state_dict.items()}
            self.model.load_state_dict(wrapped)

    def _build_model(self):
        self.model = self.model_dict[self.args.model].Model(self.args).float()
        self.model.to(self.device)

        # 多卡训练（DataParallel）
        if self.args.use_gpu and getattr(self.args, 'use_multi_gpu', False) and torch.cuda.device_count() > 1:
            device_ids = [int(d) for d in str(self.args.devices).split(',') if d.strip() != '']
            if len(device_ids) > 1:
                self.device = torch.device(f'cuda:{device_ids[0]}')
                self.model = nn.DataParallel(self.model, device_ids=device_ids)
                print(f'启用多卡训练 DataParallel, devices={device_ids}')
        
        # 如果存在checkpoint且设置了resume，则加载
        if hasattr(self.args, 'resume') and self.args.resume:
            if os.path.exists(self._get_best_path()):
                checkpoint_path = self._get_best_path()
            else:
                checkpoint_path = self._get_path()
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self._load_state_dict_compat(checkpoint['model_state_dict'])
                if 'epoch' in checkpoint:
                    self.start_epoch = checkpoint['epoch'] + 1
                print(f"Resumed from checkpoint: {checkpoint_path}, starting from epoch {self.start_epoch}")        

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def _ic_loss(self, y_pred, y_true):
        """计算IC损失（信息系数损失）"""
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        mask = ~torch.isnan(y_true)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        demeaned_pred = (y_pred - y_pred.mean()).t().squeeze()
        demeaned_true = (y_true - y_true.mean()).t()
        return -F.cosine_similarity(demeaned_pred, demeaned_true, dim=0) + 1
    
    def _turnover_loss(self, y_pred_current, y_pred_previous, use_rank=False):
        """
        计算换手率损失（预测稳定性损失）
        
        两种模式：
        1. 预测值差异 (use_rank=False): L2 = MSE(ŷ^t, ŷ^{t-1})
        2. 排名差异 (use_rank=True): L2 = MSE(rank(ŷ^t), rank(ŷ^{t-1}))
        
        Args:
            y_pred_current: 当前预测值 shape: (B,)
            y_pred_previous: 上一期预测值 shape: (B,)
            use_rank: 是否使用排名计算（更符合实际换手率）
        
        Returns:
            换手率损失标量
        """
        if y_pred_previous is None:
            return torch.tensor(0.0, device=y_pred_current.device)

        y_pred_current = y_pred_current.reshape(-1)
        y_pred_previous = y_pred_previous.reshape(-1)
        n_cur, n_prev = y_pred_current.numel(), y_pred_previous.numel()

        # 相邻日期可用股票池可能不同，先做安全对齐，避免广播警告
        if n_cur != n_prev:
            n = min(n_cur, n_prev)
            if n < 2:
                return torch.tensor(0.0, device=y_pred_current.device)
            if n_cur > n:
                keep = torch.topk(y_pred_current, k=n, largest=True).indices
                y_pred_current = y_pred_current[keep]
            if n_prev > n:
                keep = torch.topk(y_pred_previous, k=n, largest=True).indices
                y_pred_previous = y_pred_previous[keep]
        
        if use_rank:
            # 基于排名的换手率损失（归一化到 [0,1]，避免截面股票数大时量级爆炸）
            rank_current = torch.argsort(torch.argsort(y_pred_current, descending=True)).float()
            rank_previous = torch.argsort(torch.argsort(y_pred_previous, descending=True)).float()
            norm = max(rank_current.numel() - 1, 1)
            rank_current = rank_current / norm
            rank_previous = rank_previous / norm
            turnover_loss = F.mse_loss(rank_current, rank_previous)
        else:
            # 基于预测值的换手率损失（先标准化，避免尺度漂移主导总loss）
            y_pred_current = (y_pred_current - y_pred_current.mean()) / (y_pred_current.std() + 1e-6)
            y_pred_previous = (y_pred_previous - y_pred_previous.mean()) / (y_pred_previous.std() + 1e-6)
            turnover_loss = F.mse_loss(y_pred_current, y_pred_previous)
        
        return turnover_loss
    
    def _generalized_loss(self, y_pred, y_true, y_pred_previous=None, lambda_turnover=0.0, use_rank=False):
        """
        广义损失函数（结合IC损失和换手率控制）
        General Loss = L1(IC Loss) + λ × L2(Turnover Loss)
        
        Args:
            y_pred: 当前预测值
            y_true: 真实标签
            y_pred_previous: 上一期预测值（用于换手率控制）
            lambda_turnover: 换手率控制权重，建议取值0.1-2.0
            use_rank: 是否使用排名计算换手率（推荐True）
        
        Returns:
            总损失, IC损失, 换手率损失
        """
        # L1: IC损失（预测准确性）
        ic_loss = self._ic_loss(y_pred, y_true)
        
        # L2: 换手率损失（预测稳定性）
        if lambda_turnover > 0 and y_pred_previous is not None:
            turnover_loss = self._turnover_loss(y_pred, y_pred_previous, use_rank=use_rank)
            total_loss = ic_loss + lambda_turnover * turnover_loss
            return total_loss, ic_loss, turnover_loss
        else:
            return ic_loss, ic_loss, torch.tensor(0.0, device=y_pred.device)

    def _calc_cross_section_ic(self, y_pred, y_true):
        pred = np.asarray(y_pred).reshape(-1)
        true = np.asarray(y_true).reshape(-1)
        df = pd.DataFrame({'score': pred, 'label': true})
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['score', 'label'])
        if len(df) < 2:
            return np.nan, np.nan
        if df['score'].nunique() < 2 or df['label'].nunique() < 2:
            return np.nan, np.nan
        ic = df['score'].corr(df['label'], method='pearson')
        rank_ic = df['score'].corr(df['label'], method='spearman')
        return ic, rank_ic

    def _summarize_ic_metrics(self, daily_metrics):
        if daily_metrics is None or daily_metrics.empty:
            return {
                'ic_mean': np.nan,
                'ic_std': np.nan,
                'icir': np.nan,
                'icir_annualized': np.nan,
                'rank_ic_mean': np.nan,
                'rank_ic_std': np.nan,
                'rank_icir': np.nan,
                'rank_icir_annualized': np.nan,
                'ic_positive_ratio': np.nan,
                'rank_ic_positive_ratio': np.nan,
                'num_dates': 0,
            }

        ic_series = daily_metrics['ic'].dropna()
        rank_ic_series = daily_metrics['rank_ic'].dropna()

        ic_std = ic_series.std(ddof=1) if len(ic_series) > 1 else np.nan
        rank_ic_std = rank_ic_series.std(ddof=1) if len(rank_ic_series) > 1 else np.nan

        ic_mean = ic_series.mean() if len(ic_series) > 0 else np.nan
        rank_ic_mean = rank_ic_series.mean() if len(rank_ic_series) > 0 else np.nan

        icir = ic_mean / ic_std if pd.notna(ic_std) and abs(ic_std) > 1e-12 else np.nan
        rank_icir = rank_ic_mean / rank_ic_std if pd.notna(rank_ic_std) and abs(rank_ic_std) > 1e-12 else np.nan

        return {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'icir_annualized': icir * np.sqrt(252) if pd.notna(icir) else np.nan,
            'rank_ic_mean': rank_ic_mean,
            'rank_ic_std': rank_ic_std,
            'rank_icir': rank_icir,
            'rank_icir_annualized': rank_icir * np.sqrt(252) if pd.notna(rank_icir) else np.nan,
            'ic_positive_ratio': (ic_series > 0).mean() if len(ic_series) > 0 else np.nan,
            'rank_ic_positive_ratio': (rank_ic_series > 0).mean() if len(rank_ic_series) > 0 else np.nan,
            'num_dates': int(max(len(ic_series), len(rank_ic_series))),
        }

    def _evaluate_prediction_dataframe(self, df):
        if df.empty or 'label' not in df.columns:
            empty_daily = pd.DataFrame(columns=['datetime', 'ic', 'rank_ic', 'count'])
            return self._summarize_ic_metrics(empty_daily), empty_daily

        clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['score', 'label']).copy()
        if clean_df.empty:
            empty_daily = pd.DataFrame(columns=['datetime', 'ic', 'rank_ic', 'count'])
            return self._summarize_ic_metrics(empty_daily), empty_daily

        daily_rows = []
        for datetime_value, group in clean_df.groupby('datetime'):
            ic, rank_ic = self._calc_cross_section_ic(group['score'].values, group['label'].values)
            daily_rows.append({
                'datetime': datetime_value,
                'ic': ic,
                'rank_ic': rank_ic,
                'count': len(group),
            })

        daily_metrics = pd.DataFrame(daily_rows).sort_values('datetime').reset_index(drop=True)
        return self._summarize_ic_metrics(daily_metrics), daily_metrics

    def _normalize_stocklist(self, stocklist):
        if isinstance(stocklist, (pd.Index, np.ndarray)):
            stocklist = stocklist.tolist()
        elif isinstance(stocklist, tuple):
            stocklist = list(stocklist)

        if isinstance(stocklist, list) and len(stocklist) == 1 and isinstance(stocklist[0], (list, tuple, pd.Index, np.ndarray)):
            nested = stocklist[0]
            stocklist = nested.tolist() if isinstance(nested, (pd.Index, np.ndarray)) else list(nested)

        if not isinstance(stocklist, list):
            stocklist = [stocklist]

        normalized = []
        for item in stocklist:
            if isinstance(item, (list, tuple, pd.Index, np.ndarray)):
                if len(item) == 0:
                    normalized.append('')
                else:
                    normalized.append(str(item[0]))
            else:
                normalized.append(str(item))
        return normalized

    def _normalize_datetime(self, datetime_value):
        if isinstance(datetime_value, (list, tuple)):
            if len(datetime_value) == 0:
                return ''
            return self._normalize_datetime(datetime_value[0])
        if isinstance(datetime_value, pd.Timestamp):
            return datetime_value.strftime('%Y-%m-%d')
        return str(datetime_value)

    def _normalize_scores(self, scores):
        scores = np.asarray(scores, dtype=np.float32)
        if scores.size == 0:
            return scores
        mean = np.nanmean(scores)
        std = np.nanstd(scores)
        if not np.isfinite(std) or std < 1e-6:
            normalized = scores - mean
        else:
            normalized = (scores - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(normalized, -3.0, 3.0)

    def valid(self):
        """在验证集上评估并返回平均损失（IC Loss）"""
        total_loss = []
        daily_metrics = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(self.valid_loader)):
                batch_x = batch_x[0].float().to(self.device)
                batch_y = batch_y[0].float().to(self.device)
                outputs = self.model(batch_x).to(self.device)
                loss = self._ic_loss(outputs, batch_y)
                total_loss.append(loss.item())
                ic, rank_ic = self._calc_cross_section_ic(
                    outputs.detach().cpu().numpy(),
                    batch_y.detach().cpu().numpy(),
                )
                daily_metrics.append({'ic': ic, 'rank_ic': rank_ic})
        self.model.train()
        if len(total_loss) == 0:
            return float('inf'), np.nan, np.nan
        loss = sum(total_loss) / len(total_loss)
        daily_metrics_df = pd.DataFrame(daily_metrics)
        summary = self._summarize_ic_metrics(daily_metrics_df)
        logging.info(f"Valid, Loss: {loss:.4f}, IC: {summary['ic_mean']:.4f}, RankIC: {summary['rank_ic_mean']:.4f}")
        return loss, summary['ic_mean'], summary['rank_ic_mean']

    def _get_path(self):
        return self.args.checkpoint + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pth'''
    
    def _get_best_path(self):
        return self.args.checkpoint + f'''/best_alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pth'''
    
    def _get_result(self):
        return self.args.result + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pkl'''
    
    def _get_log(self):
        return self.args.log + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.log'''

    def _get_ic_summary_result(self):
        result_root, _ = os.path.splitext(self._get_result())
        return f'{result_root}_ic_summary.csv'

    def _get_daily_ic_result(self):
        result_root, _ = os.path.splitext(self._get_result())
        return f'{result_root}_daily_ic.csv'
    
    
    def train(self):
        model_optim = self._select_optimizer()
        
        # 获取换手率控制参数
        lambda_turnover = getattr(self.args, 'lambda_turnover', 0.0)
        use_rank_turnover = getattr(self.args, 'use_rank_turnover', False)
        turnover_warmup_epochs = max(1, int(getattr(self.args, 'turnover_warmup_epochs', 3)))
        
        # 简单的 early stopping / best model 逻辑
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = getattr(self.args, 'patience', 3)
        use_early_stopping = getattr(self.args, 'use_early_stopping', False)
        warned_empty_valid = False
        last_epoch = self.start_epoch - 1

        for epoch in range(self.start_epoch, self.start_epoch + self.args.train_epochs):
            last_epoch = epoch
            train_loss = []
            train_ic_loss = []
            train_turnover_loss = []
            self.model.train()
            prev_output_in_epoch = None
            lambda_turnover_epoch = lambda_turnover * min(1.0, (epoch - self.start_epoch + 1) / turnover_warmup_epochs)

            for i, (batch_x, batch_y) in enumerate(tqdm(self.train_loader)):
                model_optim.zero_grad()
                batch_x = batch_x[0].float().to(self.device)
                batch_y = batch_y[0].float().to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x).to(self.device)
                
                # 获取上一个交易日的预测值（同一epoch内相邻批次）
                y_pred_previous = prev_output_in_epoch
                
                # 计算广义损失
                if lambda_turnover > 0:
                    total_loss, ic_loss, turnover_loss = self._generalized_loss(
                        outputs, batch_y, y_pred_previous, lambda_turnover_epoch, use_rank=use_rank_turnover
                    )
                    train_turnover_loss.append(turnover_loss.item())
                else:
                    total_loss = self._ic_loss(outputs, batch_y)
                    ic_loss = total_loss
                    turnover_loss = torch.tensor(0.0, device=outputs.device)
                    
                train_loss.append(total_loss.item())
                train_ic_loss.append(ic_loss.item())
                
                # 保存当前预测值（用于下一次迭代对比）
                prev_output_in_epoch = outputs.detach()

                # 反向传播
                total_loss.backward()
                model_optim.step()

            # 计算平均损失
            avg_total_loss = sum(train_loss) / len(train_loss) if train_loss else float('inf')
            avg_ic_loss = sum(train_ic_loss) / len(train_ic_loss) if train_ic_loss else float('inf')
            
            # 验证集评估
            val_loss, val_ic, val_rank_ic = self.valid()
            is_valid_available = np.isfinite(val_loss)
            if not is_valid_available and not warned_empty_valid:
                logging.warning(
                    "Validation set is empty (Val Loss = inf). "
                    "Model selection falls back to Train IC loss. "
                    "Please set test_begin_date > valid_begin_date for proper validation."
                )
                warned_empty_valid = True

            # 日志输出
            if lambda_turnover > 0 and train_turnover_loss:
                avg_turnover_loss = sum(train_turnover_loss) / len(train_turnover_loss)
                logging.info(
                    f"Epoch {epoch}, Train Loss: {avg_total_loss:.4f} "
                    f"(IC: {avg_ic_loss:.4f}, Turnover: {avg_turnover_loss:.6f}, λ_eff: {lambda_turnover_epoch:.4f}), "
                    f"Val Loss: {val_loss:.4f}, Val IC: {val_ic:.4f}, Val RankIC: {val_rank_ic:.4f}"
                )
            else:
                logging.info(
                    f"Epoch {epoch}, Train Loss: {avg_total_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val IC: {val_ic:.4f}, Val RankIC: {val_rank_ic:.4f}"
                )

            # 保存最佳模型（基于验证集）
            monitor_loss = val_loss if is_valid_available else avg_ic_loss
            if monitor_loss < best_val_loss:
                best_val_loss = monitor_loss
                epochs_no_improve = 0
                save_path = self._get_best_path()
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch
                }, save_path)
                metric_name = "Val Loss" if is_valid_available else "Train IC Loss (fallback)"
                logging.info(f"Saved best model to {save_path} with {metric_name}: {monitor_loss:.4f}")
            else:
                epochs_no_improve += 1
                logging.info(f"No improvement for {epochs_no_improve} epoch(s)")

            if use_early_stopping and epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered. Stop at epoch {epoch} (patience={patience}).")
                break
        
        # 训练结束后，仍保存最后一次训练的模型（用于恢复/比较）
        save_path = self._get_path()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': last_epoch
        }, save_path)
    
    def test(self):
        # 优先加载 best，否则普通 ckpt
        best_path = self._get_best_path() if hasattr(self, '_get_best_path') else None
        weight_path = best_path if (best_path and os.path.exists(best_path)) else self._get_path()
        state_dict = torch.load(weight_path, map_location=self.device)
        self._load_state_dict_compat(state_dict['model_state_dict'])
        self.model.eval()

        df = pd.DataFrame(columns=['datetime', 'instrument', 'score'])

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                # 支持两种 batch 形状：
                # 训练/评估： (batch_x, batch_y, stocklist, datetime)
                # 纯推理：   (batch_x, stocklist, datetime)
                if self.is_infer:
                    batch_x, stocklist, datetime = batch
                else:
                    batch_x, batch_y, stocklist, datetime = batch

                batch_x = batch_x[0].float().to(self.device)
                outputs = self.model(batch_x).cpu().numpy()
                outputs = self._normalize_scores(outputs)
                stocklist = self._normalize_stocklist(stocklist)
                datetime_value = self._normalize_datetime(datetime)

                if self.is_infer:
                    batch_df = pd.DataFrame({
                        'datetime': [datetime_value] * len(stocklist),
                        'instrument': stocklist,
                        'score': outputs
                    })
                else:
                    batch_y = batch_y[0].cpu().numpy()
                    batch_df = pd.DataFrame({
                        'datetime': [datetime_value] * len(stocklist),
                        'instrument': stocklist,
                        'score': outputs,
                        'label': batch_y,
                    })
                df = pd.concat([df, batch_df], ignore_index=True)

        # 安全处理空 df；以及 tuple 字段的清洗
        if df.empty:
            logging.info("Test DataFrame is empty. No samples produced.")
        else:
            if df['datetime'].dtype == object and df['datetime'].apply(lambda x: isinstance(x, tuple)).any():
                df['datetime'] = df['datetime'].apply(lambda x: x[0] if isinstance(x, tuple) else x)
            if df['instrument'].dtype == object and df['instrument'].apply(lambda x: isinstance(x, tuple)).any():
                df['instrument'] = df['instrument'].apply(lambda x: x[0] if isinstance(x, tuple) else x)

        result_path = self._get_result()
        df.to_pickle(result_path)
        if not self.is_infer:
            summary, daily_metrics = self._evaluate_prediction_dataframe(df)
            pd.DataFrame([summary]).to_csv(self._get_ic_summary_result(), index=False)
            daily_metrics.to_csv(self._get_daily_ic_result(), index=False)
            logging.info(
                f"Test IC Mean: {summary['ic_mean']:.4f}, ICIR: {summary['icir']:.4f}, "
                f"RankIC Mean: {summary['rank_ic_mean']:.4f}, RankICIR: {summary['rank_icir']:.4f}"
            )
        logging.shutdown()
