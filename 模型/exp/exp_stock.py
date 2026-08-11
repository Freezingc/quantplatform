from utils.tools import EarlyStopping, adjust_learning_rate
from torch.nn.parallel import DistributedDataParallel as DDP
from models import LSTM, GRU, ALSTM, TCN, Transformer, mLSTM, SFM, MTSMatrix, MTSMixer, GAT, GAT_improve, STGAT,STGATim
import torch.nn.functional as F
from data_provider.data_stock import SDataset
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

class SExp():
    def __init__(self, args):
        self.args = args
        self.is_infer = getattr(self.args, 'inference', 0) == 1
        self.model_dict = {
            'LSTM': LSTM,
            'GRU': GRU,
            'ALSTM': ALSTM,
            'TCN': TCN,
            'Transformer': Transformer,
            'mLSTM': mLSTM,
            'SFM': SFM,
            'MTSMatrix': MTSMatrix,
            'MTSMixer': MTSMixer,
            'GAT': GAT,
            'GAT_improve': GAT_improve,
            'STGAT': STGAT,
            'STGATim': STGATim
        }
        # 根据参数选择设备
        if args.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.gpu}')
        else:
            self.device = torch.device('cpu')
        
        # 初始化起始epoch（必须在_build_model之前）
        self.start_epoch = 0
        
        self._build_model()
        if self.args.batch_size != 1:
            raise ValueError(
                f"SExp 当前实现仅支持 batch_size=1，收到 batch_size={self.args.batch_size}。"
            )
        if self.is_infer:
            self.test_dataset = SDataset(
                args,
                flag='test',
                inference=True,
                fast_inference=getattr(self.args, 'fast_inference', 1),
                infer_date=getattr(self.args, 'infer_date', ''),
            )
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=0)
        else:
            # 预处理只做一次：train先构建，valid/test复用同一份处理结果
            self.train_dataset = SDataset(args, flag='train')
            shared_bundle = self.train_dataset.get_preprocessed_bundle()
            self.valid_dataset = SDataset(args, flag='valid', preprocessed_bundle=shared_bundle)
            # test阶段统一按推理模式构造，避免被label可用性截断
            self.test_dataset = SDataset(args, flag='test', preprocessed_bundle=shared_bundle, inference=True)
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size,num_workers=8)
            self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.args.batch_size)
            # 测试时使用单进程，避免 multiprocessing 错误（特别是 nohup 后台运行时）
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size,num_workers=0)
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
        """计算IC损失（信息系数损失）- 改进版,防止NaN"""
        y_pred = y_pred.squeeze().reshape(-1)
        y_true = y_true.squeeze().reshape(-1)
        pred_finite_mask = torch.isfinite(y_pred)
        true_finite_mask = torch.isfinite(y_true)
        
        # 过滤NaN和Inf
        mask = pred_finite_mask & true_finite_mask
        if mask.sum().item() == 0:
            raise RuntimeError(
                "_ic_loss received no valid pred/label pairs. "
                f"pred_numel={y_pred.numel()}, finite_pred={pred_finite_mask.sum().item()}, "
                f"true_numel={y_true.numel()}, finite_true={true_finite_mask.sum().item()}"
            )
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        
        # 安全检查: 如果有效样本数不足,返回固定损失
        if y_pred.numel() < 2:
            return F.mse_loss(y_pred, y_true)
        
        # 去均值
        demeaned_pred = y_pred - y_pred.mean()
        demeaned_true = y_true - y_true.mean()
        
        # 安全检查: 如果标准差为0,返回固定损失
        if torch.mean(demeaned_true.square()) < 1e-12:
            return F.mse_loss(y_pred, y_true)
        
        # 确保维度正确用于cosine_similarity
        pred_norm = torch.sqrt(torch.sum(demeaned_pred.square()) + 1e-8)
        true_norm = torch.sqrt(torch.sum(demeaned_true.square()) + 1e-8)
        cos_sim = torch.sum(demeaned_pred * demeaned_true) / (pred_norm * true_norm)
        
        if not torch.isfinite(cos_sim):
            return F.mse_loss(y_pred, y_true)
        return -cos_sim + 1
    
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

    def valid(self):
        """在验证集上评估并返回平均损失（IC Loss）"""
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(self.valid_loader)):
                batch_x = batch_x[0].float().to(self.device)
                batch_y = batch_y[0].float().to(self.device)
                outputs = self.model(batch_x).to(self.device)
                loss = self._ic_loss(outputs, batch_y)
                total_loss.append(loss.item())
        self.model.train()
        if len(total_loss) == 0:
            return float('inf')
        loss = sum(total_loss) / len(total_loss)
        logging.info(f"Valid, Loss: {loss:.4f}")
        return loss

    def _get_path(self):
        return self.args.checkpoint + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pth'''

    def _get_best_path(self):
        return self.args.checkpoint + f'''/best_alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pth'''
    
    def _get_result(self):
        return self.args.result + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.pkl'''
    
    def _get_log(self):
        return self.args.log + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.log'''
    
    def _get_plot(self):
        return self.args.result + f'''/alpha{self.args.alpha}_{self.args.instruments}_seq{self.args.seq_len}_horizon{self.args.horizon}_train_begin_date{self.args.train_begin_date}_train_end_date{self.args.valid_begin_date}.png'''
    
    
    def train(self):
        model_optim = self._select_optimizer()

        # 获取换手率控制参数
        lambda_turnover = getattr(self.args, 'lambda_turnover', 0.0)
        use_rank_turnover = getattr(self.args, 'use_rank_turnover', False)
        turnover_warmup_epochs = max(1, int(getattr(self.args, 'turnover_warmup_epochs', 3)))
        use_early_stopping = getattr(self.args, 'use_early_stopping', False)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = getattr(self.args, 'patience', 3)
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
                    
                train_loss.append(total_loss.item())
                train_ic_loss.append(ic_loss.item())
                
                # 保存当前预测值（用于下一次迭代对比）
                prev_output_in_epoch = outputs.detach()

                # 反向传播
                total_loss.backward()
                model_optim.step()

            # 计算平均损失
            avg_total_loss = sum(train_loss) / len(train_loss)
            avg_ic_loss = sum(train_ic_loss) / len(train_ic_loss)
            val_loss = self.valid()
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
                    f"Epoch {epoch}, Loss: {avg_total_loss:.4f} "
                    f"(IC: {avg_ic_loss:.4f}, Turnover: {avg_turnover_loss:.6f}, λ_eff: {lambda_turnover_epoch:.4f}), "
                    f"Val Loss: {val_loss:.4f}"
                )
            else:
                logging.info(f"Epoch {epoch}, Loss: {avg_total_loss:.4f}, Val Loss: {val_loss:.4f}")

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

        # 保存模型时记录epoch信息，用于断点续训
        save_path = self._get_path()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'epoch': last_epoch
        }, save_path)
    
    def test(self):
        best_path = self._get_best_path() if hasattr(self, '_get_best_path') else None
        weight_path = best_path if (best_path and os.path.exists(best_path)) else self._get_path()
        state_dict = torch.load(weight_path, map_location=self.device)
        self._load_state_dict_compat(state_dict['model_state_dict'])

        self.model.eval()
        df = pd.DataFrame(columns=['datetime', 'instrument', 'score'])
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                if len(batch) == 3:
                    batch_x, stocklist, datetime = batch
                elif len(batch) == 4:
                    batch_x, batch_y, stocklist, datetime = batch
                else:
                    raise ValueError(f"Unexpected test batch format, len={len(batch)}")
                # DataLoader会添加batch维度,需要squeeze掉
                # batch_x: (batch_size, B, T, F) -> (B, T, F)
                # batch_y: (batch_size, B) -> (B,)
                if batch_x.dim() == 4:  # 有batch维度
                    batch_x = batch_x.squeeze(0)
                    if len(batch) == 4:
                        batch_y = batch_y.squeeze(0)
                    # stocklist和datetime也是tuple,需要取第一个元素
                    if isinstance(stocklist, (list, tuple)) and len(stocklist) > 0:
                        stocklist = stocklist[0] if isinstance(stocklist[0], list) else stocklist
                    if isinstance(datetime, (list, tuple)) and len(datetime) > 0:
                        datetime = datetime[0]
                
                # 跳过空数据或时间维度为0的样本
                if len(stocklist) == 0 or batch_x.shape[1] == 0:
                    continue
                    
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x).cpu().detach().numpy()
                batch_df = pd.DataFrame({
                    'datetime': [datetime] * len(stocklist),
                    'instrument': stocklist,
                    'score': outputs
                })
                df = pd.concat([df, batch_df], ignore_index=True)        
        
        # 检查DataFrame是否为空
        if len(df) == 0:
            print("警告：测试集没有生成任何预测结果，可能日期范围配置错误！")
            print("请检查：")
            print(f"  - 测试集日期: {self.args.test_begin_date} 到 {self.args.test_end_date}")
            print(f"  - 数据文件: {getattr(self.args, 'data_path', 'N/A')}")
            logging.warning("测试集为空，未生成预测结果")
            logging.shutdown()
            return
        
        # 处理tuple类型的datetime和instrument
        if len(df) > 0 and isinstance(df['datetime'].iloc[0], tuple):
            df['datetime'] = df['datetime'].apply(lambda x: x[0])
        if len(df) > 0 and isinstance(df['instrument'].iloc[0], tuple):
            df['instrument'] = df['instrument'].apply(lambda x: x[0])
        
        result_path = self._get_result()
        df.to_pickle(result_path)
        print(f"预测结果已保存: {result_path}")
        print(f"预测样本数: {len(df)}")
        logging.shutdown()