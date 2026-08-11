import argparse
import os
import torch
from exp.exp_stock import SExp
from exp.exp_qlib import QExp
import random
import numpy as np
import torch.distributed as dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, e.g. [LSTM, GRU, ALSTM, TCN, Transformer, SFM, MTSMatrix, MTSMixer, GAT, GAT_improve, STGAT]')
    parser.add_argument('--random_seed', type=int, default=2025, help='random seed')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--pretrain_data', type=str, default='UTSD', help='pretrain data')
    parser.add_argument('--checkpoint', type=str, help='checkpoint')
    parser.add_argument('--resume', action='store_true', default=False, help='resume training from checkpoint')
    
    # data loader
    parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
    parser.add_argument('--market_name', type=str, default='NASDAQ', help='stock market name')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='', help='data file path (for custom database training, .parquet or .csv)')
    parser.add_argument('--scale', action='store_true', default=False, help='data scale')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--train_begin_date', type=str, default='2013-01-01', help='train dataset begin date')
    parser.add_argument('--train_end_date', type=str)
    parser.add_argument('--valid_begin_date', type=str, default='2016-01-01', help='valid dataset begin date')
    parser.add_argument('--valid_end_date')
    parser.add_argument('--test_begin_date', type=str, default='2017-01-01', help='test dataset begin date')
    parser.add_argument('--test_end_date', type=str, default='2018-01-01', help='test dataset end date')
    parser.add_argument('--quantile', type=float, default=0.01, help='mask ratio of return ratio')
    parser.add_argument('--stride_dataset', type=int, default=1, help='few shot')
    
    # forecasting task
    parser.add_argument('--token_num', type=int, default=7, help='input token num')
    parser.add_argument('--input_token_len', type=int, default=96, help='input token length')
    parser.add_argument('--output_token_len', type=int, default=96, help='input token length')
    parser.add_argument('--autoregressive', action='store_true', help='autoregressive', default=False)
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--output_len', type=int, default=5, help='prediction sequence length')
    parser.add_argument('--patch_len', type=int, default=12, help='the length of patch')
    parser.add_argument('--stride', type=int, default=1, help='the length of stride')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48, help='the length of segmen-wise iteration of SegRNN')
    parser.add_argument('--graph',  action='store_true', default=False, help='use graph')
    parser.add_argument('--num_hyperedges', type=int, default=100, help='the number of hyperedge of DyHGAT')
    parser.add_argument('--hyperedges_quantile', type=float, default=0.9, help='the quantile of hyperedge of DyHGAT')
    parser.add_argument('--instruments', type=str)
    parser.add_argument('--horizon', type=int)

    # SFM model specific params
    parser.add_argument('--input_dim', type=int, default=158, help='input dimension for SFM model')
    parser.add_argument('--output_dim', type=int, default=1, help='output dimension for SFM model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension for SFM model')
    parser.add_argument('--freq_dim', type=int, default=10, help='frequency dimension for SFM model')
    parser.add_argument('--dropout_W', type=float, default=0.0, help='dropout rate for input weights in SFM')
    parser.add_argument('--dropout_U', type=float, default=0.0, help='dropout rate for recurrent weights in SFM')
    
    # LSTM/GRU/RNN model specific params
    parser.add_argument('--hidden_size', type=int, default=790, help='hidden size for LSTM/GRU models')
    parser.add_argument('--input_size', type=int, default=158, help='input size for LSTM/GRU models')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for LSTM/GRU models')
    
    # MTSMatrix model specific params
    parser.add_argument('--sampling', type=int, default=1, help='sampling factor for MTSMatrix/MTSMixer (options: 1, 2, 3, 4, 6, 8, 12)')
    parser.add_argument('--mat', type=int, default=0, help='matrix initialization for MTSMatrix (0: random, 1: identity)')
    parser.add_argument('--norm', type=int, default=1, help='use layer normalization in MTSMatrix/MTSMixer (0: False, 1: True)')
    parser.add_argument('--rev', type=int, default=1, help='use RevIN normalization in MTSMatrix/MTSMixer (0: False, 1: True)')
    
    # MTSMixer model specific params
    parser.add_argument('--fac_T', type=int, default=1, help='use factorized temporal mixing in MTSMixer (0: False, 1: True)')
    parser.add_argument('--fac_C', type=int, default=0, help='use factorized channel mixing in MTSMixer (0: False, 1: True)')
    parser.add_argument('--individual', type=int, default=0, help='use individual projection in MTSMixer (0: False, 1: True)')

    # GAT model specific params
    parser.add_argument('--gat_hidden_size', type=int, default=32, help='hidden dimension per GAT head')
    parser.add_argument('--gat_heads', type=int, default=4, help='number of attention heads in each GAT layer')
    parser.add_argument('--gat_layers', type=int, default=2, help='number of stacked GAT layers')
    parser.add_argument('--gat_topk', type=int, default=20, help='top-k neighbors for dynamic graph construction')
    parser.add_argument('--stgat_kernel_size', type=int, default=3, help='kernel size of temporal conv in STGAT')

    # loss function params (generalized loss for turnover control)
    parser.add_argument('--lambda_turnover', type=float, default=0.0, 
                        help='weight for turnover loss (0.0=no control, 0.5-2.0=recommended for neural networks)')
    parser.add_argument('--use_rank_turnover', action='store_true', default=False,
                        help='use rank-based turnover loss (recommended for better turnover control)')
    parser.add_argument('--turnover_warmup_epochs', type=int, default=3,
                        help='linearly warm up turnover weight in first N epochs')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--use_early_stopping', action='store_true', default=False,
                        help='enable early stopping based on validation loss')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--alpha', type=int, default=1, help='alpha of mse_rank_loss')
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    parser.add_argument('--logfile', type=str, default="", help="Log file name")
    parser.add_argument('--result', type=str, default="", help="Result filepath")
    parser.add_argument('--missingday', type=int, default=1, help="data missing day")
    parser.add_argument('--window_len', type=int, default=5)
    parser.add_argument('--log', type=str)
    parser.add_argument('--fill_method', type=str, default='ema',
                        help='missing value fill method for stock data: ema or fast')
    parser.add_argument('--fill_tau', type=float, default=5.0,
                        help='tau for ema fill method (used when fill_method=ema)')

   
    
    parser.add_argument('--inference', type=int, default=0, help='1: inference mode (no labels)')
    parser.add_argument('--infer_date', type=str, default='', help='inference date like 2025-11-25 (optional if you use test_begin_date==test_end_date)')
    parser.add_argument('--fast_inference', type=int, default=1, help='1: only load minimal window for inference')


    args = parser.parse_args()

    # 自动创建输出目录，避免 FileNotFoundError
    for p in [getattr(args, 'checkpoint', None), getattr(args, 'result', None), getattr(args, 'log', None)]:
        if p:
            os.makedirs(p, exist_ok=True)

    
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # 根据 data_path 参数选择使用哪个实验类
    # 如果提供了 data_path，使用 SExp（自有数据库）
    # 否则使用 QExp（qlib数据）
    if hasattr(args, 'data_path') and args.data_path:
        print(f"使用自有数据库训练模式 (SExp)")
        print(f"数据文件: {args.data_path}")
        Exp = SExp
    else:
        print(f"使用Qlib数据训练模式 (QExp)")
        print(f"Alpha版本: {args.alpha}")
        Exp = QExp
        
    exp = Exp(args)

    # inference 模式：强制不训练，只做 test()
    if getattr(args, 'inference', 0) == 1:

        args.is_training = 0

    if args.is_training:
        exp.train()

    exp.test()
