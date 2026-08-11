#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据库驱动的 STGATim 定时调仓脚本
流程：
1) 从 feature_STGAT.txt 读取所需因子
2) 从 DolphinDB 拉取最新窗口数据（默认 seq_len+buffer）
3) 生成本地推理 parquet（仅特征，不含 label）
4) 调用 run.py 进行 STGATim 推理
5) 将结果 pkl 转换为回测可用宽表 csv
"""

import argparse
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import dolphindb as ddb
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = Path('/data/zhangyudong/FDU/Quant-Platform-Prod-new')
MODEL_DIR = BASE_DIR / '模型'
SIMULATION_DIR = BASE_DIR / '模拟盘测试'

FEATURE_FILE = SIMULATION_DIR / 'feature_STGAT.txt'
DEFAULT_TMP_DIR = SIMULATION_DIR / 'tmp'
DEFAULT_LOCAL_PARQUET = DEFAULT_TMP_DIR / 'stgatim_latest_infer.parquet'
DEFAULT_LOCAL_SCORE_CSV = SIMULATION_DIR / 'STGATim_30_5_Scores.csv'
DEFAULT_SERVER_OUTPUT_DIR = Path('/data/xiahongjie/Quant-Platform-2025/输出与展示/调仓输出')
DEFAULT_SERVER_OUTPUT_FILENAME = 'MTSMatrix.csv'
DEFAULT_LOG_DIR = SIMULATION_DIR / 'logs'
DEFAULT_RESULT_DIR = DEFAULT_TMP_DIR / 'stgatim_result'
DEFAULT_CHECKPOINT_DIR = MODEL_DIR / 'data' / 'checkpoint' / 'STGATim'
DEFAULT_TRAIN_PARQUET = MODEL_DIR / 'test_data.parquet'

TRAIN_BEGIN_DATE = '2015-01-02'
TRAIN_END_DATE_FOR_NAME = '2025-01-01'


def setup_logger() -> logging.Logger:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = DEFAULT_LOG_DIR / f'auto_tiaocang_stgat_db_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger('auto_tiaocang_stgat_db')
    logger.info('日志文件: %s', log_file)
    return logger


def run_command(cmd: List[str], cwd: Path, logger: logging.Logger) -> None:
    logger.info('执行命令: %s', ' '.join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.returncode != 0:
        raise RuntimeError(f'命令执行失败，exit_code={proc.returncode}')


def read_feature_list(feature_file: Path) -> List[str]:
    factors = []
    with feature_file.open('r', encoding='utf-8') as f:
        for line in f:
            name = line.strip()
            if name:
                factors.append(name)
    if not factors:
        raise ValueError(f'因子列表为空: {feature_file}')
    return factors


def ddb_date_str(date_str: str) -> str:
    return pd.to_datetime(date_str).strftime('%Y.%m.%d')


def connect_ddb(args: argparse.Namespace):
    session = ddb.session()
    session.connect(args.ddb_host, args.ddb_port, args.ddb_user, args.ddb_password)
    return session


def get_store_table(session, factor_name: str) -> str:
    query = f'''
    db = database("dfs://YesLab")
    t = loadTable(db, "factor_metadata")
    select store_factor from t where factor_name = '{factor_name}'
    '''
    result = session.run(query)
    if result is None or result.empty:
        raise ValueError(f'因子 {factor_name} 在 factor_metadata 中不存在')
    return result.iloc[0, 0]


def resolve_infer_date(session, infer_date: str = '') -> str:
    if infer_date:
        return pd.to_datetime(infer_date).strftime('%Y-%m-%d')
    query = '''
    db = database("dfs://YesLab")
    t = loadTable(db, "stock_factor_daily")
    select max(trade_date) as max_trade_date from t
    '''
    result = session.run(query)
    if result is None or result.empty or pd.isna(result.iloc[0, 0]):
        raise RuntimeError('无法从 stock_factor_daily 获取最新交易日')
    return pd.to_datetime(result.iloc[0, 0]).strftime('%Y-%m-%d')


def resolve_window_dates(session, infer_date: str, seq_len: int, extra_days: int) -> Tuple[str, str, List[str]]:
    infer_ddb = ddb_date_str(infer_date)
    query = f'''
    db = database("dfs://YesLab")
    t = loadTable(db, "stock_factor_daily")
    select distinct trade_date from t
    where trade_date <= date(`{infer_ddb})
    order by trade_date
    '''
    result = session.run(query)
    if result is None or result.empty:
        raise RuntimeError('无法获取交易日历')
    all_dates = pd.to_datetime(result['trade_date']).dt.strftime('%Y-%m-%d').tolist()
    need = seq_len + extra_days
    if len(all_dates) < seq_len:
        raise RuntimeError(f'可用交易日不足 seq_len={seq_len}, 当前仅 {len(all_dates)}')
    use_dates = all_dates[-need:] if len(all_dates) > need else all_dates
    return use_dates[0], use_dates[-1], use_dates


def fetch_factor_df(session, factor_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    table_name = get_store_table(session, factor_name)
    start_ddb = ddb_date_str(start_date)
    end_ddb = ddb_date_str(end_date)
    query = f'''
    db = database("dfs://YesLab")
    t = loadTable(db, "{table_name}")
    select ts_code, trade_date, value as "{factor_name}" from t
    where factor_name == "{factor_name}"
      and trade_date between date(`{start_ddb}) and date(`{end_ddb})
    '''
    df = session.run(query)
    if df is None or df.empty:
        raise RuntimeError(f'因子 {factor_name} 在区间 {start_date}~{end_date} 无数据')
    df = df.drop_duplicates(subset=['trade_date', 'ts_code'], keep='first')
    return df


def build_feature_order(reference_parquet: Path, factors: List[str]) -> List[str]:
    prefixed = [f'factor_data_{x}' for x in factors]
    if reference_parquet.exists():
        schema_names = pq.ParquetFile(str(reference_parquet)).schema.names
        ref_order = [c for c in schema_names if c.startswith('factor_data_')]
        ordered = [c for c in ref_order if c in prefixed]
        remaining = sorted([c for c in prefixed if c not in ordered])
        return ordered + remaining
    return sorted(prefixed)


def build_infer_parquet(
    session,
    factors: List[str],
    start_date: str,
    end_date: str,
    output_parquet: Path,
    feature_order: List[str],
    logger: logging.Logger,
) -> pd.DataFrame:
    merged = None
    for idx, factor in enumerate(factors, start=1):
        logger.info('[%d/%d] 拉取因子: %s', idx, len(factors), factor)
        fdf = fetch_factor_df(session, factor, start_date, end_date)
        renamed_col = f'factor_data_{factor}'
        fdf = fdf.rename(columns={factor: renamed_col})
        if merged is None:
            merged = fdf
        else:
            merged = merged.merge(fdf, on=['trade_date', 'ts_code'], how='outer')

    if merged is None or merged.empty:
        raise RuntimeError('合并后数据为空，无法推理')

    merged['trade_date'] = pd.to_datetime(merged['trade_date'])
    merged = merged.sort_values(['trade_date', 'ts_code'])

    existing = [c for c in feature_order if c in merged.columns]
    missing = [c for c in feature_order if c not in merged.columns]
    for col in missing:
        merged[col] = pd.NA

    output_cols = ['trade_date', 'ts_code'] + existing + missing
    merged = merged[output_cols]

    non_null_total = int(merged[existing].notna().sum().sum()) if existing else 0
    if non_null_total == 0:
        raise RuntimeError('推理特征全部为空，停止执行')

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_parquet, index=False)
    logger.info('推理数据已保存: %s', output_parquet)
    logger.info('推理数据形状: %s, 特征列: %d, 非空特征值总数: %d', merged.shape, len(feature_order), non_null_total)
    return merged


def expected_result_pkl(result_dir: Path) -> Path:
    name = (
        'alpha1_None_seq30_horizon5_'
        f'train_begin_date{TRAIN_BEGIN_DATE}_train_end_date{TRAIN_END_DATE_FOR_NAME}.pkl'
    )
    return result_dir / name


def run_stgat_inference(
    python_bin: str,
    infer_parquet: Path,
    infer_date: str,
    result_dir: Path,
    checkpoint_dir: Path,
    log_dir: Path,
    input_size: int,
    logger: logging.Logger,
) -> Path:
    cmd = [
        python_bin,
        '-u',
        'run.py',
        '--random_seed', '2025',
        '--is_training', '0',
        '--inference', '1',
        '--infer_date', infer_date,
        '--fast_inference', '1',
        '--data_path', str(infer_parquet),
        '--model', 'STGATim',
        '--fill_method', 'fast',
        '--seq_len', '30',
        '--horizon', '5',
        '--input_size', str(input_size),
        '--hidden_size', '128',
        '--num_layers', '1',
        '--gat_hidden_size', '32',
        '--gat_heads', '4',
        '--gat_layers', '2',
        '--gat_topk', '20',
        '--dropout', '0.1',
        '--train_begin_date', TRAIN_BEGIN_DATE,
        '--valid_begin_date', TRAIN_END_DATE_FOR_NAME,
        '--test_begin_date', infer_date,
        '--test_end_date', infer_date,
        '--batch_size', '1',
        '--learning_rate', '0.0001',
        '--use_gpu', 'True',
        '--gpu', '1',
        '--checkpoint', str(checkpoint_dir),
        '--result', str(result_dir),
        '--log', str(log_dir),
    ]

    run_command(cmd, cwd=MODEL_DIR, logger=logger)

    pkl_path = expected_result_pkl(result_dir)
    if not pkl_path.exists():
        raise FileNotFoundError(f'未找到推理输出 pkl: {pkl_path}')
    logger.info('推理结果 pkl: %s', pkl_path)
    return pkl_path


def convert_ts_code(code: str) -> str:
    if pd.isna(code):
        return code
    code = str(code).strip()
    m_sh = re.match(r'(SH)(\d+)', code)
    m_sz = re.match(r'(SZ)(\d+)', code)
    if m_sh:
        return f'{m_sh.group(2)}.SH'
    if m_sz:
        return f'{m_sz.group(2)}.SZ'
    return code


def pkl_to_wide_csv(pkl_path: Path, local_csv: Path, logger: logging.Logger) -> None:
    df = pd.read_pickle(pkl_path)
    if df.empty:
        raise RuntimeError('推理结果为空，无法转换 CSV')

    df['instrument'] = df['instrument'].apply(convert_ts_code)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y%m%d')

    wide = df.pivot(index='instrument', columns='datetime', values='score').fillna(0)
    wide.insert(0, 'ts_code', wide.index)
    wide = wide.reset_index(drop=True)

    local_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(local_csv, index=False)

    logger.info('本地调仓 CSV: %s', local_csv)
    logger.info('CSV 形状: %s', wide.shape)


def copy_to_server_output(
    local_csv: Path,
    output_dir: Path,
    output_filename: str,
    logger: logging.Logger,
    no_header: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest_path = output_dir / output_filename
    if no_header:
        with local_csv.open('r', encoding='utf-8') as src, dest_path.open('w', encoding='utf-8') as dst:
            src.readline()
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(local_csv, dest_path)
    logger.info('服务器输出 CSV: %s', dest_path)
    return dest_path


def _build_cron_line(script_path: Path, python_bin: str, cron_log_file: Path) -> str:
    return (
        '0 5 * * 3 '
        f'{shlex.quote(python_bin)} {shlex.quote(str(script_path))} '
        f'>> {shlex.quote(str(cron_log_file))} 2>&1'
    )


def install_weekly_cron(script_path: Path, python_bin: str, logger: logging.Logger) -> None:
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    cron_log_file = DEFAULT_LOG_DIR / 'cron_auto_tiaocang_stgat_db.log'
    cron_line = _build_cron_line(script_path=script_path, python_bin=python_bin, cron_log_file=cron_log_file)

    existing_proc = subprocess.run(
        ['crontab', '-l'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if existing_proc.returncode == 0:
        existing_lines = [line for line in existing_proc.stdout.splitlines() if line.strip()]
    else:
        stderr = (existing_proc.stderr or '').strip().lower()
        if 'no crontab for' in stderr:
            existing_lines = []
        else:
            raise RuntimeError(f'读取当前 crontab 失败: {existing_proc.stderr.strip()}')

    target_script = str(script_path)
    filtered = [line for line in existing_lines if target_script not in line]
    filtered.append(cron_line)
    payload = '\n'.join(filtered) + '\n'

    update_proc = subprocess.run(
        ['crontab', '-'],
        input=payload,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if update_proc.returncode != 0:
        raise RuntimeError(f'写入 crontab 失败: {update_proc.stderr.strip()}')

    logger.info('已安装定时任务: %s', cron_line)


def parse_args():
    parser = argparse.ArgumentParser(description='数据库驱动 STGATim 自动调仓')
    parser.add_argument('--infer_date', type=str, default='', help='推理目标日 YYYY-MM-DD，默认数据库最新交易日')
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--extra_days', type=int, default=15, help='额外回看交易日缓冲')

    parser.add_argument('--feature_file', type=str, default=str(FEATURE_FILE))
    parser.add_argument('--tmp_parquet', type=str, default=str(DEFAULT_LOCAL_PARQUET))
    parser.add_argument('--train_reference_parquet', type=str, default=str(DEFAULT_TRAIN_PARQUET))

    parser.add_argument('--result_dir', type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument('--checkpoint_dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument('--log_dir', type=str, default=str(MODEL_DIR / 'data' / 'logs' / 'STGATim'))

    parser.add_argument('--local_score_csv', type=str, default=str(DEFAULT_LOCAL_SCORE_CSV))
    parser.add_argument('--server_output_dir', type=str, default=str(DEFAULT_SERVER_OUTPUT_DIR))
    parser.add_argument('--server_output_filename', type=str, default=DEFAULT_SERVER_OUTPUT_FILENAME)
    parser.add_argument('--server_with_header', action='store_true', help='服务器输出保留表头（默认去掉表头）')
    parser.add_argument('--skip_server_copy', action='store_true', help='仅生成本地打分 CSV，不复制到服务器目录')

    parser.add_argument('--keep_result_pkl', action='store_true', help='保留推理中间结果 pkl（默认转换后删除）')
    parser.add_argument('--install_cron', action='store_true', help='安装每周三凌晨5点自动执行的 crontab 任务')

    parser.add_argument('--python_bin', type=str, default='/data/zhangyudong/.conda/envs/quantyd/bin/python')

    parser.add_argument('--ddb_host', type=str, default=os.getenv('DDB_HOST', '10.176.55.211'))
    parser.add_argument('--ddb_port', type=int, default=int(os.getenv('DDB_PORT', '8848')))
    parser.add_argument('--ddb_user', type=str, default=os.getenv('DDB_USER', 'jiangyushi'))
    parser.add_argument('--ddb_password', type=str, default=os.getenv('DDB_PASSWORD', '123456'))

    return parser.parse_args()


def main():
    logger = setup_logger()
    args = parse_args()

    logger.info('=' * 80)
    logger.info('数据库 STGATim 自动调仓启动')
    logger.info('=' * 80)

    feature_file = Path(args.feature_file)
    tmp_parquet = Path(args.tmp_parquet)
    reference_parquet = Path(args.train_reference_parquet)

    result_dir = Path(args.result_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    model_log_dir = Path(args.log_dir)

    local_score_csv = Path(args.local_score_csv)
    server_output_dir = Path(args.server_output_dir)

    if args.install_cron:
        install_weekly_cron(script_path=Path(__file__).resolve(), python_bin=args.python_bin, logger=logger)
        return

    session = None
    try:
        factors = read_feature_list(feature_file)
        logger.info('读取因子数: %d', len(factors))

        session = connect_ddb(args)
        infer_date = resolve_infer_date(session, args.infer_date)
        start_date, end_date, use_dates = resolve_window_dates(session, infer_date, args.seq_len, args.extra_days)

        logger.info('推理日期: %s', infer_date)
        logger.info('取数窗口: %s ~ %s (交易日数=%d)', start_date, end_date, len(use_dates))

        feature_order = build_feature_order(reference_parquet, factors)
        build_infer_parquet(
            session=session,
            factors=factors,
            start_date=start_date,
            end_date=end_date,
            output_parquet=tmp_parquet,
            feature_order=feature_order,
            logger=logger,
        )

        pkl_path = run_stgat_inference(
            python_bin=args.python_bin,
            infer_parquet=tmp_parquet,
            infer_date=infer_date,
            result_dir=result_dir,
            checkpoint_dir=checkpoint_dir,
            log_dir=model_log_dir,
            input_size=len(feature_order),
            logger=logger,
        )

        pkl_to_wide_csv(
            pkl_path=pkl_path,
            local_csv=local_score_csv,
            logger=logger,
        )

        if not args.skip_server_copy:
            copy_to_server_output(
                local_csv=local_score_csv,
                output_dir=server_output_dir,
                output_filename=args.server_output_filename,
                logger=logger,
                no_header=not args.server_with_header,
            )

        if not args.keep_result_pkl and pkl_path.exists():
            pkl_path.unlink()
            logger.info('已删除中间推理 pkl: %s', pkl_path)

        logger.info('=' * 80)
        logger.info('自动调仓流程执行成功')
        logger.info('=' * 80)

    except Exception as e:
        logger.exception('执行失败: %s', e)
        sys.exit(1)
    finally:
        if session is not None:
            session.close()


if __name__ == '__main__':
    main()
