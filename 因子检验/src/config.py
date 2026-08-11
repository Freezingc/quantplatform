"""
配置文件 - 集中管理所有配置项
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")


def _env(*names, default=""):
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


# 数据库配置
DATABASE_CONFIG = {
    "host": _env("DOLPHINDB_HOST", "dolphinDB_HOST", default="10.176.55.211"),
    "port": int(_env("DOLPHINDB_PORT", "dolphinDB_PORT", default="8848")),
    "username": _env("DOLPHINDB_USERNAME", "dolphinDB_USERNAME", default="jiangyushi"),
    "password": _env("DOLPHINDB_PASSWORD", "dolphinDB_PASSWORD", default="123456")
}

# 分析参数配置
DEFAULT_ANALYSIS_PARAMS = {
    'group_num': 10,
    'factor_direction': 1,
    'benchmark': '中证500',
    'data_preprocess': True,
    'ic_horizons': [1]
}

# 基准指数映射
BENCHMARK_MAPPING = {
    "沪深300": '000300.SH',
    "中证500": '000905.SH', 
    "中证1000": '000852.SH'
}

# 数据处理配置
DATA_PREPROCESS_CONFIG = {
    'outlier_threshold': 3,  # 极值处理的标准差倍数
    'min_list_days': 252,  # 最小上市天数（去除上市不足一年的股票）
    'valid_exchanges': ['SSE', 'SZSE'],  # 有效交易所（去除北交所的票，保留沪深交易所的股票）
    'date_format': 'long'  # 数据格式，long 或 wide
}

# 可视化配置
VISUALIZATION_CONFIG = {
    'top_n_stocks': 5,  # 展示Top N只股票
    'figure_size': (12, 8),
    'matplotlib_font': ['SimHei', 'AR PL UMing CN'],
    'save_format': 'png',
    'dpi': 300
}

# 文件路径配置
PATH_CONFIG = {
    'input_folder': 'factor_data_files',
    'visualization_folder': 'visualization'
}

# 是否启用 alphalens 库
ENABLE_ALPHALENS = False
