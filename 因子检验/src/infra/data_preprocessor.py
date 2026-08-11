"""
数据预处理模块
由于仅仅是预处理，所以对外的 public 函数，其 dataframe 格式统一为：
- 接受 dataframe 的列：名称、日期、因子
- 返回 dataframe 的列：名称、日期、因子
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
import os
from typing import List, Tuple, Optional
import logging

from ..config import DATA_PREPROCESS_CONFIG
from .database import DatabaseManager

logger = logging.getLogger(__name__)
# warnings.filterwarnings('ignore')



def transform_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """转换因子数据格式从宽格式到长格式"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()
    
    # 过滤有效日期列
    valid_dates = []
    for col in df.columns:
        try:
            pd.to_datetime(col)
            valid_dates.append(col)
        except:
            logger.warning(f"跳过无效日期列：{col}")
            continue
            
    df = df[valid_dates]
    df.columns = pd.to_datetime(df.columns)
    
    # 转换长格式
    long_df = df.stack().reset_index(name='factor')
    col_map = {
        long_df.columns[0]: 'instrument',
        long_df.columns[1]: 'date'
    }
    return long_df.rename(columns=col_map)[['instrument', 'date', 'factor']]


def transform_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    转换为 {'instrument', 'date', 'factor'} 格式的 DataFrame。
    
    - 股票代码列名可以是 'instrument' 或 'ts_code'
    - 时间列名可以是 'date' 或 'trade_date'
    - 如果存在 'factor' 列，则取其作为因子值，其余列舍弃
    - 如果没有 'factor' 列，要求总列数为 3，剩余那一列视为因子值
    """
    cols = set(df.columns)

    # 允许的别名
    instrument_col = None
    for c in ['instrument', 'ts_code']:
        if c in cols:
            instrument_col = c
            break

    date_col = None
    for c in ['date', 'trade_date']:
        if c in cols:
            date_col = c
            break

    if instrument_col is None or date_col is None:
        raise ValueError("缺少必要的股票代码列或日期列 (instrument/ts_code, date/trade_date)")

    if 'factor' in cols:
        factor_col = 'factor'
    else:
        remaining_cols = cols - {instrument_col, date_col}
        if len(remaining_cols) != 1:
            raise ValueError("没有 'factor' 列时，必须正好三列 (代码, 日期, 因子值)")
        factor_col = remaining_cols.pop()

    df = df[[instrument_col, date_col, factor_col]].rename(columns={
        instrument_col: 'instrument',
        date_col: 'date',
        factor_col: 'factor'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df


def get_existing_factor_names(output_dir):
    """获取已有的因子名称"""
    existing_factor = []

    if not os.path.exists(output_dir):
        return existing_factor

    for factor_name in os.listdir(output_dir):
        existing_factor.append(factor_name)
    return existing_factor


def batch_load_and_transform_factors(input_folder: str = "factor_data_files", output_folder: Optional[str] = None) -> Tuple[List[str], List[pd.DataFrame]]:
    """批量读取因子数据文件"""
    factor_names = []
    transformed_dfs = []
    
    if not os.path.exists(input_folder):
        logger.error(f"输入文件夹不存在: {input_folder}")
        return factor_names, transformed_dfs
    
    # 获取已分析的因子名，避免重复分析
    existing_factors = []
    if output_folder:
        existing_factors = get_existing_factor_names(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.startswith("factor_data_") and filename.endswith(".csv"):
            try:
                # 提取因子名称
                factor = filename[len("factor_data_"):-len(".csv")]
                if factor in existing_factors:
                    continue

                # 读取数据
                file_path = os.path.join(input_folder, filename)

                if DATA_PREPROCESS_CONFIG['date_format'] == 'wide':
                    raw_wide_df = pd.read_csv(file_path, index_col=0) # ts_code 作为行索引
                    long_df = transform_wide_format(raw_wide_df)
                elif DATA_PREPROCESS_CONFIG['date_format'] == 'long':
                    raw_long_df = pd.read_csv(file_path)
                    long_df = transform_long_format(raw_long_df)
                else:
                    raise ValueError(f"不支持的数据格式: {DATA_PREPROCESS_CONFIG['date_format']}")
                
                # 存储因子名称
                factor_names.append(factor)

                # 存储 dataframe
                transformed_dfs.append(long_df)
                logger.info(f"成功处理：{filename}")
                
            except Exception as e:
                logger.error(f"处理 {filename} 失败：{str(e)}")
                continue
                
    return factor_names, transformed_dfs


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.config = DATA_PREPROCESS_CONFIG
    

    def _apply_filters(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """应用数据过滤条件"""
        original_count = len(factor_data)
        
        # 去除上市时间不足的股票
        factor_data = factor_data[
            (factor_data['date'] - factor_data['list_date']).dt.days > self.config['min_list_days']
        ]
        
        # 只保留指定交易所的股票
        factor_data = factor_data[
            factor_data['exchange'].isin(self.config['valid_exchanges'])
        ]
        
        # 处理异常值
        factor_data[['factor', 'low', 'high', 'open', 'close']] = factor_data[
            ['factor', 'low', 'high', 'open', 'close']
        ].replace([np.inf, -np.inf], np.nan)
        
        # 处理过大的值（过大会导致后续标准化溢出）
        MAX_VALUE_THRESHOLD = 1e10  # 100亿
        too_large_mask = factor_data['factor'].abs() > MAX_VALUE_THRESHOLD
        too_large_count = too_large_mask.sum()

        if too_large_count > 0:
            logger.warning(f"发现 {too_large_count} 个过大的值（绝对值 > {MAX_VALUE_THRESHOLD}），已将其替换为 NaN")

        factor_data['factor'] = factor_data['factor'].where(~too_large_mask, np.nan)
        
        # 删除关键字段为空的行
        factor_data.dropna(subset=['factor', 'low', 'high', 'open', 'close'], inplace=True)
        
        filtered_count = len(factor_data)
        logger.info(f"数据过滤完成: {original_count} -> {filtered_count} 行")
        
        return factor_data


    def filter_factor_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """过滤因子数据，去除不符合条件的股票"""
        start_date = factor_data.date.min().strftime('%Y-%m-%d')
        end_date = factor_data.date.max().strftime('%Y-%m-%d')
        
        # 处理股票代码格式
        factor_data['instrument'] = factor_data['instrument'].apply(lambda x: x[:9])
        
        # 获取股票基本信息
        index_info_data = self.db_manager.get_stock_basic_info(start_date, end_date)
        
        # 合并因子数据和基本信息
        factor_data = pd.merge(factor_data, index_info_data, how='left', on=['date', 'instrument'])
        
        # 应用过滤条件
        factor_data = self._apply_filters(factor_data)
        
        return factor_data[['instrument', 'date', 'factor']]
    

    def _remove_extreme_and_standardize(self, df: pd.DataFrame, col: Optional[str]) -> pd.DataFrame:
        """去极值并标准化"""

        if col is None:
            col = 'factor'

        n = self.config['outlier_threshold']
        mean = df[col].mean()
        std = df[col].std()
        lower_bound = mean - n * std
        upper_bound = mean + n * std
        
        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound
        
        # 标准化
        scaler = StandardScaler()
        df[[col]] = scaler.fit_transform(df[[col]])
        
        return df


    def _add_neutralization_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """添加中性化所需的行业和市值数据"""
        start_date = factor_data.date.min().strftime("%Y-%m-%d")
        end_date = factor_data.date.max().strftime("%Y-%m-%d")
        
        # 获取行业信息（去重：因为一个公司的行业可能有俩个）
        industry_info = self.db_manager.get_stock_industry_info()
        industry_info = industry_info.drop_duplicates(subset=["instrument"], keep="first")
        
        # 获取市值信息
        market_cap_data = self.db_manager.get_market_cap_data(start_date, end_date)
        
        # 合并数据
        factor_data = factor_data.merge(industry_info, on=['instrument'], how='inner')
        factor_data = factor_data.merge(market_cap_data, on=['date', 'instrument'], how='inner')
        
        # 计算对数市值
        factor_data['log_cap'] = np.log(factor_data['total_mv'])
        
        # 删除缺失关键信息的行
        factor_data = factor_data.dropna(subset=['l1_code', 'total_mv'])
        
        return factor_data.reset_index(drop=True)


    def _neutralize(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """行业和市值中性化"""
        # 创建行业虚拟变量
        ind_dummies = pd.get_dummies(df['l1_code'], prefix='l1_code')
        mkcap = df['log_cap']
        
        # 创建训练数据
        train = pd.concat([ind_dummies, mkcap], axis=1)
        X = train.values
        y = df[col].values
        
        # 拟合线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 计算残差作为中性化后的因子值
        residuals = y - model.predict(X)
        df[col] = residuals
        
        return df


    def preprocess_factor_data(self, factor_data: pd.DataFrame, factor_col: Optional[str] = None) -> pd.DataFrame:
        """因子数据预处理：去极值、标准化、中性化"""        
        if factor_col is None:
            factor_col = 'factor'

        # 截面标准化
        factor_data = factor_data.groupby('date')[factor_data.columns].apply(
            self._remove_extreme_and_standardize, col=factor_col
        ).reset_index(drop=True)
        
        factor_data = factor_data.fillna(0)
        
        # 获取行业和市值数据用于中性化
        factor_data = self._add_neutralization_data(factor_data)
        
        # 截面中性化
        factor_data = factor_data.dropna().groupby('date')[factor_data.columns].apply(
            self._neutralize, col=factor_col
        ).reset_index(drop=True)
        
        factor_data.sort_values(by='date', inplace=True)
        factor_data.reset_index(inplace=True, drop=True)
        
        return factor_data[['instrument', 'date', 'factor']]