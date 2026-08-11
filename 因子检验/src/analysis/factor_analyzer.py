"""
因子分析核心模块
"""

import os
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, Any, List, Optional, Tuple, Sequence
from contextlib import contextmanager

from ..infra.database import DatabaseManager
from ..infra.data_preprocessor import DataPreprocessor
from ..config import DEFAULT_ANALYSIS_PARAMS, ENABLE_ALPHALENS
from .utils.performance_calculator import PerformanceCalculator
from .utils.visualizer import Visualizer
from .utils.alphalens_calculator import save_alphalens_performance, get_alphalens_data


logger = logging.getLogger(__name__)



@contextmanager
def log_step(step_name: str):
    logger.debug(f"开始{step_name}...")
    try:
        yield
        logger.debug(f"{step_name}完成")
    except Exception as e:
        logger.error(f"{step_name}失败！: {e}")
        raise


class FactorAnalyzer:
    """因子分析器，核心分析类"""

    def __init__(self, factor_data: pd.DataFrame, factor_name: str, params: Optional[Dict] = None):
        """
        初始化因子分析器
        
        参数:
        ----------
            factor_data: 因子数据 DataFrame
            factor_name: 因子名称
            params: 分析参数，包含 horizons 列表
        """
        self.start_time = time.time()
        self.params = params or DEFAULT_ANALYSIS_PARAMS
        self.factor_name = factor_name
        self.base_output_dir = None
        
        # 初始化组件
        self.db_manager = DatabaseManager()
        self.data_preprocessor = DataPreprocessor(self.db_manager)
        self.perf_calculator = PerformanceCalculator()
        
        # 准备因子数据
        self.factor_data = self._prepare_factor_data(factor_data)
        
        # 区分计算需求：IC计算支持多持有期，完整分析仅1D
        self.ic_horizons = self.params.get('ic_horizons', [1])  # IC计算支持所有持有期
        self.full_analysis_horizons = [1]  # 完整分析和可视化仅1D

        logger.info(f"开始分析因子: {factor_name}, IC计算持有期: {self.ic_horizons}, 完整分析持有期: {self.full_analysis_horizons}")

    def _prepare_factor_data(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """准备因子数据"""
        # 应用因子方向
        factor_data['factor'] *= self.params['factor_direction']
        
        # 数据格式检查
        if factor_data['date'].dtype != 'datetime64[ns]':
            raise ValueError("date列的数据格式应为datetime格式")
        
        if not all(factor_data['instrument'].str.endswith(('.SH', '.SZ', '.BJ'))):
            raise ValueError("instrument列的数据格式应为以.SH或.SZ或.BJ结尾的字符串")
        
        if factor_data['factor'].dtype not in ['float64', 'int64']:
            raise ValueError("factor列的数据格式应为数值型")
        
        logger.info("数据格式检查通过")
        return factor_data

    def _preprocess_factor_data(self) -> pd.DataFrame:
        """处理因子数据"""
        logger.info("开始因子数据处理...")
        
        # 基础过滤
        with log_step("基础数据过滤"):
            factor_data = self.data_preprocessor.filter_factor_data(self.factor_data)
        
        # 高级预处理（如果启用）
        with log_step("高级数据预处理"):
            if self.params['data_preprocess']:
                factor_data = self.data_preprocessor.preprocess_factor_data(factor_data)
        
        logger.info("因子数据处理完成")
        return factor_data

    def _get_multi_horizon_return_data(self, horizions: Sequence) -> Dict[int, pd.DataFrame]:
        """一次性获取所有持有期的收益率数据（按需获取）"""
        logger.info("获取多持有期收益率数据...")
        
        start_date = self.factor_data.date.min().strftime('%Y-%m-%d')
        end_date = self.factor_data.date.max().strftime('%Y-%m-%d')

        # 查询数据库获取原始价格数据
        price_data = self.db_manager.get_daily_price_data(start_date, end_date)
        
        return_data = {}
        for horizon in horizions:
            # 基于同一份价格数据计算不同持有期的收益率
            return_data[horizon] = self.db_manager.compute_returns(price_data, 'daily_ret', horizon)
        
        logger.info(f"获取 {list(horizions)} 收益率数据完成")
        return return_data

    def _get_multi_horizon_benchmark_data(self, horizions: Sequence) -> Dict[int, pd.DataFrame]:
        """一次性获取所有持有期的基准收益率数据（按需获取）"""
        logger.info("获取多持有期基准收益率数据...")
        
        start_date = self.factor_data.date.min().strftime('%Y-%m-%d')
        end_date = self.factor_data.date.max().strftime('%Y-%m-%d')
        
        # 查询数据库获取基准数据
        benchmark_price_data = self.db_manager.get_benchmark_price_data(self.params['benchmark'], start_date, end_date)

        benchmark_data = {}
        for horizon in horizions:
            # 基于同一份基准数据计算不同持有期的收益率
            benchmark_ret = self.db_manager.compute_returns(
                benchmark_price_data, 'benchmark_ret', horizon=horizon
            )
            benchmark_data[horizon] = benchmark_ret
        
        logger.info(f"获取 {list(horizions)} 基准收益率数据完成")
        return benchmark_data

    def _perform_factor_grouping(self, merged_data: pd.DataFrame) -> pd.DataFrame:
        """执行因子分组"""
        def group_by_factor(df: pd.DataFrame):
            df['group'] = pd.qcut(
                df['factor'], 
                q=self.params['group_num'], 
                labels=False, 
                duplicates='drop'
            )
            df = df.dropna(subset=['group'])
            df['group'] = df['group'].astype(int).astype(str)

            # 计算实际分组数
            valid_groups = len(np.unique(df['group']))
            dropped_groups = self.params['group_num'] - valid_groups
            if dropped_groups:
                logger.warning(f"实际分组数与预期不符！预期分组{self.params['group_num']}，实际分组{valid_groups}")

            return df
        
        group_data = merged_data.groupby('date').apply(group_by_factor, include_groups=False)
        return group_data.reset_index(level=1, drop=True).reset_index()

    def _merge_and_group_data(self, return_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """合并数据并进行分组"""        
        merged_data = pd.merge(
            self.factor_data.sort_values(['date', 'instrument']),
            return_data.sort_values(['date', 'instrument']),
            on=['date', 'instrument'],
            how='inner'
        )
        
        # 因子分组
        group_data = self._perform_factor_grouping(merged_data)
        
        return merged_data, group_data

    def _calculate_ic_only_analysis(self, horizon: int, return_data: pd.DataFrame) -> Dict[str, Any]:
        """计算仅IC指标的轻量级分析（用于多持有期IC汇总表）"""
        logger.info(f"计算 {horizon}D 持有期IC分析...")

        with log_step("数据合并"):
            merged_data, group_data = self._merge_and_group_data(return_data)

        with log_step("计算IC指标"):
            daily_ic = self.perf_calculator.calculate_daily_ic(merged_data)

        logger.info(f"{horizon}D 持有期IC分析完成")

        return {
            'horizon': horizon,
            'merged_data': merged_data,
            'group_data': group_data,
            'daily_ic': daily_ic
        }

    def _calculate_full_analysis(self, horizon: int, return_data: pd.DataFrame,
                                benchmark_data: pd.DataFrame) -> Dict[str, Any]:
        """计算完整分析结果（包含所有指标，用于1D完整效果汇总表和可视化）"""
        logger.info(f"计算 {horizon}D 持有期完整分析...")
        
        with log_step("数据合并和分组"):
            merged_data, group_data = self._merge_and_group_data(return_data)
        
        with log_step("计算分组收益率"):
            group_returns = self.perf_calculator.calculate_group_returns(group_data)
        
        with log_step("计算累积收益率"):
            cumulative_returns = self.perf_calculator.calculate_cumulative_returns(
                group_returns, benchmark_data
            )
        
        with log_step("计算绩效指标"):
            performance_metrics = {}
            portfolio_types = ['long', 'short', 'long_short']
            
            for portfolio_type in portfolio_types:
                # 计算绩效指标
                perf_metrics = self.perf_calculator.calculate_performance_metrics(
                    group_returns, benchmark_data, portfolio_type
                )
                
                # 计算IC指标
                ic_metrics = self.perf_calculator.calculate_portfolio_ic_metrics(
                    group_data, portfolio_type
                )
                
                # 计算换手率
                turnover_metrics = self.perf_calculator.calculate_turnover_metrics(
                    group_data, portfolio_type
                )
                
                # 合并所有指标
                combined_metrics = {**perf_metrics, **ic_metrics, **turnover_metrics}
                performance_metrics[portfolio_type] = combined_metrics
        
        with log_step("计算年度绩效"):
            all_perf = []
            for portfolio_type in portfolio_types:
                yearly_perf = self.perf_calculator.calculate_yearly_performance(
                    group_returns, benchmark_data, portfolio_type
                )
                yearly_ic = self.perf_calculator.calculate_yearly_ic(
                    group_data, portfolio_type
                )
                yearly_perf['ic'] = yearly_ic
                yearly_perf['portfolio_type'] = portfolio_type
                all_perf.append(yearly_perf)
            
            yearly_performance = pd.concat(all_perf, axis=0)
        
        with log_step("计算IC指标"):
            daily_ic = self.perf_calculator.calculate_daily_ic(group_data)
        
        logger.info(f"{horizon}D 持有期完整分析完成")
        
        return {
            'horizon': horizon,
            'merged_data': merged_data,
            'group_data': group_data,
            'group_returns': group_returns,
            'cumulative_returns': cumulative_returns,
            'performance_metrics': performance_metrics,
            'yearly_performance': yearly_performance,
            'daily_ic': daily_ic,
            'benchmark_ret': benchmark_data
        }

    def run_analysis(self) -> Dict[str, Any]:
        """运行分层分析：IC分析支持多持有期，完整分析仅1D"""
        try:
            # 1. 数据预处理
            self.factor_data = self._preprocess_factor_data()
            
            # 2. 获取所有持有期的收益率数据
            all_horizons = set(self.ic_horizons + self.full_analysis_horizons)
            return_data_dict = self._get_multi_horizon_return_data(all_horizons)

            # 3. 获取所有持有期的基准收益率数据（仅完整分析需要）
            benchmark_data_dict = self._get_multi_horizon_benchmark_data(self.full_analysis_horizons)

            results = {
                'ic_results': [],      # IC分析结果（支持多持有期）
                'full_results': []     # 完整分析结果（仅1D）
            }

            # 4. 对所有持有期进行IC分析
            for horizon in self.ic_horizons:
                ic_result = self._calculate_ic_only_analysis(
                    horizon,
                    return_data_dict[horizon]
                )
                results['ic_results'].append(ic_result)

            # 5. 对1D持有期进行完整分析
            for horizon in self.full_analysis_horizons:
                full_result = self._calculate_full_analysis(
                    horizon,
                    return_data_dict[horizon],
                    benchmark_data_dict[horizon]
                )
                results['full_results'].append(full_result)

            logger.info(f"因子 {self.factor_name} 分析完成")
            return results
            
        except Exception as e:
            logger.error(f"因子分析失败: {str(e)}")
            raise

    def generate_summaries(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成两种汇总表：多持有期IC汇总表 + 1D完整效果汇总表"""
        end_time = time.time()
        summaries = {
            'ic_summary': [],      # 多持有期IC汇总表
            'full_summary': []     # 1D完整效果汇总表
        }

        # 1. 生成多持有期IC汇总表
        for result in results['ic_results']:
            horizon = result['horizon']
            daily_ic = result['daily_ic']

            ic = np.nanmean(daily_ic['g_ic'])
            ic_vol = np.nanstd(daily_ic['g_ic'])
            icir = ic / ic_vol if ic_vol != 0 else np.nan

            ic_summary = {
                'factor_name': self.factor_name,
                '持有期': f"{horizon}D",
                'IC均值': np.round(ic, 4),
                'IC波动率': np.round(ic_vol, 4),
                '年化ICIR': np.round(icir * np.sqrt(252 // horizon), 4),
                '分析日期': pd.Timestamp.now().strftime('%Y-%m-%d'),
                '分析耗时': f"{end_time - self.start_time:.2f}秒"
            }
            summaries['ic_summary'].append(ic_summary)

        # 2. 生成1D完整效果汇总表
        for result in results['full_results']:
            horizon = result['horizon']
            performance_metrics = result['performance_metrics']
            daily_ic = result['daily_ic']
            
            if 'long_short' not in performance_metrics:
                logger.warning(f"缺少 {horizon}D 多空组合绩效数据")
                continue
            
            ic = np.nanmean(daily_ic['g_ic'])
            ic_vol = np.nanstd(daily_ic['g_ic'])
            icir = ic / ic_vol if ic_vol != 0 else np.nan
            long_metrics = performance_metrics['long_short']
            
            full_summary = {
                'factor_name': self.factor_name,
                'IC均值': np.round(ic, 4),
                'IC波动率': np.round(ic_vol, 4),
                '年化ICIR': np.round(icir * np.sqrt(252 // horizon), 4),
                '年化收益': long_metrics.get('annual_return_ratio', np.nan),
                '夏普比率': long_metrics.get('sharp_ratio', np.nan),
                '最大回撤': long_metrics.get('max_drawdown', np.nan),
                '胜率': long_metrics.get('win_percent', np.nan),
                '换手率': long_metrics.get('turnover', np.nan),
                '分析日期': pd.Timestamp.now().strftime('%Y-%m-%d'),
                '分析耗时': f"{end_time - self.start_time:.2f}秒"
            }
            summaries['full_summary'].append(full_summary)

        # 3. 保存汇总表到文件
        if self.base_output_dir:
            # 保存多持有期IC汇总表
            if summaries['ic_summary']:
                ic_df = pd.DataFrame(summaries['ic_summary'])
                ic_file_path = os.path.join(self.base_output_dir, f"{self.factor_name}_ic_summary.csv")
                ic_df.to_csv(ic_file_path, index=False)

            # 保存1D完整效果汇总表
            if summaries['full_summary']:
                full_df = pd.DataFrame(summaries['full_summary'])
                full_file_path = os.path.join(self.base_output_dir, f"{self.factor_name}_full_summary.csv")
                full_df.to_csv(full_file_path, index=False)
        else:
            logger.warning(f"未保存{self.factor_name}汇总表！请先运行可视化函数！")

        return summaries

    def generate_visualizations(self, results: Dict[str, Any]):
        """生成可视化"""
        visualizer = Visualizer(self.factor_name)
        self.base_output_dir = visualizer.base_output_dir
        full_results = results['full_results']
        ic_results = results['ic_results']

        assert len(full_results) == 1 and full_results[0]['horizon'] == 1, f"完整分析仅支持1D，当前为{full_results}"
        
        # 可视化1D完整分析
        if full_results:
            visualizer.generate_full_comprehensive_report({
                'factor_name': self.factor_name,
                'params': self.params,
                'factor_data': self.factor_data,
                'group_data': full_results[0]['group_data'],
                'cumulative_returns': full_results[0]['cumulative_returns'],
                'performance_metrics': full_results[0]['performance_metrics'],
                'yearly_performance': full_results[0]['yearly_performance'],
                'benchmark_ret': full_results[0]['benchmark_ret']
            })
        
        # 可视化多持有期IC分析
        if ic_results:
            all_daily_ic = []
            for result in ic_results:
                df = result['daily_ic'].copy()
                df.columns = pd.MultiIndex.from_product([[result['horizon']], df.columns])
                all_daily_ic.append(df)

            visualizer.generate_ic_comprehensive_report({
                'ic_metrics': pd.concat(all_daily_ic, axis=1)
            })

        logger.info(f"可视化完成")

        # 生成 alphalens 指标
        if ENABLE_ALPHALENS:
            alphalens_data = get_alphalens_data([item["group_data"] for item in results['ic_results']], self.ic_horizons)
            save_alphalens_performance(alphalens_data, self.base_output_dir)
