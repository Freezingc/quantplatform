"""
批量处理因子
"""

import pandas as pd
import numpy as np
import traceback
import logging
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from IPython.display import display

from .infra.data_preprocessor import batch_load_and_transform_factors
from .analysis.factor_analyzer import FactorAnalyzer
from .config import PATH_CONFIG

logger = logging.getLogger(__name__)


def run_batch_factor_analysis(factor_names: List[str], factor_dataframes: List[pd.DataFrame], params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """批量分析因子，支持分层计算：IC分析支持多持有期，完整绩效分析仅1D。"""

    logger.info("开始批量因子分析...")

    # 收集所有因子的汇总结果
    all_ic_summaries = []      # 多持有期IC汇总
    all_full_summaries = []    # 1D完整效果汇总

    # 使用分析器
    with tqdm(
        zip(factor_names, factor_dataframes), 
        total=len(factor_names), 
        desc=f"分析进度"
    ) as pbar:
        for factor_name, factor_df in pbar:
            try:
                pbar.set_postfix({"分析因子": factor_name}, refresh=True)
                
                # 使用因子分析器进行分析
                analyzer = FactorAnalyzer(factor_df, factor_name, params)
                analysis_results = analyzer.run_analysis()
            
                # 生成可视化
                analyzer.generate_visualizations(analysis_results)

                # 生成汇总表
                summaries = analyzer.generate_summaries(analysis_results)

                # 收集汇总结果
                all_ic_summaries.extend(summaries['ic_summary'])
                all_full_summaries.extend(summaries['full_summary'])

            except Exception as e:
                error_info = traceback.format_exc()
                logger.error(f"因子 {factor_name} 分析失败: {str(e)}")
                logger.error(f"错误堆栈:\n{error_info}")
                continue

    logger.info(f"批量分析完成，成功分析 {len(set([s['factor_name'] for s in all_ic_summaries]))} 个因子")

    # 创建两个汇总表
    results = {}

    if all_ic_summaries:
        ic_summary_df = pd.DataFrame(all_ic_summaries)
        ic_summary_df = ic_summary_df.set_index(["factor_name", "持有期"])
        display(ic_summary_df)
        results['ic_summary'] = ic_summary_df.reset_index()

        # 保存多持有期IC汇总表到Excel
        with pd.ExcelWriter('ic_summary.xlsx', engine="openpyxl") as writer:
            ic_summary_df.to_excel(writer, sheet_name="多持有期IC汇总")
            worksheet = writer.sheets["多持有期IC汇总"]
            worksheet.freeze_panes = "A2"

    if all_full_summaries:
        full_summary_df = pd.DataFrame(all_full_summaries)
        full_summary_df = full_summary_df.set_index("factor_name")
        display(full_summary_df)
        results['full_summary'] = full_summary_df.reset_index()

        # 保存1D完整效果汇总表到Excel
        with pd.ExcelWriter('full_summary.xlsx', engine="openpyxl") as writer:
            full_summary_df.to_excel(writer, sheet_name="1D完整效果汇总")
            worksheet = writer.sheets["1D完整效果汇总"]
            worksheet.freeze_panes = "A2"

    return results


def run_factor_analysis_from_files(input_folder: str = None, params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """批处理文件"""
    input_folder = input_folder or PATH_CONFIG['input_folder']
    output_folder = PATH_CONFIG['visualization_folder']
    
    logger.info(f"开始批量处理因子文件，输入文件夹: {input_folder}")
    
    # 1. 批量读取和预处理因子数据
    factor_names, factor_dataframes = batch_load_and_transform_factors(input_folder, output_folder)
    
    if not factor_names:
        logger.warning("未找到有效的因子数据文件")
        return {}
    
    logger.info(f"发现 {len(factor_names)} 个待分析因子")
    
    # 2. 批量分析因子
    summary_results = run_batch_factor_analysis(factor_names, factor_dataframes, params)
    
    return summary_results
