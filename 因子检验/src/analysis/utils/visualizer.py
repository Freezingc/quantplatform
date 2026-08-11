"""
可视化和结果输出模块
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Any
import logging
from IPython.display import display

from ...config import VISUALIZATION_CONFIG, PATH_CONFIG

logger = logging.getLogger(__name__)



class Visualizer:
    """可视化和结果输出器"""
    
    def __init__(self, factor_name: str):
        self.factor_name = factor_name
        self.config = VISUALIZATION_CONFIG
        self.base_output_dir = self._create_output_directory()
        
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = self.config['matplotlib_font']
        plt.rcParams['axes.unicode_minus'] = False

        colors = plt.cm.tab20.colors  # 20种颜色的列表
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)


    def _create_output_directory(self) -> str:
        """创建输出目录"""
        base_dir = os.path.join(PATH_CONFIG['visualization_folder'], self.factor_name)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        return base_dir
    

    def _save_performance_metrics(self, performance_metrics: Dict[str, Dict[str, float]]):
        """保存整体绩效指标"""
        # 转换为DataFrame格式
        perf_df = pd.DataFrame.from_dict(performance_metrics, orient='index').reset_index(names="portfolio_type")
        
        # 选择关键指标
        key_columns = [
            'portfolio_type', 'ic', 'icir', 'turnover', 'return_ratio', 
            'annual_return_ratio','ex_return_ratio', 'ex_annual_return_ratio', 
            'sharp_ratio', 'return_volatility', 'information_ratio', 
            'max_drawdown', 'win_percent', 'ic_252', 'ret_252'
        ]
        
        # 过滤存在的列
        available_columns = [col for col in key_columns if col in perf_df.columns]
        perf_df = perf_df[available_columns].round(4)
        
        # 保存到CSV
        perf_df.to_csv(f'{self.base_output_dir}/整体绩效指标.csv', index=False)
        
        # 显示表格
        print("整体绩效指标:")
        display(perf_df)


    def _save_yearly_performance(self, yearly_performance: pd.DataFrame):
        """保存年度绩效指标"""
        yearly_performance = yearly_performance.reset_index(names='year')

        # 选择关键列
        key_columns = [
            'year', 'portfolio_type', 'ic', 'return_ratio', 'annual_return_ratio', 
            'ex_return_ratio', 'ex_annual_return_ratio', 'sharp_ratio', 
            'return_volatility', 'max_drawdown', 'win_percent', 'trading_days'
        ]
        
        available_columns = [col for col in key_columns if col in yearly_performance.columns]
        yearly_df = yearly_performance[available_columns].round(4)
        
        # 保存到CSV
        yearly_df.to_csv(f'{self.base_output_dir}/年度绩效指标.csv', index=False)
        
        # 显示表格
        print("年度绩效指标:")
        display(yearly_df)


    def _plot_cumulative_returns(self, cumulative_returns: pd.DataFrame):
        """绘制分组累积收益曲线"""
        fig = plt.figure(figsize=self.config['figure_size'])
        cumulative_returns = cumulative_returns.set_index('date')
        
        # 绘制所有分组的累积收益率
        for col in cumulative_returns.columns:
            if col not in ['bm']:  # 排除基准
                plt.plot(cumulative_returns.index, cumulative_returns[col], 
                        label=f'组合 {col}', linewidth=2)
        
        # 绘制基准
        if 'bm' in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns['bm'], 
                    label='基准', linewidth=2, linestyle='--', color='black')
        
        plt.title(f'{self.factor_name} - 分组累积收益率曲线', fontsize=16)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('累积收益率', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(f'{self.base_output_dir}/分组累积收益率曲线.{self.config["save_format"]}', 
                   dpi=self.config['dpi'])
        # plt.show()
        plt.close(fig)
        
        # 保存数据
        cumulative_returns.to_csv(f'{self.base_output_dir}/分组累积收益率数据.csv')


    def _plot_ic_analysis(self, ic_data: pd.DataFrame):
        """绘制IC分析图表（分子图，每个持有期一个子图）"""
        holdings = ic_data.columns.levels[0]
        n = len(holdings)

        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(12, 4 * n), sharex=True)
        fig.suptitle(f'{self.factor_name} - 多持有期 IC分析', fontsize=16, y=0.99)

        if n == 1:
            axes = [axes]  # 兼容只有一个子图的情况

        ic_data = ic_data.reset_index(names="date")

        for i, holding in enumerate(holdings):
            ax1 = axes[i]
            df = ic_data[['date']].copy()
            df['g_ic'] = ic_data[(holding, 'g_ic')]
            df['ic_roll_ma'] = ic_data[(holding, 'ic_roll_ma')]
            df['ic_cumsum'] = ic_data[(holding, 'ic_cumsum')]

            # 左轴
            ax1.bar(df['date'], df['g_ic'], alpha=0.6, label="原始IC", color="royalblue")
            ax1.plot(df['date'], df['ic_roll_ma'], color="forestgreen", label="22日均值")
            ax1.set_ylabel("IC值")
            ax1.grid(True, alpha=0.3)

            # 右轴（累计IC）
            ax2 = ax1.twinx()
            ax2.plot(df['date'], df['ic_cumsum'], color="orange", label="IC累积")
            ax2.set_ylabel("累积IC")

            # 合并图例
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc="upper left")

            ax1.set_title(f"{holding}日持有期")

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 留出标题空间
        plt.savefig(
            f"{self.base_output_dir}/IC分析图.{self.config['save_format']}",
            dpi=self.config["dpi"]
        )
        # plt.show()
        plt.close(fig)

        # 保存数据（多持有期合并在一个 CSV）
        ic_data.to_csv(f"{self.base_output_dir}/IC分析数据.csv", index=False)


    def generate_full_comprehensive_report(self, analysis_results: Dict[str, Any]):
        """生成完整综合分析报告"""
        logger.info(f"开始生成因子 {self.factor_name} 的完整可视化报告...")
        
        try:
            # 1. 保存整体绩效指标
            self._save_performance_metrics(analysis_results['performance_metrics'])
            
            # 2. 保存年度绩效指标
            self._save_yearly_performance(analysis_results['yearly_performance'])
            
            # 3. 绘制分组累积收益曲线
            self._plot_cumulative_returns(analysis_results['cumulative_returns'])
            
            logger.info(f"因子 {self.factor_name} 完整可视化报告生成完成")
            
        except Exception as e:
            logger.error(f"生成完整可视化报告失败: {str(e)}")
            raise


    def generate_ic_comprehensive_report(self, analysis_results: Dict[str, Any]):
        """生成ic综合分析报告"""
        logger.info(f"开始生成因子 {self.factor_name} 的ic可视化报告...")

        try:
            # 绘制IC分析图表
            self._plot_ic_analysis(analysis_results['ic_metrics'])
            
        except Exception as e:
            logger.error(f"生成ic可视化报告失败: {str(e)}")
            raise
