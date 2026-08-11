"""
绩效计算模块
绩效计算需要不同数据，因此注释写清楚各部分的输入输出
"""

import pandas as pd
import numpy as np
import empyrical
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)



class PerformanceCalculator:
    """绩效指标计算器"""
    
    def __init__(self):
        self.risk_free_rate = 0.035  # 无风险利率取 0.035
    

    def calculate_group_returns(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算分组收益率

        参数:
        ----------
        group_data: DataFrame
            包含分组信息和每日收益率，必须包含列：
            - 'instrument'(名称)
            - 'date'(日期)
            - 'group'(分组编号)
            - 'daily_ret'(个股日收益率)

        返回
        -------
        DataFrame
            包含各组平均收益率和多空组合收益率的表格
            示例:
                date       |   0   |   1   | ... | ls
                -----------|-------|-------|-----|-------
                2023-01-01 | 0.012 | 0.023 | ... | 0.011
                2023-01-02 |-0.005 | 0.010 | ... | 0.015

        """
        # 计算分组总数
        group_num = len(group_data["group"].unique())

        # 计算各组平均收益率
        group_data = (
            group_data[['date', 'group', 'daily_ret']].groupby(['date', 'group'])
            .apply(lambda x: np.nanmean(x)).reset_index()
        )
        group_data.rename(columns={0: 'g_ret'}, inplace=True)
        
        # 宽表形式
        group_pivot_data = group_data.pivot(index='date', values='g_ret', columns='group')
        
        # 计算多空组合收益率
        group_pivot_data['ls'] = (
            group_pivot_data[str(group_num - 1)] - group_pivot_data['0']
        )
        
        return group_pivot_data.reset_index()
    

    def calculate_cumulative_returns(
            self, group_returns: pd.DataFrame, 
            benchmark_data: Optional[pd.DataFrame] = None
        ) -> pd.DataFrame:
        """
        计算分组累积收益率（含可选基准）

        参数:
        ----------
        group_returns : DataFrame
            包含分组收益率的表格，必须包含列：
            - 'date'（日期）
            - 其他列为各分组编号对应的收益率（如'0','1','ls'等表示不同分组）
        
        benchmark_data : Optional[DataFrame], 默认None
            可选基准收益率数据框，需包含列：
            - 'date'（日期）
            - 'benchmark_ret'（基准日收益率）

        返回:
        -------
        DataFrame
            包含累积收益率的数据框，结构示例：
                date       |   0   |   1   | ... |  bm
                -----------|-------|-------|-----|-------
                2023-01-01 | 0.012 | 0.023 | ... | 0.015
                2023-01-02 | 0.007 | 0.033 | ... | 0.018
            说明：
            1. 各分组列为简单累加收益率
            2. 当提供基准数据时，包含基准累积收益率列（'bm'）
            3. 结果保留4位小数
        """

        group_returns = group_returns.set_index('date')
        
        if benchmark_data is not None:
            # 添加基准收益率
            benchmark_data = benchmark_data.set_index('date')
            group_returns['bm'] = benchmark_data['benchmark_ret']
        
        # 计算累积收益率
        group_cumulative_returns = group_returns.cumsum()
        return group_cumulative_returns.round(4).reset_index()
    

    def _get_default_metrics(self) -> Dict[str, float]:
        """返回默认的绩效指标"""
        return {
            'return_ratio': np.nan,
            'annual_return_ratio': np.nan,
            'ex_return_ratio': np.nan,
            'ex_annual_return_ratio': np.nan,
            'sharp_ratio': np.nan,
            'return_volatility': np.nan,
            'information_ratio': np.nan,
            'max_drawdown': np.nan,
            'win_percent': np.nan,
            'trading_days': 0,
            'ret_3': np.nan,
            'ret_10': np.nan,
            'ret_21': np.nan,
            'ret_63': np.nan,
            'ret_126': np.nan,
            'ret_252': np.nan
        }
    

    def _calculate_performance_metrics_by_series(self, returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        计算单个组合的绩效指标
        
        参数:
        ----------
        returns : pd.Series
            组合日收益率序列，索引为日期(datetime格式)

        benchmark_returns : pd.Series
            基准日收益率序列，索引需与returns保持一致


        返回:
        -------
        Dict[str, float]
            包含16项绩效指标的字典，结构为：
            {
                # 绝对收益指标
                'return_ratio': 总收益率,
                'annual_return_ratio': 年化收益率(按242个交易日计算),
                
                # 相对收益指标  
                'ex_return_ratio': 超额总收益(相对基准),
                'ex_annual_return_ratio': 年化超额收益,
                
                # 风险调整指标
                'sharp_ratio': 夏普比率(使用配置的无风险利率),
                'return_volatility': 年化波动率,
                'information_ratio': 信息比率,
                'max_drawdown': 最大回撤,
                'win_percent': 胜率,
                
                # 交易特征
                'trading_days': 有效交易日数,
                
                # 滚动周期收益
                'ret_3': 最近3日累计收益,
                'ret_10': 最近10日累计收益,
                'ret_21': 最近21日(月频)累计收益,
                'ret_63': 最近63日(季度)累计收益,
                'ret_126': 最近126日(半年)累计收益,
                'ret_252': 最近252日(年)累计收益
            }
        """
        returns = returns.dropna()
        benchmark_returns = benchmark_returns.dropna()
        
        # 对齐数据
        common_dates = returns.index.intersection(benchmark_returns.index)
        returns = returns.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        
        if len(returns) == 0:
            logger.warning("收益率数据为空，返回默认值")
            return self._get_default_metrics()
        
        # 基础收益指标
        total_return = returns.sum()
        annual_return = returns.sum() * 242 / len(returns)
        
        # 超额收益指标
        excess_returns = returns - benchmark_returns
        excess_total_return = excess_returns.sum()
        excess_annual_return = excess_returns.sum() * 242 / len(excess_returns)
        
        # 风险调整指标
        try:
            sharpe_ratio = np.float64(empyrical.sharpe_ratio(returns, self.risk_free_rate / 242))
            volatility = np.float64(empyrical.annual_volatility(returns))
            max_drawdown = np.float64(empyrical.max_drawdown(returns))
        except:
            sharpe_ratio = np.nan
            volatility = np.nan
            max_drawdown = np.nan
        
        information_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else np.nan
        win_rate = np.float64(len(returns[returns > 0]) / len(returns))
        
        # 不同期间收益率
        returns_filled = returns.fillna(0)
        metrics = {
            'return_ratio': total_return.round(4),
            'annual_return_ratio': annual_return.round(4),
            'ex_return_ratio': excess_total_return.round(4),
            'ex_annual_return_ratio': excess_annual_return.round(4),
            'sharp_ratio': sharpe_ratio.round(4),
            'return_volatility': volatility.round(4),
            'information_ratio': information_ratio.round(4),
            'max_drawdown': max_drawdown.round(4),
            'win_percent': win_rate.round(4),
            'trading_days': len(returns),
            'ret_3': returns_filled.rolling(3).sum().iloc[-1].round(4),
            'ret_10': returns_filled.rolling(10).sum().iloc[-1].round(4),
            'ret_21': returns_filled.rolling(21).sum().iloc[-1].round(4),
            'ret_63': returns_filled.rolling(63).sum().iloc[-1].round(4),
            'ret_126': returns_filled.rolling(126).sum().iloc[-1].round(4),
            'ret_252': returns_filled.rolling(252).sum().iloc[-1].round(4)
        }
        
        return metrics


    def calculate_performance_metrics(
        self, group_returns: pd.DataFrame, 
        benchmark_data: pd.DataFrame, 
        portfolio_type: str
    ) -> Dict[str, float]:
        """
        计算指定投资组合的绩效指标
        
        参数:
        ----------
        group_returns : DataFrame
            包含分组收益率的表格，必须包含列：
            - 'date'（日期）
            - 其他列为各分组编号对应的收益率（如'0','1','ls'等表示不同分组）

        benchmark_data : DataFrame
            基准收益率数据框，需包含列：
            - 'date'（日期）
            - 'benchmark_ret'（基准日收益率）

        portfolio_type : str
            投资组合类型，可选值：
            - 'long' : 多头组合（选择最高分组）
            - 'short' : 空头组合（选择最低分组）
            - 'long_short' : 多空组合（同时选择最高和最低分组）

        返回:
        -------
        Dict[str, float]
            包含16项绩效指标的字典，结构为：
            {
                # 绝对收益指标
                'return_ratio': 总收益率,
                'annual_return_ratio': 年化收益率(按242个交易日计算),
                
                # 相对收益指标  
                'ex_return_ratio': 超额总收益(相对基准),
                'ex_annual_return_ratio': 年化超额收益,
                
                # 风险调整指标
                'sharp_ratio': 夏普比率(使用配置的无风险利率),
                'return_volatility': 年化波动率,
                'information_ratio': 信息比率,
                'max_drawdown': 最大回撤,
                'win_percent': 胜率,
                
                # 交易特征
                'trading_days': 有效交易日数,
                
                # 滚动周期收益
                'ret_3': 最近3日累计收益,
                'ret_10': 最近10日累计收益,
                'ret_21': 最近21日(月频)累计收益,
                'ret_63': 最近63日(季度)累计收益,
                'ret_126': 最近126日(半年)累计收益,
                'ret_252': 最近252日(年)累计收益
            }
        """
        assert portfolio_type in ['long', 'short', 'long_short'], "投资组合类型错误！"
        group_returns = group_returns.set_index('date')
        benchmark_data = benchmark_data.set_index('date')

        # 根据组合类型选择分组
        series = None
        if portfolio_type == 'long':
            series = group_returns[group_returns.columns[-2]]
        elif portfolio_type == 'short':
            series = group_returns['0']
        elif portfolio_type == 'long_short':
            series = group_returns['ls']

        return self._calculate_performance_metrics_by_series(series, benchmark_data['benchmark_ret'])
    

    def _calc_ic(self, df: pd.DataFrame) -> float:
        """计算因子与收益的Spearman相关系数"""
        return df['daily_ret'].corr(df['factor'], method='spearman')


    def calculate_daily_ic(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算全组合的每日IC指标
        
        参数:
        ----------
        group_data : DataFrame
            包含因子值和收益率的分组数据，必须包含列：
            - 'date' (日期)
            - 'daily_ret' (个股日收益率)
            - 'factor' (因子值)
            示例数据格式：
                date       | daily_ret | factor | group
                -----------|-----------|--------|-------
                2023-01-01 |   0.012   |  0.85  |   0
                2023-01-01 |  -0.005   | -0.23  |   1

        返回:
        -------
        DataFrame
            包含每日IC指标的数据框，结构为：
            1. g_ic：当日横截面IC（因子值与收益率的秩相关系数）
            2. ic_cumsum：IC累计值
            3. ic_roll_ma：22日滚动平均IC（约1个月）
            4. date：日期（行索引）
        """
        ic_data = group_data[['date', 'daily_ret', 'factor']].groupby('date').apply(
            lambda x: self._calc_ic(x), include_groups=False
        ).reset_index()

        ic_data.rename(columns={0: 'g_ic'}, inplace=True)
        ic_data['ic_cumsum'] = ic_data['g_ic'].cumsum()
        ic_data['ic_roll_ma'] = ic_data['g_ic'].rolling(22).mean()
        # return ic_data.round(4).dropna().set_index("date")
        return ic_data.round(4).set_index("date")
    

    def calculate_portfolio_ic_metrics(self, group_data: pd.DataFrame, portfolio_type: str) -> Dict[str, float]:
        """
        计算指定投资组合的IC统计指标
        
        参数:
        ----------
        group_data : DataFrame
            包含分组因子数据的数据框，必须包含列：
            - 'date' (日期)
            - 'group' (分组编号，字符串格式)
            - 'daily_ret' (个股日收益率)
            - 'factor' (因子值)
            示例格式：
                date       | group | daily_ret | factor 
                -----------|-------|-----------|--------
                2023-01-01 |  '0'  |  -0.005   | -0.23  
                2023-01-01 |  '5'  |   0.012   |  0.85  

        portfolio_type : str
            投资组合类型，可选值：
            - 'long' : 多头组合（选择最高分组）
            - 'short' : 空头组合（选择最低分组）
            - 'long_short' : 多空组合（同时选择最高和最低分组）

        返回:
        -------
        Dict[str, float]
            包含IC统计指标的字典，结构为：
            {
                'ic': 历史IC均值,
                'icir': IC信息比率(IC均值/IC标准差),
                'ic_3': 最近3日IC均值,
                'ic_10': 最近10日IC均值,
                'ic_21': 最近21日(月频)IC均值,
                'ic_63': 最近63日(季度)IC均值,
                'ic_126': 最近126日(半年)IC均值,
                'ic_252': 最近252日(年)IC均值
            }
        """
        group_num = len(group_data['group'].unique())
        select_groups = []
        
        assert portfolio_type in ['long', 'short', 'long_short'], "投资组合类型错误！"

        # 根据组合类型选择分组
        if portfolio_type == 'long':
            select_groups.append(str(group_num-1))  # 选表现最好组
        elif portfolio_type == 'short':
            select_groups.append('0')  # 选表现最差组
        elif portfolio_type == 'long_short':
            select_groups.extend(['0', str(group_num-1)])
        
        # 计算选定组合的IC系列
        portfolio_data = group_data[group_data['group'].isin(select_groups)]
        ic_series = portfolio_data.groupby('date').apply(
            lambda x: self._calc_ic(x), include_groups=False
        )
        # 计算统计指标
        ic_mean = np.nanmean(ic_series)
        icir = ic_mean / np.nanstd(ic_series) if np.nanstd(ic_series) != 0 else np.nan
        
        return {
            'ic': ic_mean.round(4),
            'icir': icir.round(4),
            'ic_3': ic_series.rolling(3).mean().iloc[-1].round(4),
            'ic_10': ic_series.rolling(10).mean().iloc[-1].round(4),
            'ic_21': ic_series.rolling(21).mean().iloc[-1].round(4),
            'ic_63': ic_series.rolling(63).mean().iloc[-1].round(4),
            'ic_126': ic_series.rolling(126).mean().iloc[-1].round(4),
            'ic_252': ic_series.rolling(252).mean().iloc[-1].round(4)
        }
    

    def calculate_turnover_metrics(self, group_data: pd.DataFrame, portfolio_type: str) -> Dict[str, float]:
        """
        计算换手率
        
        参数:
        ----------
        group_data : DataFrame
            包含分组持仓数据的数据框，必须包含列：
            - 'date' (日期)
            - 'group' (分组编号，字符串格式)
            - 'instrument' (股票代码/标识符)

        portfolio_type : str
            投资组合类型，可选值：
            - 'long' : 多头组合（选择最高分组）
            - 'short' : 空头组合（选择最低分组）
            - 'long_short' : 多空组合（同时选择最高和最低分组）

        返回:
        -------
        Dict[str, float]
            包含换手率指标的字典，结构为：
            {
                'turnover': 组合平均换手率（多头/空头取单组值，多空组合取两组平均值）
            }
        """
        def calc_turnover(df):
            def count_repeat(s):
                if s.name > 0:  # 行索引
                    current_stocks = set(s['instrument'])
                    previous_stocks = set(df.loc[s.name - 1, 'instrument'])
                    return len(current_stocks.intersection(previous_stocks))
                else:
                    return np.nan
            
            df = df.groupby('date').agg({'instrument': list}).reset_index()
            df['instrument_count'] = df['instrument'].apply(len)
            df['repeat_count'] = df.apply(count_repeat, axis=1)
            df['turnover'] = 1 - df['repeat_count'] / df['instrument_count']
            return np.nanmean(df['turnover'])
        

        group_num = len(group_data['group'].unique())
        select_groups = []
        
        assert portfolio_type in ['long', 'short', 'long_short'], "投资组合类型错误！"
        if portfolio_type == 'long':
            select_groups.append(str(group_num-1))  # 选表现最好组
        elif portfolio_type == 'short':
            select_groups.append('0')  # 选表现最差组
        elif portfolio_type == 'long_short':
            select_groups.extend(['0', str(group_num-1)])

        turnover_value = 0
        for group_id in select_groups:
            filtered_data = group_data[group_data['group'] == group_id]
            turnover_value += calc_turnover(filtered_data)

        return {'turnover': round(turnover_value / len(select_groups), 4)}


    def _calculate_yearly_performance_by_series(self, returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
        """
        计算单个组合的年度绩效指标
        
        输入输出参照 func: _calculate_performance_metrics_by_series
            - 输出是"上述 func 输出"的 datafame 形式，以年份为行索引
            - 输出取消了"滚动周期收益"部分
        """
        def remove_keys_with_prefix(d, prefix):
            keys_to_remove = [k for k in d if k.startswith(prefix)]
            for k in keys_to_remove:
                d.pop(k)

        # 在某一年份下计算 returns 和 benchmark_returns 的绩效
        yearly_metrics = []
        for year, year_returns in returns.groupby(returns.index.year):
            year_benchmark = benchmark_returns[benchmark_returns.index.year == year]
            metrics = self._calculate_performance_metrics_by_series(year_returns, year_benchmark)
            
            remove_keys_with_prefix(metrics, "ret_")
            metrics['year'] = year  # 添加年份列
            yearly_metrics.append(metrics)
        
        return pd.DataFrame(yearly_metrics).set_index('year')
    

    def calculate_yearly_performance(
        self, group_returns: pd.DataFrame, 
        benchmark_data: pd.DataFrame, 
        portfolio_type: str = 'long'
    ) -> pd.DataFrame:
        """
        计算指定投资组合的年度绩效指标
        
        输入输出参照 func: calculate_performance_metrics
            - 输出是"上述 func 输出"的 datafame 形式，以年份为行索引
            - 输出取消了"滚动周期收益"部分
        """
        assert portfolio_type in ['long', 'short', 'long_short'], "投资组合类型错误！"
        group_returns = group_returns.set_index('date')
        benchmark_data = benchmark_data.set_index('date')

        # 根据组合类型选择分组
        series = None
        if portfolio_type == 'long':
            series = group_returns[group_returns.columns[-2]]
        elif portfolio_type == 'short':
            series = group_returns['0']
        elif portfolio_type == 'long_short':
            series = group_returns['ls']

        return self._calculate_yearly_performance_by_series(series, benchmark_data['benchmark_ret'])
        

    def calculate_yearly_ic(self, group_data: pd.DataFrame, portfolio_type: str = 'long') -> pd.Series:
        """
        计算指定投资组合的年度IC（信息系数）序列

        参数:
        ----------
        group_data : DataFrame
            包含分组因子数据的数据框，必须包含列：
            - 'date' (日期，需为datetime类型)
            - 'group' (分组编号，字符串格式)
            - 'daily_ret' (个股日收益率)
            - 'factor' (因子值)
            示例格式：
                date       | group | daily_ret | factor
                -----------|-------|-----------|--------
                2023-01-01 |  '0'  |  -0.005   | -0.23
                2023-01-01 |  '5'  |   0.012   |  0.85

        portfolio_type : str, 默认 'long'
            投资组合类型，可选值：
            - 'long' : 多头组合（选择最高分组）
            - 'short' : 空头组合（选择最低分组）
            - 'long_short' : 多空组合（同时选择最高和最低分组）

        返回:
        -------
        pd.Series
            按年份索引的IC均值序列，保留4位小数
        """
        group_num = len(group_data['group'].unique())
        select_groups = []
        
        assert portfolio_type in ['long', 'short', 'long_short'], "投资组合类型错误！"

        # 根据组合类型选择分组
        if portfolio_type == 'long':
            select_groups.append(str(group_num-1))  # 选表现最好组
        elif portfolio_type == 'short':
            select_groups.append('0')  # 选表现最差组
        elif portfolio_type == 'long_short':
            select_groups.extend(['0', str(group_num-1)])
        
        # 计算选定组合的IC系列
        portfolio_data = group_data[group_data['group'].isin(select_groups)]
        ic_series = portfolio_data.groupby('date').apply(
            lambda x: self._calc_ic(x), include_groups=False
        )
        ic_data = ic_series.reset_index().rename(columns={0: 'g_ic'}).dropna()
        ic_data['year'] = ic_data['date'].dt.year

        yearly_ic = ic_data.groupby('year').apply(lambda x: np.nanmean(x['g_ic']), include_groups=False)
        return yearly_ic.round(4)


        
