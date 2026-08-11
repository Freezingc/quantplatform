import os
from typing import List
import numpy as np
import pandas as pd
import alphalens as al
import matplotlib.pyplot as plt
import logging

from ...config import VISUALIZATION_CONFIG

logger = logging.getLogger(__name__)


def get_alphalens_data(group_data_all: List[pd.DataFrame], horizons: List[int]) -> pd.DataFrame:
    """整理数据，使其符合alphalens输入"""
    argmax_horizon = np.argmax(horizons)
    alphalens_data = group_data_all[argmax_horizon].copy()
    alphalens_data = alphalens_data.rename(columns={'daily_ret': f'{horizons[argmax_horizon]}D'})

    for i, (horizon, group_data) in enumerate(zip(horizons, group_data_all)):
        if i == argmax_horizon:
            continue

        # 合并当前持有期的数据
        group_data = group_data[['date', 'instrument', 'daily_ret']].rename(columns={'daily_ret': f"{horizon}D"})
        alphalens_data = alphalens_data.merge(group_data, on=['date', 'instrument'], how='left')

    alphalens_data = alphalens_data.rename(
        columns={
            'instrument': 'asset', 
            'group': 'factor_quantile'
        }
    )

    columns = [f'{horizon}D' for horizon in horizons] + ['factor', 'factor_quantile']
    alphalens_data = alphalens_data.set_index(['date', 'asset'])[columns]
    
    alphalens_data['factor_quantile'] =  alphalens_data['factor_quantile'].astype(int)
    alphalens_data.index.levels[0].freq = al.utils.infer_trading_calendar(
        alphalens_data.index.levels[0], 
        alphalens_data.index.levels[0]
    )

    # alphalens_data.to_csv("alphalens_data_for_test.csv")
    return alphalens_data


def save_returns_performance(factor_data: pd.DataFrame, output_dir:str):
    """因子分组收益"""
    factor_returns = al.performance.factor_returns(factor_data)

    mean_quant_ret, std_quantile = al.performance.mean_return_by_quantile(
        factor_data,
        by_date=False,
    ) # 计算每个 Quantile 组的平均收益

    mean_quant_rateret = mean_quant_ret.apply(
        al.utils.rate_of_return, 
        axis=0, 
        base_period=mean_quant_ret.columns[0]
    ) # 将原始周期 转化为 单位周期下的收益率

    mean_quant_ret_bydate, std_quant_daily = al.performance.mean_return_by_quantile(
        factor_data,
        by_date=True,
    ) # 每日每组的平均收益

    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        al.utils.rate_of_return,
        axis=0,
        base_period=mean_quant_ret_bydate.columns[0],
    ) # 转化为单位周期

    compstd_quant_daily = std_quant_daily.apply(
        al.utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
    ) # 转化为1日的标准差

    alpha_beta = al.performance.factor_alpha_beta(
        factor_data, factor_returns
    )

    mean_ret_spread_quant, std_spread_quant = al.performance.compute_mean_returns_spread(
        mean_quant_rateret_bydate,
        factor_data["factor_quantile"].max(),
        factor_data["factor_quantile"].min(),
        std_err=compstd_quant_daily,
    )

    fr_cols = len(factor_returns.columns)
    vertical_sections = 2 + fr_cols * 3
    gf = al.tears.GridFigure(rows=vertical_sections, cols=1)

    returns_table = al.plotting.plot_returns_table(alpha_beta, mean_quant_rateret, mean_ret_spread_quant, True).round(4)
    returns_table.to_csv(f'{output_dir}/因子分组收益.csv')

    al.plotting.plot_quantile_returns_bar(
        mean_quant_rateret,
        by_group=False,
        ylim_percentiles=None,
        ax=gf.next_row(),
    )

    al.plotting.plot_quantile_returns_violin(
        mean_quant_rateret_bydate, ylim_percentiles=(1, 99), ax=gf.next_row()
    )

    # Compute cumulative returns from daily simple returns, if '1D'
    # returns are provided.
    if "1D" in factor_returns:
        title = (
            "Factor Weighted Long/Short Portfolio Cumulative Return (1D Period)"
        )

        al.plotting.plot_cumulative_returns(
            factor_returns["1D"], period="1D", title=title, ax=gf.next_row()
        )

        al.plotting.plot_cumulative_returns_by_quantile(
            mean_quant_ret_bydate["1D"], period="1D", ax=gf.next_row()
        )

    ax_mean_quantile_returns_spread_ts = [gf.next_row() for x in range(fr_cols)]
    al.plotting.plot_mean_quantile_returns_spread_time_series(
        mean_ret_spread_quant,
        std_err=std_spread_quant,
        bandwidth=0.5,
        ax=ax_mean_quantile_returns_spread_ts,
    )

    plt.savefig(f'{output_dir}/因子分组收益.{VISUALIZATION_CONFIG["save_format"]}', 
                dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches="tight")
    gf.close()


def save_information_performance(factor_data: pd.DataFrame, output_dir: str):
    """因子信息系数(IC)"""
    ic = al.performance.factor_information_coefficient(factor_data) # 计算每日ic
    information_table = al.plotting.plot_information_table(ic, return_df=True).round(4).T
    information_table.to_csv(f'{output_dir}/IC相关指标.csv')

    columns_wide = 2
    fr_cols = len(ic.columns)
    rows_when_wide = ((fr_cols - 1) // columns_wide) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = al.tears.GridFigure(rows=vertical_sections, cols=columns_wide)

    ax_ic_ts = [gf.next_row() for _ in range(fr_cols)]
    al.plotting.plot_ic_ts(ic, ax=ax_ic_ts)

    ax_ic_hqq = [gf.next_cell() for _ in range(fr_cols * 2)]
    al.plotting.plot_ic_hist(ic, ax=ax_ic_hqq[::2])
    al.plotting.plot_ic_qq(ic, ax=ax_ic_hqq[1::2])

    mean_monthly_ic = al.performance.mean_information_coefficient(
        factor_data,
        by_time="ME",
    )
    ax_monthly_ic_heatmap = [gf.next_cell() for x in range(fr_cols)]
    al.plotting.plot_monthly_ic_heatmap(mean_monthly_ic, ax=ax_monthly_ic_heatmap)

    plt.savefig(f'{output_dir}/IC相关指标.{VISUALIZATION_CONFIG["save_format"]}', 
                dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches="tight")
    gf.close()


def save_turnover_performance(factor_data: pd.DataFrame, output_dir: str):
    """因子换手率"""
    input_periods = al.utils.get_forward_returns_columns(
        factor_data.columns, require_exact_day_multiple=True
    ).to_numpy()
    turnover_periods = al.utils.timedelta_strings_to_integers(input_periods)

    # 计算不同周期下，不同分组的每日换手率
    quantile_factor = factor_data["factor_quantile"]

    quantile_turnover = {
        p: pd.concat(
            [
                al.performance.quantile_turnover(quantile_factor, q, p)
                for q in quantile_factor.sort_values().unique().tolist()
            ],
            axis=1,
        )
        for p in turnover_periods
    }

    # 计算每日因子秩自相关
    autocorrelation = pd.concat(
        [
            al.performance.factor_rank_autocorrelation(factor_data, period)
            for period in turnover_periods
        ],
        axis=1,
    )

    # 取每日平均后，获取表格
    turnover_table, corr_table = al.plotting.plot_turnover_table(autocorrelation, quantile_turnover, return_df=True)
    turnover_table.round(4).to_csv(f'{output_dir}/换手率.csv')
    corr_table.round(4).to_csv(f'{output_dir}/秩自相关.csv')
    
    # 画图
    fr_cols = len(turnover_periods)
    columns_wide = 1
    rows_when_wide = ((fr_cols - 1) // 1) + 1
    vertical_sections = fr_cols + 3 * rows_when_wide + 2 * fr_cols
    gf = al.tears.GridFigure(rows=vertical_sections, cols=columns_wide)

    for period in turnover_periods:
        if quantile_turnover[period].isnull().all().all():
            continue
        al.plotting.plot_top_bottom_quantile_turnover(
            quantile_turnover[period], period=period, ax=gf.next_row()
        )

    for period in autocorrelation:
        if autocorrelation[period].isnull().all():
            continue
        al.plotting.plot_factor_rank_auto_correlation(
            autocorrelation[period], period=period, ax=gf.next_row()
        )

    plt.savefig(f'{output_dir}/换手率 & 秩自相关.{VISUALIZATION_CONFIG["save_format"]}', 
                dpi=VISUALIZATION_CONFIG["dpi"], bbox_inches="tight")
    gf.close()



def save_alphalens_performance(factor_data: pd.DataFrame, base_output_dir: str):
    """
    生成 alphalens 的三个主要报告
    1. 因子分组收益;
    2. 因子信息系数(IC);
    3. 因子换手率
    """
    logger.info("开始计算 alphalens 指标...")

    # 将输出定位到 因子名/alphalens 目录
    output_dir = os.path.join(base_output_dir, "alphalens")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.rcParams['axes.prop_cycle'] = plt.rcParamsDefault['axes.prop_cycle']

    quantile_statistics_table = al.plotting.plot_quantile_statistics_table(factor_data, True).round(4)
    quantile_statistics_table.to_csv(f"{output_dir}/统计信息.csv")

    save_returns_performance(factor_data, output_dir=output_dir)
    save_information_performance(factor_data, output_dir=output_dir)
    save_turnover_performance(factor_data, output_dir=output_dir)

    logger.info("alphalens 指标计算完成")


if __name__ == "__main__":
    df = pd.read_csv("alphalens_data_for_test.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date", "asset"], inplace=True)
    df.index.levels[0].freq = al.utils.infer_trading_calendar(df.index.levels[0], df.index.levels[0])
    save_alphalens_performance(df, "tmpfolder")

    # D:/programs/anaconda3/envs/quant-env/python.exe -m src.analysis.alphalens_calculator