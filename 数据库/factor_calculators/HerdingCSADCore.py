"""基于相关股票群组的 CSAD 羊群效应因子共享计算内核。"""

from __future__ import annotations

import numpy as np
import pandas as pd


RATIO_20 = "ratio_20"
RATIO_120 = "ratio_120"
RATIO_20_TO_120 = "ratio_20_to_120"
CSAD_STD_21 = "csad_std_21"
OUTPUT_NAMES = (RATIO_20, RATIO_120, RATIO_20_TO_120, CSAD_STD_21)


def _sample_std(values: np.ndarray) -> np.ndarray:
    """按列计算忽略 NaN 的样本标准差，少于两个观测时返回 NaN。"""
    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    total = np.where(valid, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        count,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=count > 0,
    )
    centered = np.where(valid, values - mean, 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        count - 1,
        out=np.full(values.shape[1], np.nan, dtype=np.float64),
        where=count > 1,
    )
    return np.sqrt(variance)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """执行有限值除法；零分母和非有限值统一返回 NaN。"""
    result = np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (np.abs(denominator) > np.finfo(np.float64).eps)
    )
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def compute_herding_csad_factors(
    returns: pd.DataFrame,
    *,
    long_window: int = 120,
    short_window: int = 20,
    csad_std_window: int = 21,
    group_size: int = 10,
    min_observations: int = 20,
    progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """计算中信建投市场羊群效应研究中的四个日频因子。

    每个交易日使用截至当日的 ``long_window`` 日收益率：

    1. 保留至少有 ``min_observations`` 个非缺失且非零收益的股票；
    2. 对每只股票选取收益相关性最高的 ``group_size - 1`` 只股票；
    3. 以该股票为中心，计算群组横截面绝对偏差 CSAD；
    4. 输出短、长周期 CSAD/收益波动率、二者之比和 CSAD 波动率。

    参数化窗口主要用于单元测试；生产口径为 120/20/21/10/20。
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns 必须是 pandas.DataFrame")
    if long_window < 2:
        raise ValueError("long_window 必须至少为 2")
    if not 2 <= short_window <= long_window:
        raise ValueError("short_window 必须在 [2, long_window] 内")
    if not 2 <= csad_std_window <= long_window:
        raise ValueError("csad_std_window 必须在 [2, long_window] 内")
    if group_size < 2:
        raise ValueError("group_size 必须至少为 2")
    if not 2 <= min_observations <= long_window:
        raise ValueError("min_observations 必须在 [2, long_window] 内")

    returns = returns.sort_index().astype(np.float64, copy=False)
    index = returns.index
    columns = returns.columns
    n_dates, n_stocks = returns.shape
    output_arrays = {
        name: np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
        for name in OUTPUT_NAMES
    }
    if n_dates < long_window or n_stocks < group_size:
        return {
            name: pd.DataFrame(values, index=index, columns=columns)
            for name, values in output_arrays.items()
        }

    values = returns.to_numpy(copy=False)
    peer_count = group_size - 1
    total_windows = n_dates - long_window + 1

    for window_number, end_pos in enumerate(
        range(long_window - 1, n_dates), start=1
    ):
        window_values = values[end_pos - long_window + 1 : end_pos + 1]
        finite = np.isfinite(window_values)
        eligible = (
            (finite.sum(axis=0) >= min_observations)
            & ((finite & (window_values != 0.0)).sum(axis=0) >= min_observations)
        )
        eligible_pos = np.flatnonzero(eligible)
        if eligible_pos.size < group_size:
            continue

        eligible_values = window_values[:, eligible_pos]
        corr = pd.DataFrame(eligible_values).corr().to_numpy(copy=True)
        finite_corr = np.isfinite(corr)
        np.fill_diagonal(finite_corr, False)

        # 目标股票必须至少有 group_size-1 个相关性可计算的同伴。
        valid_target = finite_corr.sum(axis=0) >= peer_count
        if not valid_target.any():
            continue

        scores = np.where(finite_corr, corr, -np.inf)
        kth = scores.shape[0] - peer_count
        peer_indices = np.argpartition(scores, kth=kth, axis=0)[-peer_count:]

        # 针对所有中心股票同时计算 CSAD。中心股票自身也属于10只股票的
        # 群组，因此其绝对偏差0计入横截面均值，与文章代码保持一致。
        target_values = eligible_values
        abs_sum = np.zeros_like(target_values)
        observation_count = np.zeros_like(target_values, dtype=np.int16)

        target_finite = np.isfinite(target_values)
        abs_sum += np.where(target_finite, 0.0, 0.0)
        observation_count += target_finite

        for peer_row in peer_indices:
            peer_values = eligible_values[:, peer_row]
            pair_valid = target_finite & np.isfinite(peer_values)
            abs_sum += np.where(pair_valid, np.abs(peer_values - target_values), 0.0)
            observation_count += pair_valid

        csad = np.divide(
            abs_sum,
            observation_count,
            out=np.full_like(abs_sum, np.nan),
            where=observation_count > 0,
        )

        csad_std_short = _sample_std(csad[-short_window:])
        return_std_short = _sample_std(target_values[-short_window:])
        ratio_short = _safe_divide(csad_std_short, return_std_short)

        csad_std_long = _sample_std(csad)
        return_std_long = _sample_std(target_values)
        ratio_long = _safe_divide(csad_std_long, return_std_long)
        ratio_short_to_long = _safe_divide(ratio_short, ratio_long)
        csad_std = _sample_std(csad[-csad_std_window:])

        destination = eligible_pos[valid_target]
        output_arrays[RATIO_20][end_pos, destination] = ratio_short[valid_target]
        output_arrays[RATIO_120][end_pos, destination] = ratio_long[valid_target]
        output_arrays[RATIO_20_TO_120][end_pos, destination] = (
            ratio_short_to_long[valid_target]
        )
        output_arrays[CSAD_STD_21][end_pos, destination] = csad_std[valid_target]

        if progress and (window_number % 20 == 0 or window_number == total_windows):
            print(
                "HerdingCSAD 进度: "
                f"{window_number}/{total_windows}, 当日有效股票 {valid_target.sum()}"
            )

    return {
        name: pd.DataFrame(values, index=index, columns=columns)
        for name, values in output_arrays.items()
    }

