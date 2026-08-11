import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════
# 行业联合动量因子 — 共享工具（不包含任何 Calculator 类）
# ═══════════════════════════════════════════════════════════════════════

W_TOP = np.array([0.5, 0.59, 0.7, 0.84, 1.0], dtype=np.float64)
W_BOT = 2.0 ** (-np.arange(15, dtype=np.float64) / 14.0)


def compute_industry_returns(rtn, total_mv, ts_to_l1):
    """计算每个申万一级行业的市值加权日收益率矩阵 (dates × l1_code)"""
    l1_map = pd.Series(
        {c: ts_to_l1.get(c) for c in rtn.columns}, name="l1_code"
    ).replace({None: np.nan})

    rtn_stacked = rtn.stack()
    mv_stacked = total_mv.stack()
    l1_stacked = l1_map.reindex(rtn_stacked.index, level=1)

    valid = l1_stacked.notna() & rtn_stacked.notna() & (mv_stacked > 0)
    rtn_s = rtn_stacked[valid]
    mv_s = mv_stacked[valid]
    l1_s = l1_stacked[valid]

    dates = rtn_s.index.get_level_values(0)

    num = (rtn_s * mv_s).groupby([dates, l1_s]).sum()
    den = mv_s.groupby([dates, l1_s]).sum()

    ind_ret = num / den.replace(0, np.nan)
    ind_ret = ind_ret.unstack(level=-1)

    return ind_ret.reindex(rtn.index)


def rolling_icm(key_arr, ind_ret, window):
    """ICM: 排序键最大的5天对应的行业收益率加权和"""
    T = len(key_arr)
    result = np.full(T, np.nan, dtype=np.float64)
    for i in range(window - 1, T):
        k = key_arr[i - window + 1:i + 1]
        valid_mask = ~np.isnan(k)
        if valid_mask.sum() < 5:
            continue
        valid_pos = np.where(valid_mask)[0]
        valid_k = k[valid_mask]
        sort_idx = np.argsort(valid_k)
        top5_vpos = valid_pos[sort_idx[-5:]]
        abs_top5 = i - window + 1 + top5_vpos
        ir_top = ind_ret[abs_top5]
        if np.any(np.isnan(ir_top)):
            continue
        result[i] = np.dot(ir_top, W_TOP)
    return result


def rolling_icr(key_arr, ind_ret, window):
    """ICR: 排序键最小的15天对应的行业收益率加权和"""
    T = len(key_arr)
    result = np.full(T, np.nan, dtype=np.float64)
    for i in range(window - 1, T):
        k = key_arr[i - window + 1:i + 1]
        valid_mask = ~np.isnan(k)
        if valid_mask.sum() < 15:
            continue
        valid_pos = np.where(valid_mask)[0]
        valid_k = k[valid_mask]
        sort_idx = np.argsort(valid_k)
        bot15_vpos = valid_pos[sort_idx[:15]]
        abs_bot15 = i - window + 1 + bot15_vpos
        ir_bot = ind_ret[abs_bot15]
        if np.any(np.isnan(ir_bot)):
            continue
        result[i] = np.dot(ir_bot, W_BOT)
    return result


def rolling_cmc(key_arr, ind_ret, window):
    """CMC: ICM - ICR"""
    T = len(key_arr)
    result = np.full(T, np.nan, dtype=np.float64)
    for i in range(window - 1, T):
        k = key_arr[i - window + 1:i + 1]
        valid_mask = ~np.isnan(k)
        if valid_mask.sum() < 15:
            continue
        valid_pos = np.where(valid_mask)[0]
        valid_k = k[valid_mask]
        sort_idx = np.argsort(valid_k)

        top5_vpos = valid_pos[sort_idx[-5:]]
        abs_top5 = i - window + 1 + top5_vpos
        ir_top = ind_ret[abs_top5]
        if np.any(np.isnan(ir_top)):
            continue

        bot15_vpos = valid_pos[sort_idx[:15]]
        abs_bot15 = i - window + 1 + bot15_vpos
        ir_bot = ind_ret[abs_bot15]
        if np.any(np.isnan(ir_bot)):
            continue

        result[i] = np.dot(ir_top, W_TOP) - np.dot(ir_bot, W_BOT)
    return result
