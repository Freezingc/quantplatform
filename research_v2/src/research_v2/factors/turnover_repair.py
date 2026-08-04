"""Candidate reconstruction for the 2020-09 turnover-rate source gap.

The functions here are deliberately pure.  They calculate candidates from a
market frame but never decide to overwrite a source value; callers must still
run the normal overlap, PIT, and quarantine gates.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

TURNOVER_GAP_FORMULAS: Mapping[str, str] = {
    "GP_008": "mul(turnover, cs_rank(amount))",
    "GP_011": "mul(turnover, amount)",
    "GP_013": "mul(turnover, cs_rank(cs_zscore(shift_1(amount))))",
    "GP_016": "ts_max_5(mul(cs_zscore(pct_chg), turnover))",
    "GP_018": "mul(turnover, shift_1(amount))",
    "GP_019": "mul(ts_sum_10(turnover), abs(amount))",
    "GP_026": "mul(mul(vol, ts_rank_60(cs_rank(close))), "
    "sub(turnover, ts_corr_10(open, sub(close, ts_rank_20(inv(ts_mean_5(close)))))))",
}

_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "GP_008": ("amount", "turnover_rate"),
    "GP_011": ("amount", "turnover_rate"),
    "GP_013": ("amount", "turnover_rate"),
    "GP_016": ("pct_chg", "turnover_rate"),
    "GP_018": ("amount", "turnover_rate"),
    "GP_019": ("amount", "turnover_rate"),
    "GP_026": ("close", "open", "vol", "turnover_rate"),
}


def compute_turnover_gap_candidate(
    factor_name: str,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Compute one registered GP candidate using calculator-equivalent pandas semantics."""

    if factor_name not in _REQUIRED_COLUMNS:
        raise ValueError(f"unsupported turnover-gap formula: {factor_name}")
    required = {"trade_date", "ts_code", *_REQUIRED_COLUMNS[factor_name]}
    missing = sorted(required.difference(market.columns))
    if missing:
        raise ValueError(f"market frame is missing columns for {factor_name}: {missing}")
    normalized = market.loc[:, sorted(required)].copy(deep=True)
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="raise"
    ).dt.date
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    if normalized[["trade_date", "ts_code"]].isna().any().any():
        raise ValueError("market business keys cannot be null")
    normalized = normalized.drop_duplicates().reset_index(drop=True)
    if normalized.duplicated(["trade_date", "ts_code"], keep=False).any():
        raise ValueError("market contains conflicting duplicate keys")
    for column in _REQUIRED_COLUMNS[factor_name]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    matrices = {
        column: normalized.pivot(
            index="trade_date", columns="ts_code", values=column
        ).sort_index()
        for column in _REQUIRED_COLUMNS[factor_name]
    }
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        candidate = _calculate(factor_name, matrices)
    candidate = candidate.replace([np.inf, -np.inf], np.nan)
    return _candidate_long(candidate)


def _calculate(factor_name: str, matrices: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    amount = matrices.get("amount")
    turnover = matrices.get("turnover_rate")
    if factor_name == "GP_008":
        assert amount is not None and turnover is not None
        return turnover * amount.rank(axis=1, pct=True)
    if factor_name == "GP_011":
        assert amount is not None and turnover is not None
        return turnover * amount
    if factor_name == "GP_013":
        assert amount is not None and turnover is not None
        shifted = amount.shift(1)
        zscore = shifted.sub(shifted.mean(axis=1), axis=0).div(
            shifted.std(axis=1) + 1e-10, axis=0
        )
        return turnover * zscore.rank(axis=1, pct=True)
    if factor_name == "GP_016":
        pct_chg = matrices["pct_chg"]
        assert turnover is not None
        zscore = pct_chg.sub(pct_chg.mean(axis=1), axis=0).div(
            pct_chg.std(axis=1) + 1e-10, axis=0
        )
        return (zscore * turnover).rolling(5).max()
    if factor_name == "GP_018":
        assert amount is not None and turnover is not None
        return turnover * amount.shift(1)
    if factor_name == "GP_019":
        assert amount is not None and turnover is not None
        return turnover.rolling(10).sum() * amount.abs()
    close = matrices["close"]
    open_ = matrices["open"]
    vol = matrices["vol"]
    assert turnover is not None
    rank_close = close.rank(axis=1, pct=True)
    rank_60 = rank_close.rolling(60).rank(pct=True)
    v3 = vol * rank_60
    v4 = close.rolling(5).mean()
    v5 = (1 / v4.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    v6 = v5.rolling(20).rank(pct=True)
    v7 = close - v6
    v8 = open_.rolling(10).corr(v7)
    return v3 * (turnover - v8)


def _candidate_long(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.rename_axis(index="trade_date", columns="ts_code").reset_index().melt(
        id_vars=["trade_date"], var_name="ts_code", value_name="candidate_value"
    )
