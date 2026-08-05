"""Pure signed-power variants for two invalid-domain GP expressions.

The source expressions use fractional powers of values that can be negative.
The signed-power transformation is deliberately isolated here: it creates a
new candidate formula and never overwrites the stored YesLab observations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def signed_power(base: pd.Series | pd.DataFrame, exponent: pd.Series | pd.DataFrame):
    """Return sign(base) * abs(base) ** exponent with non-finite cleanup."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.sign(base) * np.power(np.abs(base), exponent)
    return result.replace([np.inf, -np.inf], np.nan)


def _rolling_slope_matrix(values: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute equally spaced rolling slopes with cumulative sums.

    This is algebraically identical to the ordinary least-squares slope used
    by the previous rolling callback, but avoids one Python callback per
    security/window. Any non-finite value invalidates its entire window.
    """

    if window < 2:
        raise ValueError("slope window must be at least two")
    array = values.to_numpy(dtype=float)
    finite = np.isfinite(array)
    clean = np.where(finite, array, 0.0)
    index = np.arange(len(values), dtype=float)[:, None]
    clean_frame = pd.DataFrame(clean, index=values.index, columns=values.columns)
    weighted_frame = pd.DataFrame(
        clean * index, index=values.index, columns=values.columns
    )
    count_frame = pd.DataFrame(
        finite.astype(float), index=values.index, columns=values.columns
    )
    sum_y = clean_frame.rolling(window, min_periods=window).sum()
    sum_iy = weighted_frame.rolling(window, min_periods=window).sum()
    count = count_frame.rolling(window, min_periods=window).sum()
    end = np.arange(len(values), dtype=float)
    start = end - window + 1
    sum_i = window * (start + end) / 2.0
    sum_i2 = (
        end * (end + 1.0) * (2.0 * end + 1.0)
        - (start - 1.0) * start * (2.0 * (start - 1.0) + 1.0)
    ) / 6.0
    numerator = window * sum_iy - sum_i[:, None] * sum_y
    denominator = window * sum_i2 - sum_i**2
    result = numerator / denominator[:, None]
    return result.where(count.eq(float(window)))


def _matrix(market: pd.DataFrame, column: str) -> pd.DataFrame:
    return market.pivot(index="trade_date", columns="ts_code", values=column).sort_index()


def _to_long(values: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    result = (
        values.rename_axis("signal_date")
        .stack(future_stack=True)
        .rename("value")
        .reset_index()
    )
    result["factor_id"] = factor_id
    return result.loc[:, ["signal_date", "ts_code", "factor_id", "value"]]


def compute_signed_power_variants(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute GP_002 and GP_020 signed-power variants from a market panel."""

    required = {"trade_date", "ts_code", "vol", "pct_chg", "turnover_rate"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    frame = market.loc[:, sorted(required)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    for column in sorted(required.difference({"trade_date", "ts_code"})):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("market frame contains duplicate trade-date/security keys")
    vol = _matrix(frame, "vol")
    pct_chg = _matrix(frame, "pct_chg").reindex(columns=vol.columns)
    turnover = _matrix(frame, "turnover_rate").reindex(columns=vol.columns)
    slope_60 = _rolling_slope_matrix(vol, 60)
    gp002 = signed_power(slope_60, turnover)
    gp020 = signed_power(pct_chg, turnover.rolling(10).sum()) + pct_chg
    return {
        "yeslab_371": _to_long(gp002, "yeslab_371"),
        "yeslab_389": _to_long(gp020, "yeslab_389"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.input.is_file():
        raise ValueError(f"market input does not exist: {args.input}")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    market = (
        pd.read_parquet(args.input)
        if args.input.suffix == ".parquet"
        else pd.read_csv(args.input)
    )
    variants = compute_signed_power_variants(market)
    payload = {
        "scope": "SIGNED_POWER_VARIANT_COMPUTATION_V1",
        "input": str(args.input),
        "factor_ids": sorted(variants),
        "rows": {factor_id: len(frame) for factor_id, frame in variants.items()},
        "finite_rows": {
            factor_id: int(np.isfinite(frame["value"].to_numpy(float)).sum())
            for factor_id, frame in variants.items()
        },
        "database_writes": False,
        "locked_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
