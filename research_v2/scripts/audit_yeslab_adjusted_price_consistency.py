"""Audit YesLab adjusted high/low provenance against Tushare inputs.

The audit is read-only.  It separates the internal identity ``raw * adj_factor``
from cross-source differences, so a replay block cannot be attributed to
rounding without evidence.  It never reads 2026 or writes a database table.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import dolphindb as ddb
import numpy as np
import pandas as pd

LOCKED_TEST_START = pd.Timestamp("2026-01-01")
DEFAULT_SOURCE_FIELDS = (
    "ts_code",
    "trade_date",
    "high",
    "low",
    "adj_factor",
    "high_hfq",
    "low_hfq",
)
DEFAULT_CACHE_DIR = Path("research_v2/artifacts/tushare_cache")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def summarize_internal_consistency(
    frame: pd.DataFrame, *, tolerance: float = 1e-5
) -> dict[str, int | float]:
    """Measure whether YesLab raw fields reproduce its stored HFQ fields."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    _require_columns(
        frame,
        {"high", "low", "adj_factor", "high_hfq", "low_hfq"},
        "source frame",
    )
    numeric = frame.loc[:, ["high", "low", "adj_factor", "high_hfq", "low_hfq"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid_high = numeric[["high", "adj_factor", "high_hfq"]].notna().all(axis=1)
    valid_low = numeric[["low", "adj_factor", "low_hfq"]].notna().all(axis=1)
    high_diff = numeric["high_hfq"] - numeric["high"] * numeric["adj_factor"]
    low_diff = numeric["low_hfq"] - numeric["low"] * numeric["adj_factor"]
    high_abs = high_diff.abs().where(valid_high).dropna()
    low_abs = low_diff.abs().where(valid_low).dropna()
    return {
        "rows": int(len(frame)),
        "high_valid_rows": int(valid_high.sum()),
        "low_valid_rows": int(valid_low.sum()),
        "high_mismatch_rows": int((high_abs > tolerance).sum()),
        "low_mismatch_rows": int((low_abs > tolerance).sum()),
        "max_high_abs_diff": float(high_abs.max()) if len(high_abs) else float("nan"),
        "max_low_abs_diff": float(low_abs.max()) if len(low_abs) else float("nan"),
        "mean_high_abs_diff": float(high_abs.mean()) if len(high_abs) else float("nan"),
        "mean_low_abs_diff": float(low_abs.mean()) if len(low_abs) else float("nan"),
    }


def compare_market_fields(
    source: pd.DataFrame,
    market: pd.DataFrame,
    *,
    raw_tolerance: float = 1e-10,
    adjustment_tolerance: float = 1e-10,
    derived_tolerance: float = 1e-5,
) -> dict[str, int | float]:
    """Classify raw, adjustment-factor, and derived HFQ cross-source changes."""
    _require_columns(source, set(DEFAULT_SOURCE_FIELDS), "source frame")
    _require_columns(
        market, {"ts_code", "trade_date", "high", "low", "adj_factor"}, "market frame"
    )
    left = source.loc[:, list(DEFAULT_SOURCE_FIELDS)].copy()
    right = market.loc[:, ["ts_code", "trade_date", "high", "low", "adj_factor"]].copy()
    for frame in (left, right):
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        for column in ("high", "low", "adj_factor", "high_hfq", "low_hfq"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
    right = right.drop_duplicates(["ts_code", "trade_date"], keep="last")
    merged = left.merge(
        right,
        on=["ts_code", "trade_date"],
        how="inner",
        suffixes=("_source", "_market"),
    )
    source_high = merged["high_source"] * merged["adj_factor_source"]
    market_high = merged["high_market"] * merged["adj_factor_market"]
    source_low = merged["low_source"] * merged["adj_factor_source"]
    market_low = merged["low_market"] * merged["adj_factor_market"]
    raw_high = (merged["high_source"] - merged["high_market"]).abs()
    raw_low = (merged["low_source"] - merged["low_market"]).abs()
    adj_diff = (merged["adj_factor_source"] - merged["adj_factor_market"]).abs()
    derived_high = (source_high - market_high).abs()
    derived_low = (source_low - market_low).abs()
    finite = np.isfinite(
        merged[
            [
                "high_source",
                "low_source",
                "adj_factor_source",
                "high_market",
                "low_market",
                "adj_factor_market",
            ]
        ]
    ).all(axis=1)
    def count(values: pd.Series, tolerance: float) -> int:
        return int((values.where(finite).fillna(np.inf) > tolerance).sum())
    return {
        "overlap_rows": int(len(merged)),
        "raw_high_mismatch_rows": count(raw_high, raw_tolerance),
        "raw_low_mismatch_rows": count(raw_low, raw_tolerance),
        "adj_factor_mismatch_rows": count(adj_diff, adjustment_tolerance),
        "derived_high_mismatch_rows": count(derived_high, derived_tolerance),
        "derived_low_mismatch_rows": count(derived_low, derived_tolerance),
        "max_raw_high_abs_diff": float(raw_high.where(finite).max()),
        "max_raw_low_abs_diff": float(raw_low.where(finite).max()),
        "max_adj_factor_abs_diff": float(adj_diff.where(finite).max()),
        "max_derived_high_abs_diff": float(derived_high.where(finite).max()),
        "max_derived_low_abs_diff": float(derived_low.where(finite).max()),
    }


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    return frame


def load_cached_market(cache_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    daily_frames: list[pd.DataFrame] = []
    adj_frames: list[pd.DataFrame] = []
    for path in sorted(cache_dir.glob("daily_????????.json")):
        token = path.stem.removeprefix("daily_")
        try:
            day = pd.Timestamp(token)
        except ValueError:
            continue
        if start <= day <= end:
            daily_frames.append(pd.read_json(path, orient="table"))
    for path in sorted(cache_dir.glob("adj_factor_????????.json")):
        token = path.stem.removeprefix("adj_factor_")
        try:
            day = pd.Timestamp(token)
        except ValueError:
            continue
        if start <= day <= end:
            adj_frames.append(pd.read_json(path, orient="table"))
    if not daily_frames or not adj_frames:
        raise ValueError("cache has no complete daily/adj_factor observations")
    daily = _normalize_dates(pd.concat(daily_frames, ignore_index=True))
    adjustment = _normalize_dates(pd.concat(adj_frames, ignore_index=True))
    adjustment = adjustment.loc[:, ["ts_code", "trade_date", "adj_factor"]]
    adjustment["adj_factor"] = pd.to_numeric(adjustment["adj_factor"], errors="coerce")
    return daily.merge(
        adjustment,
        on=["ts_code", "trade_date"],
        how="left",
        suffixes=("", "_adj"),
        validate="one_to_one",
    )


def query_source(session: object, start: date, end: date) -> pd.DataFrame:
    query = (
        "select ts_code,trade_date,high,low,adj_factor,high_hfq,low_hfq "
        "from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}') "
        "and (endsWith(ts_code,'.SH') or endsWith(ts_code,'.SZ'))"
    )
    result = session.run(query)
    if not isinstance(result, pd.DataFrame):
        raise TypeError("source query did not return a DataFrame")
    return _normalize_dates(result)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="run the YesLab raw*adj_factor identity check without Tushare cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if start > end or end >= LOCKED_TEST_START:
        raise ValueError("date range must be before 2026 and start <= end")
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    source = query_source(session, args.start, args.end)
    source = source.loc[~source["ts_code"].str.endswith(".BJ")].copy()
    market = None if args.source_only else load_cached_market(args.cache_dir, start, end)
    if market is not None:
        market = market.loc[~market["ts_code"].str.endswith(".BJ")].copy()
    result = {
        "scope": "YESLAB_ADJUSTED_PRICE_PROVENANCE_AUDIT_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_scope": ["SH", "SZ"],
        "excluded_exchanges": ["BJ"],
        "source_rows": int(len(source)),
        "market_rows": int(len(market)) if market is not None else 0,
        "internal_consistency": summarize_internal_consistency(source),
        "cross_source": (
            {"status": "NOT_EVALUATED_SOURCE_ONLY"}
            if market is None
            else compare_market_fields(source, market)
        ),
        "market_source": "local Tushare.daily + adj_factor cache",
        "database_writes": False,
        "locked_test_read": False,
        "publication_status": "NOT_PUBLISHED",
        "model_training_started": False,
        "interpretation": (
            "Internal identity and cross-source differences are reported separately; "
            "no old source value is overwritten."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
