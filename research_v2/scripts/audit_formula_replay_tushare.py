"""Replay selected YesLab factors with Tushare as the canonical market source.

This is a bounded, read-only reconciliation command.  It deliberately limits
the stock universe so nested rolling formulas can be checked without loading
the entire A-share panel into memory.  The stored factor observations remain
the YesLab comparison target; only the market inputs used by the calculator
are replaced with Tushare ``daily`` observations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
from pathlib import Path

import audit_formula_replay_consistency as replay
import numpy as np
import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
SUPPORTED_FACTOR_IDS = (
    "yeslab_5",
    "yeslab_31",
    "yeslab_71",
    "yeslab_105",
    "yeslab_94",
    "yeslab_95",
    "yeslab_96",
    "yeslab_98",
    "yeslab_114",
    "yeslab_155",
    "yeslab_163",
    "yeslab_165",
    "yeslab_167",
    "yeslab_168",
    "yeslab_206",
    "yeslab_208",
    "yeslab_143",
    "yeslab_370",
    "yeslab_371",
    "yeslab_372",
    "yeslab_379",
    "yeslab_384",
    "yeslab_389",
    "yeslab_390",
    "yeslab_404",
    "yeslab_397",
    "yeslab_406",
    "yeslab_411",
    "yeslab_414",
    "yeslab_415",
    "yeslab_420",
    "yeslab_423",
    "yeslab_376",
    "yeslab_377",
    "yeslab_380",
    "yeslab_381",
    "yeslab_385",
    "yeslab_387",
    "yeslab_388",
    "yeslab_391",
    "yeslab_396",
    "yeslab_400",
    "yeslab_405",
    "yeslab_425",
    "yeslab_429",
    "yeslab_431",
    "yeslab_432",
    "yeslab_433",
    "yeslab_443",
    "yeslab_437",
    "yeslab_442",
    "yeslab_398",
    "yeslab_438",
    "yeslab_451",
    "yeslab_464",
    # These two remain isolated by default admission policy, but are allowed
    # here for explicit full-cross-section diagnostics.
    "yeslab_395",
    "yeslab_471",
    # Industry-normalized factors require full-market replay because their
    # cross-sectional grouping is performed inside the calculator.
    "yeslab_357",
    "yeslab_362",
    "yeslab_363",
    "yeslab_481",
    "yeslab_482",
    "yeslab_483",
    "yeslab_484",
    "yeslab_485",
    "yeslab_486",
    "yeslab_487",
)
REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"
# These calculators use same-day cross-sectional means, ranks, or z-scores.
# They must be replayed against the complete available universe in one batch.
FULL_CROSS_SECTION_FACTOR_IDS = frozenset(
    {
        "yeslab_31",
        "yeslab_71",
        "yeslab_105",
        "yeslab_94",
        "yeslab_95",
        "yeslab_96",
        "yeslab_98",
        "yeslab_114",
        "yeslab_155",
        "yeslab_167",
        "yeslab_377",
        "yeslab_376",
        "yeslab_381",
        "yeslab_385",
        "yeslab_396",
        "yeslab_400",
        "yeslab_405",
        "yeslab_425",
        "yeslab_442",
        "yeslab_432",
        "yeslab_395",
        "yeslab_443",
        "yeslab_357",
        "yeslab_362",
        "yeslab_363",
        "yeslab_481",
        "yeslab_482",
        "yeslab_483",
        "yeslab_484",
        "yeslab_485",
        "yeslab_486",
        "yeslab_487",
        # Attention spillover uses same-day market means and industry/size
        # buckets, so partial stock batches change both peer membership and
        # the resulting factor values.
        "yeslab_471",
    }
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 9, 14))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 9, 25))
    parser.add_argument("--max-ts-codes", type=int, default=64)
    parser.add_argument("--full-market", action="store_true")
    parser.add_argument("--restrict-to-v2-universe", action="store_true")
    parser.add_argument(
        "--exclude-bj",
        action="store_true",
        help="exclude .BJ securities from the replay universe",
    )
    parser.add_argument("--include-daily-basic", action="store_true")
    parser.add_argument("--stock-batch-size", type=int, default=256)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/qrv2-tushare-replay-cache"),
    )
    parser.add_argument(
        "--bj-alias-map",
        type=Path,
        help="explicit old-to-new BJ code alias artifact; disabled by default",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(args.factor_id or SUPPORTED_FACTOR_IDS)
    unsupported = sorted(set(args.factor_id).difference(SUPPORTED_FACTOR_IDS))
    if unsupported:
        raise ValueError(f"unsupported Tushare replay factor IDs: {unsupported}")
    if not args.factor_id or len(set(args.factor_id)) != len(args.factor_id):
        raise ValueError("factor IDs must be non-empty and unique")
    if args.start > args.end:
        raise ValueError("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("Tushare formula replay cannot read 2026 locked-test data")
    if args.max_ts_codes < 1:
        raise ValueError("max-ts-codes must be positive")
    if args.stock_batch_size < 1:
        raise ValueError("stock-batch-size must be positive")
    return args


def normalize_tushare_market(
    raw: pd.DataFrame,
    *,
    codes: Sequence[str],
    include_daily_basic: bool = True,
) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "open", "low", "amount", "pct_chg"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Tushare daily response is missing columns: {missing}")
    frame = raw.copy()
    frame["ts_code"] = frame["ts_code"].astype(str)
    parsed_trade_date = pd.to_datetime(
        frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    frame["trade_date"] = parsed_trade_date.fillna(
        pd.to_datetime(frame["trade_date"], errors="coerce")
    ).dt.normalize()
    for column in (
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "pct_chg",
        "turnover_rate",
        "total_mv",
        "adj_factor",
    ):
        if column == "turnover_rate" and not include_daily_basic:
            continue
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "adj_factor" in frame.columns:
        for raw_column, adjusted_column in (
            ("open", "open_hfq"),
            ("high", "high_hfq"),
            ("low", "low_hfq"),
            ("close", "close_hfq"),
            ("pre_close", "pre_close_hfq"),
        ):
            if raw_column in frame.columns:
                frame[adjusted_column] = frame[raw_column] * frame["adj_factor"]
    frame = frame.loc[frame["ts_code"].isin(set(map(str, codes)))].copy()
    frame = frame.dropna(subset=["trade_date", "ts_code"])
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    columns = ["trade_date", "ts_code"] + [
        column
        for column in (
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "vol",
            "amount",
            "pct_chg",
            "turnover_rate",
            "total_mv",
            "adj_factor",
            "open_hfq",
            "high_hfq",
            "low_hfq",
            "close_hfq",
            "pre_close_hfq",
        )
        if column in frame
        and (column not in {"turnover_rate", "total_mv"} or include_daily_basic)
    ]
    return frame.loc[:, columns].sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def normalize_full_tushare_market(raw_market: pd.DataFrame) -> pd.DataFrame:
    """Normalize a date-wise Tushare daily response without a source-code universe."""
    if "ts_code" not in raw_market.columns:
        raise ValueError("Tushare daily result is missing ts_code")
    codes = tuple(
        sorted(
            code
            for code in raw_market["ts_code"].astype(str).dropna().unique()
            if code.endswith((".SH", ".SZ"))
        )
    )
    if not codes:
        raise ValueError("Tushare daily result has no SH/SZ market rows")
    return normalize_tushare_market(raw_market, codes=codes)


def fetch_full_tushare_shsz_market(
    open_dates: Sequence[date],
    *,
    token: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch all daily Tushare SH/SZ rows and adjusted fields for open dates."""
    if not token.strip():
        raise ValueError("QRV2_TUSHARE_TOKEN is required for canonical market replay")
    expected_dates = tuple(sorted(set(open_dates)))
    if not expected_dates:
        raise ValueError("at least one open date is required")
    import tushare as ts  # type: ignore[import-untyped]

    cache_dir.mkdir(parents=True, exist_ok=True)
    pro = ts.pro_api(token)
    frames: list[pd.DataFrame] = []
    for trade_date in expected_dates:
        daily_path = cache_dir / f"daily_{trade_date:%Y%m%d}.json"
        if daily_path.is_file():
            daily = pd.read_json(daily_path, orient="table")
        else:
            daily = pro.daily(trade_date=trade_date.strftime("%Y%m%d"))
            if isinstance(daily, pd.DataFrame) and not daily.empty:
                daily_path.write_text(
                    daily.to_json(orient="table", date_format="iso"), encoding="utf-8"
                )
        if not isinstance(daily, pd.DataFrame) or daily.empty:
            raise RuntimeError(f"Tushare daily returned no rows for {trade_date:%Y-%m-%d}")
        adjustment_path = cache_dir / f"adj_factor_{trade_date:%Y%m%d}.json"
        if adjustment_path.is_file():
            adjustment = pd.read_json(adjustment_path, orient="table")
        else:
            adjustment = pro.adj_factor(trade_date=trade_date.strftime("%Y%m%d"))
            if isinstance(adjustment, pd.DataFrame) and not adjustment.empty:
                adjustment_path.write_text(
                    adjustment.to_json(orient="table", date_format="iso"), encoding="utf-8"
                )
        if not isinstance(adjustment, pd.DataFrame) or adjustment.empty:
            raise RuntimeError(
                f"Tushare adjustment factor returned no rows for {trade_date:%Y-%m-%d}"
            )
        frames.append(merge_tushare_adjustment_factor(daily, adjustment))
    normalized = normalize_full_tushare_market(pd.concat(frames, ignore_index=True))
    observed_dates = set(pd.to_datetime(normalized["trade_date"], errors="raise").dt.date)
    missing_dates = sorted(set(expected_dates).difference(observed_dates))
    if missing_dates:
        rendered = ", ".join(value.isoformat() for value in missing_dates)
        raise ValueError(f"full SH/SZ market is missing open dates: {rendered}")
    if "close_hfq" not in normalized.columns:
        raise ValueError("full SH/SZ market is missing adjusted close")
    return normalized


def merge_tushare_adjustment_factor(
    daily: pd.DataFrame, adjustment: pd.DataFrame
) -> pd.DataFrame:
    """Attach Tushare adj_factor to daily rows using the business key."""

    required_daily = {"ts_code", "trade_date"}
    required_adjustment = required_daily | {"adj_factor"}
    if missing := sorted(required_daily.difference(daily.columns)):
        raise ValueError(f"daily frame is missing columns: {missing}")
    if missing := sorted(required_adjustment.difference(adjustment.columns)):
        raise ValueError(f"adjustment frame is missing columns: {missing}")
    left = daily.copy()
    right = adjustment.loc[:, ["ts_code", "trade_date", "adj_factor"]].copy()
    for frame in (left, right):
        frame["ts_code"] = frame["ts_code"].astype(str)
        frame["trade_date"] = pd.to_datetime(
            frame["trade_date"], errors="raise"
        ).dt.normalize()
    right["adj_factor"] = pd.to_numeric(right["adj_factor"], errors="coerce")
    right = right.drop_duplicates(["ts_code", "trade_date"], keep="last")
    left = left.drop(columns=["adj_factor"], errors="ignore")
    return left.merge(right, on=["ts_code", "trade_date"], how="left", validate="one_to_one")


class TushareMarketSession:
    """DolphinDB session wrapper that substitutes only stock market queries."""

    def __init__(self, base: object, market: pd.DataFrame, codes: Sequence[str]) -> None:
        self._base = base
        self._market = market
        self._codes = set(map(str, codes))

    def run(self, script: str) -> object:
        normalized = "".join(str(script).lower().split()).replace("'", '"')
        # FactorCalculator builds this query through a database handle:
        # ``database("dfs://YesLab"); loadTable(db, "stock_factor_daily")``.
        # Match both that form and the direct loadTable form so replay never
        # silently falls back to the YesLab market table.
        if (
            "stock_factor_daily" in normalized
            and "dfs://yeslab" in normalized
            and ("loadtable(db," in normalized or "loadtable(\"dfs://yeslab\"," in normalized)
        ):
            return self._market.loc[
                self._market["ts_code"].astype(str).isin(self._codes)
            ].copy(deep=True)
        result = self._base.run(script)
        if isinstance(result, pd.DataFrame) and "ts_code" in result.columns:
            result = result.loc[result["ts_code"].astype(str).isin(self._codes)].copy()
        return result


def _query_codes(
    client: object,
    definition: replay.ReplayDefinition,
    start: date,
    end: date,
    limit: int | None,
) -> tuple[str, ...]:
    result = client.run(
        "select distinct ts_code from "
        f"loadTable('dfs://YesLab','{definition.store_factor}') "
        f"where factor_name == '{definition.factor_name}' "
        "and isValid(value) "
        f"and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}') order by ts_code"
    )
    if not isinstance(result, pd.DataFrame) or "ts_code" not in result.columns:
        raise RuntimeError("factor universe query did not return ts_code")
    codes = sorted(result["ts_code"].astype(str).dropna().unique())
    return tuple(codes if limit is None else codes[:limit])


def _query_open_dates(client: object, start: date, end: date) -> tuple[date, ...]:
    result = client.run(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open == 1 "
        f"and cal_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}') order by cal_date"
    )
    if not isinstance(result, pd.DataFrame) or "cal_date" not in result.columns:
        raise RuntimeError("trade calendar query did not return cal_date")
    dates = pd.to_datetime(result["cal_date"], errors="coerce").dropna().dt.date
    return tuple(sorted(value for value in dates if value < LOCKED_TEST_START))


def _query_v2_universe_codes(client: object, start: date, end: date) -> tuple[str, ...]:
    result = client.run(
        "select distinct ts_code from "
        "loadTable('dfs://QuantResearchV2Test','universe_daily_latest') "
        "where is_member == true "
        f"and signal_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame) or "ts_code" not in result.columns:
        raise RuntimeError("V2 universe query did not return ts_code")
    return tuple(sorted(result["ts_code"].astype(str).dropna().unique()))


def _query_stock_market_codes(
    client: object, start: date, end: date
) -> tuple[str, ...]:
    """Return the historical market universe used by cross-sectional calculators."""

    result = client.run(
        "select distinct ts_code from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame) or "ts_code" not in result.columns:
        raise RuntimeError("stock market universe query did not return ts_code")
    return tuple(sorted(result["ts_code"].astype(str).dropna().unique()))


def _calculator_market_start(
    client: object, definition: replay.ReplayDefinition, start: date, end: date
) -> date:
    calculator_class = _calculator_class(definition)
    calculator = calculator_class(client, definition.effective_replay_lookback_days)
    from factor_calculators.HerdingCSADCalculatorBase import (  # type: ignore
        HerdingCSADCalculatorBase,
    )

    if isinstance(calculator, HerdingCSADCalculatorBase):
        original_lookback = calculator.lookback_days
        calculator.lookback_days = original_lookback + 1
        try:
            processed = calculator.preprocess_time_range(
                start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
            )
        finally:
            calculator.lookback_days = original_lookback
        return date.fromisoformat(processed.replace(".", "-"))

    processed = calculator.preprocess_time_range(
        start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d")
    )
    return date.fromisoformat(processed.replace(".", "-")) - timedelta(days=30)


def _calculator_class(definition: replay.ReplayDefinition) -> type:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    module = importlib.import_module(f"factor_calculators.{definition.calculator_module}")
    return getattr(module, definition.calculator_class)


def required_market_fields(
    definitions: Sequence[replay.ReplayDefinition],
) -> tuple[str, ...]:
    """Return raw market fields declared by the selected calculator classes."""

    fields: set[str] = set()
    for definition in definitions:
        calculator = _calculator_class(definition)(None, definition.effective_replay_lookback_days)
        fields.update(str(field) for field in getattr(calculator, "required_factors", ()))
    return tuple(sorted(fields))


def requires_daily_basic(required_fields: Sequence[str]) -> bool:
    """Whether the selected calculators need fields supplied by daily_basic."""

    return bool({"turnover_rate", "total_mv"}.intersection(required_fields))


def requires_full_cross_section(factor_ids: Sequence[str]) -> bool:
    return bool(set(factor_ids).intersection(FULL_CROSS_SECTION_FACTOR_IDS))


def load_alias_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("mapping", payload)
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("BJ alias artifact must contain a non-empty mapping object")
    aliases = {str(old): str(new) for old, new in mapping.items()}
    if any(not old.endswith(".BJ") or not new.endswith(".BJ") for old, new in aliases.items()):
        raise ValueError("BJ aliases must use .BJ security codes")
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("duplicate alias target codes")
    return dict(sorted(aliases.items()))


def apply_tushare_aliases(
    market: pd.DataFrame, aliases: Mapping[str, str]
) -> pd.DataFrame:
    if not aliases:
        return market.copy(deep=True)
    inverse = {new: old for old, new in aliases.items()}
    remapped = market.copy(deep=True)
    remapped["ts_code"] = remapped["ts_code"].astype(str).map(
        lambda code: inverse.get(code, code)
    )
    if remapped.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("BJ alias remap created duplicate market business keys")
    return remapped


def source_market_coverage(
    source: pd.DataFrame, market: pd.DataFrame
) -> dict[str, int]:
    required = {"trade_date", "ts_code", "value"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"source frame is missing columns: {missing}")
    market_missing = sorted({"trade_date", "ts_code"}.difference(market.columns))
    if market_missing:
        raise ValueError(f"market frame is missing columns: {market_missing}")
    frame = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.drop_duplicates(["trade_date", "ts_code"], keep="last")
    market_keys = market.loc[:, ["trade_date", "ts_code"]].copy()
    market_keys["trade_date"] = pd.to_datetime(
        market_keys["trade_date"], errors="raise"
    ).dt.normalize()
    market_keys["ts_code"] = market_keys["ts_code"].astype(str)
    market_keys = market_keys.drop_duplicates(["trade_date", "ts_code"])
    covered = frame.loc[:, ["trade_date", "ts_code"]].merge(
        market_keys.assign(_market_present=True),
        on=["trade_date", "ts_code"],
        how="left",
        sort=False,
    )["_market_present"].eq(True).to_numpy(bool)
    finite = np.isfinite(frame["value"].to_numpy(float))
    uncovered = finite & ~covered
    orphan = frame.loc[uncovered]
    return {
        "source_finite_rows_full": int(finite.sum()),
        "source_market_uncovered_rows": int(uncovered.sum()),
        "source_market_uncovered_code_count": int(orphan["ts_code"].nunique()),
        "source_market_uncovered_date_count": int(orphan["trade_date"].nunique()),
    }


def missing_required_market_codes(
    required_keys: pd.DataFrame,
    market: pd.DataFrame,
) -> tuple[str, ...]:
    """Return codes with at least one required source key absent from market.

    A code may be present in a daily snapshot while still missing on some
    dates.  Full cross-sectional calculators need a complete code history for
    every finite source observation, so code-level fallback must cover this
    partial-gap case as well as completely absent codes.
    """

    required_columns = {"trade_date", "ts_code"}
    for label, frame in (("required_keys", required_keys), ("market", market)):
        if missing := sorted(required_columns.difference(frame.columns)):
            raise ValueError(f"{label} is missing columns: {missing}")
    left = required_keys.loc[:, ["trade_date", "ts_code"]].copy()
    right = market.loc[:, ["trade_date", "ts_code"]].copy()
    for frame in (left, right):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    left = left.drop_duplicates()
    right = right.drop_duplicates()
    covered = left.merge(
        right.assign(_market_present=True),
        on=["trade_date", "ts_code"],
        how="left",
    )["_market_present"].eq(True).to_numpy(dtype=bool)
    return tuple(sorted(left.loc[~covered, "ts_code"].unique()))


def fetch_missing_code_ranges(
    pro: object,
    codes: Sequence[str],
    start: date,
    end: date,
    *,
    cache_dir: Path,
    include_daily_basic: bool,
    include_adjusted_fields: bool = False,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch completely absent securities by code, including optional turnover data."""

    frames: list[pd.DataFrame] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    aliases = aliases or {}
    for code in codes:
        historical_code = str(code)
        request_code = aliases.get(historical_code, historical_code)
        safe_code = request_code.replace(".", "_")
        daily_cache = cache_dir / f"daily_code_{safe_code}_{start:%Y%m%d}_{end:%Y%m%d}.json"
        if daily_cache.is_file():
            daily = pd.read_json(daily_cache, orient="table")
        else:
            daily = pro.daily(
                ts_code=request_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            if isinstance(daily, pd.DataFrame):
                daily_cache.write_text(
                    daily.to_json(orient="table", date_format="iso"), encoding="utf-8"
                )
        if not isinstance(daily, pd.DataFrame) or daily.empty:
            continue
        daily = daily.copy()
        daily["ts_code"] = historical_code
        if include_adjusted_fields:
            adjustment_cache = cache_dir / (
                f"adj_factor_code_{safe_code}_{start:%Y%m%d}_{end:%Y%m%d}.json"
            )
            if adjustment_cache.is_file():
                adjustment = pd.read_json(adjustment_cache, orient="table")
            else:
                adjustment = pro.adj_factor(
                    ts_code=request_code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                )
                if isinstance(adjustment, pd.DataFrame):
                    adjustment_cache.write_text(
                        adjustment.to_json(orient="table", date_format="iso"),
                        encoding="utf-8",
                    )
            if isinstance(adjustment, pd.DataFrame) and not adjustment.empty:
                adjustment = adjustment.copy()
                adjustment["ts_code"] = historical_code
                daily = merge_tushare_adjustment_factor(daily, adjustment)
        if include_daily_basic:
            basic_cache = cache_dir / (
                f"daily_basic_code_{safe_code}_{start:%Y%m%d}_{end:%Y%m%d}.json"
            )
            if basic_cache.is_file():
                basic = pd.read_json(basic_cache, orient="table")
            else:
                basic = pd.DataFrame()
            if "total_mv" not in basic.columns:
                basic = pro.daily_basic(
                    ts_code=request_code,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    fields="ts_code,trade_date,turnover_rate,total_mv",
                )
                if isinstance(basic, pd.DataFrame):
                    basic_cache.write_text(
                        basic.to_json(orient="table", date_format="iso"), encoding="utf-8"
                    )
            if isinstance(basic, pd.DataFrame) and not basic.empty:
                basic = basic.copy()
                basic["ts_code"] = historical_code
                daily = daily.merge(
                    basic.reindex(
                        columns=["ts_code", "trade_date", "turnover_rate", "total_mv"]
                    ),
                    on=["ts_code", "trade_date"],
                    how="left",
                )
        frames.append(daily)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_tushare_market(
    codes: Sequence[str],
    start: date,
    end: date,
    *,
    token: str,
    open_dates: Sequence[date] | None = None,
    cache_dir: Path | None = None,
    include_daily_basic: bool = False,
    include_adjusted_fields: bool = False,
    aliases: Mapping[str, str] | None = None,
    required_keys: pd.DataFrame | None = None,
) -> pd.DataFrame:
    import tushare as ts  # type: ignore[import-untyped]

    if not token.strip():
        raise ValueError("QRV2_TUSHARE_TOKEN is required for canonical market replay")
    pro = ts.pro_api(token)
    frames: list[pd.DataFrame] = []
    aliases = aliases or {}
    resolved_cache = cache_dir or Path("/tmp/qrv2-tushare-replay-cache")
    resolved_cache.mkdir(parents=True, exist_ok=True)
    if open_dates is None:
        for code in codes:
            historical_code = str(code)
            request_code = aliases.get(historical_code, historical_code)
            result = pro.daily(
                ts_code=request_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            )
            if isinstance(result, pd.DataFrame) and not result.empty:
                result = result.copy()
                result["ts_code"] = historical_code
                if include_adjusted_fields:
                    adjustment_cache = resolved_cache / (
                        "adj_factor_code_"
                        f"{request_code.replace('.', '_')}_{start:%Y%m%d}_{end:%Y%m%d}.json"
                    )
                    if adjustment_cache.is_file():
                        adjustment = pd.read_json(adjustment_cache, orient="table")
                    else:
                        adjustment = pro.adj_factor(
                            ts_code=request_code,
                            start_date=start.strftime("%Y%m%d"),
                            end_date=end.strftime("%Y%m%d"),
                        )
                        if isinstance(adjustment, pd.DataFrame):
                            adjustment_cache.write_text(
                                adjustment.to_json(orient="table", date_format="iso"),
                                encoding="utf-8",
                            )
                    if isinstance(adjustment, pd.DataFrame) and not adjustment.empty:
                        adjustment = adjustment.copy()
                        adjustment["ts_code"] = historical_code
                        result = merge_tushare_adjustment_factor(result, adjustment)
                frames.append(result)
    else:
        for trade_date in open_dates:
            cache_path = resolved_cache / f"daily_{trade_date:%Y%m%d}.json"
            if cache_path.is_file():
                result = pd.read_json(cache_path, orient="table")
            else:
                result = pro.daily(trade_date=trade_date.strftime("%Y%m%d"))
                if isinstance(result, pd.DataFrame):
                    cache_path.write_text(
                        result.to_json(orient="table", date_format="iso"), encoding="utf-8"
                    )
            if isinstance(result, pd.DataFrame) and not result.empty:
                if include_adjusted_fields:
                    adjustment_cache_path = resolved_cache / (
                        f"adj_factor_{trade_date:%Y%m%d}.json"
                    )
                    if adjustment_cache_path.is_file():
                        adjustment = pd.read_json(adjustment_cache_path, orient="table")
                    else:
                        adjustment = pro.adj_factor(
                            trade_date=trade_date.strftime("%Y%m%d")
                        )
                        if isinstance(adjustment, pd.DataFrame):
                            adjustment_cache_path.write_text(
                                adjustment.to_json(orient="table", date_format="iso"),
                                encoding="utf-8",
                            )
                    if isinstance(adjustment, pd.DataFrame) and not adjustment.empty:
                        result = merge_tushare_adjustment_factor(result, adjustment)
                if include_daily_basic:
                    basic_cache_path = resolved_cache / f"daily_basic_{trade_date:%Y%m%d}.json"
                    if basic_cache_path.is_file():
                        basic = pd.read_json(basic_cache_path, orient="table")
                    else:
                        basic = pd.DataFrame()
                    if "total_mv" not in basic.columns:
                        basic = pro.daily_basic(
                            trade_date=trade_date.strftime("%Y%m%d"),
                            fields="ts_code,turnover_rate,total_mv",
                        )
                        if isinstance(basic, pd.DataFrame):
                            basic_cache_path.write_text(
                                basic.to_json(orient="table", date_format="iso"),
                                encoding="utf-8",
                            )
                    if isinstance(basic, pd.DataFrame) and not basic.empty:
                        result = result.merge(
                            basic.reindex(columns=["ts_code", "turnover_rate", "total_mv"]),
                            on="ts_code",
                            how="left",
                        )
                frames.append(result)
    if not frames:
        raise RuntimeError("Tushare daily returned no market rows")

    normalization_codes = tuple(
        sorted(set(map(str, codes)).union((aliases or {}).values()))
    )
    raw_market = pd.concat(frames, ignore_index=True)
    normalized = normalize_tushare_market(
        raw_market,
        codes=normalization_codes,
        include_daily_basic=include_daily_basic,
    )
    normalized = apply_tushare_aliases(normalized, aliases or {})
    if open_dates is not None:
        missing_codes = sorted(set(map(str, codes)).difference(normalized["ts_code"].astype(str)))
        if required_keys is not None:
            missing_codes = sorted(
                set(missing_codes)
                | set(missing_required_market_codes(required_keys, normalized))
            )
        if missing_codes:
            fallback = fetch_missing_code_ranges(
                pro,
                missing_codes,
                start,
                end,
                cache_dir=resolved_cache,
                include_daily_basic=include_daily_basic,
                include_adjusted_fields=include_adjusted_fields,
                aliases=aliases,
            )
            if not fallback.empty:
                raw_market = pd.concat([raw_market, fallback], ignore_index=True)
    return normalize_tushare_market(
        raw_market,
        codes=normalization_codes,
        include_daily_basic=include_daily_basic,
    )


def aggregate_comparisons(comparisons: Sequence[dict[str, object]]) -> dict[str, object]:
    """Aggregate independent stock-batch comparisons without weakening the replay gate."""

    if not comparisons:
        raise ValueError("at least one batch comparison is required")
    reason_codes = sorted(
        {
            str(reason)
            for comparison in comparisons
            for reason in comparison.get("reason_codes", [])
        }
    )
    mismatch_starts = [
        str(comparison["mismatch_start"])
        for comparison in comparisons
        if comparison.get("mismatch_start")
    ]
    mismatch_ends = [
        str(comparison["mismatch_end"])
        for comparison in comparisons
        if comparison.get("mismatch_end")
    ]
    finite_rows = sum(int(comparison["overlap_finite_rows"]) for comparison in comparisons)
    mismatch_rows = sum(int(comparison["mismatch_rows"]) for comparison in comparisons)
    return {
        "source_rows": sum(int(comparison["source_rows"]) for comparison in comparisons),
        "replay_rows": sum(int(comparison["replay_rows"]) for comparison in comparisons),
        "source_finite_rows": sum(
            int(comparison.get("source_finite_rows", comparison["overlap_finite_rows"]))
            for comparison in comparisons
        ),
        "replay_finite_rows": sum(
            int(comparison.get("replay_finite_rows", comparison["overlap_finite_rows"]))
            for comparison in comparisons
        ),
        "overlap_rows": sum(int(comparison["overlap_rows"]) for comparison in comparisons),
        "overlap_finite_rows": finite_rows,
        "mismatch_rows": mismatch_rows,
        "mismatch_date_count": max(
            int(comparison["mismatch_date_count"]) for comparison in comparisons
        ),
        "mismatch_start": min(mismatch_starts) if mismatch_starts else None,
        "mismatch_end": max(mismatch_ends) if mismatch_ends else None,
        "max_abs_diff": max(float(comparison["max_abs_diff"]) for comparison in comparisons),
        "max_rel_diff": max(float(comparison["max_rel_diff"]) for comparison in comparisons),
        "decision_status": (
            "FORMULA_REPLAY_PASS"
            if finite_rows > 0 and mismatch_rows == 0
            else "FORMULA_REPLAY_BLOCKED"
        ),
        "reason_codes": reason_codes,
        "batch_count": len(comparisons),
    }


def run_audit(args: argparse.Namespace, *, client: object, tushare_token: str) -> dict[str, object]:
    definitions = [replay.REPLAY_DEFINITIONS[factor_id] for factor_id in args.factor_id]
    aliases = load_alias_map(args.bj_alias_map) if args.bj_alias_map else {}
    required_fields = required_market_fields(definitions)
    effective_include_daily_basic = args.include_daily_basic or requires_daily_basic(
        required_fields
    )
    market_start = min(
        _calculator_market_start(client, definition, args.start, args.end)
        for definition in definitions
    )
    code_limit = None if args.full_market else args.max_ts_codes
    code_sets = [
        _query_codes(client, definition, args.start, args.end, code_limit)
        for definition in definitions
    ]
    codes = tuple(sorted(set().union(*map(set, code_sets))))
    market_universe_source = "factor_observation_codes"
    if (
        args.full_market
        and requires_full_cross_section(args.factor_id)
        and not args.restrict_to_v2_universe
    ):
        codes = _query_stock_market_codes(client, market_start, args.end)
        market_universe_source = "YesLab.stock_factor_daily"
    if args.restrict_to_v2_universe:
        universe_codes = set(_query_v2_universe_codes(client, args.start, args.end))
        codes = tuple(code for code in codes if code in universe_codes)
    if args.exclude_bj:
        codes = tuple(code for code in codes if not code.endswith(".BJ"))
    if not codes:
        raise RuntimeError("factor universe query returned no securities")
    if not args.full_market:
        codes = codes[: args.max_ts_codes]
    source_frames = {
        definition.factor_id: replay._query_source(
            client, definition, args.start, args.end
        )
        for definition in definitions
    }
    required_key_parts = []
    for frame in source_frames.values():
        values = pd.to_numeric(frame["value"], errors="coerce")
        finite = np.isfinite(values.to_numpy(dtype=float))
        required_key_parts.append(frame.loc[finite, ["trade_date", "ts_code"]])
    required_keys = pd.concat(required_key_parts, ignore_index=True)
    required_keys = required_keys.loc[
        required_keys["ts_code"].astype(str).isin(set(codes))
    ].drop_duplicates()
    open_dates = (
        _query_open_dates(client, market_start, args.end)
        if args.full_market or effective_include_daily_basic
        else None
    )
    market = _fetch_tushare_market(
        codes,
        market_start,
        args.end,
        token=tushare_token,
        open_dates=open_dates,
        cache_dir=args.cache_dir,
        include_daily_basic=effective_include_daily_basic,
        include_adjusted_fields=bool(
            {"open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq"}
            & set(required_fields)
        ),
        aliases=aliases,
        required_keys=required_keys,
    )
    market = apply_tushare_aliases(market, aliases)
    market_codes = tuple(sorted(set(market["ts_code"].astype(str)).intersection(codes)))
    missing_market_codes = tuple(sorted(set(codes).difference(market_codes)))
    if not market_codes:
        raise RuntimeError("Tushare market has no overlap with factor universe")
    force_full_cross_section = args.full_market and requires_full_cross_section(args.factor_id)
    batch_size = (
        len(market_codes)
        if force_full_cross_section
        else args.stock_batch_size if args.full_market else len(market_codes)
    )
    comparisons_by_factor: dict[str, list[dict[str, object]]] = {
        factor_id: [] for factor_id in args.factor_id
    }
    for offset in range(0, len(market_codes), batch_size):
        batch_codes = market_codes[offset : offset + batch_size]
        wrapped = TushareMarketSession(client, market, batch_codes)
        for factor_id in args.factor_id:
            definition = replay.REPLAY_DEFINITIONS[factor_id]
            comparisons_by_factor[factor_id].append(
                replay.compare_replay_frames(
                    replay._query_source(wrapped, definition, args.start, args.end),
                    replay._replay_factor(wrapped, definition, args.start, args.end),
                )
            )
    rows: list[dict[str, object]] = []
    for factor_id in args.factor_id:
        definition = replay.REPLAY_DEFINITIONS[factor_id]
        comparison = aggregate_comparisons(comparisons_by_factor[factor_id])
        source_full = source_frames[factor_id]
        source_full = source_full.loc[
            source_full["ts_code"].astype(str).isin(set(codes))
        ].copy()
        comparison.update(source_market_coverage(source_full, market))
        if missing_market_codes:
            comparison["decision_status"] = "FORMULA_REPLAY_BLOCKED"
            comparison["reason_codes"] = sorted(
                set(comparison["reason_codes"]) | {"TUSHARE_MARKET_MISSING_CODES"}
            )
        if comparison["source_market_uncovered_rows"]:
            comparison["decision_status"] = "FORMULA_REPLAY_BLOCKED"
            comparison["reason_codes"] = sorted(
                set(comparison["reason_codes"]) | {"TUSHARE_SOURCE_MARKET_COVERAGE_GAP"}
            )
        rows.append(
            {
                "factor_id": factor_id,
                "factor_name": definition.factor_name,
                "lookback_days": definition.lookback_days,
                "replay_lookback_days": definition.effective_replay_lookback_days,
                "max_ts_codes": None if args.full_market else args.max_ts_codes,
                **comparison,
            }
        )
    market_hash = hashlib.sha256(
        pd.util.hash_pandas_object(market, index=False).to_numpy(dtype="uint64").tobytes()
    ).hexdigest()
    return {
        "scope": "FORMULA_REPLAY_CONSISTENCY_TUSHARE_MARKET_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "market_start": market_start.isoformat(),
        "factor_ids": list(args.factor_id),
        "max_ts_codes": None if args.full_market else args.max_ts_codes,
        "universe_mode": "full_market" if args.full_market else "bounded_sample",
        "restrict_to_v2_universe": args.restrict_to_v2_universe,
        "exclude_bj": args.exclude_bj,
        "include_daily_basic": effective_include_daily_basic,
        "required_market_fields": list(required_fields),
        "bj_alias_map": str(args.bj_alias_map) if args.bj_alias_map else None,
        "bj_alias_count": len(aliases),
        "code_count": len(codes),
        "market_universe_source": market_universe_source,
        "market_code_count": len(market_codes),
        "market_missing_code_count": len(missing_market_codes),
        "stock_batch_size": batch_size,
        "cross_sectional_replay_mode": (
            "single_complete_batch" if force_full_cross_section else "configured_batch"
        ),
        "market_date_count": int(market["trade_date"].nunique()),
        "market_source": "Tushare.daily",
        "market_rows": len(market),
        "market_finite_rows": int(
            np.isfinite(market.select_dtypes(include=[np.number])).to_numpy().sum()
        ),
        "market_hash": market_hash,
        "database_writes": False,
        "locked_test_read": False,
        "rows": rows,
    }


def _connect() -> object:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(
        args,
        client=_connect(),
        tushare_token=os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, flush=True)
    return payload


if __name__ == "__main__":
    main()
