"""Replay the frozen v2 factor set for an operational window and write it to parquet.

WHY THIS EXISTS SEPARATELY

``materialize_frozen_factor_daily.py`` writes ``factor_daily`` -- the frozen trust root -- and
therefore must not be pointed at 2026: it fails loudly on any window at or after 2026-01-01, and
the shared calendar helper it reads silently drops those dates as well. Both guards exist to stop
the locked 2026 segment being touched by anything that feeds selection or tuning. They are right
and this script leaves them alone.

What the owner asked for is the opposite operation: score dates *after* the frozen window with
models that are already trained and frozen. The training window ends 2024-12-23 and the
validation year is 2025, so replaying 2026 cannot leak into any model's fit -- but it *could*
leak into a gate verdict if it were written somewhere an evaluation reads. This script therefore
writes a plain parquet file at a caller-chosen path, never ``factor_daily``, and records the
window and factor ids beside it so a consumer can see exactly what it is holding.

The replay itself is the frozen one: the same calculators, the same canonical repair variants,
the same Tushare market source and the same official ``cn_liquid_a`` universe restriction. That
is what makes the output comparable to the values the frozen models were trained on; a
"recompute it another way" path would silently change the model's inputs.

``--cross-check-factor-daily`` replays a window that overlaps the frozen table and reports how
far the replayed values sit from the stored ones. Run it before trusting a new window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "research_v2" / "src"))

FROZEN_V2_PATH = (
    REPO_ROOT / "research_v2" / "artifacts" / "diagnostics" / "factor_set_frozen_v2.json"
)
LONG_COLUMNS = [
    "signal_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "value",
    "available_date",
    "universe_id",
]


def load_frozen_v2_factors() -> tuple[str, ...]:
    """The owner-approved 15-factor v2 set; materializing anything else is not supported."""
    payload = json.loads(FROZEN_V2_PATH.read_text(encoding="utf-8"))
    return tuple(str(factor) for factor in payload["factors"])


def require_frozen_factors(factor_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Fail closed on ids outside the frozen v2 set, naming every offender at once."""
    frozen = set(load_frozen_v2_factors())
    unknown = sorted(set(factor_ids).difference(frozen))
    if unknown:
        raise ValueError(
            f"factor ids are not part of the frozen v2 set: {unknown}. Materializing them would "
            f"produce a factor set no frozen model consumes."
        )
    if not factor_ids:
        raise ValueError("at least one factor id is required")
    # Keep the requested order but drop repeats, so one factor is replayed once.
    return tuple(dict.fromkeys(factor_ids))


def window_open_dates(
    raw_dates: list[date] | tuple[date, ...], start: date, end: date
) -> tuple[date, ...]:
    """Trading dates inside ``[start, end]`` edit-free.

    This is the one place that deliberately differs from the frozen path: the shared helper
    ``_query_open_dates`` filters out ``cal_date >= 2026-01-01`` without saying so, which turns a
    2026 window into a silent empty result. Here the caller's window is honoured exactly, and an
    empty result is an error rather than a no-op.
    """
    selected = sorted({day for day in raw_dates if start <= day <= end})
    if not selected:
        raise ValueError(
            f"no open trading dates in {start} .. {end}: the calendar returned "
            f"{len(raw_dates)} dates, none inside the window"
        )
    return tuple(selected)


def assemble_long_frame(
    replay: pd.DataFrame,
    universe_by_date: dict[date, set[str]],
    *,
    factor_id: str,
    factor_version: int,
    universe_id: str = "cn_liquid_a",
) -> pd.DataFrame:
    """Restrict a replayed cross-section to official universe members with finite values.

    Produces the same schema ``factor_daily`` carries for canonical replay values, minus the
    storage bookkeeping (run ids, watermarks), plus ``available_date`` -- which for a canonical
    replay value is its own signal date, matching what the fold extractor stamps when it reads
    ``factor_daily``.
    """
    required = {"trade_date", "ts_code", "replay_value"}
    missing = sorted(required.difference(replay.columns))
    if missing:
        raise ValueError(f"replay frame is missing columns: {missing}")
    values = replay.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    values["signal_date"] = pd.to_datetime(values["trade_date"]).dt.normalize().dt.date
    values["ts_code"] = values["ts_code"].astype(str)
    values = values.loc[:, ["signal_date", "ts_code", "replay_value"]]
    if values.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError(
            f"replay produced duplicate business keys for {factor_id}; the calculator is not "
            f"emitting one value per (signal_date, ts_code)"
        )
    keep = [
        (code in universe_by_date.get(day, set())) and np.isfinite(value)
        for day, code, value in zip(
            values["signal_date"], values["ts_code"], values["replay_value"], strict=True
        )
    ]
    values = values.loc[keep]
    assembled = pd.DataFrame(
        {
            "signal_date": values["signal_date"].tolist(),
            "ts_code": values["ts_code"].tolist(),
            "factor_id": factor_id,
            "factor_version": factor_version,
            "value": values["replay_value"].astype(float).tolist(),
            "available_date": values["signal_date"].tolist(),
            "universe_id": universe_id,
        }
    )
    return assembled.loc[:, LONG_COLUMNS].reset_index(drop=True)


def compare_against_reference(
    replay: pd.DataFrame, reference: pd.DataFrame, *, tolerance: float
) -> dict[str, float | int]:
    """How closely a replayed window reproduces a stored one, on shared business keys."""
    keys = ["signal_date", "ts_code"]
    left = replay.loc[:, [*keys, "value"]].rename(columns={"value": "replay_value"})
    right = reference.loc[:, [*keys, "value"]].rename(columns={"value": "reference_value"})
    merged = left.merge(right, on=keys, how="outer", indicator=True)
    matched = merged.loc[merged["_merge"].eq("both")]
    difference = (matched["replay_value"] - matched["reference_value"]).abs()
    return {
        "matched_rows": int(len(matched)),
        "replay_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "reference_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "within_tolerance_rows": int((difference <= tolerance).sum()),
        "max_abs_diff": float(difference.max()) if len(difference) else 0.0,
    }


def _query_all_open_dates(client: object, start: date, end: date) -> tuple[date, ...]:
    """Full SSE calendar over the window, with no locked-test filter applied."""
    result = client.run(
        "select cal_date from loadTable('dfs://YesLab','trade_calendar') "
        "where exchange='SSE' and is_open == 1 "
        f"and cal_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}') "
        "order by cal_date"
    )
    if not isinstance(result, pd.DataFrame) or "cal_date" not in result.columns:
        raise RuntimeError("trade calendar query did not return cal_date")
    return tuple(
        sorted(pd.to_datetime(result["cal_date"], errors="coerce").dropna().dt.date.tolist())
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True, help="parquet path to write")
    parser.add_argument(
        "--factors", type=str, default=None, help="comma-separated subset (default all frozen v2)"
    )
    parser.add_argument(
        "--cross-check-factor-daily",
        action="store_true",
        help="also read the frozen factor_daily rows for this window and report agreement",
    )
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    if args.start > args.end:
        raise SystemExit("--start must not be after --end")

    requested = tuple(args.factors.split(",")) if args.factors else load_frozen_v2_factors()
    factors = require_frozen_factors(requested)

    import audit_formula_replay_consistency as replay
    import audit_formula_replay_tushare as tushare_replay
    import materialize_frozen_factor_daily as frozen_mod
    from audit_tushare_pit_industry import VersionedPITIndustryProvider

    client = tushare_replay._connect()
    definitions = [
        replay.REPLAY_DEFINITIONS[factor]
        for factor in factors
        if factor not in (frozen_mod.MEDIAN_206, frozen_mod.REPAIR_VARIANT_443)
    ]
    market_start = args.start - timedelta(days=120)
    if definitions:
        market_start = min(
            tushare_replay._calculator_market_start(client, definition, args.start, args.end)
            for definition in definitions
        ) - timedelta(days=90)
    fetched_dates = _query_all_open_dates(client, market_start, args.end)
    open_dates = window_open_dates(fetched_dates, args.start, args.end)
    print(
        f"window {args.start} .. {args.end}: {len(open_dates)} signal dates "
        f"({open_dates[0]} .. {open_dates[-1]}), market warm-up from {market_start}",
        flush=True,
    )

    fields = tushare_replay.required_market_fields(definitions) if definitions else ()
    fields = tuple(sorted(set(fields) | {"amount", "pct_chg"}))
    market = tushare_replay.fetch_full_tushare_shsz_market(
        fetched_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=frozen_mod.CACHE,
        include_daily_basic=tushare_replay.requires_daily_basic(fields),
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["ts_code"] = market["ts_code"].astype(str)
    print(f"market rows: {len(market)} over {market['trade_date'].nunique()} dates", flush=True)

    session = tushare_replay.TushareMarketSession(
        client, market, tuple(sorted(market["ts_code"].unique()))
    )
    provider = VersionedPITIndustryProvider.from_cache(
        frozen_mod.CACHE, cutover=frozen_mod.PIT_CUTOVER
    )
    universe = frozen_mod.load_official_universe(client, args.start, args.end)
    print(f"universe dates: {len(universe)}", flush=True)

    frames: list[pd.DataFrame] = []
    per_factor_rows: dict[str, int] = {}
    for factor in factors:
        values = frozen_mod.replay_factor_values(
            factor, session, market, provider, args.start, args.end
        )
        assembled = assemble_long_frame(
            values, universe, factor_id=factor, factor_version=1
        )
        per_factor_rows[factor] = int(len(assembled))
        print(f"{factor}: {len(assembled)} universe-aligned finite rows", flush=True)
        frames.append(assembled)

    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=LONG_COLUMNS)
    )
    result = result.sort_values(
        ["signal_date", "ts_code", "factor_id"], kind="mergesort"
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)

    sidecar = {
        "scope": "OPERATIONAL_FACTOR_REPLAY_V1",
        "note": "operational scoring inputs; not the frozen factor_daily trust root",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "factor_ids": list(factors),
        "factor_version": 1,
        "universe_id": frozen_mod.UNIVERSE_ID,
        "signal_dates": len(open_dates),
        "rows": int(len(result)),
        "per_factor_rows": per_factor_rows,
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "touched_locked_test": any(day >= date(2026, 1, 1) for day in open_dates),
    }
    if args.cross_check_factor_daily:
        sidecar["cross_check"] = _cross_check(
            client, result, factors, args.start, args.end, args.tolerance
        )
    sidecar_path = args.output.with_suffix(".meta.json")
    sidecar_path.write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {len(result)} rows to {args.output}", flush=True)
    print(f"wrote metadata to {sidecar_path}", flush=True)
    if "cross_check" in sidecar:
        for factor, summary in sidecar["cross_check"].items():
            print(f"  cross-check {factor}: {summary}", flush=True)
    client.close()
    return 0


def _cross_check(
    client: object,
    result: pd.DataFrame,
    factors: tuple[str, ...],
    start: date,
    end: date,
    tolerance: float,
) -> dict[str, dict[str, float | int]]:
    """Compare the replay against the frozen factor_daily rows on the same window."""
    import materialize_frozen_factor_daily as frozen_mod

    quoted_ids = "','".join(factors)
    reference = client.run(
        "select signal_date, ts_code, factor_id, value "
        f"from loadTable('{frozen_mod.TARGET_DATABASE}','factor_daily') "
        f"where signal_date between {start:%Y.%m.%d} and {end:%Y.%m.%d} "
        f"and factor_id in ['{quoted_ids}']"
    )
    if reference is None or not isinstance(reference, pd.DataFrame):
        return {}
    reference = reference.copy()
    reference["signal_date"] = pd.to_datetime(reference["signal_date"]).dt.normalize().dt.date
    reference["ts_code"] = reference["ts_code"].astype(str)
    summaries: dict[str, dict[str, float | int]] = {}
    for factor in factors:
        replayed = result.loc[result["factor_id"].eq(factor), ["signal_date", "ts_code", "value"]]
        stored = reference.loc[
            reference["factor_id"].astype(str).eq(factor), ["signal_date", "ts_code", "value"]
        ]
        summaries[factor] = compare_against_reference(
            replayed, stored, tolerance=tolerance
        )
    return summaries


if __name__ == "__main__":
    raise SystemExit(main())
