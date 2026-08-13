"""Compare Alpha206 median-based repair with the formal nine, read-only."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_alpha206_volume_variants import compute_variants, load_cached_market
from evaluate_alpha165_168_canonical_quality import (
    LOCKED_TEST_START,
    WARMUP_CALENDAR_DAYS,
    _connect,
    _periods,
    load_period_inputs,
)
from record_train_similarity_full_day import (
    _query_resilient,
    daily_similarity_values,
    query_clean_factor_month,
    vectorized_clean_technical_sample,
)
from research_v2.db.session import V2WriteSession, YesLabReadOnlySession
from research_v2.factors.source_backfill import build_factor_definitions

FORMAL_NINE = (
    "yeslab_380",
    "yeslab_431",
    "yeslab_437",
    "yeslab_155",
    "yeslab_98",
    "yeslab_94",
    "yeslab_376",
    "yeslab_396",
    "yeslab_442",
)
CANDIDATE = "yeslab_206_median_abs_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("similarity range must be before 2026 locked-test")
    return args


def _candidate_observations(market: pd.DataFrame, dates: Sequence[pd.Timestamp]) -> pd.DataFrame:
    panel = compute_variants(market)["median_abs"]
    selected = panel.reindex(index=pd.to_datetime(list(dates)))
    rows = selected.stack(dropna=True).rename("value").reset_index()
    rows.columns = ["signal_date", "ts_code", "value"]
    rows["signal_date"] = pd.to_datetime(rows["signal_date"], errors="raise").dt.date
    rows["factor_id"] = CANDIDATE
    rows["factor_version"] = 2
    return rows.loc[:, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]]


def run_evaluation(args: argparse.Namespace, *, client: object | None = None) -> dict[str, object]:
    own_client = client is None
    client = client or _connect()
    target = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    source = YesLabReadOnlySession(client)
    metadata = source.query(
        "select factor_id,factor_name,creator,update_frequency,store_factor,category "
        "from loadTable('dfs://YesLab','factor_metadata')"
    )
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("YesLab metadata query did not return a DataFrame")
    definitions = build_factor_definitions(metadata, FORMAL_NINE)
    grouped: dict[str, list[object]] = {}
    for definition in definitions:
        grouped.setdefault(definition.source_table, []).append(definition)
    pair_values: dict[tuple[str, str], list[float]] = {}
    candidate_rows = 0
    formal_rows = 0
    valid_dates = 0
    for period in _periods(args.start, args.end):
        period_start = pd.Period(period, freq="M").start_time.date()
        period_end = pd.Period(period, freq="M").end_time.date()
        labels, panel_codes = load_period_inputs(client, period)
        signal_dates = tuple(sorted(pd.to_datetime(labels["signal_date"]).dt.date.unique()))
        market = load_cached_market(
            args.cache_dir,
            period_start - timedelta(days=WARMUP_CALENDAR_DAYS),
            period_end,
            panel_codes,
        )
        candidate = _candidate_observations(market, signal_dates)
        formal = query_clean_factor_month(
            target,
            factor_ids=FORMAL_NINE,
            start=signal_dates[0],
            end=signal_dates[-1],
        )
        clean_ids = set(formal["factor_id"].astype(str))
        missing_ids = tuple(factor_id for factor_id in FORMAL_NINE if factor_id not in clean_ids)
        if missing_ids:
            fallback_parts: list[pd.DataFrame] = []
            for table in sorted(grouped):
                entries = [definition for definition in grouped[table] if definition.factor_id in missing_ids]
                if not entries:
                    continue
                raw = _query_resilient(
                    source,
                    table=table,
                    factor_names=tuple(definition.display_name for definition in entries),
                    dates=signal_dates,
                )
                fallback_parts.append(vectorized_clean_technical_sample(raw, tuple(entries)))
            if fallback_parts:
                formal = pd.concat([formal, *fallback_parts], ignore_index=True)
        formal = formal.loc[formal["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
        candidate = candidate.loc[candidate["ts_code"].astype(str).str.endswith((".SH", ".SZ"))].copy()
        observations = pd.concat([formal, candidate], ignore_index=True)
        for pair, values in daily_similarity_values(observations, (*FORMAL_NINE, CANDIDATE)).items():
            pair_values.setdefault(pair, []).extend(values)
        candidate_rows += len(candidate)
        formal_rows += len(formal)
        valid_dates += len(signal_dates)
        print(f"similarity month={period} progress rows={len(observations)}", flush=True)
        del market, labels, formal, candidate, observations

    rows: list[dict[str, object]] = []
    for factor_id in FORMAL_NINE:
        values = pair_values.get((factor_id, CANDIDATE), [])
        if not values:
            values = pair_values.get((CANDIDATE, factor_id), [])
        median = float(np.median(values)) if values else None
        rows.append(
            {
                "candidate_factor_id": CANDIDATE,
                "reference_factor_id": factor_id,
                "median_daily_abs_spearman": median,
                "valid_days": len(values),
                "threshold": 0.90,
                "decision_status": (
                    "INSUFFICIENT_COVERAGE"
                    if not values
                    else "REDUNDANT"
                    if median is not None and median >= 0.90
                    else "UNIQUE"
                ),
            }
        )
    pairs = pd.DataFrame(rows).sort_values("reference_factor_id", kind="mergesort")
    comparable = pairs.loc[pairs["median_daily_abs_spearman"].notna()]
    nearest = comparable.sort_values("median_daily_abs_spearman", ascending=False).iloc[0] if not comparable.empty else None
    payload = {
        "scope": "ALPHA206_MEDIAN_ABS_FULL_TRAIN_SIMILARITY_V1",
        "candidate_factor_id": CANDIDATE,
        "source_factor_id": "yeslab_206",
        "formula": "ABS(vol - rolling_median(vol,60)) / rolling_median(vol,60)",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "period": f"{args.start.isoformat()}/{args.end.isoformat()}",
        "formal_factor_ids": list(FORMAL_NINE),
        "candidate_rows": int(candidate_rows),
        "formal_rows": int(formal_rows),
        "valid_date_count": int(valid_dates),
        "threshold": 0.90,
        "nearest_reference_factor_id": str(nearest["reference_factor_id"]) if nearest is not None else None,
        "nearest_median_abs_spearman": float(nearest["median_daily_abs_spearman"]) if nearest is not None else None,
        "nearest_valid_days": int(nearest["valid_days"]) if nearest is not None else 0,
        "decision_status": (
            "INSUFFICIENT_COVERAGE"
            if nearest is None
            else "REDUNDANT"
            if float(nearest["median_daily_abs_spearman"]) >= 0.90
            else "UNIQUE"
        ),
        "database_writes": False,
        "locked_test_read": False,
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output.with_suffix(".csv"), index=False)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if own_client:
        client.close()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps(run_evaluation(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
