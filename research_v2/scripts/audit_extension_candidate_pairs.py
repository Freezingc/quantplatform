"""Read-only supplemental candidate-pair comparison for extension candidates."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from itertools import combinations, product
from pathlib import Path

import audit_formula_replay_consistency as replay
import pandas as pd
from audit_canonical_candidate_pairs import (
    CANDIDATES,
    TRAIN_END,
    TRAIN_EXPECTED_DAYS,
    TRAIN_START,
    VALIDATION_END,
    VALIDATION_EXPECTED_DAYS,
    VALIDATION_START,
    _pair_valid_dates,
    load_alpha484_checkpoint_values,
    materialize_alpha206_median_repair,
)
from audit_formula_replay_tushare import (
    TushareMarketSession,
    _calculator_market_start,
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
    required_market_fields,
    requires_daily_basic,
)
from audit_tushare_pit_industry import PITIndustryProvider, VersionedPITIndustryProvider
from evaluate_tushare_scope_validation import (
    _load_validation_inputs,
    compute_pair_daily_spearman,
)

EXTENSION_CANDIDATES: tuple[str, ...] = (
    "yeslab_363_tushare_shsz_v1",
    "yeslab_357_tushare_shsz_v1",
    "yeslab_96_canonical_replay_v1",
    "yeslab_443_tushare_shsz_v1",
)
EXTENSION_SOURCE_FACTOR_IDS: Mapping[str, str] = {
    "yeslab_363_tushare_shsz_v1": "yeslab_363",
    "yeslab_357_tushare_shsz_v1": "yeslab_357",
    "yeslab_96_canonical_replay_v1": "yeslab_96",
    "yeslab_443_tushare_shsz_v1": "yeslab_443",
}
LOCKED_TEST_START = date(2026, 1, 1)


def _validate_window(start: date, end: date) -> None:
    if start > end:
        raise ValueError("start must not be after end")
    if end >= LOCKED_TEST_START:
        raise ValueError("supplemental comparison must end before the 2026 locked test")


def _required_pairs() -> tuple[tuple[str, str], ...]:
    extension_base = product(sorted(EXTENSION_CANDIDATES), [c.candidate_id for c in CANDIDATES])
    extension_internal = combinations(sorted(EXTENSION_CANDIDATES), 2)
    return tuple(sorted((*extension_base, *extension_internal)))


def _restrict_replay_to_keys(
    replay_values: pd.DataFrame,
    comparison_keys: pd.DataFrame,
) -> pd.DataFrame:
    left = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    left["trade_date"] = pd.to_datetime(left["trade_date"], errors="raise").dt.normalize()
    left["ts_code"] = left["ts_code"].astype(str)
    right = comparison_keys.loc[:, ["trade_date", "ts_code"]].copy()
    right["trade_date"] = pd.to_datetime(right["trade_date"], errors="raise").dt.normalize()
    right["ts_code"] = right["ts_code"].astype(str)
    merged = left.merge(right, on=["trade_date", "ts_code"], how="inner", validate="one_to_one")
    return merged.drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def materialize_phase(
    client: object,
    *,
    phase_start: date,
    phase_end: date,
    cache_dir: Path,
    pit_industry_cache: Path | None,
    pit_industry_cutover: date | None,
    alpha484_checkpoint_root: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    _validate_window(phase_start, phase_end)
    inputs = _load_validation_inputs(client, phase_start, phase_end)
    keys = inputs.loc[:, ["signal_date", "ts_code"]].rename(columns={"signal_date": "trade_date"})
    definitions = [
        replay.REPLAY_DEFINITIONS[EXTENSION_SOURCE_FACTOR_IDS[candidate]]
        for candidate in EXTENSION_CANDIDATES
    ] + [
        replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
        for candidate in CANDIDATES
        if not candidate.is_alpha206_median_repair and not candidate.is_alpha484_tushare_checkpoint
    ]
    alpha206_definition = replay.REPLAY_DEFINITIONS["yeslab_206"]
    market_start = min(
        _calculator_market_start(client, definition, phase_start, phase_end)
        for definition in (*definitions, alpha206_definition)
    )
    open_dates = _query_open_dates(client, market_start, phase_end)
    required_fields = required_market_fields(definitions)
    market = fetch_full_tushare_shsz_market(
        open_dates,
        token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=cache_dir,
        include_daily_basic=requires_daily_basic(required_fields),
    )
    codes = tuple(sorted(market["ts_code"].astype(str).unique()))
    session = TushareMarketSession(client, market, codes)
    industry_provider = None
    if pit_industry_cache is not None:
        industry_provider = (
            PITIndustryProvider.from_cache(pit_industry_cache)
            if pit_industry_cutover is None
            else VersionedPITIndustryProvider.from_cache(pit_industry_cache, cutover=pit_industry_cutover)
        )
    alpha484_values = load_alpha484_checkpoint_values(alpha484_checkpoint_root)
    frames: dict[str, pd.DataFrame] = {}
    for candidate in CANDIDATES:
        if candidate.is_alpha206_median_repair:
            values = materialize_alpha206_median_repair(market, phase_start, phase_end)
        elif candidate.is_alpha484_tushare_checkpoint:
            values = alpha484_values
        else:
            definition = replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
            values = replay._replay_factor(session, definition, phase_start, phase_end, industry_provider=industry_provider)
        frames[candidate.candidate_id] = _restrict_replay_to_keys(values, keys)
    for extension_candidate in EXTENSION_CANDIDATES:
        definition = replay.REPLAY_DEFINITIONS[EXTENSION_SOURCE_FACTOR_IDS[extension_candidate]]
        values = replay._replay_factor(session, definition, phase_start, phase_end, industry_provider=industry_provider)
        frames[extension_candidate] = _restrict_replay_to_keys(values, keys)
    return frames, keys


def summarize_extension_pairs(frames: Mapping[str, pd.DataFrame], expected_dates: Sequence[date]) -> pd.DataFrame:
    expected = {pd.Timestamp(value).date() for value in expected_dates}
    rows = []
    for extension_candidate, base_candidate in _required_pairs():
        observed = _pair_valid_dates(frames[extension_candidate], frames[base_candidate])
        if observed != expected:
            missing = sorted(expected.difference(observed))
            unexpected = sorted(observed.difference(expected))
            raise ValueError(f"incomplete extension pair dates for {extension_candidate}/{base_candidate}: missing={missing}, unexpected={unexpected}")
    pairs = {
        candidate_id: frames[candidate_id]
        for candidate_id in [*EXTENSION_CANDIDATES, *[candidate.candidate_id for candidate in CANDIDATES]]
    }
    summary = compute_pair_daily_spearman(pairs)
    required = set(_required_pairs())
    selected = summary.loc[
        summary.apply(
            lambda row: (row["factor_id_left"], row["factor_id_right"]) in required
            or (row["factor_id_right"], row["factor_id_left"]) in required,
            axis=1,
        )
    ].copy()
    if len(selected) != len(required):
        raise ValueError("extension pair summary is incomplete")
    return selected.reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--pit-industry-cache", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--pit-industry-cutover", type=date.fromisoformat, default=date(2021, 12, 10))
    parser.add_argument("--alpha484-checkpoint-root", type=Path, default=Path("research_v2/artifacts/checkpoints/alpha484_canonical_v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pit_industry_cutover is not None and args.pit_industry_cutover >= LOCKED_TEST_START:
        parser.error("industry cutover must be before 2026")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    client = _connect()
    payload: dict[str, object] = {
        "scope": "SUPPLEMENTAL_CANDIDATE_PAIR_SIMILARITY_V1",
        "extension_candidates": list(EXTENSION_CANDIDATES),
        "base_candidates": [candidate.candidate_id for candidate in CANDIDATES],
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "threshold_monitor": 0.80,
        "threshold_redundant": 0.90,
        "database_writes": False,
        "locked_test_read": False,
    }
    phases = ("train", "validation") if args.phase == "all" else (args.phase,)
    for phase in phases:
        phase_start, phase_end, expected_days = (
            (TRAIN_START, TRAIN_END, TRAIN_EXPECTED_DAYS)
            if phase == "train"
            else (VALIDATION_START, VALIDATION_END, VALIDATION_EXPECTED_DAYS)
        )
        frames, keys = materialize_phase(
            client,
            phase_start=phase_start,
            phase_end=phase_end,
            cache_dir=args.cache_dir,
            pit_industry_cache=args.pit_industry_cache,
            pit_industry_cutover=args.pit_industry_cutover,
            alpha484_checkpoint_root=args.alpha484_checkpoint_root,
        )
        summary = summarize_extension_pairs(frames, sorted(keys["trade_date"].dt.date.unique()))
        payload[f"{phase}_pairs"] = summary.to_dict(orient="records")
        payload[f"{phase}_expected_days"] = expected_days
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
