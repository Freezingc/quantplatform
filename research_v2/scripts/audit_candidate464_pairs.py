"""Read-only pair comparison of yeslab_464_tushare_shsz_v1 against all ten READY candidates.

Materializes the six-candidate matrix (including the alpha206 median repair and the
alpha484 canonical checkpoint), the four extension candidates (363/357/96/443) and the
new GP_095 candidate yeslab_464_tushare_shsz_v1, then computes daily Spearman
similarity for the ten pairs that include the new candidate in both the training and
validation windows. No database writes, no locked-test reads, no admission decisions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date, timedelta
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
from audit_extension_candidate_pairs import (
    EXTENSION_CANDIDATES,
    EXTENSION_SOURCE_FACTOR_IDS,
    _restrict_replay_to_keys,
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

DEFAULT_NEW_CANDIDATE_ID = "yeslab_464_tushare_shsz_v1"
DEFAULT_NEW_CANDIDATE_SOURCE = "yeslab_464"
NEW_CANDIDATE_ID = DEFAULT_NEW_CANDIDATE_ID
NEW_CANDIDATE_SOURCE = DEFAULT_NEW_CANDIDATE_SOURCE
EXTRA_REFERENCE_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("yeslab_464_tushare_shsz_v1", "yeslab_464"),
    ("yeslab_370_tushare_shsz_v1", "yeslab_370"),
    ("yeslab_375_tushare_shsz_v1", "yeslab_375"),
    ("yeslab_389_tushare_shsz_v1", "yeslab_389"),
    ("yeslab_390_tushare_shsz_v1", "yeslab_390"),
    ("yeslab_398_tushare_shsz_v1", "yeslab_398"),
    ("yeslab_471_tushare_pit_v1", "yeslab_471"),
    ("yeslab_395_tushare_shsz_v1", "yeslab_395"),
)
LOCKED_TEST_START = date(2026, 1, 1)
THRESHOLD_MONITOR = 0.80
THRESHOLD_REDUNDANT = 0.90


def _validate_window(start: date, end: date) -> None:
    if start > end:
        raise ValueError("start must not be after end")
    if end >= LOCKED_TEST_START:
        raise ValueError("comparison must end before the 2026 locked test")


def _reference_candidate_ids() -> list[str]:
    others = [c.candidate_id for c in CANDIDATES] + list(EXTENSION_CANDIDATES)
    others.extend(cid for cid, _ in EXTRA_REFERENCE_CANDIDATES)
    return [candidate_id for candidate_id in others if candidate_id != NEW_CANDIDATE_ID]


def _required_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(sorted((NEW_CANDIDATE_ID, other) for other in _reference_candidate_ids()))


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
        replay.REPLAY_DEFINITIONS[NEW_CANDIDATE_SOURCE],
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
    # Safety buffer: some calculators (e.g. AttentionSpillover/yeslab_471) need
    # more warmup than their declared lookback_days; extend the market start so
    # every factor has values from phase_start.
    market_start = market_start - timedelta(days=90)
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
    for extra_candidate_id, extra_source_factor_id in EXTRA_REFERENCE_CANDIDATES:
        if extra_candidate_id == NEW_CANDIDATE_ID or extra_candidate_id in frames:
            continue
        definition = replay.REPLAY_DEFINITIONS[extra_source_factor_id]
        values = replay._replay_factor(session, definition, phase_start, phase_end, industry_provider=industry_provider)
        frames[extra_candidate_id] = _restrict_replay_to_keys(values, keys)
    definition = replay.REPLAY_DEFINITIONS[NEW_CANDIDATE_SOURCE]
    values = replay._replay_factor(session, definition, phase_start, phase_end, industry_provider=industry_provider)
    frames[NEW_CANDIDATE_ID] = _restrict_replay_to_keys(values, keys)
    return frames, keys


def summarize_pairs(frames: Mapping[str, pd.DataFrame], expected_dates: Sequence[date]) -> pd.DataFrame:
    expected = {pd.Timestamp(value).date() for value in expected_dates}
    for _, other in _required_pairs():
        observed = _pair_valid_dates(frames[NEW_CANDIDATE_ID], frames[other])
        if observed != expected:
            missing = sorted(expected.difference(observed))
            unexpected = sorted(observed.difference(expected))
            raise ValueError(f"incomplete pair dates for {NEW_CANDIDATE_ID}/{other}: missing={missing[:5]}..., unexpected={unexpected[:5]}...")
    summary = compute_pair_daily_spearman(dict(frames))
    required = set(_required_pairs())
    selected = summary.loc[
        summary.apply(
            lambda row: (row["factor_id_left"], row["factor_id_right"]) in required
            or (row["factor_id_right"], row["factor_id_left"]) in required,
            axis=1,
        )
    ].copy()
    if len(selected) != len(required):
        raise ValueError("candidate-464 pair summary is incomplete")
    return selected.reset_index(drop=True)


def classify(value: float) -> str:
    if value >= THRESHOLD_REDUNDANT:
        return "REDUNDANT_NO_JOINT_ADMISSION"
    if value >= THRESHOLD_MONITOR:
        return "ELEVATED_PAIR_MONITOR"
    return "PAIR_UNIQUE"


def run_phase(client: object, phase: str, args: argparse.Namespace) -> dict[str, object]:
    if phase == "train":
        start, end, expected_days = TRAIN_START, TRAIN_END, TRAIN_EXPECTED_DAYS
    else:
        start, end, expected_days = VALIDATION_START, VALIDATION_END, VALIDATION_EXPECTED_DAYS
    frames, keys = materialize_phase(
        client,
        phase_start=start,
        phase_end=end,
        cache_dir=args.cache_dir,
        pit_industry_cache=args.pit_industry_cache,
        pit_industry_cutover=args.pit_industry_cutover,
        alpha484_checkpoint_root=args.alpha484_checkpoint_root,
    )
    summary = summarize_pairs(frames, sorted(keys["trade_date"].dt.date.unique()))
    pairs = []
    for row in summary.itertuples(index=False):
        pairs.append(
            {
                "factor_id_left": row.factor_id_left,
                "factor_id_right": row.factor_id_right,
                "median_abs_spearman": float(row.median_abs_spearman),
                "max_abs_spearman": float(row.max_abs_spearman),
                "valid_day_count": int(row.valid_day_count),
            }
        )
    return {
        "phase": phase,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "expected_day_count": expected_days,
        "pairs": pairs,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--pit-industry-cache", type=Path, default=Path("research_v2/artifacts/tushare_cache"))
    parser.add_argument("--pit-industry-cutover", type=date.fromisoformat, default=date(2021, 12, 10))
    parser.add_argument("--alpha484-checkpoint-root", type=Path, default=Path("research_v2/artifacts/checkpoints/alpha484_canonical_v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", default=DEFAULT_NEW_CANDIDATE_ID)
    parser.add_argument("--candidate-source", default=DEFAULT_NEW_CANDIDATE_SOURCE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    global NEW_CANDIDATE_ID, NEW_CANDIDATE_SOURCE
    NEW_CANDIDATE_ID = args.candidate_id
    NEW_CANDIDATE_SOURCE = args.candidate_source
    if NEW_CANDIDATE_SOURCE not in replay.REPLAY_DEFINITIONS:
        raise ValueError(f"unsupported candidate source factor id: {NEW_CANDIDATE_SOURCE}")
    client = _connect()
    phases = ("train", "validation") if args.phase == "all" else (args.phase,)
    payload: dict[str, object] = {
        "scope": "NEW_CANDIDATE_PAIR_SIMILARITY_V1",
        "new_candidate_id": NEW_CANDIDATE_ID,
        "source_factor_id": NEW_CANDIDATE_SOURCE,
        "reference_candidates": _reference_candidate_ids(),
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "threshold_monitor": THRESHOLD_MONITOR,
        "threshold_redundant": THRESHOLD_REDUNDANT,
        "database_writes": False,
        "locked_test_read": False,
    }
    results = {}
    for phase in phases:
        results[phase] = run_phase(client, phase, args)
    payload.update({f"{phase}_pairs": results[phase]["pairs"] for phase in results})
    all_pairs = [p for phase in results for p in results[phase]["pairs"]]
    maximum = max(all_pairs, key=lambda p: p["median_abs_spearman"]) if all_pairs else None
    payload["training_maximum_median_abs_spearman"] = (
        max(results["train"]["pairs"], key=lambda p: p["median_abs_spearman"])["median_abs_spearman"] if "train" in results else None
    )
    payload["validation_maximum_median_abs_spearman"] = (
        max(results["validation"]["pairs"], key=lambda p: p["median_abs_spearman"])["median_abs_spearman"] if "validation" in results else None
    )
    elevated = [
        p for phase in results for p in results[phase]["pairs"]
        if p["median_abs_spearman"] >= THRESHOLD_MONITOR
    ]
    payload["elevated_or_redundant_pairs"] = elevated
    payload["status"] = "PASS" if not elevated else "PASS_WITH_MONITORING"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "maximum_pair": maximum}, ensure_ascii=False))
    return payload


if __name__ == "__main__":
    main()
