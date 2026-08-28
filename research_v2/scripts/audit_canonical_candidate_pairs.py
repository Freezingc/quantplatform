"""Define the read-only canonical candidate-pair comparison contract."""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations
from pathlib import Path

import audit_formula_replay_consistency as replay
import numpy as np
import pandas as pd
from audit_formula_replay_tushare import (
    TushareMarketSession,
    _calculator_market_start,
    _fetch_tushare_market,
    _query_open_dates,
    required_market_fields,
    requires_daily_basic,
)
from evaluate_alpha206_volume_variants import compute_variants
from evaluate_tushare_scope_validation import (
    _load_validation_inputs,
    compute_pair_daily_spearman,
)

TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)
TRAIN_EXPECTED_DAYS = 1212
VALIDATION_EXPECTED_DAYS = 243


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    source_factor_id: str
    is_alpha206_median_repair: bool = False


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("yeslab_114_tushare_shsz_v2", "yeslab_114"),
    CandidateSpec("yeslab_381_tushare_shsz_v1", "yeslab_381"),
    CandidateSpec("yeslab_405_tushare_shsz_v1", "yeslab_405"),
    CandidateSpec("yeslab_425_tushare_shsz_v2", "yeslab_425"),
    CandidateSpec("yeslab_484_tushare_shsz_v1", "yeslab_484"),
    CandidateSpec("yeslab_206_median_abs_v2", "yeslab_206", is_alpha206_median_repair=True),
)


def _candidate_ids() -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in CANDIDATES)


def _source_definitions() -> tuple[replay.ReplayDefinition, ...]:
    return tuple(
        replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
        for candidate in CANDIDATES
        if not candidate.is_alpha206_median_repair
    )


def _validate_window(
    start: date | pd.Timestamp, end: date | pd.Timestamp
) -> tuple[pd.Timestamp, pd.Timestamp]:
    normalized_start = pd.Timestamp(start).normalize()
    normalized_end = pd.Timestamp(end).normalize()
    if normalized_start > normalized_end:
        raise ValueError("comparison start must not be after end")
    if normalized_end.date() >= LOCKED_TEST_START:
        raise ValueError("comparison range must end before the 2026 locked test")
    return normalized_start, normalized_end


def _comparison_keys(comparison_keys: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "ts_code"}
    if missing := sorted(required.difference(comparison_keys.columns)):
        raise ValueError(f"comparison keys are missing columns: {missing}")
    keys = comparison_keys.loc[:, ["signal_date", "ts_code"]].copy()
    keys["trade_date"] = pd.to_datetime(keys.pop("signal_date"), errors="raise").dt.normalize()
    keys["ts_code"] = keys["ts_code"].astype(str)
    if keys["trade_date"].dt.date.ge(LOCKED_TEST_START).any():
        raise ValueError("comparison keys reach the 2026 locked test")
    if not keys["ts_code"].str.endswith((".SH", ".SZ")).all():
        raise ValueError("comparison keys must contain only SH/SZ codes")
    if keys.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("comparison keys contain duplicate business keys")
    return keys.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)


def load_comparison_keys(client: object, start: date, end: date) -> pd.DataFrame:
    """Read and normalize the V2 member plus valid-label comparison panel."""
    _validate_window(start, end)
    return _comparison_keys(_load_validation_inputs(client, start, end))


def _normalize_replay_frame(replay_values: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "replay_value"}
    if missing := sorted(required.difference(replay_values.columns)):
        raise ValueError(f"replay values are missing columns: {missing}")
    frame = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["replay_value"] = pd.to_numeric(frame["replay_value"], errors="coerce")
    if frame["trade_date"].dt.date.ge(LOCKED_TEST_START).any():
        raise ValueError("replay values reach the 2026 locked test")
    if frame.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("replay values contain duplicate business keys")
    return frame


def _restrict_replay_to_month_keys(
    replay_values: pd.DataFrame,
    comparison_keys: pd.DataFrame,
    month_start: date | pd.Timestamp,
    month_end: date | pd.Timestamp,
) -> pd.DataFrame:
    start, end = _validate_window(month_start, month_end)
    keys = _comparison_keys(comparison_keys)
    keys = keys.loc[keys["trade_date"].between(start, end)].copy()
    if keys.empty:
        raise ValueError("comparison keys have no overlap with the requested month")
    values = _normalize_replay_frame(replay_values)
    values = values.loc[values["trade_date"].between(start, end)]
    restricted = keys.merge(
        values,
        on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )
    finite = np.isfinite(restricted["replay_value"].to_numpy(dtype=float))
    if not finite.all():
        missing = int((~finite).sum())
        raise ValueError(f"replay values are missing or non-finite for {missing} comparison keys")
    return restricted.loc[:, ["trade_date", "ts_code", "replay_value"]].reset_index(drop=True)


def materialize_alpha206_median_repair(
    market: pd.DataFrame,
    month_start: date | pd.Timestamp,
    month_end: date | pd.Timestamp,
) -> pd.DataFrame:
    """Materialize only Alpha206's approved median-deviation repair."""
    start, end = _validate_window(month_start, month_end)
    panel = compute_variants(market)["median_abs"]
    replay_values = (
        panel.rename_axis("trade_date")
        .reset_index()
        .melt(id_vars=["trade_date"], var_name="ts_code", value_name="replay_value")
    )
    replay_values = _normalize_replay_frame(replay_values)
    return replay_values.loc[
        replay_values["trade_date"].between(start, end),
        ["trade_date", "ts_code", "replay_value"],
    ].reset_index(drop=True)


def load_candidate_month_market(
    client: object,
    *,
    comparison_codes: Sequence[str],
    month_start: date,
    month_end: date,
    cache_dir: Path,
    token: str | None = None,
) -> pd.DataFrame:
    """Load one pre-2026 Tushare month with calculator-specific warmup."""
    _validate_window(month_start, month_end)
    codes = tuple(sorted({str(code) for code in comparison_codes}))
    if not codes or not all(code.endswith((".SH", ".SZ")) for code in codes):
        raise ValueError("comparison code union must contain only SH/SZ codes")
    definitions = _source_definitions()
    alpha206_definition = replay.REPLAY_DEFINITIONS["yeslab_206"]
    market_start = min(
        _calculator_market_start(client, definition, month_start, month_end)
        for definition in (*definitions, alpha206_definition)
    )
    open_dates = _query_open_dates(client, market_start, month_end)
    market_fields = required_market_fields(definitions)
    return _fetch_tushare_market(
        codes,
        market_start,
        month_end,
        token=token if token is not None else os.getenv("QRV2_TUSHARE_TOKEN", ""),
        open_dates=open_dates,
        cache_dir=cache_dir,
        include_daily_basic=requires_daily_basic(market_fields),
        include_adjusted_fields=bool(
            {"open_hfq", "high_hfq", "low_hfq", "close_hfq", "pre_close_hfq"}
            & set(market_fields)
        ),
    )


def materialize_candidate_month(
    client: object,
    market: pd.DataFrame,
    comparison_keys: pd.DataFrame,
    *,
    comparison_codes: Sequence[str],
    month_start: date,
    month_end: date,
) -> dict[str, pd.DataFrame]:
    """Replay all six candidates, emitting only the month's V2 comparison keys."""
    _validate_window(month_start, month_end)
    codes = tuple(sorted({str(code) for code in comparison_codes}))
    if not codes or not all(code.endswith((".SH", ".SZ")) for code in codes):
        raise ValueError("comparison code union must contain only SH/SZ codes")
    session = TushareMarketSession(client, market, codes)
    frames: dict[str, pd.DataFrame] = {}
    for candidate in CANDIDATES:
        if candidate.is_alpha206_median_repair:
            replay_values = materialize_alpha206_median_repair(market, month_start, month_end)
        else:
            definition = replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
            replay_values = replay._replay_factor(session, definition, month_start, month_end)
        frames[candidate.candidate_id] = _restrict_replay_to_month_keys(
            replay_values,
            comparison_keys,
            month_start,
            month_end,
        )
    return frames


def summarize_phase_pairs(
    replay_frames: Mapping[str, pd.DataFrame], expected_days: int
) -> pd.DataFrame:
    """Compute and fail closed on the complete fixed candidate-pair matrix."""
    if expected_days < 1:
        raise ValueError("expected_days must be positive")
    candidate_ids = _candidate_ids()
    supplied_ids = set(replay_frames)
    expected_ids = set(candidate_ids)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids.difference(supplied_ids))
        unexpected = sorted(supplied_ids.difference(expected_ids))
        raise ValueError(
            "incomplete candidate pair matrix: "
            f"missing_candidates={missing}, unexpected_candidates={unexpected}"
        )
    frames = {
        candidate_id: _normalize_replay_frame(replay_frames[candidate_id])
        for candidate_id in candidate_ids
    }
    summary = compute_pair_daily_spearman(frames)
    pair_columns = ["factor_id_left", "factor_id_right"]
    expected_pairs = set(combinations(sorted(candidate_ids), 2))
    actual_pairs = set(map(tuple, summary.loc[:, pair_columns].itertuples(index=False, name=None)))
    if len(summary) != len(actual_pairs) or actual_pairs != expected_pairs:
        missing = sorted(expected_pairs.difference(actual_pairs))
        unexpected = sorted(actual_pairs.difference(expected_pairs))
        raise ValueError(
            "incomplete candidate pair matrix: "
            f"missing_pairs={missing}, unexpected_pairs={unexpected}"
        )
    days = pd.to_numeric(summary["valid_day_count"], errors="coerce")
    if not days.eq(expected_days).all():
        incomplete = summary.loc[
            ~days.eq(expected_days), pair_columns + ["valid_day_count"]
        ].to_dict(orient="records")
        raise ValueError(
            "incomplete candidate pair matrix: "
            f"expected_days={expected_days}, incomplete_pairs={incomplete}"
        )
    summary = summary.copy()
    summary["status"] = "COMPLETE"
    return summary.sort_values(pair_columns, kind="mergesort").reset_index(drop=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one read-only comparison range that cannot reach the locked test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("comparison range must end before the 2026 locked test")
    return args


def classify_pair(
    median_train: float,
    days_train: int,
    median_validation: float,
    days_validation: int,
) -> str:
    """Classify one pair only when both fixed comparison phases are complete."""
    if days_train != TRAIN_EXPECTED_DAYS or days_validation != VALIDATION_EXPECTED_DAYS:
        return "INCOMPLETE"
    maximum = max(median_train, median_validation)
    if maximum >= 0.90:
        return "REDUNDANT"
    if maximum >= 0.80:
        return "ELEVATED_PAIR_REDUNDANCY_MONITOR"
    return "PAIR_UNIQUE"
