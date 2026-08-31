"""Define the read-only canonical candidate-pair comparison contract."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    _connect,
    _query_open_dates,
    fetch_full_tushare_shsz_market,
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
PAIR_MONITOR_THRESHOLD = 0.80
PAIR_REDUNDANCY_THRESHOLD = 0.90
DEFAULT_ALPHA484_CHECKPOINT_ROOT = Path(
    "research_v2/artifacts/checkpoints/alpha484_canonical_v1"
)
CHECKPOINT_INTEGRITY_FILENAME = "integrity.json"
CHECKPOINT_INTEGRITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    source_factor_id: str
    is_alpha206_median_repair: bool = False
    is_alpha484_tushare_checkpoint: bool = False


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec("yeslab_114_tushare_shsz_v2", "yeslab_114"),
    CandidateSpec("yeslab_381_tushare_shsz_v1", "yeslab_381"),
    CandidateSpec("yeslab_405_tushare_shsz_v1", "yeslab_405"),
    CandidateSpec("yeslab_425_tushare_shsz_v2", "yeslab_425"),
    CandidateSpec(
        "yeslab_484_tushare_shsz_v1",
        "yeslab_484",
        is_alpha484_tushare_checkpoint=True,
    ),
    CandidateSpec("yeslab_206_median_abs_v2", "yeslab_206", is_alpha206_median_repair=True),
)


def _candidate_ids() -> tuple[str, ...]:
    return tuple(candidate.candidate_id for candidate in CANDIDATES)


def _source_definitions() -> tuple[replay.ReplayDefinition, ...]:
    return tuple(
        replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
        for candidate in CANDIDATES
        if not candidate.is_alpha206_median_repair
        and not candidate.is_alpha484_tushare_checkpoint
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
    date_columns = {"signal_date", "trade_date"}.intersection(comparison_keys.columns)
    if len(date_columns) != 1:
        raise ValueError("comparison keys must contain exactly one of signal_date or trade_date")
    date_column = next(iter(date_columns))
    required = {date_column, "ts_code"}
    if missing := sorted(required.difference(comparison_keys.columns)):
        raise ValueError(f"comparison keys are missing columns: {missing}")
    keys = comparison_keys.loc[:, [date_column, "ts_code"]].copy()
    keys["trade_date"] = pd.to_datetime(keys.pop(date_column), errors="raise").dt.normalize()
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _date_set_digest(values: pd.Series) -> str:
    dates = sorted(pd.to_datetime(values, errors="raise").dt.date.astype(str).unique())
    return hashlib.sha256("\n".join(dates).encode("utf-8")).hexdigest()


def _load_checkpoint_integrity(checkpoint_root: Path) -> Mapping[str, object]:
    path = checkpoint_root / CHECKPOINT_INTEGRITY_FILENAME
    if not path.is_file():
        raise ValueError(f"Alpha484 checkpoint integrity file is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != CHECKPOINT_INTEGRITY_SCHEMA_VERSION
    ):
        raise ValueError(f"Alpha484 checkpoint integrity schema is invalid: {path}")
    phases = payload.get("phases")
    if not isinstance(phases, Mapping):
        raise ValueError(f"Alpha484 checkpoint integrity phases are missing: {path}")
    expected_identity = {
        "candidate_id": "yeslab_484_tushare_shsz_v1",
        "source_factor_id": "yeslab_484",
        "market_scope": "SH/SZ",
        "label": "forward_open_5d:v1",
    }
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Alpha484 checkpoint integrity identity mismatch: {path}")
    return payload


def load_alpha484_checkpoint_values(checkpoint_root: Path) -> pd.DataFrame:
    """Load the completed Tushare Alpha484 replay with bound file integrity."""
    integrity = _load_checkpoint_integrity(checkpoint_root)
    integrity_phases = integrity["phases"]
    if not isinstance(integrity_phases, Mapping):
        raise ValueError("Alpha484 checkpoint integrity phases are invalid")
    required_fingerprint = {
        "candidate_id": "yeslab_484_tushare_shsz_v1",
        "source_factor_id": "yeslab_484",
        "market_scope": "SH/SZ",
        "label": "forward_open_5d:v1",
    }
    phase_bounds = (
        ("train", TRAIN_START, TRAIN_END),
        ("validation", VALIDATION_START, VALIDATION_END),
    )
    frames = []
    for phase, start, end in phase_bounds:
        directory = checkpoint_root / phase
        manifest_path = directory / "manifest.json"
        values_path = directory / "candidate_values.pkl"
        if not manifest_path.is_file() or not values_path.is_file():
            raise ValueError(f"Alpha484 Tushare checkpoint is incomplete for {phase}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        fingerprint = manifest.get("fingerprint") if isinstance(manifest, dict) else None
        expected = {
            **required_fingerprint,
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        if (
            not isinstance(fingerprint, Mapping)
            or manifest.get("schema_version") != 1
            or manifest.get("status") != "complete"
            or any(fingerprint.get(key) != value for key, value in expected.items())
        ):
            raise ValueError(f"Alpha484 Tushare checkpoint fingerprint mismatch for {phase}")
        code_commit = fingerprint.get("code_commit")
        cache_manifest = fingerprint.get("cache_manifest")
        if not isinstance(code_commit, str) or not code_commit.strip():
            raise ValueError(f"Alpha484 checkpoint provenance is missing for {phase}")
        if (
            not isinstance(cache_manifest, Mapping)
            or not isinstance(cache_manifest.get("files"), list)
            or not cache_manifest["files"]
            or any(
                not isinstance(item, Mapping)
                or not str(item.get("name", "")).strip()
                or not isinstance(item.get("size"), int)
                or item["size"] < 0
                for item in cache_manifest["files"]
            )
        ):
            raise ValueError(f"Alpha484 checkpoint cache provenance is missing for {phase}")
        phase_integrity = integrity_phases.get(phase)
        if not isinstance(phase_integrity, Mapping):
            raise ValueError(f"Alpha484 checkpoint integrity is missing for {phase}")
        if phase_integrity.get("manifest_sha256") != _sha256_file(manifest_path):
            raise ValueError(f"Alpha484 checkpoint manifest integrity mismatch for {phase}")
        if phase_integrity.get("producer_code_commit") != code_commit:
            raise ValueError(f"Alpha484 checkpoint producer mismatch for {phase}")
        if phase_integrity.get("fingerprint_sha256") != _json_digest(fingerprint):
            raise ValueError(f"Alpha484 checkpoint fingerprint integrity mismatch for {phase}")
        if phase_integrity.get("cache_manifest_sha256") != _json_digest(cache_manifest):
            raise ValueError(f"Alpha484 checkpoint cache integrity mismatch for {phase}")
        if phase_integrity.get("values_sha256") != _sha256_file(values_path):
            raise ValueError(f"Alpha484 checkpoint values integrity mismatch for {phase}")
        values = _normalize_replay_frame(pd.read_pickle(values_path))
        if values.empty:
            raise ValueError(f"Alpha484 checkpoint values are empty for {phase}")
        finite = pd.to_numeric(values["replay_value"], errors="coerce")
        finite_count = int(finite.replace([float("inf"), float("-inf")], pd.NA).notna().sum())
        observed = {
            "row_count": len(values),
            "date_count": values["trade_date"].nunique(),
            "code_count": values["ts_code"].nunique(),
            "finite_value_count": finite_count,
            "min_trade_date": values["trade_date"].min().date().isoformat(),
            "max_trade_date": values["trade_date"].max().date().isoformat(),
            "trade_dates_sha256": _date_set_digest(values["trade_date"]),
        }
        if any(phase_integrity.get(key) != value for key, value in observed.items()):
            raise ValueError(f"Alpha484 checkpoint value coverage integrity mismatch for {phase}")
        frames.append(values)
    values = pd.concat(frames, ignore_index=True)
    if not values["ts_code"].str.endswith((".SH", ".SZ")).all():
        raise ValueError("Alpha484 Tushare checkpoint contains non-SH/SZ values")
    return values


def _normalize_replay_frame(replay_values: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "ts_code", "replay_value"}
    if missing := sorted(required.difference(replay_values.columns)):
        raise ValueError(f"replay values are missing columns: {missing}")
    frame = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    if not frame["ts_code"].str.endswith((".SH", ".SZ")).all():
        raise ValueError("replay values must contain only SH/SZ codes")
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


def load_candidate_phase_market(
    client: object,
    *,
    phase_start: date,
    phase_end: date,
    cache_dir: Path,
    token: str | None = None,
) -> pd.DataFrame:
    """Load the canonical full-SH/SZ market once for one evaluation phase."""
    _validate_window(phase_start, phase_end)
    definitions = _source_definitions()
    alpha206_definition = replay.REPLAY_DEFINITIONS["yeslab_206"]
    market_start = min(
        _calculator_market_start(client, definition, phase_start, phase_end)
        for definition in (*definitions, alpha206_definition)
    )
    open_dates = _query_open_dates(client, market_start, phase_end)
    return fetch_full_tushare_shsz_market(
        open_dates,
        token=token if token is not None else os.getenv("QRV2_TUSHARE_TOKEN", ""),
        cache_dir=cache_dir,
        include_daily_basic=False,
    )


def materialize_candidate_phase(
    client: object,
    market: pd.DataFrame,
    comparison_keys: pd.DataFrame,
    *,
    phase_start: date,
    phase_end: date,
    alpha484_checkpoint_values: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Replay all candidates once per phase, then restrict to V2 output keys."""
    _validate_window(phase_start, phase_end)
    market_codes = tuple(sorted(market["ts_code"].astype(str).unique()))
    if not market_codes or not all(code.endswith((".SH", ".SZ")) for code in market_codes):
        raise ValueError("phase market must contain only SH/SZ codes")
    session = TushareMarketSession(client, market, market_codes)
    frames: dict[str, pd.DataFrame] = {}
    for candidate in CANDIDATES:
        if candidate.is_alpha206_median_repair:
            replay_values = materialize_alpha206_median_repair(market, phase_start, phase_end)
        elif candidate.is_alpha484_tushare_checkpoint:
            if alpha484_checkpoint_values is None:
                raise ValueError("Alpha484 requires a completed Tushare replay checkpoint")
            replay_values = alpha484_checkpoint_values
        else:
            definition = replay.REPLAY_DEFINITIONS[candidate.source_factor_id]
            replay_values = replay._replay_factor(session, definition, phase_start, phase_end)
        frames[candidate.candidate_id] = _restrict_replay_to_month_keys(
            replay_values,
            comparison_keys,
            phase_start,
            phase_end,
        )
    return frames


def _pair_valid_dates(
    left: pd.DataFrame, right: pd.DataFrame, *, minimum_cross_section: int = 30
) -> set[date]:
    merged = left.merge(
        right,
        on=["trade_date", "ts_code"],
        how="inner",
        validate="one_to_one",
        suffixes=("_left", "_right"),
    )
    valid_dates: set[date] = set()
    for trade_date, group in merged.groupby("trade_date", sort=True):
        left_values = pd.to_numeric(group["replay_value_left"], errors="coerce")
        right_values = pd.to_numeric(group["replay_value_right"], errors="coerce")
        finite = np.isfinite(left_values.to_numpy(float)) & np.isfinite(
            right_values.to_numpy(float)
        )
        finite_group = group.loc[finite].copy()
        if len(finite_group) < minimum_cross_section:
            continue
        left_rank = left_values.loc[finite].rank(method="average")
        right_rank = right_values.loc[finite].rank(method="average")
        correlation = left_rank.corr(right_rank)
        if pd.notna(correlation):
            valid_dates.add(pd.Timestamp(trade_date).date())
    return valid_dates


def summarize_phase_pairs(
    replay_frames: Mapping[str, pd.DataFrame],
    expected_days: int,
    *,
    expected_dates: Sequence[date | pd.Timestamp],
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
    normalized_expected_dates = {pd.Timestamp(value).date() for value in expected_dates}
    if len(normalized_expected_dates) != expected_days:
        raise ValueError(
            "incomplete candidate pair matrix: expected date set size does not match "
            f"expected_days={expected_days}"
        )
    if any(value >= LOCKED_TEST_START for value in normalized_expected_dates):
        raise ValueError("expected comparison dates reach the 2026 locked test")
    for left_id, right_id in combinations(sorted(candidate_ids), 2):
        observed_dates = _pair_valid_dates(frames[left_id], frames[right_id])
        if observed_dates != normalized_expected_dates:
            raise ValueError(
                "incomplete candidate pair matrix: expected date set mismatch for "
                f"{left_id}/{right_id}; missing="
                f"{sorted(normalized_expected_dates.difference(observed_dates))}, "
                f"unexpected={sorted(observed_dates.difference(normalized_expected_dates))}"
            )
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


def build_audit_payload(
    train_replay_frames: Mapping[str, pd.DataFrame],
    validation_replay_frames: Mapping[str, pd.DataFrame],
    *,
    train_expected_dates: Sequence[date | pd.Timestamp],
    validation_expected_dates: Sequence[date | pd.Timestamp],
) -> dict[str, object]:
    """Build comparison-only evidence after both fixed phases are complete."""
    train_pairs = summarize_phase_pairs(
        train_replay_frames,
        TRAIN_EXPECTED_DAYS,
        expected_dates=train_expected_dates,
    )
    validation_pairs = summarize_phase_pairs(
        validation_replay_frames,
        VALIDATION_EXPECTED_DAYS,
        expected_dates=validation_expected_dates,
    )
    pair_columns = ["factor_id_left", "factor_id_right"]
    joined = train_pairs.merge(
        validation_pairs,
        on=pair_columns,
        how="inner",
        validate="one_to_one",
        suffixes=("_train", "_validation"),
    )
    if len(joined) != len(train_pairs) or len(joined) != len(validation_pairs):
        raise ValueError("complete phase pair matrices do not have matching pair keys")
    classifications = []
    for row in joined.sort_values(pair_columns, kind="mergesort").itertuples(index=False):
        classifications.append(
            {
                "factor_id_left": str(row.factor_id_left),
                "factor_id_right": str(row.factor_id_right),
                "training_median_abs_spearman": float(row.median_abs_spearman_train),
                "training_max_abs_spearman": float(row.max_abs_spearman_train),
                "training_valid_day_count": int(row.valid_day_count_train),
                "validation_median_abs_spearman": float(row.median_abs_spearman_validation),
                "validation_max_abs_spearman": float(row.max_abs_spearman_validation),
                "validation_valid_day_count": int(row.valid_day_count_validation),
                "status": classify_pair(
                    float(row.median_abs_spearman_train),
                    int(row.valid_day_count_train),
                    float(row.median_abs_spearman_validation),
                    int(row.valid_day_count_validation),
                ),
            }
        )
    return {
        "scope": "TUSHARE_CANONICAL_CANDIDATE_PAIR_SIMILARITY_V1",
        "label": "forward_open_5d:v1 raw five-trading-day adjusted-open return",
        "comparison_scope": {
            "output_keys": "V2 is_member=true and label_valid=true",
            "formula_input_market": "Tushare daily SH/SZ full panel",
            "excluded_exchanges": ["BJ"],
        },
        "thresholds": {
            "pair_monitor_median_abs_spearman": PAIR_MONITOR_THRESHOLD,
            "pair_redundancy_median_abs_spearman": PAIR_REDUNDANCY_THRESHOLD,
        },
        "decision": "COMPARISON_ONLY_NO_ADMISSION",
        "candidate_registry": [candidate.candidate_id for candidate in CANDIDATES],
        "formal_factor_set_changed": False,
        "model_eligible": False,
        "database_writes": False,
        "locked_test_read": False,
        "candidate_value_sources": {
            "yeslab_484_tushare_shsz_v1": "completed_tushare_shsz_replay_checkpoint",
            "all_other_candidates": "local_tushare_market_replay",
        },
        "training": {
            "start": TRAIN_START.isoformat(),
            "end": TRAIN_END.isoformat(),
            "expected_valid_day_count": TRAIN_EXPECTED_DAYS,
        },
        "validation": {
            "start": VALIDATION_START.isoformat(),
            "end": VALIDATION_END.isoformat(),
            "expected_valid_day_count": VALIDATION_EXPECTED_DAYS,
        },
        "train_pairs": train_pairs.to_dict(orient="records"),
        "validation_pairs": validation_pairs.to_dict(orient="records"),
        "pair_classifications": classifications,
    }


def _output_paths(output: Path) -> tuple[Path, Path]:
    """Keep the JSON summary beside the directory containing pair CSV files."""
    if output.suffix == ".json":
        return output, output.with_suffix("")
    return output.with_suffix(".json"), output


def run_audit(args: argparse.Namespace, *, client: object) -> dict[str, object]:
    """Run the pre-2026 candidate comparison and write only complete evidence."""
    _validate_window(args.start, args.end)
    comparison_keys = load_comparison_keys(client, args.start, args.end)
    alpha484_checkpoint_values = load_alpha484_checkpoint_values(
        args.alpha484_checkpoint_root
    )
    phase_frames: dict[str, dict[str, pd.DataFrame]] = {}
    phase_expected_dates: dict[str, tuple[date, ...]] = {}
    phase_bounds = (
        ("train", TRAIN_START, TRAIN_END),
        ("validation", VALIDATION_START, VALIDATION_END),
    )
    for phase_name, fixed_start, fixed_end in phase_bounds:
        phase_start = max(args.start, fixed_start)
        phase_end = min(args.end, fixed_end)
        if phase_start > phase_end:
            phase_expected_dates[phase_name] = ()
            phase_frames[phase_name] = {
                candidate_id: pd.DataFrame(
                    columns=["trade_date", "ts_code", "replay_value"]
                )
                for candidate_id in _candidate_ids()
            }
            continue
        market = load_candidate_phase_market(
            client,
            phase_start=phase_start,
            phase_end=phase_end,
            cache_dir=args.cache_dir,
        )
        market_dates = pd.to_datetime(market["trade_date"], errors="raise").dt.date
        phase_expected_dates[phase_name] = tuple(
            sorted(set(market_dates[market_dates.between(phase_start, phase_end)]))
        )
        phase_frames[phase_name] = materialize_candidate_phase(
            client,
            market,
            comparison_keys,
            phase_start=phase_start,
            phase_end=phase_end,
            alpha484_checkpoint_values=alpha484_checkpoint_values,
        )
    payload = build_audit_payload(
        phase_frames["train"],
        phase_frames["validation"],
        train_expected_dates=phase_expected_dates.get("train"),
        validation_expected_dates=phase_expected_dates.get("validation"),
    )
    summary_path, pairs_directory = _output_paths(args.output)
    pairs_directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(payload["train_pairs"]).to_csv(
        pairs_directory / "train_pairs.csv", index=False
    )
    pd.DataFrame(payload["validation_pairs"]).to_csv(
        pairs_directory / "validation_pairs.csv", index=False
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one read-only comparison range that cannot reach the locked test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache")
    )
    parser.add_argument(
        "--alpha484-checkpoint-root",
        type=Path,
        default=DEFAULT_ALPHA484_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.start > args.end:
        parser.error("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        parser.error("comparison range must end before the 2026 locked test")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run the comparison through a read-only DolphinDB connection."""
    return run_audit(parse_args(argv), client=_connect())


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
    if maximum >= PAIR_REDUNDANCY_THRESHOLD:
        return "REDUNDANT"
    if maximum >= PAIR_MONITOR_THRESHOLD:
        return "ELEVATED_PAIR_REDUNDANCY_MONITOR"
    return "PAIR_UNIQUE"


if __name__ == "__main__":
    main()
