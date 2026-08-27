"""Evaluate the isolated Tushare SH/SZ canonical Alpha484 candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import date
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
from evaluate_tushare_scope_validation import (
    _load_validation_inputs,
    compute_daily_rank_ic,
    compute_pair_daily_spearman,
    summarize_rank_ic,
)

from research_v2.compact.publisher import read_latest_partition
from research_v2.db.session import V2WriteSession

CANDIDATE_ID = "yeslab_484_tushare_shsz_v1"
SOURCE_FACTOR_ID = "yeslab_484"
TRAIN_START = date(2020, 1, 2)
TRAIN_END = date(2024, 12, 31)
VALIDATION_START = date(2025, 1, 2)
VALIDATION_END = date(2025, 12, 31)
LOCKED_TEST_START = date(2026, 1, 1)
MIN_SIGNED_RANK_IC = 0.02
MIN_SIGNED_ICIR = 0.50
MIN_DIRECTIONAL_SHARE = 0.55
MIN_ANNUAL_GATE_YEARS = 4
MIN_MEDIAN_COVERAGE = 0.80
MIN_P05_COVERAGE = 0.70
MIN_MONTHLY_COVERAGE = 0.70
MIN_TRAIN_SIMILARITY_DAYS = 1000
MIN_VALIDATION_SIMILARITY_DAYS = 200
MAX_MEDIAN_ABS_SPEARMAN = 0.90
FORMAL_FACTOR_IDS = (
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
CHECKPOINT_SCHEMA_VERSION = 1


def build_input_fingerprint(
    *,
    start: date,
    end: date,
    cache_manifest: Mapping[str, object],
    code_commit: str,
) -> dict[str, object]:
    """Build the immutable inputs that make a checkpoint reusable."""
    return {
        "candidate_id": CANDIDATE_ID,
        "source_factor_id": SOURCE_FACTOR_ID,
        "market_scope": "SH/SZ",
        "label": "forward_open_5d:v1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "code_commit": str(code_commit),
        "cache_manifest": dict(cache_manifest),
    }


def _write_checkpoint_state(path: Path, state: Mapping[str, object]) -> None:
    """Atomically write one JSON checkpoint beside the eventual output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(dict(state), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _load_checkpoint_state(path: Path) -> dict[str, object]:
    """Read one checkpoint and reject malformed JSON rather than guessing."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a JSON object: {path}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    return payload


def _assert_checkpoint_fingerprint(
    state: Mapping[str, object], expected: Mapping[str, object]
) -> None:
    """Fail closed when a checkpoint was produced from different inputs."""
    actual = state.get("fingerprint")
    if not isinstance(actual, Mapping):
        raise ValueError("checkpoint input fingerprint is missing")
    actual_core = {key: value for key, value in actual.items() if key != "cache_manifest"}
    expected_core = {
        key: value for key, value in expected.items() if key != "cache_manifest"
    }
    if actual_core != expected_core:
        raise ValueError("checkpoint input fingerprint does not match current inputs")
    actual_files = {
        str(item["name"]): int(item["size"])
        for item in actual.get("cache_manifest", {}).get("files", [])
        if isinstance(item, Mapping) and "name" in item and "size" in item
    }
    expected_files = {
        str(item["name"]): int(item["size"])
        for item in expected.get("cache_manifest", {}).get("files", [])
        if isinstance(item, Mapping) and "name" in item and "size" in item
    }
    if any(expected_files.get(name) != size for name, size in actual_files.items()):
        raise ValueError("checkpoint input fingerprint does not match current inputs")


def _current_code_commit() -> str:
    """Return a stable source revision without making it a runtime requirement."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _cache_manifest(cache_dir: Path) -> dict[str, object]:
    """Capture cache file names and sizes used as a cheap input manifest."""
    if not cache_dir.is_dir():
        return {"files": []}
    files = sorted(
        {path.name: path.stat().st_size for path in cache_dir.glob("*.json")}.items()
    )
    return {"files": [{"name": name, "size": size} for name, size in files]}


def _append_log(path: Path | None, event: str, **details: object) -> None:
    """Append a small JSON-lines event without exposing credentials."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"event": event, **details}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a bounded Alpha484 canonical evaluation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=TRAIN_START)
    parser.add_argument("--end", type=date.fromisoformat, default=VALIDATION_END)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("research_v2/artifacts/tushare_cache")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="directory for resumable per-window and per-month checkpoints",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "train", "validation"),
        default="all",
        help="run only one evaluation phase when staging a long audit",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-file", type=Path)
    args = parser.parse_args(argv)
    if args.start > args.end or args.end >= LOCKED_TEST_START:
        parser.error("evaluation range must end before the 2026 locked test")
    return args


def validate_full_market_dates(
    market: pd.DataFrame, expected_dates: tuple[date, ...]
) -> None:
    """Require one complete canonical SH/SZ market result per open date."""
    if "trade_date" not in market.columns:
        raise ValueError("full SH/SZ market is missing trade_date")
    observed = set(pd.to_datetime(market["trade_date"], errors="raise").dt.date)
    missing = sorted(set(expected_dates).difference(observed))
    if missing:
        rendered = ", ".join(value.isoformat() for value in missing)
        raise ValueError(f"full SH/SZ market is missing open dates: {rendered}")


def freeze_training_direction(daily_rank_ic: pd.DataFrame) -> int:
    """Freeze factor direction from finite training RankIC values only."""
    if "rank_ic" not in daily_rank_ic.columns:
        raise ValueError("training RankIC is missing rank_ic")
    values = pd.to_numeric(daily_rank_ic["rank_ic"], errors="coerce").dropna()
    if values.empty:
        raise ValueError("training RankIC has no finite observations")
    return -1 if float(values.mean()) < 0.0 else 1


def summarize_coverage(
    replay_values: pd.DataFrame, labels: pd.DataFrame
) -> dict[str, object]:
    """Measure finite candidate coverage on canonical valid-label keys."""
    value_required = {"trade_date", "ts_code", "replay_value"}
    label_required = {"signal_date", "ts_code"}
    if missing := sorted(value_required.difference(replay_values.columns)):
        raise ValueError(f"replay values are missing columns: {missing}")
    if missing := sorted(label_required.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    value_frame = replay_values.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    value_frame["signal_date"] = pd.to_datetime(
        value_frame.pop("trade_date"), errors="raise"
    ).dt.normalize()
    value_frame["ts_code"] = value_frame["ts_code"].astype(str)
    value_frame["replay_value"] = pd.to_numeric(
        value_frame["replay_value"], errors="coerce"
    )
    if value_frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("replay values contain duplicate signal-date keys")
    label_frame = labels.loc[:, ["signal_date", "ts_code"]].copy()
    label_frame["signal_date"] = pd.to_datetime(
        label_frame["signal_date"], errors="raise"
    ).dt.normalize()
    label_frame["ts_code"] = label_frame["ts_code"].astype(str)
    if label_frame.empty:
        raise ValueError("labels contain no canonical keys")
    if label_frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("labels contain duplicate signal-date keys")
    merged = label_frame.merge(
        value_frame, on=["signal_date", "ts_code"], how="left", validate="one_to_one"
    )
    merged["is_finite"] = merged["replay_value"].notna()
    daily = (
        merged.groupby("signal_date", sort=True)["is_finite"]
        .agg(expected_key_count="size", finite_key_count="sum")
        .reset_index()
    )
    daily["coverage_ratio"] = (
        daily["finite_key_count"] / daily["expected_key_count"]
    )
    daily["month"] = daily["signal_date"].dt.to_period("M").astype(str)
    monthly = (
        daily.groupby("month", sort=True)[["expected_key_count", "finite_key_count"]]
        .sum()
        .reset_index()
    )
    monthly["coverage_ratio"] = (
        monthly["finite_key_count"] / monthly["expected_key_count"]
    )
    median_ratio = float(daily["coverage_ratio"].median())
    p05_ratio = float(daily["coverage_ratio"].quantile(0.05))
    monthly_min_ratio = float(monthly["coverage_ratio"].min())
    status = (
        "PASS"
        if median_ratio >= MIN_MEDIAN_COVERAGE
        and p05_ratio >= MIN_P05_COVERAGE
        and monthly_min_ratio >= MIN_MONTHLY_COVERAGE
        else "FAIL"
    )
    return {
        "status": status,
        "valid_day_count": int(len(daily)),
        "label_key_count": int(len(label_frame)),
        "finite_key_count": int(merged["is_finite"].sum()),
        "median_ratio": median_ratio,
        "p05_ratio": p05_ratio,
        "monthly_min_ratio": monthly_min_ratio,
        "daily": [
            {
                "signal_date": row.signal_date.date().isoformat(),
                "expected_key_count": int(row.expected_key_count),
                "finite_key_count": int(row.finite_key_count),
                "coverage_ratio": float(row.coverage_ratio),
            }
            for row in daily.itertuples(index=False)
        ],
        "monthly": [
            {
                "month": str(row.month),
                "expected_key_count": int(row.expected_key_count),
                "finite_key_count": int(row.finite_key_count),
                "coverage_ratio": float(row.coverage_ratio),
            }
            for row in monthly.itertuples(index=False)
        ],
    }


def _passes_predictive_gate(metrics: Mapping[str, object]) -> bool:
    """Return whether one signed RankIC summary clears the fixed gate."""
    try:
        return (
            float(metrics["signed_mean_rank_ic"]) > MIN_SIGNED_RANK_IC
            and float(metrics["signed_icir"]) >= MIN_SIGNED_ICIR
            and float(metrics["directional_day_share"]) >= MIN_DIRECTIONAL_SHARE
        )
    except (KeyError, TypeError, ValueError):
        return False


def build_review(
    *,
    training: Mapping[str, object],
    validation: Mapping[str, object],
    coverage: Mapping[str, object],
    train_similarity: Mapping[str, object],
    validation_similarity: Mapping[str, object],
) -> dict[str, object]:
    """Build a candidate review that never performs an admission mutation."""
    try:
        annual_pass = int(training["annual_gate_years"]) >= MIN_ANNUAL_GATE_YEARS
    except (KeyError, TypeError, ValueError):
        annual_pass = False
    gates_pass = (
        _passes_predictive_gate(training)
        and _passes_predictive_gate(validation)
        and annual_pass
        and coverage.get("status") == "PASS"
        and train_similarity.get("status") == "PASS"
        and validation_similarity.get("status") == "PASS"
    )
    return {
        "candidate_id": CANDIDATE_ID,
        "source_factor_id": SOURCE_FACTOR_ID,
        "training": dict(training),
        "validation": dict(validation),
        "coverage": dict(coverage),
        "train_similarity": dict(train_similarity),
        "validation_similarity": dict(validation_similarity),
        "status": (
            "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
            if gates_pass
            else "REJECT_CANONICAL_GATE"
        ),
        "database_writes": False,
        "formal_factor_set_changed": False,
        "model_eligible": False,
        "model_training_started": False,
        "locked_test_read": False,
    }


def _load_formal_values(
    client: object, *, start: date, end: date
) -> dict[str, pd.DataFrame]:
    """Load current formal-factor model values from V2 latest partitions."""
    session = V2WriteSession(client, target_database="dfs://QuantResearchV2Test")
    factor_keys = tuple((factor_id, 1) for factor_id in FORMAL_FACTOR_IDS)
    frames: list[pd.DataFrame] = []
    for period in pd.period_range(start, end, freq="M"):
        values = read_latest_partition(
            session,
            object_name="factor_observation_clean_latest",
            logical_partition=str(period),
            columns=(
                "batch_id",
                "signal_date",
                "ts_code",
                "factor_id",
                "factor_version",
                "value",
            ),
            factor_keys=factor_keys,
        )
        if not values.empty:
            frames.append(values)
    if not frames:
        raise ValueError("formal-factor latest partitions contain no values")
    combined = pd.concat(frames, ignore_index=True)
    combined["signal_date"] = pd.to_datetime(
        combined["signal_date"], errors="raise"
    ).dt.normalize()
    combined["ts_code"] = combined["ts_code"].astype(str)
    combined["factor_id"] = combined["factor_id"].astype(str)
    combined["value"] = pd.to_numeric(combined["value"], errors="coerce")
    combined = combined.loc[
        combined["ts_code"].str.endswith((".SH", ".SZ"))
        & combined["factor_id"].isin(FORMAL_FACTOR_IDS)
    ].copy()
    result: dict[str, pd.DataFrame] = {}
    for factor_id in FORMAL_FACTOR_IDS:
        frame = combined.loc[
            combined["factor_id"] == factor_id,
            ["signal_date", "ts_code", "value"],
        ].rename(columns={"signal_date": "trade_date", "value": "replay_value"})
        if frame.empty:
            raise ValueError(f"formal-factor latest values are missing {factor_id}")
        if frame.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"formal-factor latest values duplicate keys for {factor_id}")
        result[factor_id] = frame.reset_index(drop=True)
    return result


def _replay_formal_values(
    client: object,
    market: pd.DataFrame,
    *,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """Replay all formal factors from the same complete Tushare market panel."""
    market_codes = tuple(sorted(market["ts_code"].astype(str).unique()))
    session = TushareMarketSession(client, market, market_codes)
    result: dict[str, pd.DataFrame] = {}
    for factor_id in FORMAL_FACTOR_IDS:
        definition = replay.REPLAY_DEFINITIONS[factor_id]
        values = replay._replay_factor(session, definition, start, end)
        values = values.loc[
            values["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
        ].copy()
        if values.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError(f"canonical formal replay duplicates keys for {factor_id}")
        result[factor_id] = values
    return result


def _write_similarity_checkpoint(path: Path, pairs: pd.DataFrame) -> None:
    """Persist one monthly similarity result using an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                json.loads(pairs.to_json(orient="records")),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_similarity_checkpoint(path: Path) -> pd.DataFrame:
    """Read a monthly similarity checkpoint with its fixed schema."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"similarity checkpoint is not a list: {path}")
    columns = [
        "factor_id_left",
        "factor_id_right",
        "median_abs_spearman",
        "max_abs_spearman",
        "valid_day_count",
        "daily_spearman",
    ]
    return pd.DataFrame(payload, columns=columns)


def _write_frame_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    """Atomically persist a pandas frame without requiring an optional parquet engine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            frame.to_pickle(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_frame_checkpoint(path: Path) -> pd.DataFrame:
    """Read a frame checkpoint and fail closed on unexpected payloads."""
    frame = pd.read_pickle(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"frame checkpoint is not a DataFrame: {path}")
    return frame


def _collapse_similarity_chunks(chunks: pd.DataFrame) -> pd.DataFrame:
    """Combine monthly summaries conservatively for the fixed similarity gate."""
    columns = [
        "factor_id_left",
        "factor_id_right",
        "median_abs_spearman",
        "max_abs_spearman",
        "valid_day_count",
    ]
    if chunks.empty:
        return pd.DataFrame(columns=columns)
    required = set(columns)
    if missing := sorted(required.difference(chunks.columns)):
        raise ValueError(f"similarity chunks are missing columns: {missing}")
    selected_columns = columns + (["daily_spearman"] if "daily_spearman" in chunks else [])
    working = chunks.loc[:, selected_columns].copy()
    working["factor_id_left"] = working["factor_id_left"].astype(str)
    working["factor_id_right"] = working["factor_id_right"].astype(str)
    for column in ("median_abs_spearman", "max_abs_spearman", "valid_day_count"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    rows: list[dict[str, object]] = []
    for (left_id, right_id), group in working.groupby(
        ["factor_id_left", "factor_id_right"], sort=True
    ):
        daily_values: list[float] = []
        if "daily_spearman" in group:
            for value in group["daily_spearman"]:
                if isinstance(value, (list, tuple)):
                    daily_values.extend(
                        float(item) for item in value if np.isfinite(item)
                    )
        if daily_values:
            absolute = np.abs(np.asarray(daily_values, dtype=float))
            rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "median_abs_spearman": float(np.median(absolute)),
                    "max_abs_spearman": float(np.max(absolute)),
                    "valid_day_count": int(len(absolute)),
                }
            )
        else:
            rows.append(
                {
                    "factor_id_left": left_id,
                    "factor_id_right": right_id,
                    "median_abs_spearman": float(group["median_abs_spearman"].max()),
                    "max_abs_spearman": float(group["max_abs_spearman"].max()),
                    "valid_day_count": int(group["valid_day_count"].sum()),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _compute_incremental_similarity(
    client: object,
    candidate: pd.DataFrame,
    *,
    start: date,
    end: date,
    checkpoint_dir: Path,
    resume: bool,
    formal_values: Mapping[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute candidate/formal similarities one month at a time.

    A completed month is immutable for a given run and is reused only when its
    checkpoint exists. The caller is responsible for validating the run-level
    input fingerprint before invoking this helper.
    """
    required = {"trade_date", "ts_code", "replay_value"}
    if missing := sorted(required.difference(candidate.columns)):
        raise ValueError(f"candidate values are missing columns: {missing}")
    working = candidate.loc[:, sorted(required)].copy()
    working["trade_date"] = pd.to_datetime(working["trade_date"], errors="raise").dt.normalize()
    working["ts_code"] = working["ts_code"].astype(str)
    month_dir = checkpoint_dir / "similarity"
    frames: list[pd.DataFrame] = []
    for period in pd.period_range(start, end, freq="M"):
        month_start = max(start, period.start_time.date())
        month_end = min(end, period.end_time.date())
        if month_start > month_end:
            continue
        path = month_dir / f"{period}.json"
        if resume and path.is_file():
            frames.append(_read_similarity_checkpoint(path))
            continue
        candidate_month = working.loc[
            working["trade_date"].dt.date.between(month_start, month_end)
        ].copy()
        if formal_values is None:
            month_formal_values = _load_formal_values(
                client, start=month_start, end=month_end
            )
        else:
            month_formal_values = {
                factor_id: frame.loc[
                    pd.to_datetime(frame["trade_date"], errors="raise")
                    .dt.date.between(month_start, month_end)
                ].copy()
                for factor_id, frame in formal_values.items()
            }
        pairs = compute_pair_daily_spearman(
            {CANDIDATE_ID: candidate_month, **month_formal_values},
            include_daily_values=True,
        )
        candidate_pairs = pairs.loc[
            (pairs["factor_id_left"] == CANDIDATE_ID)
            | (pairs["factor_id_right"] == CANDIDATE_ID)
        ].reset_index(drop=True)
        _write_similarity_checkpoint(path, candidate_pairs)
        frames.append(candidate_pairs)
    if not frames:
        return pd.DataFrame(
            columns=[
                "factor_id_left",
                "factor_id_right",
                "median_abs_spearman",
                "max_abs_spearman",
                "valid_day_count",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _evaluate_window(
    client: object,
    *,
    start: date,
    end: date,
    cache_dir: Path,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Evaluate one fixed window using the full canonical Tushare market."""
    inputs = _load_validation_inputs(client, start, end)
    candidate_path = (
        checkpoint_dir / "candidate_values.pkl" if checkpoint_dir is not None else None
    )
    definition = replay.REPLAY_DEFINITIONS[SOURCE_FACTOR_ID]
    market_start = _calculator_market_start(client, definition, start, end)
    open_dates = _query_open_dates(client, market_start, end)
    market: pd.DataFrame | None = None
    market_codes: tuple[str, ...] = ()

    def load_market() -> tuple[pd.DataFrame, tuple[str, ...]]:
        nonlocal market, market_codes
        if market is None:
            market = fetch_full_tushare_shsz_market(
                open_dates,
                token=os.getenv("QRV2_TUSHARE_TOKEN", ""),
                cache_dir=cache_dir,
                include_daily_basic=True,
            )
            validate_full_market_dates(market, open_dates)
            market_codes = tuple(sorted(market["ts_code"].astype(str).unique()))
        return market, market_codes

    if resume and candidate_path is not None and candidate_path.is_file():
        candidate_values = _read_frame_checkpoint(candidate_path)
    else:
        market, market_codes = load_market()
        candidate_values = replay._replay_factor(
            TushareMarketSession(client, market, market_codes), definition, start, end
        )
        candidate_values = candidate_values.loc[
            candidate_values["ts_code"].astype(str).str.endswith((".SH", ".SZ"))
        ].copy()
        if candidate_values.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("canonical Alpha484 values duplicate market keys")
        if candidate_path is not None:
            _write_frame_checkpoint(candidate_path, candidate_values)
    daily = compute_daily_rank_ic(candidate_values, inputs)
    coverage = summarize_coverage(candidate_values, inputs)
    if checkpoint_dir is None:
        market, market_codes = load_market()
        formal_values = _replay_formal_values(
            client, market, start=start, end=end
        )
        replay_frames = {CANDIDATE_ID: candidate_values, **formal_values}
        pairs = compute_pair_daily_spearman(replay_frames)
        candidate_pairs = pairs.loc[
            (pairs["factor_id_left"] == CANDIDATE_ID)
            | (pairs["factor_id_right"] == CANDIDATE_ID)
        ].reset_index(drop=True)
    else:
        market, _ = load_market()
        formal_values = _replay_formal_values(
            client, market, start=start, end=end
        )
        candidate_pairs = _compute_incremental_similarity(
            client,
            candidate_values,
            start=start,
            end=end,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            formal_values=formal_values,
        )
        candidate_pairs = _collapse_similarity_chunks(candidate_pairs)
    return daily, coverage, candidate_pairs


def _summarize_signed(daily: pd.DataFrame, direction: int) -> dict[str, object]:
    """Summarize RankIC with a direction frozen from the training window."""
    summary = summarize_rank_ic(daily, direction=direction)
    summary["signed_icir"] = float(summary["icir"]) * direction
    return summary


def _summarize_training(daily: pd.DataFrame) -> dict[str, object]:
    """Freeze direction and count annual predictive-gate passes."""
    direction = freeze_training_direction(daily)
    summary = _summarize_signed(daily, direction)
    dated = daily.copy()
    dated["signal_date"] = pd.to_datetime(dated["signal_date"], errors="raise")
    annual: dict[str, dict[str, object]] = {}
    annual_gate_years = 0
    for year, group in dated.groupby(dated["signal_date"].dt.year, sort=True):
        year_summary = _summarize_signed(group, direction)
        annual[str(int(year))] = year_summary
        if _passes_predictive_gate(year_summary):
            annual_gate_years += 1
    return {
        **summary,
        "annual": annual,
        "annual_gate_years": annual_gate_years,
    }


def _summarize_similarity(
    pairs: pd.DataFrame, *, minimum_valid_days: int
) -> dict[str, object]:
    """Fail closed unless all formal-nine candidate similarities are complete."""
    if len(pairs) != len(FORMAL_FACTOR_IDS):
        return {
            "status": "FAIL",
            "reason": "missing formal-factor similarity pairs",
            "observed_pair_count": int(len(pairs)),
        }
    working = pairs.copy()
    working["reference_factor_id"] = working["factor_id_left"].where(
        working["factor_id_right"] == CANDIDATE_ID,
        working["factor_id_right"],
    )
    expected = set(FORMAL_FACTOR_IDS)
    observed = set(working["reference_factor_id"].astype(str))
    if observed != expected:
        return {
            "status": "FAIL",
            "reason": "unexpected formal-factor similarity references",
            "observed_factor_ids": sorted(observed),
        }
    working["median_abs_spearman"] = pd.to_numeric(
        working["median_abs_spearman"], errors="coerce"
    )
    working["valid_day_count"] = pd.to_numeric(
        working["valid_day_count"], errors="coerce"
    )
    complete = (
        working["median_abs_spearman"].notna().all()
        and working["valid_day_count"].notna().all()
        and (working["valid_day_count"] >= minimum_valid_days).all()
    )
    maximum = float(working["median_abs_spearman"].max()) if complete else float("nan")
    nearest = working.sort_values(
        ["median_abs_spearman", "reference_factor_id"], ascending=[False, True]
    ).iloc[0]
    return {
        "status": "PASS" if complete and maximum < MAX_MEDIAN_ABS_SPEARMAN else "FAIL",
        "observed_pair_count": int(len(working)),
        "minimum_valid_days": int(working["valid_day_count"].min()) if complete else 0,
        "maximum_median_abs_spearman": maximum,
        "nearest_reference_factor": str(nearest["reference_factor_id"]),
        "threshold": MAX_MEDIAN_ABS_SPEARMAN,
    }


def _evaluate_window_kwargs(
    client: object,
    *,
    start: date,
    end: date,
    cache_dir: Path,
    checkpoint_dir: Path | None,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Call the window evaluator while keeping legacy test doubles compatible."""
    if checkpoint_dir is None and not resume:
        return _evaluate_window(client, start=start, end=end, cache_dir=cache_dir)
    return _evaluate_window(
        client,
        start=start,
        end=end,
        cache_dir=cache_dir,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )


def run_evaluation(
    args: argparse.Namespace, *, client: object | None = None
) -> dict[str, object]:
    """Run fixed train and validation audits without writing any database table."""
    own_client = client is None
    client = client or _connect()
    try:
        checkpoint_root = args.checkpoint_dir
        phase = getattr(args, "phase", "all")

        def phase_fingerprint(start: date, end: date) -> dict[str, object]:
            return build_input_fingerprint(
                start=start,
                end=end,
                cache_manifest=_cache_manifest(args.cache_dir),
                code_commit=_current_code_commit(),
            )

        def evaluate_window(
            *, phase: str, start: date, end: date
        ) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
            phase_dir = checkpoint_root / phase if checkpoint_root is not None else None
            expected = phase_fingerprint(start, end)
            _append_log(
                args.log_file,
                "phase_start",
                phase=phase,
                start=start.isoformat(),
                end=end.isoformat(),
                resume=bool(args.resume),
            )
            if phase_dir is not None and args.resume:
                manifest_path = phase_dir / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError(
                        "resume requested but checkpoint manifest is missing: "
                        f"{manifest_path}"
                    )
                _assert_checkpoint_fingerprint(
                    _load_checkpoint_state(manifest_path), expected
                )
            elif phase_dir is not None:
                _write_checkpoint_state(
                    phase_dir / "manifest.json",
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "fingerprint": expected,
                        "status": "running",
                    },
                )
            values = _evaluate_window_kwargs(
                client,
                start=start,
                end=end,
                cache_dir=args.cache_dir,
                checkpoint_dir=phase_dir,
                resume=args.resume,
            )
            if phase_dir is not None:
                _write_checkpoint_state(
                    phase_dir / "manifest.json",
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "fingerprint": expected,
                        "status": "complete",
                    },
                )
            _append_log(
                args.log_file,
                "phase_complete",
                phase=phase,
                start=start.isoformat(),
                end=end.isoformat(),
            )
            return values

        def write_phase_artifacts(
            *,
            phase_name: str,
            daily: pd.DataFrame,
            coverage: Mapping[str, object],
            summary: Mapping[str, object],
            pairs: pd.DataFrame | None = None,
        ) -> None:
            output = args.output
            output.mkdir(parents=True, exist_ok=True)
            daily.assign(factor_id=CANDIDATE_ID, phase=phase_name).to_csv(
                output / "daily_rank_ic.csv", index=False
            )
            (output / "coverage_summary.json").write_text(
                json.dumps(dict(coverage), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            if pairs is not None:
                pairs.assign(phase=phase_name).to_json(
                    output / "similarity_summary.json", orient="records", indent=2
                )
            (output / "quality_summary.json").write_text(
                json.dumps(
                    {
                        "scope": "TUSHARE_SHSZ_ALPHA484_CANONICAL_V1",
                        "candidate_id": CANDIDATE_ID,
                        "source_factor_id": SOURCE_FACTOR_ID,
                        phase_name: dict(summary),
                        "phase": phase_name,
                        "database_writes": False,
                        "formal_factor_set_changed": False,
                        "model_eligible": False,
                        "model_training_started": False,
                        "locked_test_read": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        if phase == "train":
            train_daily, train_coverage, _train_pairs = evaluate_window(
                phase="train", start=TRAIN_START, end=TRAIN_END
            )
            training = _summarize_training(train_daily)
            write_phase_artifacts(
                phase_name="training_2020_2024",
                daily=train_daily,
                coverage={"status": train_coverage.get("status"), "training": train_coverage},
                summary=training,
            )
            if checkpoint_root is not None:
                _write_checkpoint_state(
                    checkpoint_root / "train" / "summary.json",
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "training": training,
                        "coverage": train_coverage,
                    },
                )
            return {
                "candidate_id": CANDIDATE_ID,
                "phase": "train",
                "status": "TRAIN_PHASE_COMPLETE",
                "direction": training["direction"],
                "database_writes": False,
                "model_eligible": False,
                "locked_test_read": False,
            }

        if phase == "validation":
            if checkpoint_root is None:
                raise ValueError("validation phase requires --checkpoint-dir from a train phase")
            train_summary_path = checkpoint_root / "train" / "summary.json"
            if not train_summary_path.is_file():
                raise ValueError(f"validation phase requires train summary: {train_summary_path}")
            train_state = _load_checkpoint_state(train_summary_path)
            training_state = train_state.get("training")
            if not isinstance(training_state, dict) or "direction" not in training_state:
                raise ValueError("train checkpoint is missing frozen direction")
            validation_daily, validation_coverage, validation_pairs = evaluate_window(
                phase="validation", start=VALIDATION_START, end=VALIDATION_END
            )
            validation = _summarize_signed(
                validation_daily, int(training_state["direction"])
            )
            write_phase_artifacts(
                phase_name="validation_2025",
                daily=validation_daily,
                coverage={
                    "status": validation_coverage.get("status"),
                    "validation": validation_coverage,
                },
                summary=validation,
                pairs=validation_pairs,
            )
            return {
                "candidate_id": CANDIDATE_ID,
                "phase": "validation",
                "status": "VALIDATION_PHASE_COMPLETE",
                "direction": training_state["direction"],
                "database_writes": False,
                "model_eligible": False,
                "locked_test_read": False,
            }

        train_daily, train_coverage, train_pairs = evaluate_window(
            phase="train", start=TRAIN_START, end=TRAIN_END
        )
        training = _summarize_training(train_daily)
        validation_daily, validation_coverage, validation_pairs = evaluate_window(
            phase="validation", start=VALIDATION_START, end=VALIDATION_END
        )
        validation = _summarize_signed(validation_daily, int(training["direction"]))
        train_similarity = _summarize_similarity(
            train_pairs, minimum_valid_days=MIN_TRAIN_SIMILARITY_DAYS
        )
        validation_similarity = _summarize_similarity(
            validation_pairs, minimum_valid_days=MIN_VALIDATION_SIMILARITY_DAYS
        )
        coverage = {
            "status": (
                "PASS"
                if train_coverage.get("status") == "PASS"
                and validation_coverage.get("status") == "PASS"
                else "FAIL"
            ),
            "training": train_coverage,
            "validation": validation_coverage,
        }
        review = build_review(
            training=training,
            validation=validation,
            coverage=coverage,
            train_similarity=train_similarity,
            validation_similarity=validation_similarity,
        )
        review.update(
            {
                "phase": "all",
                "scope": "TUSHARE_SHSZ_ALPHA484_CANONICAL_V1",
                "market_scope": "SH/SZ",
                "excluded_exchanges": ["BJ"],
                "label": "forward_open_5d:v1 raw five-trading-day adjusted-open return",
                "training_window": f"{TRAIN_START.isoformat()}/{TRAIN_END.isoformat()}",
                "validation_window": f"{VALIDATION_START.isoformat()}/{VALIDATION_END.isoformat()}",
            }
        )
        output = args.output
        output.mkdir(parents=True, exist_ok=True)
        pd.concat(
            [
                train_daily.assign(factor_id=CANDIDATE_ID, phase="train"),
                validation_daily.assign(factor_id=CANDIDATE_ID, phase="validation"),
            ],
            ignore_index=True,
        ).to_csv(output / "daily_rank_ic.csv", index=False)
        pd.concat(
            [train_pairs.assign(phase="train"), validation_pairs.assign(phase="validation")],
            ignore_index=True,
        ).to_json(output / "similarity_summary.json", orient="records", indent=2)
        (output / "coverage_summary.json").write_text(
            json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        quality = {
            "scope": "TUSHARE_SHSZ_ALPHA484_CANONICAL_V1",
            "candidate_id": CANDIDATE_ID,
            "source_factor_id": SOURCE_FACTOR_ID,
            "training_2020_2024": training,
            "validation_2025": validation,
            "database_writes": False,
            "formal_factor_set_changed": False,
            "model_eligible": False,
            "model_training_started": False,
            "locked_test_read": False,
        }
        (output / "quality_summary.json").write_text(
            json.dumps(quality, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "admission_review.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return review
    finally:
        if own_client:
            client.close()


def main(argv: list[str] | None = None) -> int:
    """Run the canonical Alpha484 evaluation and render its review."""
    args = parse_args(argv)
    try:
        print(json.dumps(run_evaluation(args), ensure_ascii=False, indent=2))
    except Exception as exc:
        _append_log(
            args.log_file,
            "run_failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
