"""Shared DolphinDB assembly helpers for the real-data research drivers.

Used by run_real_walk_forward.py (B0-B3 baselines) and
run_real_candidate_training.py (M1-M3 candidates) so both consume identical
point-in-time dataset assembly: connection/retry policy, published-version
resolution, per-fold extraction and the deterministic fold cache key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import dolphindb as ddb
import pandas as pd

from research_v2.datasets.builder import (
    DatasetBuildRequest,
    FactorInput,
    build_point_in_time_dataset,
)
from research_v2.datasets.splits import SplitResolver, annual_folds
from research_v2.pit.calendar import TradingCalendar

TARGET_DATABASE = "dfs://QuantResearchV2Test"
FOLD_CACHE_ROOT = Path("/tmp/walk_forward_cache_v1")
SPLIT_ROLE_REQUEST: dict[str, Literal["TRAIN", "VALIDATION", "TEST"]] = {
    "train": "TRAIN",
    "validation": "VALIDATION",
    "test": "TEST",
}


def fold_cache_key(fold_id: str, role: str, versions: Mapping[str, Mapping[str, str]]) -> str:
    """Deterministic cache key bound to the published versions consumed."""
    payload = json.dumps(
        {"fold_id": fold_id, "role": role, "versions": {k: dict(v) for k, v in versions.items()}},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connect_db() -> Any:
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return session


def run_query_with_retry(
    connect: Callable[[], Any],
    session: Any,
    script: str,
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
) -> tuple[Any, Any]:
    """Run one read-only query, reconnecting and retrying on transient failures.

    Returns (session, frame); the session may have been replaced after a failure.
    """
    last_error: Exception | None = None
    current_session = session
    for attempt in range(attempts):
        try:
            return current_session, current_session.run(script)
        except Exception as error:  # noqa: BLE001 - transient DB faults must not kill a 40h run
            last_error = error
            if attempt + 1 >= attempts:
                break
            delay = base_delay * (2**attempt)
            print(
                f"  query failed (attempt {attempt + 1}/{attempts}): {error}; "
                f"reconnecting in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            try:
                current_session = connect()
            except Exception as reconnect_error:  # noqa: BLE001
                print(f"  reconnect failed: {reconnect_error}", flush=True)
    raise RuntimeError(f"query failed after {attempts} attempts: {last_error}") from last_error


def query(holder: dict[str, Any], script: str) -> tuple[Any, Any]:
    """Run one query through the retry helper, keeping the holder's session fresh."""
    session, frame = run_query_with_retry(holder["connect"], holder["session"], script)
    holder["session"] = session
    return session, frame


def new_holder() -> dict[str, Any]:
    return {"connect": connect_db, "session": connect_db()}


def load_calendar(holder: dict[str, Any]) -> TradingCalendar:
    _, frame = query(
        holder,
        "select cal_date from loadTable('dfs://YesLab', 'trade_calendar') "
        "where exchange='SSE' and is_open=1 and cal_date between 2013.01.01 and 2026.12.31 "
        "order by cal_date",
    )
    return TradingCalendar.from_dates([pd.Timestamp(v).date() for v in frame["cal_date"]])


def current_versions(holder: dict[str, Any], object_name: str) -> dict[str, str]:
    _, frame = query(
        holder,
        f"select logical_partition, data_version, published_at "
        f"from loadTable('{TARGET_DATABASE}','published_version') "
        f"where object_name='{object_name}'",
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    frame = frame.sort_values("published_at", kind="mergesort")
    return {
        str(row.logical_partition): str(row.data_version)
        for row in frame.itertuples(index=False)
    }


def filter_current(
    frame: pd.DataFrame, date_column: str, versions: Mapping[str, str]
) -> pd.DataFrame:
    partitions = pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m")
    expected = partitions.map(versions)
    return frame.loc[expected.notna() & (frame["data_version"] == expected)].copy()


def factor_members_for(factor_ids: tuple[str, ...]) -> tuple[FactorInput, ...]:
    return tuple(
        FactorInput(factor_id=factor_id, factor_version=1, feature_name=f"f{index:02d}")
        for index, factor_id in enumerate(factor_ids)
    )


def extract_dataset(
    holder: dict[str, Any],
    resolver: SplitResolver,
    fold_id: str,
    role: str,
    factor_members: tuple[FactorInput, ...],
    versions: dict[str, dict[str, str]],
) -> pd.DataFrame:
    cache_path = FOLD_CACHE_ROOT / f"{fold_cache_key(fold_id, role, versions)}.parquet"
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        expected = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
        if not expected.issubset(cached.columns):
            raise ValueError(f"cached dataset for {fold_id}/{role} is corrupt")
        print(f"  loaded cached dataset for {fold_id}/{role} ({len(cached)} rows)", flush=True)
        return cached
    split_role = cast(Literal["train", "validation", "test"], role)
    segment = resolver.resolve(fold_id, split_role)
    start = min(segment.signal_dates)
    end = max(segment.signal_dates)
    fold_definition = next(fold for fold in annual_folds() if fold.fold_id == fold_id)
    role_years = {
        "train": fold_definition.train_years,
        "validation": fold_definition.validation_years,
        "test": fold_definition.test_years,
    }[role]
    # Train/validation exits must stay inside their years (no leak into the next
    # segment); test-year signals legitimately exit into the following January.
    if role == "test":
        segment_end = date(max(role_years) + 1, 1, 31)
    else:
        segment_end = date(max(role_years), 12, 31)
    years = list(range(start.year, end.year + 1))
    universe_parts = []
    for year in years:
        _, part = query(holder,
            f"select signal_date, ts_code, data_version "
            f"from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
            f"and is_member=1"
        )
        if isinstance(part, pd.DataFrame) and not part.empty:
            universe_parts.append(part)
    universe_rows = pd.concat(universe_parts, ignore_index=True)
    universe_rows = filter_current(universe_rows, "signal_date", versions["universe_daily"])
    factor_parts = []
    for member in factor_members:
        for year in years:
            _, part = query(holder,
                f"select signal_date, ts_code, factor_id, factor_version, value, "
                f"available_date, data_version "
                f"from loadTable('{TARGET_DATABASE}','factor_observation_clean') "
                f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
                f"and factor_id='{member.factor_id}'"
            )
            if isinstance(part, pd.DataFrame) and not part.empty:
                factor_parts.append(part)
        print(f"  extracted factor {member.factor_id} for {fold_id}/{role}", flush=True)
    factor_rows = pd.concat(factor_parts, ignore_index=True)
    factor_rows = filter_current(factor_rows, "signal_date", versions["factor_observation_clean"])
    label_parts = []
    for year in years:
        _, part = query(holder,
            f"select signal_date, ts_code, raw_return, label_valid, exit_date, data_version "
            f"from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31')"
        )
        if isinstance(part, pd.DataFrame) and not part.empty:
            label_parts.append(part)
    label_rows = pd.concat(label_parts, ignore_index=True)
    label_rows = filter_current(label_rows, "signal_date", versions["label_daily"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    universe_rows = universe_rows.loc[
        (universe_rows["signal_date"] >= start_ts) & (universe_rows["signal_date"] <= end_ts)
    ]
    factor_rows = factor_rows.loc[
        (factor_rows["signal_date"] >= start_ts) & (factor_rows["signal_date"] <= end_ts)
    ]
    label_rows = label_rows.loc[
        (label_rows["signal_date"] >= start_ts) & (label_rows["signal_date"] <= end_ts)
    ]
    dataset = build_point_in_time_dataset(
        universe_rows.loc[:, ["signal_date", "ts_code"]],
        factor_rows.loc[
            :, ["signal_date", "ts_code", "factor_id", "factor_version", "value", "available_date"]
        ],
        label_rows.loc[:, ["signal_date", "ts_code", "raw_return", "label_valid", "exit_date"]],
        factor_members=factor_members,
        segment_end=segment_end,
        request=DatasetBuildRequest(fold_id=fold_id, split_role=SPLIT_ROLE_REQUEST[role]),
    )
    FOLD_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".parquet.tmp")
    dataset.to_parquet(tmp_path, index=False)
    tmp_path.replace(cache_path)
    print(f"  cached dataset for {fold_id}/{role} at {cache_path.name}", flush=True)
    return dataset


