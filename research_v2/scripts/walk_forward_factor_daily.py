"""Fold extraction backed by the materialized factor_daily table (plan Task 1).

Mirrors ``walk_forward_common.extract_dataset`` column-for-column, but sources
factor rows from ``factor_daily`` (isolated canonical replay values for the
frozen factor set) instead of ``factor_observation_clean``. Version pinning
uses the ``published_version`` registrations written by
``publish_factor_daily_versions.py`` (versions["factor_daily"]). Materialized
replay values are canonical at their signal date, so ``available_date`` is set
to ``signal_date``, satisfying the builder's availability gate.

Cache keys embed the full versions mapping (table names included), so this
path can never collide with the legacy ``factor_observation_clean`` cache
entries.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from walk_forward_common import (  # noqa: E402
    FOLD_CACHE_ROOT,
    SPLIT_ROLE_REQUEST,
    TARGET_DATABASE,
    factor_members_for,
    filter_current,
    fold_cache_key,
    query,
)

from research_v2.datasets.builder import (  # noqa: E402
    DatasetBuildRequest,
    FactorInput,
    build_point_in_time_dataset,
)
from research_v2.datasets.splits import SplitResolver, annual_folds  # noqa: E402

__all__ = ["extract_dataset_factor_daily", "factor_members_for"]


def extract_dataset_factor_daily(
    holder: dict[str, Any],
    resolver: SplitResolver,
    fold_id: str,
    role: str,
    factor_members: tuple[FactorInput, ...],
    versions: dict[str, dict[str, str]],
    *,
    query_fn: Callable[[dict[str, Any], str], Any] = query,
    cache_root: Path = FOLD_CACHE_ROOT,
) -> pd.DataFrame:
    """Assemble one fold split from universe_daily x factor_daily x label_daily."""
    cache_path = cache_root / f"{fold_cache_key(fold_id, role, versions)}.parquet"
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
    if role == "test":
        segment_end = date(max(role_years) + 1, 1, 31)
    else:
        segment_end = date(max(role_years), 12, 31)
    years = list(range(start.year, end.year + 1))

    universe_parts = []
    for year in years:
        _, part = query_fn(
            holder,
            f"select signal_date, ts_code, data_version "
            f"from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
            f"and is_member=1",
        )
        if isinstance(part, pd.DataFrame) and not part.empty:
            universe_parts.append(part)
    universe_rows = pd.concat(universe_parts, ignore_index=True)
    universe_rows = filter_current(universe_rows, "signal_date", versions["universe_daily"])

    factor_parts = []
    for member in factor_members:
        for year in years:
            _, part = query_fn(
                holder,
                f"select signal_date, ts_code, factor_id, factor_version, value, data_version "
                f"from loadTable('{TARGET_DATABASE}','factor_daily') "
                f"where signal_date between date('{year}.01.01') and date('{year}.12.31') "
                f"and factor_id='{member.factor_id}'",
            )
            if isinstance(part, pd.DataFrame) and not part.empty:
                factor_parts.append(part)
        print(f"  extracted factor {member.factor_id} for {fold_id}/{role}", flush=True)
    factor_rows = pd.concat(factor_parts, ignore_index=True)
    factor_rows = filter_current(factor_rows, "signal_date", versions["factor_daily"])
    # Canonical replay values are available at their signal date.
    factor_rows["available_date"] = factor_rows["signal_date"]

    label_parts = []
    for year in years:
        _, part = query_fn(
            holder,
            f"select signal_date, ts_code, raw_return, label_valid, exit_date, data_version "
            f"from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date between date('{year}.01.01') and date('{year}.12.31')",
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
    cache_root.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".parquet.tmp")
    dataset.to_parquet(tmp_path, index=False)
    tmp_path.replace(cache_path)
    print(f"  cached dataset for {fold_id}/{role} at {cache_path.name}", flush=True)
    return dataset
