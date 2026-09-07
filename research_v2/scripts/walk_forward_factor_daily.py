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
    WALK_FORWARD_FACTOR_IDS,
    extract_dataset,
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

# Frozen factor set v1 (2026-09-03 admission decision; factor_set_frozen_v1.json).
# Order is canonical: it fixes feature_name assignments (f00..f16).
FROZEN_FACTOR_SET_V1: tuple[str, ...] = (
    "yeslab_206",
    "yeslab_357",
    "yeslab_372",
    "yeslab_375",
    "yeslab_381",
    "yeslab_384",
    "yeslab_389",
    "yeslab_390",
    "yeslab_398",
    "yeslab_405",
    "yeslab_406",
    "yeslab_425",
    "yeslab_443_reg30_max20_v1",
    "yeslab_464",
    "yeslab_471",
    "yeslab_484",
    "yeslab_96",
)

# Frozen factor set v2 (2026-09-07 owner-approved; factor_set_frozen_v2.json).
# FROZEN_FACTOR_SET_V1 stays untouched above: it remains the frozen-set governance
# contract the coverage gates screen against, not the training input. V2 is the
# owner-confirmed training set -- 15 factors (9 dense baseline-eligible + 6 sparse
# candidate-only), with yeslab_405 and yeslab_398 removed by the W1 watch-pair
# review. Order is canonical: it fixes feature_name assignments (f00..f14).
FROZEN_FACTOR_SET_V2: tuple[str, ...] = (
    "yeslab_206",
    "yeslab_357",
    "yeslab_372",
    "yeslab_375",
    "yeslab_381",
    "yeslab_384",
    "yeslab_389",
    "yeslab_390",
    "yeslab_406",
    "yeslab_425",
    "yeslab_443_reg30_max20_v1",
    "yeslab_464",
    "yeslab_471",
    "yeslab_484",
    "yeslab_96",
)

# The 9 dense baseline-eligible factors of v2 (contract field ``baseline_eligible``).
# Baselines B0-B3 must not see the six sparse candidate-only factors: those are NaN
# on roughly half of all days by formula (their sparsity IS the candidate signal),
# and the dense-only baseline keeps the G4 bar an honest simple-model reference.
FROZEN_FACTOR_SET_V2_DENSE: tuple[str, ...] = (
    "yeslab_206",
    "yeslab_357",
    "yeslab_381",
    "yeslab_425",
    "yeslab_443_reg30_max20_v1",
    "yeslab_464",
    "yeslab_471",
    "yeslab_484",
    "yeslab_96",
)

# Factor source switch (reconstruction plan Task 2.3). Both real-data drivers
# (run_real_walk_forward.py and run_real_candidate_training.py) resolve their
# factor ids, version tables and fold extractor through these helpers so the
# legacy/frozen contract has exactly one definition.
FACTOR_SOURCES: tuple[str, ...] = ("legacy", "frozen", "frozen_v2", "frozen_v2_baseline")
LEGACY_VERSION_TABLES: tuple[str, ...] = (
    "universe_daily",
    "factor_observation_clean",
    "label_daily",
)
FROZEN_VERSION_TABLES: tuple[str, ...] = (
    "universe_daily",
    "factor_daily",
    "label_daily",
)


def factor_source_plan(source: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(factor_ids, version_tables)`` for a factor source.

    ``legacy`` reproduces the exact pre-flag contract that produced the
    criterion-303 evidence: the same 14 factor ids and the same three version
    tables. Because ``fold_cache_key`` hashes the whole versions mapping (table
    names included), keeping that mapping identical is precisely what makes the
    legacy path bit-identical and keeps its existing fold cache hits valid.
    ``frozen`` switches to the frozen 17 ids backed by ``factor_daily``; its
    mapping differs by table name, so its cache keys are disjoint from legacy and
    both chains may share one ``FOLD_CACHE_ROOT``.
    """
    if source == "legacy":
        return WALK_FORWARD_FACTOR_IDS, LEGACY_VERSION_TABLES
    if source == "frozen":
        return FROZEN_FACTOR_SET_V1, FROZEN_VERSION_TABLES
    if source == "frozen_v2":
        return FROZEN_FACTOR_SET_V2, FROZEN_VERSION_TABLES
    if source == "frozen_v2_baseline":
        return FROZEN_FACTOR_SET_V2_DENSE, FROZEN_VERSION_TABLES
    raise ValueError(f"unknown factor source: {source}")


def extract_fn_for(source: str) -> Callable[..., Any]:
    """Return the fold extractor bound to a factor source."""
    if source == "legacy":
        return extract_dataset
    if source in {"frozen", "frozen_v2", "frozen_v2_baseline"}:
        return extract_dataset_factor_daily
    raise ValueError(f"unknown factor source: {source}")


__all__ = [
    "extract_dataset_factor_daily",
    "extract_fn_for",
    "factor_members_for",
    "factor_source_plan",
    "FACTOR_SOURCES",
    "FROZEN_FACTOR_SET_V1",
    "FROZEN_FACTOR_SET_V2",
    "FROZEN_FACTOR_SET_V2_DENSE",
    "FROZEN_VERSION_TABLES",
    "LEGACY_VERSION_TABLES",
    "fold_cache_key_for",
    "scoped_cache_key",
]


def scoped_cache_key(
    fold_id: str,
    role: str,
    factor_members: tuple[FactorInput, ...],
    versions: Mapping[str, Mapping[str, str]],
) -> str:
    """Fold cache key that additionally binds the requested factor set.

    ``fold_cache_key`` hashes only ``(fold_id, role, versions)``. The frozen factor
    set is a module constant rather than part of that payload, so if
    ``FROZEN_FACTOR_SET_V1`` were ever changed -- for instance shrunk to drop the six
    direction-selected factors -- the key would be unchanged and a stale 17-factor
    frame would be served silently, with the wrong columns and no error. That is the
    same failure class as an unregistered month, so it is closed here.

    The scope is folded in as one extra pseudo-table entry, which keeps
    ``walk_forward_common`` untouched (plan Task 2.3 requires zero changes there) and
    preserves disjointness from legacy keys, whose mapping names
    ``factor_observation_clean`` instead of ``factor_daily``.
    """
    scope = {
        str(member.factor_id): f"{member.factor_version}:{member.feature_name}"
        for member in factor_members
    }
    return fold_cache_key(fold_id, role, {**versions, "factor_set": scope})


CacheKeyFn = Callable[
    [str, str, tuple[FactorInput, ...], Mapping[str, Mapping[str, str]]], str
]


def fold_cache_key_for(source: str) -> CacheKeyFn:
    """Return the cache-key function that ``source``'s own extractor writes with.

    Any tool that resolves a cached frame by recomputing its key must ask for the
    source's function rather than picking one, because the two chains key differently:
    the legacy extractor hashes ``(fold_id, role, versions)`` alone, while the frozen
    extractor folds the requested factor set in as well (see :func:`scoped_cache_key`).
    A wrong key is indistinguishable from an absent frame -- the calibration audit made
    exactly that mistake and reported all eighteen frozen segments missing when three
    were present, which then propagated into a documented claim that Task 3.1 would have
    to cold-assemble every segment.

    The factor set is part of the signature even though the legacy branch ignores it, so
    one call shape serves both sources and a caller cannot pass the wrong arity.
    """
    if source == "legacy":
        def legacy_key(
            fold_id: str,
            role: str,
            _factor_members: tuple[FactorInput, ...],
            versions: Mapping[str, Mapping[str, str]],
        ) -> str:
            return fold_cache_key(fold_id, role, versions)

        return legacy_key
    if source in {"frozen", "frozen_v2", "frozen_v2_baseline"}:
        return scoped_cache_key
    raise ValueError(f"unknown factor source: {source}")
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
    cache_key = scoped_cache_key(fold_id, role, factor_members, versions)
    cache_path = cache_root / f"{cache_key}.parquet"
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
