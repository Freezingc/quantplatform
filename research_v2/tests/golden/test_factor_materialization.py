"""Golden equality evidence for the factor pipeline (C plan C8.3 / C8.4).

Runs the production path -- ``build_technical_factor_batches`` for the technical factors and
``expand_fundamental_events`` for the fundamental one -- over the 50x60 fixture and compares
against expectations computed by the hand-written reference in
``scripts/compute_factor_golden_expected.py``, which imports nothing from
``research_v2.factors``. That independence is the whole point: a golden sample produced by
snapshotting the implementation would pin whatever the code happens to do, defects included.

Only business-meaningful columns are compared. ``created_at`` is a wall-clock timestamp and
``record_hash`` derives from it, so neither is reproducible; the B plan's market golden test
excludes its own non-deterministic fields the same way.

The fixture digests are pinned here so that regenerating the fixtures against a defective
implementation cannot make this test pass by moving the expectation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.contracts.factor import AvailabilityPolicy, FactorDefinition, FactorType
from research_v2.factors.availability import expand_fundamental_events
from research_v2.factors.materializer import build_technical_factor_batches
from research_v2.factors.similarity import compute_factor_similarity
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext

_DATA = Path(__file__).parent / "data"
_CONTEXT = RunContext(
    run_id="00000000-0000-0000-0000-000000000001", data_version="golden-factor-v1"
)
_WATERMARK = "golden-watermark"

TECHNICAL_FACTORS: tuple[tuple[str, str], ...] = (
    ("yeslab_901", "alpha_dense"),
    ("yeslab_902", "alpha_dup"),
    ("yeslab_903", "alpha_single"),
    ("yeslab_904", "alpha_conflict"),
    ("yeslab_905", "alpha_mirror"),
    ("yeslab_906", "alpha_copy"),
)

CLEAN_COLUMNS = [
    "signal_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "value",
    "dedup_resolution",
]
QUARANTINE_COLUMNS = [
    "observation_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "reason",
    "distinct_values_json",
]
FUNDAMENTAL_COLUMNS = [
    "signal_date",
    "ts_code",
    "raw_value",
    "report_period",
    "ann_date",
    "available_date",
    "source_revision_id",
    "stale_days",
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_parquet(_DATA / name)


def _canonical_rows(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    ordered = frame.loc[:, columns].sort_values(columns[:3], kind="mergesort")
    rows: list[dict[str, object]] = []
    for record in ordered.to_dict(orient="records"):
        rows.append(
            {
                key: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
                for key, value in record.items()
            }
        )
    return rows


def _digest(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _definition(factor_id: str, factor_name: str) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        factor_version=1,
        display_name=factor_name,
        source_database="dfs://YesLab",
        source_table="factor_data9",
        source_value_column="value",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="DAILY",
        expected_direction=0,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="UNKNOWN",
        owner="research",
    )


def _production_clean_and_quarantine() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = _read("factors_50x60.parquet")
    clean_parts: list[pd.DataFrame] = []
    quarantine_parts: list[pd.DataFrame] = []
    for factor_id, factor_name in TECHNICAL_FACTORS:
        batches = build_technical_factor_batches(
            source,
            _definition(factor_id, factor_name),
            source_factor_name=factor_name,
            source_watermark_id=_WATERMARK,
            context=_CONTEXT,
        )
        clean_parts.append(batches.clean_batch.frame.loc[:, CLEAN_COLUMNS])
        if batches.quarantine_batch is not None:
            quarantine_parts.append(
                batches.quarantine_batch.frame.loc[:, QUARANTINE_COLUMNS]
            )
    clean = pd.concat(clean_parts, ignore_index=True)
    quarantine = (
        pd.concat(quarantine_parts, ignore_index=True)
        if quarantine_parts
        else pd.DataFrame(columns=QUARANTINE_COLUMNS)
    )
    return clean, quarantine


def test_technical_clean_matches_the_independent_expectation() -> None:
    clean, _ = _production_clean_and_quarantine()
    expected = _read("expected_factor_observations.parquet")

    actual_rows = _canonical_rows(clean, CLEAN_COLUMNS)
    expected_rows = _canonical_rows(expected, CLEAN_COLUMNS)

    assert len(actual_rows) == len(expected_rows) == 17900
    assert _digest(actual_rows) == _digest(expected_rows)
    assert actual_rows == expected_rows


def test_technical_quarantine_matches_the_independent_expectation() -> None:
    _, quarantine = _production_clean_and_quarantine()
    expected = _read("expected_factor_quarantine.parquet")

    actual_rows = _canonical_rows(quarantine, QUARANTINE_COLUMNS)
    expected_rows = _canonical_rows(expected, QUARANTINE_COLUMNS)

    assert len(actual_rows) == len(expected_rows) == 25
    assert _digest(actual_rows) == _digest(expected_rows)
    assert actual_rows == expected_rows


def test_the_eight_required_cases_are_each_present_in_the_expectation() -> None:
    """Guard the fixture itself: a case that quietly vanished would weaken every test above."""
    clean = _read("expected_factor_observations.parquet")
    quarantine = _read("expected_factor_quarantine.parquet")
    fundamental = _read("expected_fundamental_expansion.parquet")

    # 1 daily technical missing: 5 securities x 5 days absent, never forward-filled.
    dense = clean.loc[clean["factor_id"] == "yeslab_901"]
    assert len(dense) == 2975 == 3000 - 25
    # 2 Friday announcement becomes available the next open day.
    first = fundamental.loc[fundamental["source_revision_id"] == "r1"]
    assert first["ann_date"].iloc[0] == date(2025, 1, 3)
    assert first["available_date"].iloc[0] == date(2025, 1, 6)
    # 3 180-day stale: the September block publishes nothing at all.
    assert not (fundamental["signal_date"] >= date(2025, 9, 1)).any()
    # 4 revision: later signal dates see r2, earlier ones keep r1.
    revised = fundamental.loc[fundamental["source_revision_id"] == "r2"]
    assert len(first) == 500 and len(revised) == 900
    assert max(first["signal_date"]) < min(revised["signal_date"])
    assert revised["ann_date"].iloc[0] == date(2025, 1, 17)
    assert revised["available_date"].iloc[0] == date(2025, 1, 20)
    # 5 exact duplicate collapses; 6 single-nonnull publishes the finite value.
    resolutions = clean["dedup_resolution"].value_counts().to_dict()
    assert resolutions["IDENTICAL_COLLAPSE"] == 25
    assert resolutions["SINGLE_NONNULL"] == 25
    # 7 conflict quarantines the whole group and publishes nothing for those keys.
    assert len(quarantine) == 25
    assert set(quarantine["reason"]) == {"CONFLICT_QUARANTINE"}
    conflicted = set(
        map(tuple, quarantine.loc[:, ["observation_date", "ts_code"]].to_numpy().tolist())
    )
    published = clean.loc[clean["factor_id"] == "yeslab_904"]
    published_keys = set(
        map(tuple, published.loc[:, ["signal_date", "ts_code"]].to_numpy().tolist())
    )
    assert not (conflicted & published_keys), "a conflicted key was published"


def test_identical_and_sign_opposite_factors_are_indistinguishable_by_similarity() -> None:
    """Case 8: a threshold on |Spearman| cannot separate a copy or a mirror from its origin.

    Both reach exactly 1.0, which is why representative selection has to break such ties on
    something other than similarity -- the reason C7.4 ranks representatives deterministically
    rather than taking an arbitrary member of the cluster.
    """
    clean = _read("expected_factor_observations.parquet").loc[
        :, ["signal_date", "ts_code", "factor_id", "factor_version", "value"]
    ]
    keep = clean.loc[clean["factor_id"].isin(["yeslab_901", "yeslab_905", "yeslab_906"])]

    similarity = compute_factor_similarity(
        keep,
        start_date=date(2025, 1, 2),
        end_date=date(2025, 10, 31),
        minimum_common_members=30,
        minimum_valid_days=20,
    )

    pairs = {
        (str(row["factor_id_left"]), str(row["factor_id_right"])): float(
            row["median_daily_abs_spearman"]
        )
        for row in similarity.to_dict(orient="records")
    }
    mirror = ("yeslab_901", "yeslab_905")
    copy = ("yeslab_901", "yeslab_906")
    # Not exactly 1.0: a rank correlation over 50 names accumulates float error and the
    # mirror pair measures 0.9999999999999999. The claim is that neither is separable from
    # its origin at any threshold the pipeline would actually use, so assert against the
    # C7 default of 0.90 with a tight tolerance on the value itself.
    assert pairs[mirror] == pytest.approx(1.0, abs=1e-12), (
        f"sign-opposite factor was separable: {pairs[mirror]}"
    )
    assert pairs[copy] == pytest.approx(1.0, abs=1e-12), (
        f"identical factor was separable: {pairs[copy]}"
    )
    assert pairs[mirror] >= 0.90 and pairs[copy] >= 0.90


def test_fundamental_expansion_matches_the_independent_expectation() -> None:
    events = _read("fundamental_events_50x60.parquet")
    universe = _read("universe_50x60.parquet")
    calendar = TradingCalendar.from_dates(
        [ts.date() for ts in pd.bdate_range("2025-01-01", "2025-12-31")]
    )

    expanded = expand_fundamental_events(events, universe, calendar)
    expected = _read("expected_fundamental_expansion.parquet")

    actual_rows = _canonical_rows(expanded, FUNDAMENTAL_COLUMNS)
    expected_rows = _canonical_rows(expected, FUNDAMENTAL_COLUMNS)

    assert len(actual_rows) == len(expected_rows) == 1400
    assert _digest(actual_rows) == _digest(expected_rows)
    assert actual_rows == expected_rows


def test_the_fixtures_are_the_approved_ones() -> None:
    """Pin the expectations, so regenerating them cannot silently move the goalposts.

    Without this, fixing a defect by re-running the generator would update the expectation
    and the test would pass either way. If a change to the pipeline is correct, these digests
    must be updated deliberately in the same commit, with the reason recorded.
    """
    digests = {
        name: hashlib.sha256((_DATA / name).read_bytes()).hexdigest()
        for name in (
            "factors_50x60.parquet",
            "fundamental_events_50x60.parquet",
            "universe_50x60.parquet",
            "expected_factor_observations.parquet",
            "expected_factor_quarantine.parquet",
            "expected_fundamental_expansion.parquet",
        )
    }
    approved = json.loads((_DATA / "approved_factor_golden_digests.json").read_text("utf-8"))

    assert digests == approved
