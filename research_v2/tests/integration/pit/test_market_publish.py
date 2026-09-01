"""Market materialization and publish integration evidence (B7.4)."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from research_v2.contracts.source import SourceBatch
from research_v2.db.migrations import MigrationRunner
from research_v2.db.publish import PublishValidationError, V2BatchPublisher
from research_v2.materializers.market import build_market_batch
from research_v2.pit.calendar import TradingCalendar
from research_v2.quality.engine import CheckResult, CheckStatus, Severity
from research_v2.runs.context import RunContext

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run PIT integration tests",
    ),
]

_MIGRATION_DIRECTORY = Path(__file__).resolve().parents[3] / "migrations"
_DAYS = [day.date() for day in pd.bdate_range("2024-01-02", periods=60)]
_CODES = tuple(f"s{index:02d}" for index in range(50))
_CALENDAR = TradingCalendar.from_dates(_DAYS)
_CLEAN_DQ = [CheckResult("market_row_count", Severity.INFO, CheckStatus.PASS)]


def _source_frame() -> pd.DataFrame:
    rows = []
    for day in _DAYS:
        for index, code in enumerate(_CODES):
            base = 10.0 + index * 0.1
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "open": base,
                    "high": base + 0.5,
                    "low": base - 0.5,
                    "close": base + 0.2,
                    "vol": 1000 + index,
                    "amount": 1_000_000.0 + index,
                    "adj_factor": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _security_scd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": _CODES,
            "list_date": [date(2020, 1, 1)] * len(_CODES),
            "valid_from": [date(2020, 1, 1)] * len(_CODES),
            "valid_to": [None] * len(_CODES),
        }
    )


def _source_batch(frame: pd.DataFrame, watermark_id: str) -> SourceBatch:
    return SourceBatch(
        frame=frame,
        source_name="yeslab",
        table_name="stock_factor_daily",
        watermark_id=watermark_id,
        max_observation_date=_DAYS[-1],
        source_row_count=len(frame),
        schema_fingerprint="sha256:" + "f" * 64,
        extracted_at=datetime(2024, 4, 1, tzinfo=UTC),
    )


def _context(watermark_id: str) -> RunContext:
    return RunContext.build(
        config={"source_watermark": watermark_id},
        watermarks=(watermark_id,),
        code_commit="sha256:" + "a" * 64,
        schema_version="013",
    )


def _publish_monthly(database, batch, quality_results) -> None:
    publisher = V2BatchPublisher(database)
    for monthly_batch in batch.split_by_month():
        publisher.publish(monthly_batch, quality_results)


def _bootstrapped_database(fake_v2_test_database):
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)
    runner.apply()
    return fake_v2_test_database


def test_market_materialize_publish_rerun_is_idempotent(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    context = _context("wm-1")
    batch = build_market_batch(
        _source_batch(_source_frame(), "wm-1"), _security_scd(), _CALENDAR, context
    )

    assert len(batch.frame) == len(_DAYS) * len(_CODES)
    _publish_monthly(database, batch, _CLEAN_DQ)

    table = database.tables["market_daily_clean"]
    assert len(table) == len(_DAYS) * len(_CODES)
    assert len(database.tables["published_version"]) == 3

    rerun = build_market_batch(
        _source_batch(_source_frame(), "wm-1"), _security_scd(), _CALENDAR, _context("wm-1")
    )
    assert rerun.data_version == batch.data_version
    _publish_monthly(database, rerun, _CLEAN_DQ)

    assert len(database.tables["market_daily_clean"]) == len(_DAYS) * len(_CODES)
    assert len(database.tables["published_version"]) == 3


def test_single_source_row_change_scopes_to_the_recomputation_window(
    fake_v2_test_database,
) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    context_v1 = _context("wm-1")
    first = build_market_batch(
        _source_batch(_source_frame(), "wm-1"), _security_scd(), _CALENDAR, context_v1
    )
    _publish_monthly(database, first, _CLEAN_DQ)

    changed_frame = _source_frame()
    change_day = _DAYS[25]
    changed_frame.loc[
        (changed_frame["trade_date"] == change_day) & (changed_frame["ts_code"] == "s00"),
        "close",
    ] = 10.49
    context_v2 = _context("wm-2")
    window_end = _DAYS[_DAYS.index(change_day) + 19]
    scoped = build_market_batch(
        _source_batch(changed_frame, "wm-2"),
        _security_scd(),
        _CALENDAR,
        context_v2,
        publish_start=change_day,
        publish_end=window_end,
    )

    scoped_dates = set(scoped.frame["trade_date"])
    assert scoped_dates == {day for day in _DAYS if change_day <= day <= window_end}
    assert scoped.data_version != first.data_version

    _publish_monthly(database, scoped, _CLEAN_DQ)

    table = database.tables["market_daily_clean"]
    rows_by_version = table.groupby("data_version").size()
    assert len(rows_by_version) == 2
    assert rows_by_version[context_v2.data_version] == len(scoped_dates) * len(_CODES)
    # Partitions touched by the window point to V2; untouched months keep V1,
    # and V1 history rows remain append-only visible.
    window_months = {day.strftime("%Y-%m") for day in scoped_dates}
    pointers = database.tables["published_version"].sort_values("published_at")
    current_by_partition = {
        row["logical_partition"]: row["data_version"] for _, row in pointers.iterrows()
    }
    for partition, version in current_by_partition.items():
        if partition in window_months:
            assert version == context_v2.data_version
        else:
            assert version == context_v1.data_version
    assert rows_by_version[context_v1.data_version] > 0


def test_stage_duplicate_keys_keep_old_version_visible(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    context = _context("wm-1")
    batch = build_market_batch(
        _source_batch(_source_frame(), "wm-1"), _security_scd(), _CALENDAR, context
    )
    _publish_monthly(database, batch, _CLEAN_DQ)
    table_before = database.tables["market_daily_clean"].copy()

    poisoned = batch.split_by_month()[0]
    duplicate_row = poisoned.frame.iloc[[0]].copy()
    poisoned.frame = pd.concat([poisoned.frame, duplicate_row], ignore_index=True)
    poisoned.data_version = "sha256:" + "9" * 64

    publisher = V2BatchPublisher(database)
    with pytest.raises(PublishValidationError):
        publisher.publish(poisoned, _CLEAN_DQ)

    pd.testing.assert_frame_equal(
        database.tables["market_daily_clean"].reset_index(drop=True),
        table_before.reset_index(drop=True),
    )
    assert all(
        row["data_version"] == context.data_version
        for _, row in database.tables["published_version"].iterrows()
    )
