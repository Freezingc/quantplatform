from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.materializers.security_master import build_security_master_batch
from research_v2.runs.context import RunContext


def _batch(table_name: str, frame: pd.DataFrame, watermark: str) -> SourceBatch:
    return SourceBatch(
        frame=frame,
        source_name="YesLab",
        table_name=table_name,
        watermark_id=watermark,
        max_observation_date=date(2025, 1, 3),
        source_row_count=len(frame),
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )


def test_security_master_batch_has_v2_audit_fields_and_scd_business_keys() -> None:
    snapshot = _batch(
        "stock_info_table",
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": [date(2020, 1, 2)],
                "delist_date": [pd.NaT],
                "name": ["Alpha"],
                "list_status": ["L"],
            }
        ),
        "snapshot-watermark",
    )
    names = _batch(
        "stock_used_name",
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["*ST Alpha"],
                "start_date": [date(2020, 2, 3)],
                "end_date": [date(2020, 2, 7)],
            }
        ),
        "name-watermark",
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_security_master_batch(snapshot, names, context, as_of_date=date(2025, 1, 3))

    assert batch.object_name == "security_master_scd"
    assert batch.logical_partition == "2025-01-03"
    assert batch.business_keys == ("ts_code", "valid_from")
    assert set({"run_id", "data_version", "source_watermark_id", "record_hash"}).issubset(
        batch.frame.columns
    )
    assert batch.frame.loc[1, "is_st"]
    assert batch.frame.loc[1, "data_version"] == "version-a"


def test_security_master_batch_clips_an_interval_crossing_coverage_start() -> None:
    snapshot = _batch(
        "stock_info_table",
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": [pd.Timestamp("2010-01-02")],
                "delist_date": [pd.NaT],
                "name": ["Alpha"],
                "list_status": ["L"],
            }
        ),
        "snapshot-watermark",
    )
    names = _batch(
        "stock_used_name",
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "name-watermark",
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_security_master_batch(
        snapshot,
        names,
        context,
        as_of_date=date(2025, 1, 3),
        coverage_start=date(2013, 1, 1),
    )

    assert batch.frame.loc[0, "valid_from"] == date(2013, 1, 1)


def test_security_master_batch_augments_primary_snapshot_with_missing_history() -> None:
    snapshot = _batch(
        "stock_info_table",
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": [date(2020, 1, 2)],
                "delist_date": [pd.NaT],
                "name": ["Alpha"],
                "list_status": ["L"],
            }
        ),
        "primary-watermark",
    )
    names = _batch(
        "stock_used_name",
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "name-watermark",
    )
    historical = _batch(
        "tushare_stock_basic",
        pd.DataFrame(
            {
                "ts_code": ["000004.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": ["20120101"],
                "delist_date": ["20240521"],
                "name": ["Historic"],
                "list_status": ["D"],
            }
        ),
        "secondary-watermark",
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_security_master_batch(
        snapshot,
        names,
        context,
        as_of_date=date(2025, 1, 3),
        coverage_start=date(2013, 1, 1),
        historical_snapshot=historical,
    )

    historical_row = batch.frame.loc[batch.frame["ts_code"] == "000004.SZ"].iloc[0]
    assert historical_row["listing_status"] == "D"
    assert historical_row["list_date"] == date(2012, 1, 1)
    assert historical_row["delist_date"] == date(2024, 5, 21)
    assert historical_row["source_watermark_id"] != "primary-watermark"


def test_security_master_batch_uses_market_derived_profile_only_after_both_masters() -> None:
    snapshot = _batch(
        "stock_info_table",
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": [date(2020, 1, 2)],
                "delist_date": [pd.NaT],
                "name": ["Alpha"],
                "list_status": ["L"],
            }
        ),
        "primary-watermark",
    )
    names = _batch(
        "stock_used_name",
        pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date"]),
        "name-watermark",
    )
    derived = _batch(
        "derived_market_security_profile",
        pd.DataFrame(
            {
                "ts_code": ["300114.SZ"],
                "exchange": ["SZSE"],
                "market": ["创业板"],
                "list_date": [date(2013, 1, 4)],
                "delist_date": [None],
                "name": [None],
                "list_status": ["L"],
            }
        ),
        "derived-watermark",
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_security_master_batch(
        snapshot,
        names,
        context,
        as_of_date=date(2025, 1, 3),
        coverage_start=date(2013, 1, 1),
        derived_snapshot=derived,
    )

    assert set(batch.frame["ts_code"]) == {"000001.SZ", "300114.SZ"}
