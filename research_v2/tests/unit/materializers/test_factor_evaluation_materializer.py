from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.materializers.factor_evaluation import (
    build_factor_rank_ic_daily_batch,
    build_factor_rank_ic_summary_batch,
)
from research_v2.runs.context import RunContext


def _context() -> RunContext:
    return RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_id": ["yeslab_1"],
            "factor_name": ["Alpha1"],
            "factor_version": [1],
        }
    )


def test_factor_rank_ic_daily_batch_maps_catalog_and_partition() -> None:
    daily = pd.DataFrame(
        {
            "factor_name": ["Alpha1"],
            "signal_date": [date(2023, 1, 3)],
            "rank_ic": [0.12],
            "sample_count": [100],
        }
    )

    batch = build_factor_rank_ic_daily_batch(
        daily,
        _catalog(),
        _context(),
        evaluation_scope="2023-01",
        source_watermark_id="source-v1",
    )

    assert batch.object_name == "factor_rank_ic_daily"
    assert batch.logical_partition == "2023-01"
    assert batch.frame.loc[0, "factor_id"] == "yeslab_1"
    assert batch.business_keys == ("signal_date", "factor_id", "factor_version")


def test_factor_rank_ic_summary_batch_rejects_unknown_factor() -> None:
    summary = pd.DataFrame(
        {
            "factor_name": ["unknown"],
            "valid_day_count": [10],
            "mean_rank_ic": [0.1],
            "rank_ic_std": [0.2],
            "icir": [0.5],
            "positive_day_share": [0.6],
            "mean_sample_count": [100.0],
        }
    )

    try:
        build_factor_rank_ic_summary_batch(
            summary,
            _catalog(),
            _context(),
            as_of_date=date(2023, 12, 29),
            evaluation_scope="2023",
            source_watermark_id="source-v1",
        )
    except ValueError as error:
        assert "unknown factor_name" in str(error)
    else:
        raise AssertionError("unknown factor should be rejected")

