from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.evaluation.rank_ic import RankICSummary
from research_v2.materializers.oos import (
    build_model_oos_daily_batch,
    build_model_oos_summary_batch,
)
from research_v2.runs.context import RunContext


def _context() -> RunContext:
    return RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")


def test_model_oos_daily_batch_binds_fold_identity() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "rank_ic": [0.1],
            "sample_count": [100],
        }
    )

    batch = build_model_oos_daily_batch(
        daily,
        _context(),
        evaluation_id="F06_stable27_rank_composite_v1",
        model_id="stable27_rank_composite_v1",
        fold_id="F06",
        source_watermark_id="source-v1",
    )

    assert batch.object_name == "model_oos_rank_ic_daily"
    assert batch.logical_partition == "2025-01"
    assert batch.frame.loc[0, "fold_id"] == "F06"
    assert batch.business_keys == ("signal_date", "evaluation_id", "model_id")


def test_model_oos_summary_batch_writes_summary_metrics() -> None:
    batch = build_model_oos_summary_batch(
        RankICSummary(243, 0.0948, 0.1076, 0.8809),
        _context(),
        as_of_date=date(2025, 12, 31),
        evaluation_id="F06_stable27_rank_composite_v1",
        model_id="stable27_rank_composite_v1",
        fold_id="F06",
        source_watermark_id="source-v1",
        positive_day_share=0.8065,
        mean_sample_count=1000.0,
    )

    assert batch.object_name == "model_oos_rank_ic_summary"
    assert batch.frame.loc[0, "icir"] == 0.8809
    assert batch.frame.loc[0, "positive_day_share"] == 0.8065

