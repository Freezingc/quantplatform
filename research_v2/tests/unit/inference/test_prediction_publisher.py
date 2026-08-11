from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.inference.publisher import build_prediction_batch
from research_v2.runs.context import RunContext


def test_prediction_batch_uses_stable_score_desc_code_asc_rank_order() -> None:
    ensemble = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3,
            "ts_code": ["b", "a", "c"],
            "ensemble_score": [0.9, 0.9, 0.1],
        }
    )
    batch = build_prediction_batch(
        ensemble,
        model_id="model-1",
        model_version=1,
        universe_id="cn_liquid_a",
        universe_version=1,
        inference_manifest_id="manifest-1",
        source_watermark_id="watermark-1",
        context=RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
    )

    assert batch.object_name == "prediction_daily"
    assert batch.frame.sort_values("rank")["ts_code"].tolist() == ["a", "b", "c"]
    assert batch.frame.sort_values("rank")["rank"].tolist() == [1, 2, 3]
    assert batch.business_keys == ("signal_date", "model_id", "model_version", "ts_code")
