from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from research_v2.inference.publisher import build_prediction_batch
from research_v2.runs.context import RunContext


@settings(deadline=None, max_examples=50)
@given(
    st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
                min_size=1,
                max_size=8,
            ),
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                width=32,
                min_value=-1_000.0,
                max_value=1_000.0,
            ),
        ),
        min_size=1,
        max_size=8,
        unique_by=lambda item: item[0],
    )
)
def test_prediction_batch_is_idempotent_under_row_reordering(
    rows: list[tuple[str, float]],
) -> None:
    signal_date = date(2025, 1, 2)
    frame = pd.DataFrame(
        {
            "signal_date": [signal_date] * len(rows),
            "ts_code": [ts_code for ts_code, _ in rows],
            "ensemble_score": [score for _, score in rows],
        }
    )
    reordered = frame.iloc[::-1].reset_index(drop=True)
    context = RunContext(
        run_id="00000000-0000-0000-0000-000000000001",
        data_version="data-version-1",
    )

    first = build_prediction_batch(
        frame,
        model_id="model-1",
        model_version=1,
        universe_id="cn_liquid_a",
        universe_version=1,
        inference_manifest_id="manifest-1",
        source_watermark_id="watermark-1",
        context=context,
    )
    second = build_prediction_batch(
        reordered,
        model_id="model-1",
        model_version=1,
        universe_id="cn_liquid_a",
        universe_version=1,
        inference_manifest_id="manifest-1",
        source_watermark_id="watermark-1",
        context=context,
    )

    assert first.object_name == second.object_name == "prediction_daily"
    assert first.logical_partition == second.logical_partition
    assert first.data_version == second.data_version == context.data_version
    assert first.business_keys == second.business_keys == (
        "signal_date",
        "model_id",
        "model_version",
        "ts_code",
    )
    pd.testing.assert_frame_equal(first.frame, second.frame)


def test_prediction_batch_rejects_duplicate_business_keys() -> None:
    ensemble = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "ensemble_score": [0.9, 0.8],
        }
    )

    with pytest.raises(ValueError, match="unique business keys"):
        build_prediction_batch(
            ensemble,
            model_id="model-1",
            model_version=1,
            universe_id="cn_liquid_a",
            universe_version=1,
            inference_manifest_id="manifest-1",
            source_watermark_id="watermark-1",
            context=RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
        )
