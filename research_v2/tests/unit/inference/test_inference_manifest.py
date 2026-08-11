from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.inference.manifest import build_inference_manifest


def test_inference_manifest_is_content_addressed_and_label_free() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["b", "a"],
            "f1": [2.0, 1.0],
        }
    )
    first = build_inference_manifest(
        frame,
        model_id="model-1",
        universe_id="universe-1",
        factor_set_id="factors-1",
        preprocessor_hash="pre-1",
        input_versions={"market": "v1", "factor:f1": "v2"},
    )
    second = build_inference_manifest(
        frame.iloc[::-1].reset_index(drop=True),
        model_id="model-1",
        universe_id="universe-1",
        factor_set_id="factors-1",
        preprocessor_hash="pre-1",
        input_versions={"factor:f1": "v2", "market": "v1"},
    )

    assert first.manifest_id == second.manifest_id
    assert not first.contains_labels
    with pytest.raises(ValueError, match="label"):
        build_inference_manifest(
            frame.assign(raw_return_5d=0.01),
            model_id="model-1",
            universe_id="universe-1",
            factor_set_id="factors-1",
            preprocessor_hash="pre-1",
            input_versions={"market": "v1"},
        )
