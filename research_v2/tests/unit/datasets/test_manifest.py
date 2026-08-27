from __future__ import annotations

import pandas as pd

from research_v2.datasets.manifest import (
    build_dataset_manifest,
    build_split_dataset_manifest,
)


def test_manifest_is_content_addressed_and_invariant_to_input_row_order() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": ["2025-01-03", "2025-01-02"],
            "ts_code": ["b", "a"],
            "f1": [2.0, 1.0],
            "label": [0.02, 0.01],
        }
    )
    metadata = {
        "fold_id": "F01",
        "factor_order": ["f1"],
        "preprocessor_state_hash": "state",
        "source_versions": {"market": "v1", "labels": "v2"},
    }

    first = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata=metadata,
    )
    second = build_dataset_manifest(
        frame.iloc[::-1].reset_index(drop=True),
        business_keys=("signal_date", "ts_code"),
        metadata=metadata,
    )

    assert first.dataset_id == second.dataset_id
    assert first.content_hash == second.content_hash
    assert first.row_count == 2


def test_manifest_rejects_duplicate_or_null_business_keys() -> None:
    frame = pd.DataFrame({"signal_date": ["2025-01-02", "2025-01-02"], "ts_code": ["a", "a"]})

    try:
        build_dataset_manifest(frame, business_keys=("signal_date", "ts_code"), metadata={})
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("expected duplicate business-key rejection")


def test_split_manifest_identity_includes_split_assignment() -> None:
    train = pd.DataFrame({"signal_date": ["2025-01-02"], "ts_code": ["a"], "f1": [1.0]})
    test = pd.DataFrame({"signal_date": ["2025-01-03"], "ts_code": ["b"], "f1": [2.0]})

    first = build_split_dataset_manifest(
        {"train": train, "test": test},
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    swapped = build_split_dataset_manifest(
        {"train": test, "test": train},
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )

    assert first.dataset_id != swapped.dataset_id
    assert dict(first.split_content_hashes) != dict(swapped.split_content_hashes)
