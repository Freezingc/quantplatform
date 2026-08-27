from __future__ import annotations

import pandas as pd

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.evaluation.leakage import (
    assert_train_scope_isolation,
    fingerprint_train_scope,
)


def test_train_scope_fingerprint_is_stable_when_future_rows_are_added_outside_fit() -> None:
    train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0],
        }
    )
    future = pd.DataFrame({"f1": [10_000.0], "f2": [-10_000.0]})

    original = fingerprint_train_scope(
        train_features=train,
        factor_order=("f1", "f2"),
    )
    contaminated_source = pd.concat([train, future], ignore_index=True)
    contaminated = fingerprint_train_scope(
        train_features=contaminated_source.iloc[: len(train)].copy(),
        factor_order=("f1", "f2"),
    )

    assert_train_scope_isolation(original, contaminated)
    assert original.digest == contaminated.digest


def test_preprocessor_canary_changes_when_future_rows_are_illegally_fit() -> None:
    train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    future = pd.DataFrame({"f1": [10_000.0]})
    first = FoldPreprocessor(("f1",))
    second = FoldPreprocessor(("f1",))
    first.fit(train, role="TRAIN")
    second.fit(pd.concat([train, future], ignore_index=True), role="TRAIN")

    assert first.state_hash != second.state_hash


def test_train_scope_fingerprint_reuses_existing_stable_hash_contract() -> None:
    train = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})

    fingerprint = fingerprint_train_scope(
        train_features=train,
        factor_order=("f1",),
    )

    assert len(fingerprint.preprocessor_hash) == 64
    assert len(fingerprint.digest) == 64
