from __future__ import annotations

import pandas as pd
import pytest

from research_v2.datasets.preprocessing import FoldPreprocessor


def test_preprocessor_fits_only_train_and_emits_missing_masks() -> None:
    train = pd.DataFrame({"f1": [0.0, 1.0, 2.0], "f2": [1.0, None, 3.0]})
    validation = pd.DataFrame({"f1": [1000.0], "f2": [None]})
    preprocessor = FoldPreprocessor(("f1", "f2"))

    preprocessor.fit(train, role="TRAIN")
    transformed = preprocessor.transform(validation)

    assert transformed.values.loc[0, "f1"] == pytest.approx(1.0)
    assert transformed.values.loc[0, "f2"] == 0.0
    assert transformed.missing_mask.loc[0, "f2"]
    same = FoldPreprocessor(("f1", "f2"))
    same.fit(train, role="TRAIN")
    assert preprocessor.state_hash == same.state_hash


def test_preprocessor_rejects_nontrain_fit_and_schema_order_drift() -> None:
    preprocessor = FoldPreprocessor(("f1", "f2"))

    with pytest.raises(ValueError, match="TRAIN"):
        preprocessor.fit(pd.DataFrame({"f1": [1.0], "f2": [2.0]}), role="VALIDATION")
    preprocessor.fit(pd.DataFrame({"f1": [1.0], "f2": [2.0]}), role="TRAIN")
    with pytest.raises(ValueError, match="order"):
        preprocessor.transform(pd.DataFrame({"f2": [2.0], "f1": [1.0]}))
