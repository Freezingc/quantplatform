from __future__ import annotations

import pandas as pd
import pytest

from research_v2.factors.admission import (
    FactorAdmissionContract,
    FactorAdmissionError,
    admit_feature_frame,
)
from research_v2.factors.factor_sets import (
    FactorSetStatus,
    create_factor_set,
    transition_factor_set,
)


def _frozen_set(*, factor_id: str = "yeslab_1"):
    draft = create_factor_set(
        [(factor_id, 1, 1, f"{factor_id}-definition")],
        selection_scope_id="F04_SCOPE",
        rules_version=2,
    )
    return transition_factor_set(
        transition_factor_set(draft, FactorSetStatus.VALIDATED),
        FactorSetStatus.FROZEN,
    )


def _contract():
    return FactorAdmissionContract.from_factor_set(
        _frozen_set(),
        feature_names_by_factor_id={"yeslab_1": "alpha"},
        rules_hash="a" * 64,
        preprocessor_hash="b" * 64,
    )


def test_admission_accepts_exact_frozen_factor_order() -> None:
    contract = _contract()
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
            "ts_code": ["000001.SZ", "000002.SZ"],
            "alpha": [0.1, -0.2],
        }
    )

    result = admit_feature_frame(contract, frame, role="TRAIN")

    assert result.factor_set_id == contract.factor_set_id
    assert result.row_count == 2
    assert result.factor_names == ("alpha",)


def test_admission_rejects_wrong_order_or_missing_factor() -> None:
    contract = _contract()
    frame = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"],
            "ts_code": ["000001.SZ"],
            "wrong": [0.1],
        }
    )

    with pytest.raises(FactorAdmissionError, match="feature columns"):
        admit_feature_frame(contract, frame, role="TRAIN")


def test_admission_rejects_non_frozen_or_synthetic_factor_set() -> None:
    draft = create_factor_set(
        [("yeslab_1", 1, 1, "definition")],
        selection_scope_id="F04_SCOPE",
        rules_version=2,
    )
    with pytest.raises(FactorAdmissionError, match="FROZEN"):
        FactorAdmissionContract.from_factor_set(
            draft,
            feature_names_by_factor_id={"yeslab_1": "alpha"},
            rules_hash="a" * 64,
            preprocessor_hash="b" * 64,
        )

    with pytest.raises(FactorAdmissionError, match="excluded"):
        FactorAdmissionContract.from_factor_set(
            _frozen_set(factor_id="yeslab_0"),
            feature_names_by_factor_id={"yeslab_0": "synthetic"},
            rules_hash="a" * 64,
            preprocessor_hash="b" * 64,
        )


def test_admission_rejects_low_finite_ratio_and_duplicate_keys() -> None:
    contract = _contract()
    low_finite = pd.DataFrame(
        {
            "signal_date": ["2025-01-02", "2025-01-03"],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "alpha": [0.1, float("nan")],
        }
    )
    with pytest.raises(FactorAdmissionError, match="finite ratio"):
        admit_feature_frame(contract, low_finite, role="TRAIN", minimum_finite_ratio=0.75)

    duplicate = pd.concat([low_finite.iloc[[0]], low_finite.iloc[[0]]], ignore_index=True)
    duplicate.loc[:, "alpha"] = 0.1
    with pytest.raises(FactorAdmissionError, match="duplicate"):
        admit_feature_frame(contract, duplicate, role="TRAIN")
