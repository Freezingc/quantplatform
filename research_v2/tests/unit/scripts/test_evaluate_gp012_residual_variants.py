from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate_gp012_residual_variants as evaluation


def test_cross_sectional_residual_removes_linear_exposure() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    y = pd.Series([3.0, 5.0, 7.0, 9.0], index=list("abcd"))

    result = evaluation.cross_sectional_residual(y, x)

    assert np.allclose(result.to_numpy(), 0.0)


def test_cross_sectional_residual_preserves_missing_values() -> None:
    x = pd.Series([1.0, 2.0, np.nan, 4.0], index=list("abcd"))
    y = pd.Series([3.0, 5.0, 7.0, 9.0], index=list("abcd"))

    result = evaluation.cross_sectional_residual(y, x)

    assert np.isnan(result.loc["c"])
    assert result.loc[["a", "b", "d"]].notna().all()


def test_build_variants_requires_matching_panels() -> None:
    dates = pd.date_range("2025-01-01", periods=2)
    left = pd.DataFrame([[1.0]], index=dates[:1], columns=["a"])
    right = pd.DataFrame([[1.0]], index=dates[:1], columns=["b"])

    with pytest.raises(ValueError, match="matching"):
        evaluation.build_variants(left, right)
