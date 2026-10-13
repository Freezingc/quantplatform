"""Unit tests for the score-diagnostic script's pure helpers.

Both helpers feed the platform's own RankIC evaluator, so a mistake here does not show up as an
error -- it shows up as a plausible-looking IC computed on the wrong rows. The ensemble helper in
particular must align on business keys, because two models emit their rows in score order.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import diagnose_operational_scores as diagnostic


def test_returns_to_labels_marks_only_finite_returns_valid() -> None:
    returns = {
        date(2026, 8, 3): {"a": 0.01, "b": float("nan")},
        date(2026, 8, 4): {"a": -0.02},
    }
    labels = diagnostic.returns_to_labels(returns)
    assert set(labels.columns) == {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
    assert len(labels) == 3
    valid = labels.loc[labels["label_valid"]]
    assert len(valid) == 2
    assert set(valid["ts_code"]) == {"a"}


def test_returns_to_labels_emits_nothing_for_an_empty_window() -> None:
    labels = diagnostic.returns_to_labels({})
    assert labels.empty
    assert set(labels.columns) == {"signal_date", "ts_code", "raw_return_5d", "label_valid"}


def test_ensemble_score_averages_on_business_keys_not_position() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2026, 8, 3)] * 2,
            "ts_code": ["a", "b"],
            "m0r": [0.9, 0.1],
            "m1": [0.2, 0.8],
        }
    )
    ensemble = diagnostic.ensemble_score(frame, ("m0r", "m1"))
    by_code = ensemble.set_index("ts_code")["score"]
    assert by_code.loc["a"] == pytest.approx(0.55)
    assert by_code.loc["b"] == pytest.approx(0.45)
    assert ensemble.columns.tolist() == ["signal_date", "ts_code", "score"]


def test_ensemble_score_drops_rows_missing_either_member() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2026, 8, 3)] * 2,
            "ts_code": ["a", "b"],
            "m0r": [0.9, np.nan],
            "m1": [0.1, 0.5],
        }
    )
    ensemble = diagnostic.ensemble_score(frame, ("m0r", "m1"))
    assert ensemble["ts_code"].tolist() == ["a"]
