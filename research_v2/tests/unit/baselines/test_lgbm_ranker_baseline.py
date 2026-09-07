from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.lgbm_ranker import (
    LGBMHyperparameters,
    LGBMRankerBaseline,
    LGBMRankerCandidate,
)


def test_lgbm_ranker_is_train_only_and_scores_frozen_feature_order() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(5):
        signal_date = date(2020, 1, 2) + timedelta(days=day_offset)
        for index in range(30):
            feature_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "f1": float(index)}
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)
    baseline = LGBMRankerBaseline.fit(
        features,
        labels,
        factor_order=("f1",),
        role="TRAIN",
        num_boost_round=30,
    )

    scores = baseline.predict(features)

    assert np.isfinite(scores).all()
    assert np.corrcoef(scores, features["f1"])[0, 1] > 0.7
    with pytest.raises(PermissionError, match="TRAIN"):
        LGBMRankerBaseline.fit(features, labels, factor_order=("f1",), role="TEST")


def test_lgbm_ranker_supports_cross_sections_larger_than_default_label_gain() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(2):
        signal_date = date(2020, 2, 3) + timedelta(days=day_offset)
        for index in range(64):
            feature_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "f1": float(index)}
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
    baseline = LGBMRankerBaseline.fit(
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
        factor_order=("f1",),
        role="TRAIN",
        num_boost_round=5,
    )

    scores = baseline.predict(pd.DataFrame(feature_rows))

    assert np.isfinite(scores).all()


def test_lgbm_candidate_selects_fixed_grid_inside_train_and_records_groups() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for year in range(2014, 2019):
        for day_offset in range(10):
            signal_date = date(year, 3, 2) + timedelta(days=day_offset)
            for index in range(30):
                key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
                feature_rows.append({**key, "f1": float(index)})
                label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})
    train = BaselineDataset(
        "TRAIN",
        "F01",
        "train-v1",
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
    )
    validation = BaselineDataset(
        "VALIDATION",
        "F01",
        "validation-v1",
        train.features.copy(),
        train.labels.copy(),
    )

    fitted = LGBMRankerCandidate(
        factor_order=("f1",),
        hyperparameter_grid=(LGBMHyperparameters(15, 4, 0.05, 1),),
        num_boost_round=5,
        minimum_cross_section=30,
        early_stopping_patience=2,
    ).fit(train, validation)

    assert fitted.model.hyperparameters == LGBMHyperparameters(15, 4, 0.05, 1)
    assert fitted.model.training_day_count == 50
    assert fitted.model.eligible_training_day_count == 50
    assert fitted.model.num_boost_round <= 5


def test_lgbm_excludes_small_training_days_but_keeps_coverage_counts() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset, stock_count in enumerate((29, 30)):
        signal_date = date(2020, 4, 1) + timedelta(days=day_offset)
        for index in range(stock_count):
            key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
            feature_rows.append({**key, "f1": float(index)})
            label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})

    model = LGBMRankerBaseline.fit(
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
        factor_order=("f1",),
        role="TRAIN",
        num_boost_round=3,
        minimum_cross_section=30,
        hyperparameters=LGBMHyperparameters(7, 4, 0.05, 1),
    )

    assert model.training_day_count == 2
    assert model.eligible_training_day_count == 1


def test_lgbm_inner_grid_does_not_consult_outer_validation_labels() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for year in range(2014, 2019):
        for day_offset in range(10):
            signal_date = date(year, 5, 1) + timedelta(days=day_offset)
            for index in range(30):
                key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
                feature_rows.append({**key, "f1": float(index)})
                label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})
    train = BaselineDataset(
        "TRAIN",
        "F01",
        "train-v1",
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
    )
    validation = BaselineDataset(
        "VALIDATION",
        "F01",
        "validation-v1",
        train.features.copy(),
        train.labels.copy(),
    )
    changed_labels = validation.labels.copy()
    changed_labels["raw_return_5d"] = -changed_labels["raw_return_5d"]
    changed_validation = BaselineDataset(
        "VALIDATION",
        "F01",
        "validation-v2",
        validation.features.copy(),
        changed_labels,
    )
    candidate = LGBMRankerCandidate(
        factor_order=("f1",),
        hyperparameter_grid=(LGBMHyperparameters(15, 4, 0.05, 1),),
        num_boost_round=5,
        minimum_cross_section=30,
        early_stopping_patience=2,
    )

    fitted_a = candidate.fit(train, validation)
    fitted_b = candidate.fit(train, changed_validation)

    assert fitted_a.model.hyperparameters == fitted_b.model.hyperparameters
    assert fitted_a.model.num_boost_round == fitted_b.model.num_boost_round
    assert np.allclose(
        fitted_a.model.predict(train.features),
        fitted_b.model.predict(train.features),
    )
def _seed_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(6):
        signal_date = date(2020, 3, 2) + timedelta(days=day_offset)
        for index in range(40):
            feature_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "f1": float(rng.standard_normal()),
                    "f2": float(rng.standard_normal()),
                }
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(rng.standard_normal()),
                    "label_valid": True,
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows)


def test_lgbm_ranker_seed_kwarg_keeps_fit_deterministic() -> None:
    """The M0 three-seed contract needs per-seed fits that are reproducible."""
    features, labels = _seed_fixture()
    kwargs = {
        "factor_order": ("f1", "f2"),
        "role": "TRAIN",
        "num_boost_round": 20,
        "hyperparameters": LGBMHyperparameters(
            num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=10
        ),
        "seed": 20260802,
    }

    first = LGBMRankerBaseline.fit(features, labels, **kwargs)
    second = LGBMRankerBaseline.fit(features, labels, **kwargs)

    assert first.booster.model_to_string() == second.booster.model_to_string()


def test_lgbm_ranker_seed_changes_the_fit_when_feature_fraction_below_one() -> None:
    """Seeds must actually vary the model, or the three-seed contract is vacuous."""
    rng = np.random.default_rng(11)
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(6):
        signal_date = date(2020, 3, 2) + timedelta(days=day_offset)
        for index in range(40):
            feature_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    **{name: float(rng.standard_normal()) for name in ("f1", "f2", "f3", "f4")},
                }
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(rng.standard_normal()),
                    "label_valid": True,
                }
            )
    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)
    hyperparameters = LGBMHyperparameters(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=10,
        feature_fraction=0.5,
    )
    fitted = {
        seed: LGBMRankerBaseline.fit(
            features,
            labels,
            factor_order=("f1", "f2", "f3", "f4"),
            role="TRAIN",
            num_boost_round=20,
            hyperparameters=hyperparameters,
            seed=seed,
        )
        for seed in (20260801, 20260802, 20260803)
    }
    predictions = {seed: model.predict(features) for seed, model in fitted.items()}

    # Guard the guard: the fixture is informative enough that all three seeds diverge,
    # so a vacuous seed contract fails loudly here instead of passing silently.
    # (A two-feature fixture lets two seeds collide on the same feature-subset
    # sequence, which is a fixture degeneracy, not a vacuous seed contract.)
    assert not np.allclose(predictions[20260801], predictions[20260802])
    assert not np.allclose(predictions[20260801], predictions[20260803])
    assert not np.allclose(predictions[20260802], predictions[20260803])
