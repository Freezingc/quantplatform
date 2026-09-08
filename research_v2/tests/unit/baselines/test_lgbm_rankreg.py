"""Contract tests for the M0R rank-regression carrier.

The carrier exists because the LambdaRank one caps out below the G4 bar; see
docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md. These tests pin the parts
that make its measured numbers meaningful: the target is a per-day percentile rank, early
stopping watches the daily RankIC the gates actually measure, the three-seed contract is real
rather than vacuous, and a fit does not depend on which fits ran before it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.lgbm_ranker import LGBMHyperparameters
from research_v2.baselines.lgbm_rankreg import (
    LGBM_RANKREG_DEFAULT_HYPERPARAMETERS,
    RANK_IC_METRIC_NAME,
    DailyRankICMetric,
    LGBMRankRegBaseline,
    _percentile_rank_targets,
    _rankreg_training_parameters,
)

FACTORS = ("f0", "f1", "f2", "f3", "f4")
SEEDS = (20260801, 20260802, 20260803)


def _fixture(
    *, days: int = 8, per_day: int = 60, nan_factor: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Days x names with a learnable signal in f0, so boosting has something to find."""
    rng = np.random.default_rng(23)
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(days):
        signal_date = date(2021, 5, 3) + timedelta(days=day_offset)
        for index in range(per_day):
            signal = float(rng.standard_normal())
            values = {name: float(rng.standard_normal()) for name in FACTORS}
            values["f0"] = signal
            if nan_factor is not None and day_offset % 2 == 1:
                values[nan_factor] = float("nan")
            feature_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", **values}
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": 0.8 * signal + 0.2 * float(rng.standard_normal()),
                    "label_valid": True,
                }
            )
    return pd.DataFrame(feature_rows), pd.DataFrame(label_rows)


def _hp(**overrides: object) -> LGBMHyperparameters:
    base = dict(
        num_leaves=15,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=10,
        feature_fraction=0.6,
        num_threads=2,
    )
    base.update(overrides)
    return LGBMHyperparameters(**base)  # type: ignore[arg-type]


def _fit(features: pd.DataFrame, labels: pd.DataFrame, seed: int, **kwargs: object):
    return LGBMRankRegBaseline.fit(
        features,
        labels,
        factor_order=FACTORS,
        role="TRAIN",
        num_boost_round=int(kwargs.pop("num_boost_round", 30)),
        minimum_cross_section=30,
        hyperparameters=_hp(**kwargs),
        seed=seed,
    )


def test_rankreg_is_train_only() -> None:
    features, labels = _fixture(days=2)
    for role in ("VALIDATION", "TEST", "LOCKED"):
        with pytest.raises(PermissionError, match="TRAIN"):
            LGBMRankRegBaseline.fit(
                features, labels, factor_order=FACTORS, role=role, num_boost_round=5  # type: ignore[arg-type]
            )


def test_rankreg_parameters_pin_the_carrier() -> None:
    """objective/metric are the whole point of this carrier; they must not drift."""
    params = _rankreg_training_parameters(_hp(lambda_l2=0.0), seed=20260802)
    assert params["objective"] == "regression"
    # metric="None" suppresses the objective's default RMSE so the RankIC feval is the only
    # thing early stopping can watch -- RMSE of a percentile rank is not what G1/G4 measure.
    assert params["metric"] == "None"
    assert "label_gain" not in params
    assert "lambdarank_truncation_level" not in params
    assert "lambda_l2" not in params  # omitted at 0.0, exactly like the ranker
    assert params["deterministic"] is True
    assert params["force_col_wise"] is True
    assert params["feature_pre_filter"] is False
    assert params["seed"] == params["feature_fraction_seed"] == params["bagging_seed"] == 20260802
    assert params["feature_fraction"] == 0.6

    regularized = _rankreg_training_parameters(_hp(lambda_l2=10.0, feature_fraction=1.0))
    assert regularized["lambda_l2"] == 10.0
    assert "feature_fraction" not in regularized
    assert regularized["seed"] == 20260801


def test_percentile_rank_targets_are_computed_per_day() -> None:
    """The regression target must be a within-day rank, never a cross-day one."""
    dates = np.array([date(2021, 1, 1)] * 4 + [date(2021, 1, 2)] * 4, dtype=object)
    # Day 2's returns are uniformly larger than day 1's; a global rank would leak that.
    returns = np.array([0.01, 0.02, 0.03, 0.04, 5.0, 6.0, 7.0, 8.0])
    target = _percentile_rank_targets(dates, returns)
    assert np.allclose(target[:4], [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(target[4:], [0.25, 0.5, 0.75, 1.0])
    assert target.max() == 1.0 and target.min() == 0.25


def test_daily_rank_ic_metric_matches_a_direct_spearman() -> None:
    dates = np.array([date(2021, 1, 1)] * 40 + [date(2021, 1, 2)] * 40, dtype=object)
    rng = np.random.default_rng(5)
    returns = rng.standard_normal(80)
    scores = 0.7 * returns + 0.3 * rng.standard_normal(80)
    metric = DailyRankICMetric.build(dates, returns, minimum_cross_section=30)

    frame = pd.DataFrame({"signal_date": dates, "r": returns, "s": scores})
    expected = float(
        np.mean(
            [
                group["s"].corr(group["r"], method="spearman")
                for _, group in frame.groupby("signal_date", sort=False)
            ]
        )
    )
    assert metric.evaluate(scores) == pytest.approx(expected, abs=1e-12)
    name, value, higher_is_better = metric(scores, None)  # type: ignore[arg-type]
    assert (name, higher_is_better) == (RANK_IC_METRIC_NAME, True)
    assert value == pytest.approx(expected, abs=1e-12)


def test_daily_rank_ic_metric_skips_days_below_the_floor() -> None:
    """A day too thin to score must not drag the mean, matching compute_daily_rank_ic."""
    dates = np.array([date(2021, 1, 1)] * 40 + [date(2021, 1, 2)] * 5, dtype=object)
    returns = np.arange(45, dtype=float)
    metric = DailyRankICMetric.build(dates, returns, minimum_cross_section=30)
    wide_only = DailyRankICMetric.build(dates[:40], returns[:40], minimum_cross_section=30)
    assert metric.evaluate(returns) == pytest.approx(wide_only.evaluate(returns[:40]), abs=1e-12)


def test_daily_rank_ic_metric_rejects_misaligned_inputs() -> None:
    dates = np.array([date(2021, 1, 1)] * 4, dtype=object)
    with pytest.raises(ValueError, match="same length"):
        DailyRankICMetric.build(dates, np.zeros(5))
    with pytest.raises(ValueError, match="finite"):
        DailyRankICMetric.build(dates, np.array([0.0, np.nan, 1.0, 2.0]))
    metric = DailyRankICMetric.build(dates, np.arange(4.0), minimum_cross_section=2)
    with pytest.raises(ValueError, match="prediction count"):
        metric.evaluate(np.zeros(3))


def test_rankreg_seed_kwarg_keeps_fit_deterministic() -> None:
    features, labels = _fixture()
    first = _fit(features, labels, 20260802)
    second = _fit(features, labels, 20260802)
    assert first.booster.model_to_string() == second.booster.model_to_string()


def test_rankreg_seeds_differ_when_feature_fraction_below_one() -> None:
    """The three-seed contract is vacuous unless the seeds really do diverge."""
    features, labels = _fixture()
    predictions = {seed: _fit(features, labels, seed).predict(features) for seed in SEEDS}
    assert not np.allclose(predictions[20260801], predictions[20260802])
    assert not np.allclose(predictions[20260801], predictions[20260803])
    assert not np.allclose(predictions[20260802], predictions[20260803])


def test_rankreg_fit_is_independent_of_preceding_seed_fits() -> None:
    """Guard the invariant the shared-Dataset hazard breaks.

    Reusing one lgb.Dataset across lgb.train calls makes seed N's model depend on the fits
    that ran before it. That divergence only appears at production scale (~2.3M rows), so
    this fixture-scale test cannot catch the real bug; it pins the property anyway, so a
    future caching refactor that breaks it at ANY scale fails here rather than silently
    changing three boosters in a promotion run.
    """
    features, labels = _fixture()
    alone = {seed: _fit(features, labels, seed) for seed in (20260803,)}
    for seed in SEEDS[:-1]:
        _fit(features, labels, seed)
    after = _fit(features, labels, 20260803)
    assert after.booster.model_to_string() == alone[20260803].booster.model_to_string()


def test_rankreg_fits_and_predicts_with_nan_features_by_default() -> None:
    """The six sparse frozen-v2 factors are ~50% NaN; native missing handling is required."""
    features, labels = _fixture(nan_factor="f3")
    assert features["f3"].isna().any()
    model = _fit(features, labels, 20260801)
    scores = model.predict(features)
    assert np.isfinite(scores.to_numpy()).all()
    assert model.allow_nan is True


def test_rankreg_rejects_infinite_features_even_with_allow_nan() -> None:
    features, labels = _fixture(days=2)
    features.loc[0, "f1"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        _fit(features, labels, 20260801)


def test_rankreg_rejects_a_mismatched_feature_schema() -> None:
    features, labels = _fixture(days=2)
    with pytest.raises(ValueError, match="feature order"):
        LGBMRankRegBaseline.fit(
            features.loc[:, ["signal_date", "ts_code", "f4", "f0", "f1", "f2", "f3"]],
            labels,
            factor_order=FACTORS,
            role="TRAIN",
            num_boost_round=5,
        )


def test_rankreg_requires_evaluation_frames_in_pairs() -> None:
    features, labels = _fixture(days=2)
    with pytest.raises(ValueError, match="supplied together"):
        LGBMRankRegBaseline.fit(
            features,
            labels,
            factor_order=FACTORS,
            role="TRAIN",
            num_boost_round=5,
            evaluation_features=features,
        )


def test_rankreg_early_stops_on_the_validation_frame_and_records_the_iteration() -> None:
    """With a validation frame the booster stops early and predict honours best_iteration."""
    train_features, train_labels = _fixture(days=8)
    val_features, val_labels = _fixture(days=3, per_day=60)
    model = LGBMRankRegBaseline.fit(
        train_features,
        train_labels,
        factor_order=FACTORS,
        role="TRAIN",
        num_boost_round=500,
        minimum_cross_section=30,
        hyperparameters=_hp(),
        evaluation_features=val_features,
        evaluation_labels=val_labels,
        early_stopping_patience=5,
        seed=20260801,
    )
    assert model.best_iteration is not None and model.best_iteration >= 1
    assert model.best_iteration < 500, "early stopping never fired"
    assert model.booster.num_trees() == model.best_iteration
    assert model.training_day_count == 8 == model.eligible_training_day_count
    # a frozen round budget with no validation frame trains the whole budget instead
    unbounded = _fit(train_features, train_labels, 20260801, num_boost_round=12)
    assert unbounded.best_iteration is None
    assert unbounded.booster.num_trees() == 12


def test_rankreg_default_hyperparameters_are_the_measured_config() -> None:
    """The +0.00915 pooled delta was measured at exactly this config, untuned."""
    assert LGBMHyperparameters(
        num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=200,
        feature_fraction=0.85,
    ) == LGBM_RANKREG_DEFAULT_HYPERPARAMETERS
