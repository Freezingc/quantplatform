"""Train-only LightGBM rank-regression candidate: regress the per-day return percentile rank.

Why this lives next to ``lgbm_ranker`` instead of replacing it: measured on the frozen-v2
15-factor set over the chain's 3,417,963 paired TEST rows, the LambdaRank carrier
(``objective=lambdarank``, 5 relevance grades, LightGBM's default
``lambdarank_truncation_level``) reached a pooled OOS delta of only **+0.00109** at one fixed
config, while this carrier reached **+0.00915** on identical rows and the identical config,
and passed all six promotion gates (G1 0.094376, G2 HAC t 12.905, G3 1.00, G4 +0.009149,
G5 0.2791, G6 0.9752). It won every fold: +0.01249 / +0.00319 / +0.01392 / +0.00645.

The mechanism is the truncation. With ~2400-3800 names per trading day, a default
``lambdarank_truncation_level`` of 20 means only the top 20 positions of each day contribute
gradients, while G1/G4 measure a full-cross-section daily Spearman RankIC. Raising the
truncation to 200 alone -- leaving the LambdaRank objective untouched -- is worth +0.0057
(+0.00679, also passing G4), which is the same diagnosis from the other side.

Neither carrier is a hyperparameter story: the LambdaRank side plateaued at +0.004202 only
after a 24-config per-fold search, and the numbers above needed no search at all, so they
carry no validation-selection bias. Full attribution, including the negative results (finer
relevance grades, explicit sparse-missingness flags, ``rank_xendcg``, and swapping the
early-stopping metric alone) is in
``docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md``.

The frozen B3 baseline and the M0 candidate keep using ``lgbm_ranker`` unchanged; this module
is additive so no existing evidence, model_code_version, or bundle digest is disturbed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_v2.baselines.lgbm_ranker import (
    LGBMHyperparameters,
    _finite_matrix,
    _normalize_dates,
    _prepare_ranker_rows,
    _validate_feature_schema,
)
from research_v2.baselines.linear import FitRole

LGBM_RANKREG_MAX_ESTIMATORS = 500
LGBM_RANKREG_EARLY_STOPPING_PATIENCE = 50
LGBM_RANKREG_DEFAULT_SEED = 20260801
RANK_IC_METRIC_NAME = "rank_ic"
# The config the carrier was measured with (fixed, no search): num_leaves 15, max_depth 4,
# lr 0.05, min_child_samples 200, feature_fraction 0.85, lambda_l2 0. It is the dataclass
# default so a booster read back from a verified bundle can be reconstructed without it.
LGBM_RANKREG_DEFAULT_HYPERPARAMETERS = LGBMHyperparameters(
    num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=200, feature_fraction=0.85
)


@dataclass(frozen=True)
class DailyRankICMetric:
    """Mean daily cross-sectional Spearman RankIC, usable as a LightGBM ``feval``.

    This is the metric the promotion gates actually measure, so it is what early stopping
    should watch. Vectorised through ``np.bincount`` because LightGBM calls it once per
    boosting round over the whole evaluation frame (~900k rows, ~240 days on the v2 folds).
    """

    group_ids: np.ndarray
    label_ranks: np.ndarray
    group_sizes: np.ndarray
    eligible_groups: np.ndarray
    _label_rank_sum: np.ndarray
    _label_rank_squares: np.ndarray

    @classmethod
    def build(
        cls, signal_dates: np.ndarray, raw_returns: np.ndarray, *, minimum_cross_section: int = 30
    ) -> DailyRankICMetric:
        """Build the metric from an evaluation frame already filtered to valid labels."""
        if minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")
        if len(signal_dates) != len(raw_returns):
            raise ValueError("signal_dates and raw_returns must have the same length")
        if not np.isfinite(raw_returns).all():
            raise ValueError("rank-regression targets must be finite")
        codes, uniques = pd.factorize(pd.Series(signal_dates), sort=False)
        group_ids = codes.astype(np.int64)
        if len(group_ids) and not bool((np.diff(group_ids) >= 0).all()):
            raise ValueError("evaluation dates must arrive in contiguous per-day blocks")
        group_sizes = np.bincount(group_ids, minlength=len(uniques)).astype(float)
        label_ranks = (
            pd.Series(raw_returns).groupby(group_ids).rank(method="average").to_numpy(dtype=float)
        )
        return cls(
            group_ids=group_ids,
            label_ranks=label_ranks,
            group_sizes=group_sizes,
            eligible_groups=group_sizes >= minimum_cross_section,
            _label_rank_sum=np.bincount(group_ids, weights=label_ranks, minlength=len(uniques)),
            _label_rank_squares=np.bincount(
                group_ids, weights=label_ranks * label_ranks, minlength=len(uniques)
            ),
        )

    def evaluate(self, predictions: np.ndarray) -> float:
        """Mean daily Spearman RankIC over the days wide enough to score."""
        if len(predictions) != len(self.group_ids):
            raise ValueError("prediction count does not match the evaluation frame")
        score_ranks = (
            pd.Series(np.asarray(predictions, dtype=float))
            .groupby(self.group_ids)
            .rank(method="average")
            .to_numpy(dtype=float)
        )
        count = self.group_sizes
        score_sum = np.bincount(self.group_ids, weights=score_ranks, minlength=len(count))
        score_squares = np.bincount(
            self.group_ids, weights=score_ranks * score_ranks, minlength=len(count)
        )
        cross = np.bincount(
            self.group_ids, weights=score_ranks * self.label_ranks, minlength=len(count)
        )
        covariance = cross - score_sum * self._label_rank_sum / count
        score_variance = score_squares - score_sum * score_sum / count
        label_variance = self._label_rank_squares - self._label_rank_sum**2 / count
        with np.errstate(invalid="ignore", divide="ignore"):
            daily = covariance / np.sqrt(score_variance * label_variance)
        usable = self.eligible_groups & np.isfinite(daily)
        if not usable.any():
            return 0.0
        return float(daily[usable].mean())

    def __call__(self, predictions: np.ndarray, dataset: lgb.Dataset) -> tuple[str, float, bool]:
        """LightGBM ``feval`` protocol; higher is better."""
        del dataset  # the ranks are precomputed from the evaluation frame
        return RANK_IC_METRIC_NAME, self.evaluate(predictions), True


def _rankreg_training_parameters(
    parameters: LGBMHyperparameters, *, seed: int | None = None
) -> dict[str, Any]:
    """LightGBM params for the rank-regression carrier.

    ``metric="None"`` suppresses the objective's default metric so ``DailyRankICMetric`` is
    the only thing early stopping can watch; without it a regression objective would stop on
    RMSE of the percentile rank, which is not what the gates measure.
    """
    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "deterministic": True,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "learning_rate": parameters.learning_rate,
        "max_depth": parameters.max_depth,
        "min_child_samples": parameters.min_child_samples,
        "num_leaves": parameters.num_leaves,
        "num_threads": parameters.num_threads,
        "objective": "regression",
        "metric": "None",
        "seed": LGBM_RANKREG_DEFAULT_SEED,
        "verbosity": -1,
    }
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("LightGBM seed must be a non-negative int")
        params["seed"] = seed
        params["feature_fraction_seed"] = seed
        params["bagging_seed"] = seed
    if parameters.lambda_l2 != 0.0:
        params["lambda_l2"] = parameters.lambda_l2
    if parameters.feature_fraction != 1.0:
        params["feature_fraction"] = parameters.feature_fraction
    if parameters.bagging_fraction != 1.0:
        params["bagging_fraction"] = parameters.bagging_fraction
        params["bagging_freq"] = parameters.bagging_freq
    return params


def _percentile_rank_targets(signal_dates: np.ndarray, raw_returns: np.ndarray) -> np.ndarray:
    """Per-day percentile rank of the 5-day return -- the continuous regression target."""
    return (
        pd.Series(raw_returns)
        .groupby(pd.factorize(pd.Series(signal_dates), sort=False)[0].astype(np.int64))
        .rank(pct=True)
        .to_numpy(dtype=float)
    )


@dataclass(frozen=True)
class LGBMRankRegBaseline:
    """A LightGBM booster regressing the per-day return percentile rank.

    Deliberately the same shape as ``LGBMRankerBaseline`` so the candidate driver's bundle,
    checkpoint and seed-ensemble machinery serves both carriers without a second code path.
    """

    factor_order: tuple[str, ...]
    booster: object
    num_boost_round: int
    hyperparameters: LGBMHyperparameters = LGBM_RANKREG_DEFAULT_HYPERPARAMETERS
    training_day_count: int = 0
    eligible_training_day_count: int = 0
    best_iteration: int | None = None
    # NaN features go to LightGBM natively (it learns a split direction for "missing"), which
    # is what the six sparse frozen-v2 factors need: "no value on a down day" stays
    # information instead of becoming an imputation.
    allow_nan: bool = True

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        factor_order: tuple[str, ...],
        role: FitRole,
        num_boost_round: int = LGBM_RANKREG_MAX_ESTIMATORS,
        minimum_cross_section: int = 30,
        hyperparameters: LGBMHyperparameters | None = None,
        evaluation_features: pd.DataFrame | None = None,
        evaluation_labels: pd.DataFrame | None = None,
        early_stopping_patience: int = LGBM_RANKREG_EARLY_STOPPING_PATIENCE,
        seed: int | None = None,
        allow_nan: bool = True,
    ) -> LGBMRankRegBaseline:
        """Fit on TRAIN only; early-stop on the evaluation frame's daily RankIC."""
        if role != "TRAIN":
            raise PermissionError("LGBMRankRegBaseline.fit is permitted only for TRAIN data")
        if not factor_order or len(set(factor_order)) != len(factor_order):
            raise ValueError("factor_order must be a non-empty sequence of unique names")
        if num_boost_round < 1 or minimum_cross_section < 2 or early_stopping_patience < 1:
            raise ValueError("training parameters are invalid")
        selected = hyperparameters or LGBM_RANKREG_DEFAULT_HYPERPARAMETERS
        if (evaluation_features is None) != (evaluation_labels is None):
            raise ValueError("evaluation_features and evaluation_labels must be supplied together")
        _validate_feature_schema(features, factor_order)
        joined, training_day_count, eligible_training_day_count = _prepare_ranker_rows(
            _normalize_dates(features),
            _normalize_dates(labels),
            factor_order=factor_order,
            minimum_cross_section=minimum_cross_section,
        )
        values = _finite_matrix(joined, factor_order, allow_nan=allow_nan)
        target = _percentile_rank_targets(
            joined["signal_date"].to_numpy(),
            pd.to_numeric(joined["raw_return_5d"], errors="coerce").to_numpy(dtype=float),
        )
        # A FRESH Dataset per fit is mandatory, never a hoisted or cached one. Reusing a
        # single lgb.Dataset across lgb.train calls silently changes the fitted model:
        # measured on fold F1 (2,283,604 prepared rows, feature_fraction 0.85,
        # num_threads 16), a Dataset shared across the three REQUIRED_SEEDS reproduced the
        # first seed's booster but grew 79 and 64 trees for the other two where fresh
        # Datasets grow 2 and 2. It is not a NaN effect and does not appear below ~18k rows,
        # so no fixture-scale test can catch it. See lgbm_ranker.fit for the same guard.
        dataset = lgb.Dataset(values, label=target)
        evaluation_set: lgb.Dataset | None = None
        feval: Callable[[np.ndarray, lgb.Dataset], tuple[str, float, bool]] | None = None
        callbacks: list[Callable[..., Any]] = []
        if evaluation_features is not None and evaluation_labels is not None:
            _validate_feature_schema(evaluation_features, factor_order)
            evaluation_joined, _, _ = _prepare_ranker_rows(
                _normalize_dates(evaluation_features),
                _normalize_dates(evaluation_labels),
                factor_order=factor_order,
                minimum_cross_section=minimum_cross_section,
            )
            evaluation_values = _finite_matrix(evaluation_joined, factor_order, allow_nan=allow_nan)
            evaluation_target = _percentile_rank_targets(
                evaluation_joined["signal_date"].to_numpy(),
                pd.to_numeric(evaluation_joined["raw_return_5d"], errors="coerce").to_numpy(
                    dtype=float
                ),
            )
            evaluation_set = lgb.Dataset(
                evaluation_values, label=evaluation_target, reference=dataset
            )
            feval = DailyRankICMetric.build(
                evaluation_joined["signal_date"].to_numpy(),
                pd.to_numeric(evaluation_joined["raw_return_5d"], errors="coerce").to_numpy(
                    dtype=float
                ),
                minimum_cross_section=minimum_cross_section,
            )
            callbacks.append(lgb.early_stopping(early_stopping_patience, verbose=False))
        booster = lgb.train(
            _rankreg_training_parameters(selected, seed=seed),
            dataset,
            num_boost_round=num_boost_round,
            valid_sets=[evaluation_set] if evaluation_set is not None else None,
            valid_names=["inner_validation"] if evaluation_set is not None else None,
            feval=feval,
            callbacks=callbacks or None,
        )
        best_iteration = getattr(booster, "best_iteration", None)
        return cls(
            factor_order=factor_order,
            booster=booster,
            num_boost_round=num_boost_round,
            hyperparameters=selected,
            training_day_count=training_day_count,
            eligible_training_day_count=eligible_training_day_count,
            best_iteration=int(best_iteration) if best_iteration else None,
            allow_nan=allow_nan,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Score a validation or test frame with the immutable trained booster."""
        _validate_feature_schema(features, self.factor_order)
        values = _finite_matrix(features, self.factor_order, allow_nan=self.allow_nan)
        predict = getattr(self.booster, "predict", None)
        if not callable(predict):
            raise ValueError("stored LightGBM booster cannot predict")
        scores = cast(
            np.ndarray, predict(values, num_iteration=self.best_iteration or None)
        )
        if not np.isfinite(scores).all():
            raise ValueError("LightGBM emitted non-finite scores")
        return pd.Series(scores, index=features.index, name="score")
