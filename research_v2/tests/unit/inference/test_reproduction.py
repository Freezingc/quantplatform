"""Tests for the prediction-reproduction verifier.

The property that matters most is non-vacuity: a verifier that always passes is worse than none,
because it turns "never checked" into "checked and matched". So the sensitivity cases are pinned
alongside the happy path, including exactness -- there is no float tolerance to hide behind.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from research_v2.datasets.manifest import build_split_dataset_manifest
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.inference import reproduction
from research_v2.inference.artifact_loader import load_bundle_for_evaluation
from research_v2.inference.reproduction import (
    FoldReproduction,
    ReproductionError,
    _compare_scores,
    _normalized,
)
from research_v2.inference.torch_scoring import concat_date_scores, prepare_frame, score_batch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.registry.registry_store import RegistryStoreError
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore
from research_v2.training.static_batches import build_static_training_batches

KEYS = ["signal_date", "ts_code"]
DAYS = [date(2025, 1, 2), date(2025, 1, 3)]


def _scores(values: list[float]) -> pd.DataFrame:
    rows = [
        {"signal_date": day, "ts_code": f"S{i:03d}", "score": value}
        for day in DAYS
        for i, value in enumerate(values[:3])
    ]
    return pd.DataFrame(rows)


def _compare(left: pd.DataFrame, right: pd.DataFrame) -> FoldReproduction:
    return _compare_scores(
        left, right, fold_id="F1", model_artifact_id="m0r-f1", family="LGBM_RANKREG"
    )


def test_identical_scores_reproduce_exactly() -> None:
    frame = _scores([0.25, 0.5, 0.75])
    result = _compare(frame, frame.copy())
    assert result.identical is True
    assert result.max_absolute_difference == 0.0
    assert result.row_count == 6
    assert result.fold_id == "F1"
    assert result.model_artifact_id == "m0r-f1"
    assert result.family == "LGBM_RANKREG"


def test_a_single_changed_value_is_refused() -> None:
    """The core sensitivity check: one row out of six must fail, not be averaged away."""
    stored = _scores([0.25, 0.5, 0.75])
    stored.loc[3, "score"] = 0.99
    with pytest.raises(ReproductionError, match="max absolute difference"):
        _compare(_scores([0.25, 0.5, 0.75]), stored)


def test_the_comparison_has_no_float_tolerance() -> None:
    """A one-ulp difference is still a difference; an epsilon here would hide real divergence."""
    stored = _scores([0.25, 0.5, 0.75])
    stored.loc[0, "score"] = stored.loc[0, "score"] + 1e-16
    with pytest.raises(ReproductionError, match="differ from what bundle") as info:
        _compare(_scores([0.25, 0.5, 0.75]), stored)
    assert "e-16" in str(info.value), "a one-ulp difference must be reported, not tolerated"


def test_a_missing_stored_row_is_refused_with_counts() -> None:
    stored = _scores([0.25, 0.5, 0.75]).iloc[:-1]
    with pytest.raises(ReproductionError, match="row sets differ"):
        _compare(_scores([0.25, 0.5, 0.75]), stored)


def test_an_extra_stored_row_is_refused_with_counts() -> None:
    recomputed = _scores([0.25, 0.5, 0.75]).iloc[:-1]
    with pytest.raises(ReproductionError, match="row sets differ"):
        _compare(recomputed, _scores([0.25, 0.5, 0.75]))


def test_a_permuted_row_order_still_reproduces() -> None:
    """Comparison is on business keys, so row order must not matter."""
    frame = _scores([0.25, 0.5, 0.75])
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    assert _compare(frame, shuffled).identical is True


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(lambda f: f.drop(columns=["score"]), "missing columns", id="no_score"),
        pytest.param(lambda f: f.drop(columns=["ts_code"]), "missing columns", id="no_key"),
        pytest.param(
            lambda f: pd.concat([f, f.head(1)], ignore_index=True),
            "duplicate business keys",
            id="duplicates",
        ),
        pytest.param(
            lambda f: f.assign(score=["x"] * len(f)), "non-numeric", id="non_numeric"
        ),
    ],
)
def test_normalized_refuses_a_malformed_score_frame(mutate, match: str) -> None:
    with pytest.raises(ReproductionError, match=match):
        _normalized(mutate(_scores([0.25, 0.5, 0.75])), "recomputed")


def test_normalized_aligns_key_types_so_the_join_is_real() -> None:
    """Timestamps and date objects must meet as one type, or the join silently matches nothing."""
    frame = _scores([0.25, 0.5, 0.75])
    as_timestamps = frame.assign(signal_date=pd.to_datetime(frame["signal_date"]))
    left = _normalized(frame, "recomputed")
    right = _normalized(as_timestamps, "stored")
    assert _compare(left, right).row_count == 6


# ---------------------------------------------------------------------------------------------
# The audit's authority: the registry status is reported, never enforced.
# ---------------------------------------------------------------------------------------------


class _StubRecord:
    def __init__(self, status: str) -> None:
        self.status = _StubStatus(status)


class _StubStatus:
    def __init__(self, value: str) -> None:
        self.value = value


class _StubRegistry:
    def __init__(self, records: dict[str, _StubRecord]) -> None:
        self._records = records

    def get(self, model_id: str) -> _StubRecord:
        try:
            return self._records[model_id]
        except KeyError as error:
            raise ValueError(f"unknown model {model_id}") from error


class _StubStore:
    def __init__(self, records: dict[str, _StubRecord]) -> None:
        self.registry = _StubRegistry(records)


class _CorruptStore:
    @property
    def registry(self) -> Any:
        raise RegistryStoreError("registry signature does not verify")


def _identity(store: Any, monkeypatch: pytest.MonkeyPatch) -> reproduction.RunIdentity:
    monkeypatch.setattr(
        reproduction, "walk_forward_model_id", lambda workspace, report: "sha256:" + "a" * 64
    )
    return reproduction._run_identity(Path("/unused"), Path("/unused/report.json"), store)


def test_an_unregistered_run_is_reported_rather_than_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runs worth auditing most are the ones that failed a gate and never reached registry."""
    identity = _identity(_StubStore({}), monkeypatch)
    assert identity.model_id == "sha256:" + "a" * 64
    assert identity.status == "UNREGISTERED"


@pytest.mark.parametrize("status", ["DRAFT", "WALK_FORWARD_PASSED", "RETIRED"])
def test_a_registered_run_reports_the_status_the_registry_holds(
    status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_id = "sha256:" + "a" * 64
    identity = _identity(_StubStore({model_id: _StubRecord(status)}), monkeypatch)
    assert identity.status == status


def test_a_corrupt_registry_is_not_reported_as_unregistered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Swallowing RegistryStoreError would turn "cannot open the registry" into "not registered"."""
    with pytest.raises(RegistryStoreError, match="signature does not verify"):
        _identity(_CorruptStore(), monkeypatch)


# ---------------------------------------------------------------------------------------------
# The bundle must belong to this fold's report entry.
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("recorded", ["ds-1", "ds-1:train"])
def test_a_card_dataset_id_derived_from_the_report_is_accepted(recorded: str) -> None:
    """LightGBM records the train split it fitted on; torch records the bare test dataset id."""
    card = {"dataset_id": recorded}
    assert reproduction._checked_card_dataset_id(card, "ds-1", fold_id="F1") == recorded


@pytest.mark.parametrize("recorded", ["ds-2", "ds-1:test", "ds-11:train", "", None, 7])
def test_a_card_dataset_id_from_anywhere_else_is_refused(recorded: object) -> None:
    card = {"dataset_id": recorded}
    with pytest.raises(ReproductionError, match="F1 bundle records dataset_id") as info:
        reproduction._checked_card_dataset_id(card, "ds-1", fold_id="F1")
    message = str(info.value)
    assert "'ds-1'" in message and "'ds-1:train'" in message


@pytest.mark.parametrize("card", [{}, {"dataset_id": 7}, {"family": "LGBM_RANKER"}])
def test_a_card_without_a_dataset_id_is_refused(card: dict[str, object]) -> None:
    with pytest.raises(ReproductionError, match="dataset_id"):
        reproduction._checked_card_dataset_id(card, "ds-1", fold_id="F1")


@pytest.mark.parametrize("card", [{}, {"family": ""}, {"family": 7}, {"family": None}])
def test_a_card_without_a_usable_family_is_refused(card: dict[str, object]) -> None:
    with pytest.raises(ReproductionError, match="F1 bundle model card records no usable family"):
        reproduction._family_of(card, "F1")


# ---------------------------------------------------------------------------------------------
# Family routing: each family reaches its own scorer, and nothing else.
# ---------------------------------------------------------------------------------------------


def _bundle(
    family: str,
    *,
    dataset_id: str = "ds-1",
    test: pd.DataFrame | None = None,
    store: ModelArtifactBundleStore | None = None,
    workspace: Path | None = None,
) -> reproduction._FoldBundle:
    return reproduction._FoldBundle(
        fold_id="F1",
        model_artifact_id="m1-f1",
        family=family,
        dataset_id=dataset_id,
        report_dataset_id=dataset_id,
        factor_order=("f1", "f2"),
        store=store if store is not None else ModelArtifactBundleStore(Path("/nonexistent-models")),
        workspace=workspace if workspace is not None else Path("/nonexistent-workspace"),
        test=_scores([0.25, 0.5, 0.75]) if test is None else test,
    )


def _recorders(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace both scorers with recorders, so routing is observable and nothing heavy runs."""
    calls: dict[str, list[Any]] = {"lgbm": [], "torch": [], "temporal": []}

    def lgbm(bundle: reproduction._FoldBundle) -> pd.DataFrame:
        calls["lgbm"].append(bundle)
        return _scores([0.25, 0.5, 0.75])

    def static_torch(bundle: reproduction._FoldBundle) -> pd.DataFrame:
        calls["torch"].append(bundle)
        return _scores([0.25, 0.5, 0.75])

    def temporal_torch(bundle: reproduction._FoldBundle) -> pd.DataFrame:
        calls["temporal"].append(bundle)
        return _scores([0.25, 0.5, 0.75])

    monkeypatch.setattr(reproduction, "_recompute_lgbm", lgbm)
    monkeypatch.setattr(reproduction, "_recompute_static_torch", static_torch)
    monkeypatch.setattr(reproduction, "_recompute_temporal_torch", temporal_torch)
    return calls


@pytest.mark.parametrize("family", ["LGBM_RANKER", "LGBM_RANKREG"])
def test_the_lgbm_families_route_to_the_lgbm_scorer_only(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorders(monkeypatch)
    reproduction._recompute_fold_scores(_bundle(family))
    assert len(calls["lgbm"]) == 1
    assert calls["torch"] == [] and calls["temporal"] == []


@pytest.mark.parametrize("family", ["RESIDUAL_MLP", "SET_ATTENTION"])
def test_the_static_torch_families_route_to_the_torch_scorer_only(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorders(monkeypatch)
    reproduction._recompute_fold_scores(_bundle(family))
    assert len(calls["torch"]) == 1
    assert calls["lgbm"] == [] and calls["temporal"] == []


@pytest.mark.parametrize("family", ["TEMPORAL_GRU", "TEMPORAL_TCN"])
def test_the_temporal_families_route_to_the_temporal_scorer(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2's card is named by the winning config, so both spellings have to reach the same scorer.

    Routing a temporal bundle to the static scorer would not fail loudly: it would score the frozen
    TEST frame without its history window and produce numbers that merely look like ranks.
    """
    calls = _recorders(monkeypatch)
    reproduction._recompute_fold_scores(_bundle(family))
    assert len(calls["temporal"]) == 1
    assert calls["lgbm"] == [] and calls["torch"] == []


@pytest.mark.parametrize("family", ["SOMETHING_NEW", "residual_mlp", "LGBM"])
def test_an_unallowlisted_family_is_refused_by_name(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _recorders(monkeypatch)
    with pytest.raises(ReproductionError, match="not allowlisted for reproduction") as info:
        reproduction._recompute_fold_scores(_bundle(family))
    assert repr(family) in str(info.value)
    assert calls == {"lgbm": [], "torch": [], "temporal": []}


def test_the_lgbm_path_loads_for_evaluation_and_passes_no_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The serving loader refuses an unpromoted run, so the audit must not go through it."""
    captured: dict[str, Any] = {}
    scores = _scores([0.25, 0.5, 0.75])

    def fake_load(store: Any, model_id: str, **kwargs: Any) -> object:
        captured.update(kwargs)
        captured["model_id"] = model_id
        return object()

    monkeypatch.setattr(reproduction, "load_lgbm_bundle_for_evaluation", fake_load)
    monkeypatch.setattr(reproduction, "score_lgbm_artifact", lambda loaded, frame: scores)
    out = reproduction._recompute_lgbm(_bundle("LGBM_RANKREG", dataset_id="ds-1:train"))
    assert out is scores
    assert captured["model_id"] == "m1-f1"
    assert captured["expected_dataset_id"] == "ds-1:train"
    assert captured["expected_feature_order"] == ("f1", "f2")
    assert "promoted_run" not in captured


# ---------------------------------------------------------------------------------------------
# The torch path must repeat the driver's scoring steps, not a similar-looking ones.
# ---------------------------------------------------------------------------------------------


class _StubLoaded:
    def __init__(self, preprocessor: object, models: tuple[object, ...] = ()) -> None:
        self.preprocessor = preprocessor
        self.models = models


def test_the_torch_path_repeats_the_driver_scoring_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load for evaluation on cpu, zero the label, preprocess, batch per date, average the seeds."""
    test = _scores([0.25, 0.5, 0.75]).assign(label_valid=True)
    steps: list[tuple[str, Any]] = []
    preprocessor = object()
    models = (object(), object(), object())
    prepared = _scores([0.1, 0.2, 0.3])
    batches = {date(2025, 1, 2): object()}

    def fake_load(store: Any, model_id: str, **kwargs: Any) -> _StubLoaded:
        steps.append(("load", kwargs))
        return _StubLoaded(preprocessor, models)

    def fake_prepare(frame: pd.DataFrame, fitted: Any, factor_order: tuple[str, ...]) -> Any:
        steps.append(("prepare", (frame, fitted, factor_order)))
        return prepared

    def fake_batches(frame: Any, *, factor_order: tuple[str, ...]) -> Any:
        steps.append(("batches", (frame, factor_order)))
        return batches

    def fake_concat(loaded_models: Any, built: Any) -> pd.DataFrame:
        steps.append(("concat", (loaded_models, built)))
        return _scores([0.25, 0.5, 0.75])

    monkeypatch.setattr(reproduction, "load_bundle_for_evaluation", fake_load)
    monkeypatch.setattr(reproduction, "prepare_frame", fake_prepare)
    monkeypatch.setattr(reproduction, "build_static_training_batches", fake_batches)
    monkeypatch.setattr(reproduction, "concat_date_scores", fake_concat)
    monkeypatch.setattr(reproduction, "_recorded_preprocessor_hash", lambda bundle: "state-h")

    reproduction._recompute_static_torch(_bundle("RESIDUAL_MLP", test=test))

    assert [name for name, _ in steps] == ["load", "prepare", "batches", "concat"]
    load_kwargs = steps[0][1]
    assert load_kwargs["expected_preprocessor_hash"] == "state-h"
    assert str(load_kwargs["device"]) == "cpu"
    assert load_kwargs["expected_feature_order"] == ("f1", "f2")

    prepared_frame, fitted, factor_order = steps[1][1]
    assert fitted is preprocessor
    assert factor_order == ("f1", "f2")
    # The driver scores with the label column zeroed: the batch builder requires a target, and
    # scoring must not be able to read the thing being predicted.
    assert (prepared_frame["raw_return_5d"] == 0.0).all()
    assert list(prepared_frame["ts_code"]) == list(test["ts_code"])

    assert steps[2][1] == (prepared, ("f1", "f2"))
    assert steps[3][1] == (models, batches)


def test_a_torch_bundle_that_restored_no_preprocessing_state_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Weights without the fitted state cannot put a frame into the space they were trained in."""
    monkeypatch.setattr(
        reproduction, "load_bundle_for_evaluation", lambda *a, **k: _StubLoaded(None)
    )
    monkeypatch.setattr(reproduction, "_recorded_preprocessor_hash", lambda bundle: "state-h")
    with pytest.raises(ReproductionError, match="restored no preprocessing state"):
        reproduction._recompute_static_torch(_bundle("SET_ATTENTION"))


def test_a_bundle_with_no_readable_preprocessing_artifact_is_refused(tmp_path: Path) -> None:
    bundle = _bundle("RESIDUAL_MLP", store=ModelArtifactBundleStore(tmp_path))
    with pytest.raises(ReproductionError, match="no readable preprocessing artifact"):
        reproduction._recorded_preprocessor_hash(bundle)


@pytest.mark.parametrize("payload", [{"state_hash": ""}, {"state_hash": 7}, {}])
def test_a_preprocessing_artifact_without_a_usable_hash_is_refused(
    payload: object, tmp_path: Path
) -> None:
    """An empty or absent hash would let a bundle with no fitted state reach the loader."""
    bundle_root = tmp_path / "m1-f1"
    bundle_root.mkdir(parents=True)
    (bundle_root / "preprocessing.json").write_text(json.dumps(payload), encoding="utf-8")
    bundle = _bundle("RESIDUAL_MLP", store=ModelArtifactBundleStore(tmp_path))
    with pytest.raises(ReproductionError, match="records no preprocessor state hash"):
        reproduction._recorded_preprocessor_hash(bundle)


def test_a_preprocessing_artifact_that_is_not_an_object_is_refused(tmp_path: Path) -> None:
    bundle_root = tmp_path / "m1-f1"
    bundle_root.mkdir(parents=True)
    (bundle_root / "preprocessing.json").write_text("[]", encoding="utf-8")
    bundle = _bundle("RESIDUAL_MLP", store=ModelArtifactBundleStore(tmp_path))
    with pytest.raises(ReproductionError, match="preprocessing artifact is not an object"):
        reproduction._recorded_preprocessor_hash(bundle)


# ---------------------------------------------------------------------------------------------
# The static torch path against a genuine bundle: real weights, real fitted preprocessing state,
# real forward passes. Everything above this line stubs the loader; this is what the m1_v2 audit
# will do on production artifacts, exercised here so the chain is known to close before then.
# ---------------------------------------------------------------------------------------------

_TWO_FACTORS = ("factor_a", "factor_b")
_TWO_DAYS = (date(2025, 1, 2), date(2025, 1, 3))
_STOCKS_PER_DAY = 32


def _real_bundle(tmp_path: Path) -> tuple[ModelArtifactBundleStore, str, str]:
    """Write a three-seed RESIDUAL_MLP bundle the way the driver does, weights and state alike."""
    rng = np.random.default_rng(7)
    fitting_frame = pd.DataFrame({name: rng.standard_normal(256) for name in _TWO_FACTORS})
    fitted = FoldPreprocessor(factor_order=_TWO_FACTORS)
    fitted.fit(fitting_frame, role="TRAIN")
    config: dict[str, Any] = {
        "raw_factor_count": len(_TWO_FACTORS),
        "hidden_dim": 8,
        "num_blocks": 1,
        "dropout": 0.0,
    }
    seed_checkpoints: dict[int, Path] = {}
    prototype: ResidualMLP | None = None
    for seed in REQUIRED_SEEDS:
        torch.manual_seed(seed)
        model = ResidualMLP(ResidualMLPConfig(**config))
        prototype = model if prototype is None else prototype
        path = tmp_path / f"seed-{seed}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        seed_checkpoints[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")
    assert prototype is not None
    store.write(
        "m1-f1",
        model_card={
            "dataset_id": "ds-1",
            "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "residual-mlp-1",
        },
        config=config,
        signature=asdict(prototype.artifact_signature()),
        feature_order=_TWO_FACTORS,
        preprocessing={
            "state_hash": fitted.state_hash,
            "state_payload": fitted.state_payload(),
        },
        seed_checkpoints=seed_checkpoints,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    return store, "m1-f1", fitted.state_hash


def _real_frame(*, labels: float | None = None) -> pd.DataFrame:
    """A PIT-shaped TEST frame wide enough for the batch builder's 30-stock minimum per date."""
    rng = np.random.default_rng(11)
    rows: list[dict[str, Any]] = []
    for day in _TWO_DAYS:
        for index in range(_STOCKS_PER_DAY):
            # The label draw happens whether or not it is used, so a frame built with a constant
            # label consumes the same random numbers as one with real labels and differs from it in
            # the label column only. Otherwise "scores do not depend on labels" would be comparing
            # two different factor matrices and passing or failing for the wrong reason.
            drawn = float(rng.standard_normal()) * 0.01
            row: dict[str, Any] = {
                "signal_date": day,
                "ts_code": f"S{index:03d}",
                "raw_return_5d": drawn if labels is None else labels,
                "label_valid": True,
            }
            for name in _TWO_FACTORS:
                row[name] = float(rng.standard_normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _torch_bundle(tmp_path: Path, test: pd.DataFrame) -> reproduction._FoldBundle:
    store, model_id, _ = _real_bundle(tmp_path)
    return reproduction._FoldBundle(
        fold_id="F1",
        model_artifact_id=model_id,
        family="RESIDUAL_MLP",
        dataset_id="ds-1",
        report_dataset_id="ds-1",
        factor_order=_TWO_FACTORS,
        store=store,
        workspace=tmp_path,
        test=test,
    )


def test_the_static_torch_path_closes_on_a_real_bundle(tmp_path: Path) -> None:
    scores = reproduction._recompute_static_torch(_torch_bundle(tmp_path, _real_frame()))
    assert set(scores.columns) == {"signal_date", "ts_code", "score"}
    assert len(scores) == len(_TWO_DAYS) * _STOCKS_PER_DAY
    assert not scores.duplicated(["signal_date", "ts_code"]).any()
    for _, group in scores.groupby("signal_date"):
        assert len(group) == _STOCKS_PER_DAY
        assert group["score"].between(0.0, 1.0).all()
        assert group["score"].nunique() > 1, "a collapsed cross-section is not a ranking"


def test_the_recomputation_is_a_function_of_the_bundle_and_the_frame(tmp_path: Path) -> None:
    """Exact determinism is the whole claim: an audit that drifted could not accuse anyone."""
    bundle = _torch_bundle(tmp_path, _real_frame())
    first = reproduction._recompute_static_torch(bundle)
    second = reproduction._recompute_static_torch(bundle)
    pd.testing.assert_frame_equal(first, second, check_exact=True)


def test_the_audit_scores_the_seed_ensemble_and_not_one_convenient_seed(tmp_path: Path) -> None:
    store, model_id, state_hash = _real_bundle(tmp_path)
    test = _real_frame()
    loaded = load_bundle_for_evaluation(
        store,
        model_id,
        expected_dataset_id="ds-1",
        expected_feature_order=_TWO_FACTORS,
        expected_preprocessor_hash=state_hash,
        device=torch.device("cpu"),
    )
    assert loaded.preprocessor is not None
    batches = build_static_training_batches(
        prepare_frame(test.assign(raw_return_5d=0.0), loaded.preprocessor, _TWO_FACTORS),
        factor_order=_TWO_FACTORS,
    )
    expected = concat_date_scores(loaded.models, batches)
    bundle = reproduction._FoldBundle(
        fold_id="F1",
        model_artifact_id=model_id,
        family="RESIDUAL_MLP",
        dataset_id="ds-1",
        report_dataset_id="ds-1",
        factor_order=_TWO_FACTORS,
        store=store,
        workspace=tmp_path,
        test=test,
    )
    audited = reproduction._recompute_static_torch(bundle)
    merged = expected.merge(
        audited, on=["signal_date", "ts_code"], suffixes=("_expected", "_audited"),
        validate="one_to_one",
    )
    assert len(merged) == len(expected)
    assert float((merged["score_expected"] - merged["score_audited"]).abs().max()) == 0.0
    # And the ensemble is not degenerate: three identically-initialised seeds would make the mean
    # equal to any one seed, which would hide a scorer that only ran the first.
    day = _TWO_DAYS[0]
    ranks = [
        score_batch(model, batches[day])
        .set_index("ts_code")["percentile_rank"]
        .sort_index()
        .to_numpy()
        for model in loaded.models
    ]
    assert not all(np.array_equal(ranks[0], other) for other in ranks[1:])


def test_the_label_column_is_zeroed_so_an_unusable_test_label_cannot_drop_rows(
    tmp_path: Path,
) -> None:
    """The driver scores TEST with raw_return_5d zeroed; the audit has to do the same.

    A frozen TEST split whose forward returns are not yet knowable carries a non-finite label, and
    the batch builder keeps only rows with a finite target. Zeroing the column is what makes the
    row set a property of the universe rather than of the labels -- drop the assignment and this
    frame scores zero rows instead of sixty-four.
    """
    unusable = _real_frame(labels=float("nan"))
    scores = reproduction._recompute_static_torch(_torch_bundle(tmp_path, unusable))
    assert len(scores) == len(_TWO_DAYS) * _STOCKS_PER_DAY


def test_the_scores_do_not_depend_on_the_stored_labels_at_all(tmp_path: Path) -> None:
    # Separate roots: a bundle is immutable, so the two frames cannot share one workspace.
    (tmp_path / "real").mkdir()
    (tmp_path / "zeroed").mkdir()
    real = reproduction._recompute_static_torch(
        _torch_bundle(tmp_path / "real", _real_frame())
    )
    zeroed = reproduction._recompute_static_torch(
        _torch_bundle(tmp_path / "zeroed", _real_frame(labels=0.0))
    )
    pd.testing.assert_frame_equal(real, zeroed, check_exact=True)


# ---------------------------------------------------------------------------------------------
# The temporal path: a real TemporalGRU bundle over a real published fold, whose TEST scores come
# from a window panel rather than from the frozen frame alone.
# ---------------------------------------------------------------------------------------------

_TEMPORAL_TRAIN_DAYS = [date(2025, 1, day) for day in range(1, 21)]
_TEMPORAL_VALIDATION_DAYS = [date(2025, 1, day) for day in range(21, 31)]
_TEMPORAL_TEST_DAYS = [date(2025, 2, day) for day in range(1, 6)]
# Two stocks per TEST date carry an unusable label, so the temporal panel drops them and the
# remaining thirty sit exactly on the batch builder's per-date minimum.
_TEMPORAL_UNLABELLED = 2


def _temporal_frame(
    days: list[date], rng: np.random.Generator, *, unlabelled: int = 0
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in days:
        for index in range(_STOCKS_PER_DAY):
            row: dict[str, Any] = {
                "signal_date": day,
                "ts_code": f"S{index:03d}",
                "raw_return_5d": float(rng.standard_normal()) * 0.01,
                "label_valid": index >= unlabelled,
            }
            for name in _TWO_FACTORS:
                row[name] = float(rng.standard_normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _temporal_bundle(tmp_path: Path) -> reproduction._FoldBundle:
    """Publish a fold and freeze a real three-seed TemporalGRU bundle over it."""
    rng = np.random.default_rng(17)
    frames = {
        "train": _temporal_frame(_TEMPORAL_TRAIN_DAYS, rng),
        "validation": _temporal_frame(_TEMPORAL_VALIDATION_DAYS, rng),
        "test": _temporal_frame(_TEMPORAL_TEST_DAYS, rng, unlabelled=_TEMPORAL_UNLABELLED),
    }
    preprocessor = FoldPreprocessor(factor_order=_TWO_FACTORS)
    preprocessor.fit(frames["train"].loc[:, list(_TWO_FACTORS)], role="TRAIN")
    manifest = build_split_dataset_manifest(
        frames,
        business_keys=("signal_date", "ts_code"),
        metadata={
            "fold_id": "F1",
            "factor_order": list(_TWO_FACTORS),
            "data_versions": {"label_daily": {"2025-01": "v1"}},
        },
    )
    DatasetArtifactStore(tmp_path / "datasets").write(
        manifest, frames, preprocessing={"assembler": "test"}
    )
    config: dict[str, Any] = {
        "raw_factor_count": len(_TWO_FACTORS),
        "hidden_dim": 4,
        "dropout": 0.0,
    }
    seed_checkpoints: dict[int, Path] = {}
    prototype: TemporalGRU | None = None
    for seed in REQUIRED_SEEDS:
        torch.manual_seed(seed)
        model = TemporalGRU(TemporalGRUConfig(**config))
        prototype = model if prototype is None else prototype
        path = tmp_path / f"seed-{seed}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        seed_checkpoints[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")
    assert prototype is not None
    store.write(
        "m2-f1",
        model_card={
            "dataset_id": manifest.dataset_id,
            "family": "TEMPORAL_GRU",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "temporal-gru-1",
        },
        config=config,
        signature=asdict(prototype.artifact_signature()),
        feature_order=_TWO_FACTORS,
        preprocessing={
            "state_hash": preprocessor.state_hash,
            "state_payload": preprocessor.state_payload(),
        },
        seed_checkpoints=seed_checkpoints,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-21"], "ts_code": ["S002"], "percentile_rank": [0.5]}
        ),
    )
    return reproduction._FoldBundle(
        fold_id="F1",
        model_artifact_id="m2-f1",
        family="TEMPORAL_GRU",
        dataset_id=manifest.dataset_id,
        report_dataset_id=manifest.dataset_id,
        factor_order=_TWO_FACTORS,
        store=store,
        workspace=tmp_path,
        test=frames["test"],
    )


def test_the_temporal_path_closes_on_a_real_bundle(tmp_path: Path) -> None:
    bundle = _temporal_bundle(tmp_path)
    scores = reproduction._recompute_temporal_torch(bundle)
    assert set(scores.columns) == {"signal_date", "ts_code", "score"}
    days = set(pd.to_datetime(scores["signal_date"]).dt.date)
    # History-only dates must not leak in: the panel contains train and validation rows too, and
    # only the filter to the TEST dates keeps this a TEST scoring.
    assert days == set(_TEMPORAL_TEST_DAYS)
    labelled = _STOCKS_PER_DAY - _TEMPORAL_UNLABELLED
    assert len(scores) == len(_TEMPORAL_TEST_DAYS) * labelled
    pd.testing.assert_frame_equal(
        scores, reproduction._recompute_temporal_torch(bundle), check_exact=True
    )


def test_the_temporal_row_set_is_the_labelled_subset(tmp_path: Path) -> None:
    """M2 does not zero the label column, so unlabelled TEST rows never entered the panel.

    That row set belongs to the run being audited. An audit that "helpfully" scored every TEST row
    would report a row-set difference on every temporal fold and would be right to -- but for the
    wrong reason, and it would never reproduce a genuine M2 run at all.
    """
    scores = reproduction._recompute_temporal_torch(_temporal_bundle(tmp_path))
    assert len(scores) < len(_TEMPORAL_TEST_DAYS) * _STOCKS_PER_DAY
    unlabelled_codes = [f"S{index:03d}" for index in range(_TEMPORAL_UNLABELLED)]
    assert not scores["ts_code"].isin(unlabelled_codes).any()


def test_scoring_is_bitwise_stable_across_thread_counts(tmp_path: Path) -> None:
    """Why the audit may compare exactly without pinning threads.

    Training is NOT thread-stable -- the plan measured three distinct checkpoint digests at 1/4/8
    threads, which is why the search records bind the count. A forward pass is, so an audit run on a
    host with a different core count still reproduces a run trained elsewhere. Measured rather than
    assumed, because a torch upgrade could change it and the failure would look like tampering.
    """
    store, model_id, state_hash = _real_bundle(tmp_path)
    loaded = load_bundle_for_evaluation(
        store,
        model_id,
        expected_dataset_id="ds-1",
        expected_feature_order=_TWO_FACTORS,
        expected_preprocessor_hash=state_hash,
        device=torch.device("cpu"),
    )
    assert loaded.preprocessor is not None
    batches = build_static_training_batches(
        prepare_frame(
            _real_frame().assign(raw_return_5d=0.0), loaded.preprocessor, _TWO_FACTORS
        ),
        factor_order=_TWO_FACTORS,
    )
    day = _TWO_DAYS[0]
    original = torch.get_num_threads()
    try:
        runs = []
        for threads in (1, max(2, min(original, 8))):
            torch.set_num_threads(threads)
            scored = score_batch(loaded.models[0], batches[day])
            runs.append(scored.sort_values("ts_code").to_numpy())
        assert np.array_equal(runs[0], runs[1])
    finally:
        torch.set_num_threads(original)
