from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

import pandas as pd
import pytest

from research_v2.baselines.orchestration import BaselineSelectionRecord, BaselineValidationScore
from research_v2.datasets.manifest import build_dataset_manifest, build_split_dataset_manifest
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_walk_forward import (
    ModelOosFold,
    _temporal_history_ineligible_keys,
    evaluate_model_walk_forward,
)
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionStore,
    SelectionEvidenceStore,
)
from research_v2.models.families import TEMPORAL_TORCH_FAMILIES
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

_SELECTION_KEY = b"test-selection-signing-key"
_SELECTION_KEY_ENV = "QRV2_SELECTION_EVIDENCE_HMAC_KEY"
_PREDICTION_KEY = b"test-prediction-signing-key"
_PREDICTION_KEY_ENV = "QRV2_PREDICTION_EVIDENCE_HMAC_KEY"


@pytest.fixture(autouse=True)
def _configure_evidence_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    monkeypatch.setenv(_PREDICTION_KEY_ENV, _PREDICTION_KEY.decode("utf-8"))


_Coverage = Literal["all", "labelled", "labelled_missing_one", "labelled_plus_foreign"]
"""Which rows the fixture's candidate predictions cover.

``all`` is the LightGBM carriers' row set and ``labelled`` is the torch carriers': the static and
temporal batch builders both drop rows without a usable label before scoring, so on the real F1 fold
a torch candidate holds 803,572 of the TEST frame's 804,575 rows. The other two are the guards --
a candidate that silently drops a row the metric could have used, and one that names a row the
verified TEST frame does not contain.
"""


def _test_frame(year: int, *, unlabelled_per_day: int = 0) -> pd.DataFrame:
    # Two test dates per fold so four folds clear the >5 valid-day OOS gate.
    frame = pd.DataFrame(
        {
            "signal_date": [date(year, 1, 2)] * 30 + [date(year, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "raw_return_5d": [float(index) for index in range(30)] * 2,
            "label_valid": [True] * 60,
            "f1": [float(index) for index in range(30)] * 2,
        }
    )
    if unlabelled_per_day:
        # Rows a torch carrier never scores: no usable label, so they cannot enter the metric
        # either. Each day keeps its 30 valid rows, so the valid-day count does not move.
        extra = pd.DataFrame(
            {
                "signal_date": [
                    date(year, 1, day)
                    for day in (2, 3)
                    for _ in range(unlabelled_per_day)
                ],
                "ts_code": [
                    f"U{day}{index:03d}"
                    for day in (2, 3)
                    for index in range(unlabelled_per_day)
                ],
                "raw_return_5d": [float("nan")] * (2 * unlabelled_per_day),
                "label_valid": [False] * (2 * unlabelled_per_day),
                "f1": [float(index) for index in range(2 * unlabelled_per_day)],
            }
        )
        frame = pd.concat([frame, extra], ignore_index=True)
    return frame


def _write_dataset(
    store: DatasetArtifactStore,
    year: int,
    *,
    unlabelled_per_day: int = 0,
    temporal_history: bool = False,
) -> str:
    frame = _test_frame(year, unlabelled_per_day=unlabelled_per_day)
    if temporal_history:
        codes = [f"S{index:03d}" for index in range(1, 30)]
        train = pd.DataFrame(
            {
                "signal_date": [date(year - 2, 1, day) for day in range(1, 15) for _ in codes],
                "ts_code": codes * 14,
                "raw_return_5d": [1.0] * (14 * len(codes)),
                "label_valid": [True] * (14 * len(codes)),
                "f1": [1.0] * (14 * len(codes)),
            }
        )
        validation = pd.DataFrame(
            {
                "signal_date": [date(year - 1, 1, 2)] * len(codes),
                "ts_code": codes,
                "raw_return_5d": [1.0] * len(codes),
                "label_valid": [True] * len(codes),
                "f1": [1.0] * len(codes),
            }
        )
        splits = {"train": train, "validation": validation, "test": frame}
        manifest = build_split_dataset_manifest(
            splits,
            business_keys=("signal_date", "ts_code"),
            metadata={"fold_id": f"F{year - 2021}"},
        )
        store.write(manifest, splits, preprocessing={})
        return manifest.dataset_id
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": f"F{year - 2021}"},
    )
    store.write(manifest, {"test": frame}, preprocessing={})
    return manifest.dataset_id


def _baseline_record(fold_id: str, *, selected_baseline_id: str = "B0") -> BaselineSelectionRecord:
    return BaselineSelectionRecord(
        fold_id=fold_id,
        selected_baseline_id=selected_baseline_id,
        validation_scores=(
            BaselineValidationScore("B0", 0.10, 100, 0),
            BaselineValidationScore("B1", 0.05, 100, 0),
            BaselineValidationScore("B2", 0.04, 100, 0),
            BaselineValidationScore("B3", 0.03, 100, 0),
        ),
        unselected_reasons=(
            ("B1", "LOWER_VALIDATION_RANKIC"),
            ("B2", "LOWER_VALIDATION_RANKIC"),
            ("B3", "LOWER_VALIDATION_RANKIC"),
        ),
    )


def _write_model_bundle(
    store: ModelArtifactBundleStore,
    model_id: str,
    training_dataset_id: str,
    tmp_path: object,
    card_dataset_id: str | None = None,
    card_family: str = "M1",
) -> str:
    checkpoints = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"{model_id}-{seed}.pt"  # type: ignore[operator]
        path.write_bytes(f"checkpoint-{model_id}-{seed}".encode())  # type: ignore[attr-defined]
        checkpoints[seed] = path
    store.write(
        model_id,
        model_card={
            "family": card_family,
            "dataset_id": (
                training_dataset_id if card_dataset_id is None else card_dataset_id
            ),
        },
        config={"raw_factor_count": 1},
        signature={"raw_factor_count": 1},
        feature_order=("f1",),
        preprocessing={"state_hash": "preprocessor"},
        seed_checkpoints=checkpoints,
        validation_predictions=pd.DataFrame(
            {
                "signal_date": [date(2019, 1, 2)],
                "ts_code": ["S000"],
                "percentile_rank": [0.5],
            }
        ),
    )
    return store.bundle_digest(model_id)


def _fold(
    year: int,
    *,
    dataset_store: DatasetArtifactStore,
    model_store: ModelArtifactBundleStore,
    selection_store: SelectionEvidenceStore,
    prediction_store: FrozenPredictionStore,
    tmp_path: object,
    card_form: Literal["training", "bare", "unrelated"] = "training",
    unlabelled_per_day: int = 0,
    candidate_coverage: _Coverage = "all",
    selection_family: str = "M1",
    card_family: str = "M1",
    temporal_history: bool = False,
) -> ModelOosFold:
    fold_id = f"F{year - 2021}"
    test_dataset_id = _write_dataset(
        dataset_store,
        year,
        unlabelled_per_day=unlabelled_per_day,
        temporal_history=temporal_history,
    )
    model_id = f"model-{fold_id.lower()}"
    # What every real writer emits: the selection names the train split of the same dataset
    # artifact, not an unrelated string. The fixture used to say "train-F1" here, a shape no
    # writer produces, which is why the two legitimate card conventions were never both covered.
    training_dataset_id = f"{test_dataset_id}:train"
    if card_form == "training":
        # The LightGBM convention: the card records the TRAIN split it fitted on.
        card_dataset_id = training_dataset_id
    elif card_form == "bare":
        # The torch convention: the card records the bare dataset id, which is what the
        # reproduction audit then loads the TEST split with.
        card_dataset_id = test_dataset_id
    else:
        card_dataset_id = "dataset-from-another-run"
    model_bundle_digest = _write_model_bundle(
        model_store,
        model_id,
        training_dataset_id,
        tmp_path,
        card_dataset_id=card_dataset_id,
        card_family=card_family,
    )
    baseline = selection_store.write_baseline_selection(
        record=_baseline_record(fold_id),
        test_dataset_id=test_dataset_id,
    )
    model = selection_store.write_model_selection(
        fold_id=fold_id,
        model_family=selection_family,
        model_artifact_id=model_id,
        model_bundle_digest=model_bundle_digest,
        training_dataset_id=training_dataset_id,
        validation_dataset_id=f"validation-{fold_id}",
        test_dataset_id=test_dataset_id,
        baseline_selection_id=baseline.evidence_id,
        validation_rank_ic=0.20,
    )
    frame = _test_frame(year, unlabelled_per_day=unlabelled_per_day)
    keys = frame.loc[:, ["signal_date", "ts_code"]]
    labelled = frame["label_valid"].astype(bool).to_numpy()
    labelled_keys = keys.loc[labelled].reset_index(drop=True)
    if candidate_coverage == "all":
        candidate_keys = keys.reset_index(drop=True)
    elif candidate_coverage == "labelled":
        candidate_keys = labelled_keys
    elif candidate_coverage == "labelled_missing_one":
        candidate_keys = labelled_keys.iloc[1:].reset_index(drop=True)
    else:
        candidate_keys = pd.concat(
            [
                labelled_keys,
                pd.DataFrame(
                    {
                        "signal_date": [frame["signal_date"].iloc[0]],
                        "ts_code": ["NOT_IN_THE_TEST_FRAME"],
                    }
                ),
            ],
            ignore_index=True,
        )
    candidate = prediction_store.write(
        selection_id=model.evidence_id,
        kind="candidate",
        fold_id=fold_id,
        dataset_id=test_dataset_id,
        model_id=model_id,
        scores=candidate_keys.assign(
            score=[float(index % 30) for index in range(len(candidate_keys))]
        ),
        selections=selection_store,
    )
    best_simple = prediction_store.write(
        selection_id=baseline.evidence_id,
        kind="best_simple",
        fold_id=fold_id,
        dataset_id=test_dataset_id,
        model_id="B0",
        # The frozen simple baseline is written over the labelled rows, as the real one is, which
        # also keeps it inside the candidate's coverage -- what the baseline check next door asks.
        scores=labelled_keys.assign(
            score=[float(-(index % 30)) for index in range(len(labelled_keys))]
        ),
        selections=selection_store,
    )
    return ModelOosFold(
        model_selection_id=model.evidence_id,
        candidate_prediction_id=candidate.prediction_id,
        best_simple_prediction_id=best_simple.prediction_id,
    )


def _folds(
    tmp_path: object,
    *,
    signing_key: bytes = _SELECTION_KEY,
    prediction_signing_key: bytes = _PREDICTION_KEY,
    card_form: Literal["training", "bare", "unrelated"] = "training",
    unlabelled_per_day: int = 0,
    candidate_coverage: _Coverage = "all",
    selection_family: str = "M1",
    card_family: str = "M1",
    temporal_history: bool = False,
    missing_first_row_in_fold: str | None = None,
) -> tuple[
    dict[str, ModelOosFold],
    DatasetArtifactStore,
    ModelArtifactBundleStore,
    SelectionEvidenceStore,
    FrozenPredictionStore,
]:
    dataset_store = DatasetArtifactStore(tmp_path / "datasets")  # type: ignore[operator]
    model_store = ModelArtifactBundleStore(tmp_path / "models")  # type: ignore[operator]
    selection_store = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=signing_key,
    )
    prediction_store = FrozenPredictionStore(
        tmp_path / "predictions",
        signing_key=prediction_signing_key,
    )
    folds = {
        f"F{index}": _fold(
            2021 + index,
            dataset_store=dataset_store,
            model_store=model_store,
            selection_store=selection_store,
            prediction_store=prediction_store,
            tmp_path=tmp_path,
            card_form=card_form,
            unlabelled_per_day=unlabelled_per_day,
            candidate_coverage=(
                "labelled_missing_one"
                if f"F{index}" == missing_first_row_in_fold
                else candidate_coverage
            ),
            selection_family=selection_family,
            card_family=card_family,
            temporal_history=temporal_history,
        )
        for index in range(1, 5)
    }
    return folds, dataset_store, model_store, selection_store, prediction_store


def test_model_walk_forward_loads_verified_evidence_and_test_manifests(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)

    result = evaluate_model_walk_forward(
        folds,
        dataset_store=datasets,
        model_store=models,
        selection_root=tmp_path / "selections",  # type: ignore[operator]
        prediction_root=tmp_path / "predictions",  # type: ignore[operator]
    )

    assert result.model_family == "M1"
    assert [selection.fold_id for selection in result.selections] == [
        "F1",
        "F2",
        "F3",
        "F4",
    ]
    assert result.evaluation.delta_rank_ic == pytest.approx(2.0)


def test_temporal_history_eligibility_counts_observed_rows_across_splits() -> None:
    train = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, day) for day in range(1, 15)],
            "ts_code": ["READY"] * 14,
        }
    )
    validation = pd.DataFrame(
        {
            "signal_date": [date(2021, 1, 2), date(2021, 1, 2)],
            "ts_code": ["READY", "NEW"],
        }
    )
    test = pd.DataFrame(
        {
            "signal_date": [date(2022, 1, 2), date(2022, 1, 3)] * 2,
            "ts_code": ["READY", "READY", "NEW", "NEW"],
        }
    )

    assert _temporal_history_ineligible_keys(
        train.iloc[::-1], validation.iloc[::-1], test.iloc[::-1]
    ) == frozenset(
        {(date(2022, 1, 2), "NEW"), (date(2022, 1, 3), "NEW")}
    )


def test_m2_walk_forward_discloses_verified_history_gap(tmp_path: object) -> None:
    folds, datasets, models, _, _ = _folds(
        tmp_path,
        card_form="bare",
        selection_family="M2",
        card_family="TEMPORAL_GRU",
        missing_first_row_in_fold="F1",
        temporal_history=True,
    )

    result = evaluate_model_walk_forward(
        folds,
        dataset_store=datasets,
        model_store=models,
        selection_root=tmp_path / "selections",  # type: ignore[operator]
        prediction_root=tmp_path / "predictions",  # type: ignore[operator]
    )

    assert result.model_family == "M2"
    assert result.unscored_scoreable_rows_by_fold == {"F1": 1, "F2": 0, "F3": 0, "F4": 0}
    assert result.evaluation.summary.valid_day_count == 7


def test_model_walk_forward_rejects_tampered_model_bundle_before_test_evaluation(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    bundle = models.bundle_path("model-f1")
    (bundle / "config.json").write_text("tampered", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_selection_evidence_content_hash_rejects_post_selection_mutation(tmp_path: object) -> None:
    selections = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=b"test-selection-signing-key",
    )
    baseline = selections.write_baseline_selection(
        record=_baseline_record("F1"),
        test_dataset_id="dataset-f01",
    )
    record_path = (
        tmp_path  # type: ignore[operator]
        / "selections"
        / "baseline"
        / f"{baseline.evidence_id}.json"
        / "record.json"
    )
    document = json.loads(record_path.read_text(encoding="utf-8"))
    document["payload"]["selected_baseline_id"] = "B1"
    record_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="signature"):
        selections.load_baseline_selection(baseline.evidence_id)


def test_selection_evidence_rejects_a_record_signed_by_a_different_key(tmp_path: object) -> None:
    trusted = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=b"trusted-selection-signing-key",
    )
    baseline = trusted.write_baseline_selection(
        record=_baseline_record("F1"),
        test_dataset_id="dataset-f01",
    )
    untrusted = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=b"untrusted-selection-signing-key",
    )

    with pytest.raises(ValueError, match="signature"):
        untrusted.load_baseline_selection(baseline.evidence_id)


def test_baseline_selection_evidence_rejects_a_non_winning_validation_record(
    tmp_path: object,
) -> None:
    selections = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=b"test-selection-signing-key",
    )

    with pytest.raises(ValueError, match="winner"):
        selections.write_baseline_selection(
            record=_baseline_record("F1", selected_baseline_id="B1"),
            test_dataset_id="dataset-f01",
        )


def test_prediction_evidence_rejects_unselected_simple_baseline_before_evaluation(
    tmp_path: object,
) -> None:
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    current = folds["F1"]
    model = selections.load_model_selection(current.model_selection_id)
    baseline = selections.load_baseline_selection(model.baseline_selection_id)
    test_keys = datasets.load_split(model.test_dataset_id, "test").loc[
        :, ["signal_date", "ts_code"]
    ]
    with pytest.raises(ValueError, match="model_id"):
        predictions.write(
            selection_id=baseline.evidence_id,
            kind="best_simple",
            fold_id="F1",
            dataset_id=model.test_dataset_id,
            model_id="B3",
            scores=test_keys.assign(score=[float(index) for index in range(30)] * 2),
            selections=selections,
        )


def test_model_walk_forward_rejects_tampered_prediction_evidence(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    current = folds["F1"]
    prediction_path = tmp_path / "predictions" / current.candidate_prediction_id / "scores.parquet"  # type: ignore[operator]
    prediction_path.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="prediction"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_tampered_test_dataset_before_metrics(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    model = selections.load_model_selection(folds["F1"].model_selection_id)
    test_shard = next(datasets.artifact_path(model.test_dataset_id).glob("split=test/**/*.parquet"))
    test_shard.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="dataset"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_test_manifest_from_a_different_fold(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    original = folds["F1"]
    original_selection = selections.load_model_selection(original.model_selection_id)
    other_test_dataset_id = selections.load_model_selection(
        folds["F2"].model_selection_id
    ).test_dataset_id
    baseline = selections.write_baseline_selection(
        record=_baseline_record("F1"),
        test_dataset_id=other_test_dataset_id,
    )
    replacement = selections.write_model_selection(
        fold_id="F1",
        model_family="M1",
        model_artifact_id=original_selection.model_artifact_id,
        model_bundle_digest=original_selection.model_bundle_digest,
        training_dataset_id=original_selection.training_dataset_id,
        validation_dataset_id=original_selection.validation_dataset_id,
        test_dataset_id=other_test_dataset_id,
        baseline_selection_id=baseline.evidence_id,
        validation_rank_ic=original_selection.validation_rank_ic,
    )
    test_keys = datasets.load_split(other_test_dataset_id, "test").loc[
        :, ["signal_date", "ts_code"]
    ]
    candidate = predictions.write(
        selection_id=replacement.evidence_id,
        kind="candidate",
        fold_id="F1",
        dataset_id=other_test_dataset_id,
        model_id=replacement.model_artifact_id,
        scores=test_keys.assign(score=[float(index) for index in range(30)] * 2),
        selections=selections,
    )
    best_simple = predictions.write(
        selection_id=baseline.evidence_id,
        kind="best_simple",
        fold_id="F1",
        dataset_id=other_test_dataset_id,
        model_id="B0",
        scores=test_keys.assign(score=[float(-index) for index in range(30)] * 2),
        selections=selections,
    )
    folds["F1"] = ModelOosFold(
        model_selection_id=replacement.evidence_id,
        candidate_prediction_id=candidate.prediction_id,
        best_simple_prediction_id=best_simple.prediction_id,
    )

    with pytest.raises(ValueError, match="fold"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_selection_signed_by_the_caller(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    attacker = SelectionEvidenceStore(  # type: ignore[operator]
        tmp_path / "selections",
        signing_key=b"attacker-controlled-selection-key",
    )
    baseline = attacker.write_baseline_selection(
        record=_baseline_record("F1"),
        test_dataset_id="attacker-controlled-test-dataset",
    )
    replacement = attacker.write_model_selection(
        fold_id="F1",
        model_family="M1",
        model_artifact_id="attacker-controlled-model",
        model_bundle_digest="0" * 64,
        training_dataset_id="attacker-controlled-training-dataset",
        validation_dataset_id="attacker-controlled-validation-dataset",
        test_dataset_id="attacker-controlled-test-dataset",
        baseline_selection_id=baseline.evidence_id,
        validation_rank_ic=0.99,
    )
    original = folds["F1"]
    folds["F1"] = ModelOosFold(
        model_selection_id=replacement.evidence_id,
        candidate_prediction_id=original.candidate_prediction_id,
        best_simple_prediction_id=original.best_simple_prediction_id,
    )

    with pytest.raises(ValueError, match="signature"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_a_republished_model_bundle(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    model = selections.load_model_selection(folds["F1"].model_selection_id)
    bundle = models.bundle_path(model.model_artifact_id)
    config_path = bundle / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["raw_factor_count"] = 2
    config_path.write_text(json.dumps(config), encoding="utf-8")
    checksum_lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in sorted(bundle.iterdir())
        if path.is_file() and path.name != "checksums.sha256"
    ]
    (bundle / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_prediction_signed_by_the_caller(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path)
    original = folds["F1"]
    model = selections.load_model_selection(original.model_selection_id)
    test_keys = datasets.load_split(model.test_dataset_id, "test").loc[
        :, ["signal_date", "ts_code"]
    ]
    attacker = FrozenPredictionStore(
        tmp_path / "predictions",  # type: ignore[operator]
        signing_key=b"attacker-controlled-prediction-key",
    )
    forged = attacker.write(
        selection_id=model.evidence_id,
        kind="candidate",
        fold_id="F1",
        dataset_id=model.test_dataset_id,
        model_id=model.model_artifact_id,
        scores=test_keys.assign(score=[float(-index) for index in range(30)] * 2),
        selections=selections,
    )
    folds["F1"] = ModelOosFold(
        model_selection_id=original.model_selection_id,
        candidate_prediction_id=forged.prediction_id,
        best_simple_prediction_id=original.best_simple_prediction_id,
    )

    with pytest.raises(ValueError, match="prediction signature"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_accepts_a_bundle_that_records_the_bare_dataset_id(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torch bundle records the bare dataset id, and that is a convention, not a mismatch.

    ``inference.reproduction._checked_card_dataset_id`` already documents the split: the LightGBM
    path records the TRAIN split it fitted on (``<dataset>:train``) while the torch path records the
    bare dataset id it later loads the TEST split with, and both are legitimate so the check there
    is membership rather than equality. This evaluator enforced equality against the selection's
    ``training_dataset_id`` alone, so every torch run trained and froze all four folds and then died
    at the evaluation step -- which is what happened to both real M1 runs, ``m1_v2`` after ~44 h and
    ``m1_parallel_v1`` after ~20 h, with no ``report.json`` and therefore no verdict for either.

    Membership of a set built from this fold's own signed selection is the rule, and it is not
    weaker: the set is derived from evidence this fold signed, so a bundle dropped in from another
    fold or another run names a different hash and is still refused.
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path, card_form="bare")

    result = evaluate_model_walk_forward(
        folds,
        dataset_store=datasets,
        model_store=models,
        selection_root=tmp_path / "selections",  # type: ignore[operator]
        prediction_root=tmp_path / "predictions",  # type: ignore[operator]
    )

    assert result.model_family == "M1"
    assert [selection.fold_id for selection in result.selections] == ["F1", "F2", "F3", "F4"]
    assert result.evaluation.delta_rank_ic == pytest.approx(2.0)


def test_model_walk_forward_still_rejects_a_bundle_naming_an_unrelated_dataset(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepting two conventions must not become accepting anything.

    This is the non-triviality guard for the test above. Widening equality to membership is only
    safe because the accepted set comes from this fold's signed selection, so a card naming some
    other dataset must still be refused by the same check -- if this test ever passes for the wrong
    reason, the test above proves nothing.
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(tmp_path, card_form="unrelated")

    with pytest.raises(ValueError, match="does not match frozen model selection"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


@pytest.mark.parametrize("card_family", sorted(TEMPORAL_TORCH_FAMILIES))
def test_model_walk_forward_accepts_the_family_an_m2_bundle_actually_records(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
    card_family: str,
) -> None:
    """M2's card is named by the winning config, and the evaluator used to accept neither spelling.

    ``run_full_m2`` writes ``f"TEMPORAL_{best_config['family']}"`` into the card because the grid
    searches GRU and TCN alike, so a real M2 bundle records ``TEMPORAL_GRU`` or ``TEMPORAL_TCN``.
    ``_MODEL_CARD_FAMILIES["M2"]`` held ``{"M2", "TEMPORAL"}`` -- a spelling no writer produces --
    so every M2 run would have trained for days, frozen and signed four bundles, and then died
    here on a family mismatch. That is the failure which cost both real M1 runs their verdicts
    three times over, and the evaluator's own comment above ``MODEL_FAMILIES`` records the same
    drift having already killed a completed run once, when M0R was added to the driver only.

    Parametrised over the set derived from the model classes rather than naming one architecture, so
    a third temporal family has to be allowed deliberately.
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(
        tmp_path, selection_family="M2", card_family=card_family
    )

    result = evaluate_model_walk_forward(
        folds,
        dataset_store=datasets,
        model_store=models,
        selection_root=tmp_path / "selections",  # type: ignore[operator]
        prediction_root=tmp_path / "predictions",  # type: ignore[operator]
    )

    assert result.model_family == "M2"
    assert result.evaluation.delta_rank_ic == pytest.approx(2.0)


def test_model_walk_forward_rejects_an_m2_selection_vouching_for_a_static_bundle(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Widening M2's allowlist must not make it accept any torch family.

    The non-triviality guard for the test above: M2 now accepts two architecture names, and the
    point is that it accepts exactly the two its own bundles can carry. A static carrier's bundle
    dropped into an M2 selection has to be refused by the same check, or the allowlist has stopped
    saying anything.
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(
        tmp_path, selection_family="M2", card_family="RESIDUAL_MLP"
    )

    with pytest.raises(ValueError, match="family does not match frozen model selection"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_accepts_a_candidate_scoring_only_the_labelled_rows(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A torch candidate covers the labelled rows only, and that is a convention, not a gap.

    The static and temporal batch builders both drop rows without a usable label before scoring, so
    a torch run's frozen candidate holds fewer rows than the verified TEST frame: on the real F1
    fold, 803,572 of 804,575, the 1,003 missing ones being ``label_valid=False``. The LightGBM
    carriers score the whole cross-section instead, and this evaluator required the candidate's
    business keys to be IDENTICAL to the unfiltered label frame -- so it encoded one family's
    convention as a universal rule and no torch run could ever reach a verdict. Both real M1 runs
    died here, at the second of two blockers in the same evaluation, after the dataset_id one.

    The metric agrees that the unlabelled rows are irrelevant: ``_valid_label_day_count`` and
    ``evaluate_frozen_oos`` use only rows that are ``label_valid`` with a finite return. And the
    neighbouring ``_require_baseline_covered_by_candidate`` settled the same question for the
    candidate/baseline pair, its docstring recording that requiring identical keys there "rejected
    exactly that case".
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(
        tmp_path, unlabelled_per_day=4, candidate_coverage="labelled"
    )

    result = evaluate_model_walk_forward(
        folds,
        dataset_store=datasets,
        model_store=models,
        selection_root=tmp_path / "selections",  # type: ignore[operator]
        prediction_root=tmp_path / "predictions",  # type: ignore[operator]
    )

    assert result.model_family == "M1"
    assert result.evaluation.delta_rank_ic == pytest.approx(2.0)


def test_model_walk_forward_rejects_a_candidate_that_drops_a_scoreable_row(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Coverage in one direction only would let the metric silently shrink.

    This is the non-triviality guard for the test above. Accepting a subset is safe only because the
    subset is required to include every row the metric can actually use; a candidate missing one of
    those would compute RankIC over a quietly smaller cross-section and still report a number.
    """
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(
        tmp_path, unlabelled_per_day=4, candidate_coverage="labelled_missing_one"
    )

    with pytest.raises(ValueError, match="usable label are not scored"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )


def test_model_walk_forward_rejects_a_candidate_row_outside_the_test_frame(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction: a candidate may not name a row the verified TEST frame lacks."""
    monkeypatch.setenv(_SELECTION_KEY_ENV, _SELECTION_KEY.decode("utf-8"))
    folds, datasets, models, selections, predictions = _folds(
        tmp_path, unlabelled_per_day=4, candidate_coverage="labelled_plus_foreign"
    )

    with pytest.raises(ValueError, match="not in the verified TEST frame"):
        evaluate_model_walk_forward(
            folds,
            dataset_store=datasets,
            model_store=models,
            selection_root=tmp_path / "selections",  # type: ignore[operator]
            prediction_root=tmp_path / "predictions",  # type: ignore[operator]
        )
