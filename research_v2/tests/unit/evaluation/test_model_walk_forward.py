from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

from research_v2.baselines.orchestration import BaselineSelectionRecord, BaselineValidationScore
from research_v2.datasets.manifest import build_dataset_manifest
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_walk_forward import (
    ModelOosFold,
    evaluate_model_walk_forward,
)
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionStore,
    SelectionEvidenceStore,
)
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


def _test_frame(year: int) -> pd.DataFrame:
    # Two test dates per fold so four folds clear the >5 valid-day OOS gate.
    return pd.DataFrame(
        {
            "signal_date": [date(year, 1, 2)] * 30 + [date(year, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "raw_return_5d": [float(index) for index in range(30)] * 2,
            "label_valid": [True] * 60,
            "f1": [float(index) for index in range(30)] * 2,
        }
    )


def _write_dataset(store: DatasetArtifactStore, year: int) -> str:
    frame = _test_frame(year)
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
) -> str:
    checkpoints = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"{model_id}-{seed}.pt"  # type: ignore[operator]
        path.write_bytes(f"checkpoint-{model_id}-{seed}".encode())  # type: ignore[attr-defined]
        checkpoints[seed] = path
    store.write(
        model_id,
        model_card={"family": "M1", "dataset_id": training_dataset_id},
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
) -> ModelOosFold:
    fold_id = f"F{year - 2021}"
    test_dataset_id = _write_dataset(dataset_store, year)
    model_id = f"model-{fold_id.lower()}"
    training_dataset_id = f"train-{fold_id}"
    model_bundle_digest = _write_model_bundle(
        model_store,
        model_id,
        training_dataset_id,
        tmp_path,
    )
    baseline = selection_store.write_baseline_selection(
        record=_baseline_record(fold_id),
        test_dataset_id=test_dataset_id,
    )
    model = selection_store.write_model_selection(
        fold_id=fold_id,
        model_family="M1",
        model_artifact_id=model_id,
        model_bundle_digest=model_bundle_digest,
        training_dataset_id=training_dataset_id,
        validation_dataset_id=f"validation-{fold_id}",
        test_dataset_id=test_dataset_id,
        baseline_selection_id=baseline.evidence_id,
        validation_rank_ic=0.20,
    )
    keys = _test_frame(year).loc[:, ["signal_date", "ts_code"]]
    candidate = prediction_store.write(
        selection_id=model.evidence_id,
        kind="candidate",
        fold_id=fold_id,
        dataset_id=test_dataset_id,
        model_id=model_id,
        scores=keys.assign(score=[float(index) for index in range(30)] * 2),
        selections=selection_store,
    )
    best_simple = prediction_store.write(
        selection_id=baseline.evidence_id,
        kind="best_simple",
        fold_id=fold_id,
        dataset_id=test_dataset_id,
        model_id="B0",
        scores=keys.assign(score=[float(-index) for index in range(30)] * 2),
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
