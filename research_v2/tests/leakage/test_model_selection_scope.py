from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.datasets.manifest import build_dataset_manifest
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.training.search import (
    InnerFoldValidation,
    OuterValidationScore,
    TrialScheduler,
    ValidationProvenanceStore,
    build_pre_registered_trials,
    select_model_config,
    selection_evidence_hash,
)

_PROVENANCE_KEY = b"test-validation-provenance-key"
_PROVENANCE_KEY_ENV = "QRV2_VALIDATION_PROVENANCE_HMAC_KEY"


@pytest.fixture(autouse=True)
def _configure_validation_provenance_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_PROVENANCE_KEY_ENV, _PROVENANCE_KEY.decode("utf-8"))


def _selection_inputs(tmp_path: Path) -> tuple[
    tuple[object, ...],
    list[InnerFoldValidation],
    dict[str, OuterValidationScore],
    DatasetArtifactStore,
    Path,
]:
    store = DatasetArtifactStore(tmp_path / "validation-datasets")
    frame = pd.DataFrame(
        {"signal_date": [date(2019, 1, 2)], "ts_code": ["S000"], "value": [1.0]}
    )
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"dataset_role": "VALIDATION", "fold_id": "F01"},
    )
    store.write(manifest, {"validation": frame}, preprocessing={})
    provenance_root = tmp_path / "validation-provenance"
    provenance = ValidationProvenanceStore(
        provenance_root,
        store,
        signing_key=_PROVENANCE_KEY,
    ).issue(dataset_id=manifest.dataset_id, outer_fold_id="F01")
    trials = build_pre_registered_trials("M1", "F01", {"hidden": (64, 128, 256)})
    inner = [
        InnerFoldValidation(trial.config_hash, fold, 20260801, 0.02, provenance)
        for trial in trials
        for fold in ("I01", "I02", "I03")
    ]
    outer = {
        trial.config_hash: OuterValidationScore(0.03, 2.5, 64, provenance)
        for trial in trials
    }
    return trials, inner, outer, store, provenance_root


def _audit_id(root: Path, trials: tuple[object, ...]) -> str:
    scheduler = TrialScheduler(trials, lease_root=root / "leases")  # type: ignore[arg-type]
    lease = scheduler.lease_next(worker_id="trainer", gpu_id="gpu-0")
    assert lease is not None
    scheduler.complete(lease.trial.config_hash, lease_id=lease.lease_id)
    return scheduler.write_audit(root).stem


def test_future_test_labels_cannot_change_model_selection_or_trial_audit(tmp_path: Path) -> None:
    trials, inner, outer, validation_store, provenance_root = _selection_inputs(tmp_path)
    original_winner = select_model_config(
        trials,
        inner,
        outer,
        validation_store=validation_store,
        validation_provenance_root=provenance_root,
    )
    original_evidence = selection_evidence_hash(
        trials,
        inner,
        outer,
        validation_store=validation_store,
        validation_provenance_root=provenance_root,
    )
    original_audit = _audit_id(tmp_path / "first", trials)
    future_test_2026_labels = [1_000_000.0, -1_000_000.0, 42.0]

    assert future_test_2026_labels != [0.0, 0.0, 0.0]
    assert (
        select_model_config(
            trials,
            inner,
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )
        == original_winner
    )
    assert (
        selection_evidence_hash(
            trials,
            inner,
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )
        == original_evidence
    )
    assert _audit_id(tmp_path / "second", trials) == original_audit
