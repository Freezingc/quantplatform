from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import research_v2.training.search as search
from research_v2.datasets.manifest import build_dataset_manifest
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.training.search import (
    InnerFoldValidation,
    OuterValidationScore,
    TrialScheduler,
    ValidationProvenance,
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


def test_trials_are_fold_bound_content_ordered_and_capped() -> None:
    first = build_pre_registered_trials(
        "RESIDUAL_MLP",
        "F01",
        {"hidden": (64, 128), "dropout": (0.0, 0.1), "lr": (0.001,)},
        maximum_trials=3,
    )
    reordered = build_pre_registered_trials(
        "RESIDUAL_MLP",
        "F01",
        {"lr": (0.001,), "dropout": (0.0, 0.1), "hidden": (64, 128)},
        maximum_trials=3,
    )
    other_fold = build_pre_registered_trials(
        "RESIDUAL_MLP",
        "F02",
        {"hidden": (64, 128), "dropout": (0.0, 0.1), "lr": (0.001,)},
        maximum_trials=3,
    )

    assert first == reordered
    assert len(first) == 3
    assert {trial.fold_id for trial in first} == {"F01"}
    assert {trial.config_hash for trial in first}.isdisjoint(
        {trial.config_hash for trial in other_fold}
    )
    with pytest.raises(TypeError):
        first[0].config["hidden"] = 512  # type: ignore[index]


def test_scheduler_and_selection_reject_mixed_fold_trials(tmp_path: object) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256)})
    mixed = replace(trials[1], fold_id="F02")

    with pytest.raises(ValueError, match="one fold"):
        TrialScheduler((trials[0], mixed), lease_root=tmp_path / "leases")  # type: ignore[operator]
    with pytest.raises(ValueError, match="one fold"):
        select_model_config(
            (trials[0], mixed),
            [],
            {},
            validation_store=DatasetArtifactStore(tmp_path / "unused"),  # type: ignore[operator]
            validation_provenance_root=tmp_path / "provenance",  # type: ignore[operator]
        )


def test_scheduler_rejects_more_than_thirty_declared_trials(tmp_path: object) -> None:
    trials = build_pre_registered_trials(
        "M1",
        "F01",
        {"hidden": tuple(range(31))},
        maximum_trials=30,
    )
    with pytest.raises(ValueError, match="30"):
        TrialScheduler(
            (*trials, replace(trials[0], config_hash="f" * 64)),
            lease_root=tmp_path / "leases",  # type: ignore[operator]
        )


def test_trial_scheduler_records_exclusive_gpu_leases_and_completion_audit(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128)})
    scheduler = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]

    first = scheduler.lease_next(worker_id="worker-a", gpu_id="gpu-0")
    with pytest.raises(ValueError, match="gpu-0"):
        scheduler.lease_next(worker_id="worker-b", gpu_id="gpu-0")
    second = scheduler.lease_next(worker_id="worker-b", gpu_id="gpu-1")

    assert first is not None and second is not None
    assert first.trial.config_hash != second.trial.config_hash
    with pytest.raises(ValueError, match="lease"):
        scheduler.complete(first.trial.config_hash, lease_id="wrong")
    scheduler.complete(first.trial.config_hash, lease_id=first.lease_id)
    assert [event.event for event in scheduler.audit()] == ["LEASED", "LEASED", "COMPLETED"]
    assert all(event.occurred_at.endswith("+00:00") for event in scheduler.audit())
    scheduler.close()


def test_trial_scheduler_atomically_rejects_concurrent_leases_on_one_gpu(
    tmp_path: object,
) -> None:
    scheduler = TrialScheduler(
        build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256)}),
        lease_root=tmp_path / "leases",  # type: ignore[operator]
    )

    def lease() -> object:
        try:
            return scheduler.lease_next(worker_id="worker", gpu_id="gpu-0")
        except ValueError as error:
            return error

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: lease(), range(4)))

    assert sum(result is not None and not isinstance(result, ValueError) for result in results) == 1
    assert sum(isinstance(result, ValueError) for result in results) == 3
    scheduler.close()


def test_trial_scheduler_shares_gpu_and_trial_claims_across_scheduler_instances(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256)})
    first = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]
    second = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]

    claimed = first.lease_next(worker_id="worker-a", gpu_id="gpu-0")

    assert claimed is not None
    with pytest.raises(ValueError, match="gpu-0"):
        second.lease_next(worker_id="worker-b", gpu_id="gpu-0")
    first.close()


def test_trial_scheduler_cannot_reissue_a_trial_completed_by_another_instance(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64,)})
    first = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]
    second = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]

    lease = first.lease_next(worker_id="worker-a", gpu_id="gpu-0")

    assert lease is not None
    first.complete(lease.trial.config_hash, lease_id=lease.lease_id)
    assert second.lease_next(worker_id="worker-b", gpu_id="gpu-1") is None


def test_trial_scheduler_reclaims_an_expired_lease_after_a_worker_crash(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64,)})
    started_at = datetime(2026, 8, 27, 12, tzinfo=UTC)
    first = TrialScheduler(
        trials,
        lease_root=tmp_path / "leases",  # type: ignore[operator]
        lease_duration=timedelta(minutes=1),
        clock=lambda: started_at,
    )
    recovered = TrialScheduler(
        trials,
        lease_root=tmp_path / "leases",  # type: ignore[operator]
        lease_duration=timedelta(minutes=1),
        clock=lambda: started_at + timedelta(minutes=2),
    )

    abandoned = first.lease_next(worker_id="worker-a", gpu_id="gpu-0")
    replacement = recovered.lease_next(worker_id="worker-b", gpu_id="gpu-1")

    assert abandoned is not None and replacement is not None
    assert replacement.trial.config_hash == abandoned.trial.config_hash
    with pytest.raises(ValueError, match="active lease"):
        first.complete(abandoned.trial.config_hash, lease_id=abandoned.lease_id)
    recovered.complete(replacement.trial.config_hash, lease_id=replacement.lease_id)


def test_trial_scheduler_uses_host_global_gpu_exclusivity_across_experiment_roots(
    tmp_path: object,
) -> None:
    first = TrialScheduler(
        build_pre_registered_trials("M1", "F1", {"hidden": (64,)}),
        lease_root=tmp_path / "experiment-a",  # type: ignore[operator]
    )
    second = TrialScheduler(
        build_pre_registered_trials("M2", "F1", {"hidden": (64,)}),
        lease_root=tmp_path / "experiment-b",  # type: ignore[operator]
    )

    lease = first.lease_next(worker_id="worker-a", gpu_id="gpu-global-exclusive")

    assert lease is not None
    with pytest.raises(ValueError, match="gpu-global-exclusive"):
        second.lease_next(worker_id="worker-b", gpu_id="gpu-global-exclusive")
    first.complete(lease.trial.config_hash, lease_id=lease.lease_id)


def test_trial_scheduler_reads_shared_audit_events_from_all_scheduler_instances(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128)})
    first = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]
    second = TrialScheduler(trials, lease_root=tmp_path / "leases")  # type: ignore[operator]

    first_lease = first.lease_next(worker_id="worker-a", gpu_id="gpu-0")
    second_lease = second.lease_next(worker_id="worker-b", gpu_id="gpu-1")

    assert first_lease is not None and second_lease is not None
    assert {event.config_hash for event in first.audit()} == {
        first_lease.trial.config_hash,
        second_lease.trial.config_hash,
    }
    first.complete(first_lease.trial.config_hash, lease_id=first_lease.lease_id)
    second.complete(second_lease.trial.config_hash, lease_id=second_lease.lease_id)


def test_trial_scheduler_can_finish_a_completion_interrupted_after_its_marker(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = TrialScheduler(
        build_pre_registered_trials("M1", "F1", {"hidden": (64,)}),
        lease_root=tmp_path / "leases",  # type: ignore[operator]
    )
    lease = scheduler.lease_next(worker_id="worker-a", gpu_id="gpu-0")
    assert lease is not None
    original_remove_claim = search._remove_claim
    monkeypatch.setattr(search, "_remove_claim", lambda *_: (_ for _ in ()).throw(OSError("disk")))

    with pytest.raises(OSError, match="disk"):
        scheduler.complete(lease.trial.config_hash, lease_id=lease.lease_id)

    monkeypatch.setattr(search, "_remove_claim", original_remove_claim)
    scheduler.complete(lease.trial.config_hash, lease_id=lease.lease_id)
    assert scheduler.audit()[-1].event == "COMPLETED"


def test_trial_audit_publication_is_exclusive_under_concurrent_writers(tmp_path: object) -> None:
    scheduler = TrialScheduler(
        build_pre_registered_trials("M1", "F1", {"hidden": (64,)}),
        lease_root=tmp_path / "leases",  # type: ignore[operator]
    )

    def write_audit() -> object:
        try:
            return scheduler.write_audit(tmp_path / "audits")  # type: ignore[operator]
        except FileExistsError as error:
            return error

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: write_audit(), range(4)))

    assert sum(not isinstance(result, FileExistsError) for result in results) == 1
    assert sum(isinstance(result, FileExistsError) for result in results) == 3


def test_trial_scheduler_writes_an_immutable_content_addressed_experiment_audit(
    tmp_path: object,
) -> None:
    scheduler = TrialScheduler(
        build_pre_registered_trials("M1", "F1", {"hidden": (64,)}),
        lease_root=tmp_path / "leases",  # type: ignore[operator]
    )
    lease = scheduler.lease_next(worker_id="worker-a", gpu_id="gpu-0")
    assert lease is not None
    scheduler.complete(lease.trial.config_hash, lease_id=lease.lease_id)

    path = scheduler.write_audit(tmp_path / "audits")  # type: ignore[operator]
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name.endswith(".json")
    assert payload["fold_id"] == "F1"
    assert payload["events"][-1]["event"] == "COMPLETED"
    with pytest.raises(FileExistsError):
        scheduler.write_audit(tmp_path / "audits")  # type: ignore[operator]


def test_model_selection_uses_two_inner_folds_then_outer_validation_tie_breaks(
    tmp_path: object,
) -> None:
    validation_store, validation, provenance_root = _validation_provenance(tmp_path)
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256, 512)})
    inner = [
        InnerFoldValidation(
            trial.config_hash,
            inner_fold,
            20260801,
            score,
            validation,
        )
        for trial, score in zip(trials, (0.05, 0.04, 0.03, 0.01), strict=True)
        for inner_fold in ("I01", "I02")
    ]
    outer = {
        trials[0].config_hash: OuterValidationScore(
            0.03, 2.2, 200, validation
        ),
        trials[1].config_hash: OuterValidationScore(
            0.03, 2.2, 100, validation
        ),
        trials[2].config_hash: OuterValidationScore(
            0.02, 9.0, 1, validation
        ),
    }

    winner = select_model_config(
        trials,
        inner,
        outer,
        validation_store=validation_store,
        validation_provenance_root=provenance_root,
    )

    assert winner.config_hash == trials[1].config_hash
    with pytest.raises(ValueError, match="two inner"):
        select_model_config(
            trials,
            inner[:-1],
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )


def test_model_selection_requires_three_inner_candidates(tmp_path: object) -> None:
    validation_store, validation, provenance_root = _validation_provenance(tmp_path)
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128)})
    inner = [
        InnerFoldValidation(
            trial.config_hash,
            fold,
            20260801,
            0.02,
            validation,
        )
        for trial in trials
        for fold in ("I01", "I02")
    ]
    outer = {
        trial.config_hash: OuterValidationScore(
            0.02, 2.1, 64, validation
        )
        for trial in trials
    }

    with pytest.raises(ValueError, match="top three"):
        select_model_config(
            trials,
            inner,
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )


def test_model_selection_rejects_test_evidence_and_is_invariant_to_future_labels(
    tmp_path: object,
) -> None:
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256)})
    validation_store, validation, provenance_root = _validation_provenance(tmp_path)
    inner = [
        InnerFoldValidation(trial.config_hash, fold, 20260801, 0.02, validation)
        for trial in trials
        for fold in ("I01", "I02")
    ]
    outer = {
        trial.config_hash: OuterValidationScore(0.02, 2.1, 64, validation)
        for trial in trials
    }
    original = selection_evidence_hash(
        trials,
        inner,
        outer,
        validation_store=validation_store,
        validation_provenance_root=provenance_root,
    )
    future_test_labels = [1_000_000.0, -1_000_000.0]

    assert future_test_labels != [0.0, 0.0]
    assert (
        selection_evidence_hash(
            trials,
            list(reversed(inner)),
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )
        == original
    )
    assert (
        select_model_config(
            trials,
            inner,
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )
        == select_model_config(
            trials,
            list(reversed(inner)),
            outer,
            validation_store=validation_store,
            validation_provenance_root=provenance_root,
        )
    )
    with pytest.raises(ValueError, match="VALIDATION"):
        _write_provenance(
            validation_store,
            "TEST",
            provenance_root=tmp_path / "test-provenance",  # type: ignore[operator]
        )


def test_model_selection_rejects_a_validation_artifact_from_the_test_year(
    tmp_path: object,
) -> None:
    validation_store = DatasetArtifactStore(tmp_path / "datasets")  # type: ignore[operator]
    with pytest.raises(ValueError, match="expected validation year"):
        _write_provenance(
            validation_store,
            "VALIDATION",
            provenance_root=tmp_path / "provenance",  # type: ignore[operator]
            year=2022,
        )


def test_model_selection_rejects_provenance_signed_with_an_untrusted_key(
    tmp_path: object,
) -> None:
    validation_store, legitimate, _ = _validation_provenance(tmp_path)
    trials = build_pre_registered_trials("M1", "F1", {"hidden": (64, 128, 256)})
    attacker_root = tmp_path / "attacker-provenance"  # type: ignore[operator]
    forged = ValidationProvenanceStore(
        attacker_root,
        validation_store,
        signing_key=b"untrusted-key",
    ).issue(
        dataset_id=legitimate.dataset_id,
        outer_fold_id="F1",
    )
    inner = [
        InnerFoldValidation(trial.config_hash, fold, 20260801, 0.02, forged)
        for trial in trials
        for fold in ("I01", "I02")
    ]
    outer = {
        trial.config_hash: OuterValidationScore(0.02, 2.1, 64, forged)
        for trial in trials
    }

    with pytest.raises(ValueError, match="signature"):
        select_model_config(
            trials,
            inner,
            outer,
            validation_store=validation_store,
            validation_provenance_root=attacker_root,
        )


def _validation_provenance(
    tmp_path: object,
) -> tuple[DatasetArtifactStore, ValidationProvenance, Path]:
    store = DatasetArtifactStore(tmp_path / "validation-datasets")  # type: ignore[operator]
    return _write_provenance(
        store,
        "VALIDATION",
        provenance_root=tmp_path / "provenance",  # type: ignore[operator]
    )


def _write_provenance(
    store: DatasetArtifactStore,
    dataset_role: str,
    *,
    provenance_root: Path,
    year: int = 2021,
) -> tuple[DatasetArtifactStore, ValidationProvenance, Path]:
    frame = pd.DataFrame(
        {"signal_date": [date(year, 1, 2)], "ts_code": ["S000"], "value": [1.0]}
    )
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"dataset_role": dataset_role, "fold_id": "F1"},
    )
    store.write(manifest, {"validation": frame}, preprocessing={})
    provenance = ValidationProvenanceStore(
        provenance_root,
        store,
        signing_key=_PROVENANCE_KEY,
    ).issue(dataset_id=manifest.dataset_id, outer_fold_id="F1")
    return store, provenance, provenance_root
