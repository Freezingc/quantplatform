from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest
import torch

import research_v2.inference.artifact_loader as artifact_loader
from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

_INFERENCE_CONTRACT = {
    "factor_set_id": "factor-set-1",
    "factor_set_status": "FROZEN",
    "universe_id": "cn_liquid_a",
    "universe_version": 1,
    "label_id": "forward_open_5d",
    "label_version": 1,
    "split_id": "walk-forward-1",
    "code_version": "research-code-1",
    "model_code_version": "residual-mlp-1",
}


def test_allowlisted_loader_reads_only_verified_three_seed_state_dicts(tmp_path: object) -> None:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    seeds = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"  # type: ignore[operator]
        torch.save({"model_state_dict": model.state_dict()}, path)
        seeds[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")  # type: ignore[operator]
    store.write(
        "model-1",
        model_card={
            "dataset_id": "dataset-1",
            "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED",
        },
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=tuple(f"f{index}" for index in range(15)),
        preprocessing={"state_hash": "pre-1"},
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    loaded = load_inference_artifact(
        store,
        "model-1",
        expected_dataset_id="dataset-1",
        expected_feature_order=tuple(f"f{index}" for index in range(15)),
        expected_preprocessor_hash="pre-1",
        device=torch.device("cpu"),
    )

    assert len(loaded.models) == 3
    assert all(not candidate.training for candidate in loaded.models)


def test_allowlisted_loader_reconstructs_gru_and_set_attention_bundles(tmp_path: object) -> None:
    cases = (
        (
            "gru-model",
            "TEMPORAL_GRU",
            TemporalGRU(TemporalGRUConfig(raw_factor_count=2, hidden_dim=4, dropout=0.0)),
            {"raw_factor_count": 2, "hidden_dim": 4, "dropout": 0.0},
            TemporalGRU,
        ),
        (
            "set-model",
            "SET_ATTENTION",
            SetAttention(
                SetAttentionConfig(
                    raw_factor_count=2,
                    hidden_dim=4,
                    num_heads=2,
                    inducing_points=2,
                    num_blocks=1,
                    dropout=0.0,
                )
            ),
            {
                "raw_factor_count": 2,
                "hidden_dim": 4,
                "num_heads": 2,
                "inducing_points": 2,
                "num_blocks": 1,
                "dropout": 0.0,
            },
            SetAttention,
        ),
    )
    store = ModelArtifactBundleStore(tmp_path / "models")  # type: ignore[operator]
    for model_id, family, model, config, expected_type in cases:
        seeds = {}
        for seed in REQUIRED_SEEDS:
            path = tmp_path / f"{model_id}-{seed}.pt"  # type: ignore[operator]
            torch.save({"model_state_dict": model.state_dict()}, path)
            seeds[seed] = path
        store.write(
            model_id,
            model_card={
                "dataset_id": f"{model_id}-dataset",
                "family": family,
                "model_status": "WALK_FORWARD_PASSED",
            },
            config=config,
            signature=asdict(model.artifact_signature()),
            feature_order=("factor_a", "factor_b"),
            preprocessing={"state_hash": f"{model_id}-preprocessor"},
            seed_checkpoints=seeds,
            validation_predictions=pd.DataFrame(
                {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
            ),
        )

        loaded = load_inference_artifact(
            store,
            model_id,
            expected_dataset_id=f"{model_id}-dataset",
            expected_feature_order=("factor_a", "factor_b"),
            expected_preprocessor_hash=f"{model_id}-preprocessor",
            device=torch.device("cpu"),
        )

        assert len(loaded.models) == 3
        assert all(isinstance(candidate, expected_type) for candidate in loaded.models)


@pytest.mark.parametrize("field", tuple(_INFERENCE_CONTRACT))
def test_contract_mismatch_is_rejected_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    expected_contract = dict(_INFERENCE_CONTRACT)
    expected_contract[field] = "unexpected-value"

    def fail_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("model factory must not run for a contract mismatch")

    monkeypatch.setattr(artifact_loader, "_allowlisted_factory", fail_factory)
    with pytest.raises(ValueError, match=field):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
            expected_inference_contract=expected_contract,
        )


def test_missing_requested_contract_field_is_rejected_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_exclusions=("factor_set_id",),
    )

    def fail_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("model factory must not run for a missing contract field")

    monkeypatch.setattr(artifact_loader, "_allowlisted_factory", fail_factory)
    with pytest.raises(ValueError, match="factor_set_id"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
            expected_inference_contract=_INFERENCE_CONTRACT,
        )


def test_non_frozen_factor_set_is_rejected(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_overrides={"factor_set_status": "CANDIDATE"},
    )

    with pytest.raises(ValueError, match="FROZEN"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
        )


def test_unknown_model_code_version_is_rejected(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_overrides={"model_code_version": "untrusted-implementation"},
    )

    with pytest.raises(ValueError, match="model_code_version"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
        )


def test_tampered_bundle_is_rejected_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    config_path = store.bundle_path(model_id) / "config.json"
    config_path.write_bytes(config_path.read_bytes() + b"tampered")

    def fail_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("model factory must not run for a tampered bundle")

    monkeypatch.setattr(artifact_loader, "_allowlisted_factory", fail_factory)
    with pytest.raises(ValueError, match="integrity"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
        )


def test_missing_seed_is_rejected_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    (store.bundle_path(model_id) / f"seed-{REQUIRED_SEEDS[0]}.pt").unlink()

    def fail_factory(*args: object, **kwargs: object) -> object:
        raise AssertionError("model factory must not run for a missing seed")

    monkeypatch.setattr(artifact_loader, "_allowlisted_factory", fail_factory)
    with pytest.raises(ValueError, match="integrity"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    (
        ("keys", "state_dict keys do not match"),
        ("shape", "state_dict shape mismatch"),
        ("dtype", "state_dict dtype mismatch"),
    ),
)
def test_seed_state_schema_is_verified_before_device_load(
    tmp_path: Path,
    mutation: str,
    expected_message: str,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    seed_path = store.bundle_path(model_id) / f"seed-{REQUIRED_SEEDS[0]}.pt"
    payload = torch.load(seed_path, map_location="cpu", weights_only=True)
    state = dict(payload["model_state_dict"])
    key = next(key for key, tensor in state.items() if tensor.ndim >= 2)
    if mutation == "keys":
        state.pop(key)
    elif mutation == "shape":
        state[key] = state[key].reshape(-1)
    else:
        state[key] = state[key].to(dtype=torch.float64)
    torch.save({"model_state_dict": state}, seed_path)
    _refresh_checksums(store.bundle_path(model_id))

    with pytest.raises(ValueError, match=expected_message):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash="pre-1",
            device=torch.device("cpu"),
        )


def test_loader_records_non_sensitive_load_audit_metadata(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    loaded = load_inference_artifact(
        store,
        model_id,
        expected_dataset_id="dataset-1",
        expected_feature_order=feature_order,
        expected_preprocessor_hash="pre-1",
        device=torch.device("cpu"),
        expected_inference_contract=_INFERENCE_CONTRACT,
        run_id="run-1",
    )

    assert loaded.audit.model_id == model_id
    assert loaded.audit.bundle_digest == store.bundle_digest(model_id)
    assert loaded.audit.artifact_hashes["config.json"] == _sha256(
        store.bundle_path(model_id) / "config.json"
    )
    assert loaded.audit.artifact_hashes[f"seed-{REQUIRED_SEEDS[0]}.pt"] == _sha256(
        store.bundle_path(model_id) / f"seed-{REQUIRED_SEEDS[0]}.pt"
    )
    assert loaded.audit.device == "cpu"
    assert loaded.audit.code_version == "research-code-1"
    assert loaded.audit.loaded_at.tzinfo is UTC
    assert loaded.audit.run_id == "run-1"
    assert "model_state_dict" not in repr(loaded.audit)
    assert "secret" not in repr(loaded.audit)


def _write_residual_bundle(
    root: Path,
    *,
    model_card_overrides: Mapping[str, object] | None = None,
    model_card_exclusions: tuple[str, ...] = (),
    model_id: str = "model-1",
) -> tuple[ModelArtifactBundleStore, str, tuple[str, ...]]:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    seeds = {}
    for seed in REQUIRED_SEEDS:
        path = root / f"seed-{seed}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        seeds[seed] = path
    store = ModelArtifactBundleStore(root / "models")
    model_card = {
        "dataset_id": "dataset-1",
        "family": "RESIDUAL_MLP",
        "model_status": "WALK_FORWARD_PASSED",
        **_INFERENCE_CONTRACT,
    }
    if model_card_overrides:
        model_card.update(model_card_overrides)
    for field in model_card_exclusions:
        model_card.pop(field, None)
    feature_order = tuple(f"f{index}" for index in range(15))
    store.write(
        model_id,
        model_card=model_card,
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=feature_order,
        preprocessing={"state_hash": "pre-1"},
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    return store, model_id, feature_order


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_checksums(root: Path) -> None:
    entries = [
        path for path in root.iterdir() if path.is_file() and path.name != "checksums.sha256"
    ]
    lines = [f"{_sha256(path)}  {path.name}" for path in sorted(entries)]
    (root / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
