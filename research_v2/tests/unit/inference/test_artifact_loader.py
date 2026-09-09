from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import research_v2.inference.artifact_loader as artifact_loader
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.inference.artifact_loader import load_bundle_for_evaluation
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore


def _fitted_preprocessing(factor_order: tuple[str, ...]) -> dict[str, object]:
    """A real fitted state, because that is what the driver now persists.

    Fixtures that wrote only ``{"state_hash": "pre-1"}`` modelled an artifact the chain never
    produces and are now refused -- the same fixture/production divergence that hid the
    inference-contract guards. The hash is whatever the fitted state actually digests to, so the
    load sites compare against a real value rather than a label.
    """
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({name: rng.standard_normal(64) for name in factor_order})
    fitted = FoldPreprocessor(factor_order=factor_order)
    fitted.fit(frame, role="TRAIN")
    return {"state_hash": fitted.state_hash, "state_payload": fitted.state_payload()}


_F15 = tuple(f"f{index}" for index in range(15))
_PRE_F15 = _fitted_preprocessing(_F15)
_PRE_F15_HASH = str(_PRE_F15["state_hash"])
_PRE_AB = _fitted_preprocessing(("factor_a", "factor_b"))
_PRE_AB_HASH = str(_PRE_AB["state_hash"])

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
            "model_code_version": "residual-mlp-1",
        },
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=_F15,
        preprocessing=_PRE_F15,
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    loaded = load_bundle_for_evaluation(
        store,
        "model-1",
        expected_dataset_id="dataset-1",
        expected_feature_order=tuple(f"f{index}" for index in range(15)),
        expected_preprocessor_hash=_PRE_F15_HASH,
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
                "model_code_version": artifact_loader._MODEL_CODE_VERSIONS[family],
            },
            config=config,
            signature=asdict(model.artifact_signature()),
            feature_order=("factor_a", "factor_b"),
            preprocessing=_PRE_AB,
            seed_checkpoints=seeds,
            validation_predictions=pd.DataFrame(
                {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
            ),
        )

        loaded = load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id=f"{model_id}-dataset",
            expected_feature_order=("factor_a", "factor_b"),
            expected_preprocessor_hash=_PRE_AB_HASH,
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
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
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
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
            expected_inference_contract=_INFERENCE_CONTRACT,
        )


def test_non_frozen_factor_set_is_rejected(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_overrides={"factor_set_status": "CANDIDATE"},
    )

    with pytest.raises(ValueError, match="FROZEN"):
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_unknown_model_code_version_is_rejected(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_overrides={"model_code_version": "untrusted-implementation"},
    )

    with pytest.raises(ValueError, match="model_code_version"):
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
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
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
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
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
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
        load_bundle_for_evaluation(
            store,
            model_id,
            expected_dataset_id="dataset-1",
            expected_feature_order=feature_order,
            expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_loader_records_non_sensitive_load_audit_metadata(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    loaded = load_bundle_for_evaluation(
        store,
        model_id,
        expected_dataset_id="dataset-1",
        expected_feature_order=feature_order,
        expected_preprocessor_hash=_PRE_F15_HASH,
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


def test_loader_uses_one_verified_bundle_snapshot_for_seed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)

    def forbidden_path_lookup(unused_model_id: str) -> Path:
        raise AssertionError("loader must not reload seed files by path")

    monkeypatch.setattr(store, "bundle_path", forbidden_path_lookup)

    loaded = load_bundle_for_evaluation(
        store,
        model_id,
        expected_dataset_id="dataset-1",
        expected_feature_order=feature_order,
        expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"),
    )

    assert len(loaded.models) == len(REQUIRED_SEEDS)


def _write_residual_bundle(
    root: Path,
    *,
    model_card_overrides: Mapping[str, object] | None = None,
    model_card_exclusions: tuple[str, ...] = (),
    model_id: str = "model-1",
    preprocessing_override: Mapping[str, object] | None = None,
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
    feature_order = _F15
    store.write(
        model_id,
        model_card=model_card,
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=feature_order,
        preprocessing=dict(_PRE_F15 if preprocessing_override is None else preprocessing_override),
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


# ---------- serving path: promotion authority comes from the registry, not the card ----------

_ORDER = _F15


def _write_mlp_bundle(tmp_path: Path, model_id: str = "model-1") -> ModelArtifactBundleStore:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    seeds = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"seed-{model_id}-{seed}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        seeds[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")
    store.write(
        model_id,
        model_card={
            "dataset_id": "dataset-1",
            "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "residual-mlp-1",
        },
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=_ORDER,
        preprocessing=_PRE_F15,
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    return store


def _promoted(*artifact_ids: str):
    from research_v2.registry.model_registry import ModelStatus
    from research_v2.registry.promotion_evidence import PromotedRun

    ids = artifact_ids or ("model-1", "model-2", "model-3", "model-4")
    return PromotedRun(
        model_id="sha256:" + "a" * 64,
        status=ModelStatus.WALK_FORWARD_PASSED,
        fold_artifact_ids=dict(zip(("F1", "F2", "F3", "F4"), ids, strict=True)),
    )


def test_serving_load_refuses_without_registry_verified_promotion(tmp_path: Path) -> None:
    """The card says WALK_FORWARD_PASSED for every bundle ever written, so it proves nothing."""
    store = _write_mlp_bundle(tmp_path)
    with pytest.raises(ValueError, match="requires a PromotedRun"):
        artifact_loader.load_inference_artifact(
            store, "model-1", expected_dataset_id="dataset-1",
            expected_feature_order=_ORDER, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_serving_load_refuses_a_bundle_outside_the_authorised_run(tmp_path: Path) -> None:
    """One promoted run's authority must not serve another run's bundles."""
    store = _write_mlp_bundle(tmp_path, model_id="other-run-f1")
    with pytest.raises(ValueError, match="does not belong to promoted run"):
        artifact_loader.load_inference_artifact(
            store, "other-run-f1", expected_dataset_id="dataset-1",
            expected_feature_order=_ORDER, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"), promoted_run=_promoted(),
        )


def test_serving_load_records_the_authority_that_permitted_it(tmp_path: Path) -> None:
    store = _write_mlp_bundle(tmp_path)
    loaded = artifact_loader.load_inference_artifact(
        store, "model-1", expected_dataset_id="dataset-1",
        expected_feature_order=_ORDER, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"), promoted_run=_promoted(),
    )
    assert len(loaded.models) == 3
    assert loaded.promotion_model_id == "sha256:" + "a" * 64
    assert loaded.promotion_status == "WALK_FORWARD_PASSED"


def test_evaluation_load_carries_no_promotion_claim(tmp_path: Path) -> None:
    """The driver's pre-evaluation load must not look like an authorised serving load."""
    store = _write_mlp_bundle(tmp_path)
    loaded = load_bundle_for_evaluation(
        store, "model-1", expected_dataset_id="dataset-1",
        expected_feature_order=_ORDER, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"),
    )
    assert loaded.promotion_model_id == ""
    assert loaded.promotion_status == ""


# ---------- which bindings a load actually verified ----------


def test_a_production_shaped_card_reports_both_bindings_unverified(tmp_path: Path) -> None:
    """The driver writes four card fields; the contract guards need nine. Say so, don't imply more.

    ``_write_residual_bundle`` defaults to a card carrying every _INFERENCE_CONTRACT field, which
    is NOT what the candidate driver writes -- it writes dataset_id, family, model_status and
    model_code_version and nothing else. Excluding the contract fields reproduces the production
    shape, and the load must report that neither the factor-set nor the contract binding fired.
    """
    # Production writes exactly four card fields: dataset_id, family, model_status and
    # model_code_version. model_code_version is part of _INFERENCE_CONTRACT but is NOT omitted by
    # the driver, so excluding it here would model an artifact the chain never produces.
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_exclusions=tuple(set(_INFERENCE_CONTRACT) - {"model_code_version"}),
    )
    loaded = load_bundle_for_evaluation(
        store, model_id, expected_dataset_id="dataset-1",
        expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"),
    )
    assert loaded.unverified_bindings == ("factor_set_status", "inference_contract")


def test_a_complete_card_reports_no_unverified_bindings(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    loaded = load_bundle_for_evaluation(
        store, model_id, expected_dataset_id="dataset-1",
        expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"), expected_inference_contract=_INFERENCE_CONTRACT,
    )
    assert loaded.unverified_bindings == ()


def test_supplying_only_the_contract_still_leaves_the_factor_set_unverified(tmp_path: Path) -> None:
    """factor_set_status is a separate guard; a contract does not silently cover for it."""
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path, model_card_exclusions=("factor_set_status",)
    )
    contract = {k: v for k, v in _INFERENCE_CONTRACT.items() if k != "factor_set_status"}
    loaded = load_bundle_for_evaluation(
        store, model_id, expected_dataset_id="dataset-1",
        expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"), expected_inference_contract=contract,
    )
    assert loaded.unverified_bindings == ("factor_set_status",)


# ---------- family and code version are both mandatory, not conditional ----------


def test_a_card_omitting_model_code_version_is_refused(tmp_path: Path) -> None:
    """This used to load: the check was `if model_code_version is not None and ...`.

    Omitting the field skipped the allowlist entirely, and the audit then reported the inference
    contract's research code_version as though it were the model's, because
    _model_card_code_version falls back across the two fields. An artifact of unstated provenance
    must be refused rather than loaded with a borrowed identity.
    """
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path, model_card_exclusions=("model_code_version",)
    )
    with pytest.raises(ValueError, match="model_code_version is not allowlisted"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_a_card_with_a_foreign_code_version_is_refused(tmp_path: Path) -> None:
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path, model_card_overrides={"model_code_version": "temporal-gru-1"}
    )
    with pytest.raises(ValueError, match="model_code_version is not allowlisted"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_an_unknown_family_is_refused_by_name(tmp_path: Path) -> None:
    """The family message must stay accurate now that the family is checked first."""
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_overrides={"family": "LGBM_RANKREG", "model_code_version": "lgbm-rankreg-1"},
    )
    with pytest.raises(ValueError, match="model family is not allowlisted"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_the_torch_code_version_allowlist_is_pinned() -> None:
    assert artifact_loader._MODEL_CODE_VERSIONS == {
        "RESIDUAL_MLP": "residual-mlp-1",
        "TEMPORAL_GRU": "temporal-gru-1",
        "SET_ATTENTION": "set-attention-1",
    }


def test_an_empty_inference_contract_is_refused(tmp_path: Path) -> None:
    """A contract naming no fields verifies nothing and must not look like it verified something.

    Before this was refused, passing {} loaded a card stripped of every contract field and
    reported unverified_bindings == ("factor_set_status",) -- dropping inference_contract from the
    list, i.e. the loader actively claimed a binding it had not checked.
    """
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path,
        model_card_exclusions=tuple(set(_INFERENCE_CONTRACT) - {"model_code_version"}),
    )
    with pytest.raises(ValueError, match="at least one field"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"), expected_inference_contract={},
        )


def test_unverified_bindings_treats_an_empty_contract_as_unverified(tmp_path: Path) -> None:
    """Defence in depth: the reporter must not depend on the validator having run first."""
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    card = {"family": "RESIDUAL_MLP", "model_code_version": "residual-mlp-1"}
    assert artifact_loader.unverified_bindings(card, None) == (
        "factor_set_status", "inference_contract",
    )
    assert artifact_loader.unverified_bindings(card, {}) == (
        "factor_set_status", "inference_contract",
    )
    complete = dict(card, factor_set_status="FROZEN")
    assert artifact_loader.unverified_bindings(complete, {"factor_set_id": "fs"}) == ()
    assert artifact_loader.unverified_bindings(complete, {}) == ("inference_contract",)


# ---------- the fitted preprocessing state must travel with the bundle ----------


def test_the_restored_preprocessor_transforms_like_the_fitted_one(tmp_path: Path) -> None:
    """Weights plus a state hash are not a servable model; the state itself is what was missing."""
    store, model_id, feature_order = _write_residual_bundle(tmp_path)
    loaded = load_bundle_for_evaluation(
        store, model_id, expected_dataset_id="dataset-1",
        expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
        device=torch.device("cpu"),
    )
    assert loaded.preprocessor is not None
    assert loaded.preprocessor.fitted_factor_order == feature_order
    assert loaded.preprocessor.state_hash == _PRE_F15_HASH

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({name: rng.standard_normal(50) for name in feature_order})
    expected = FoldPreprocessor(factor_order=feature_order)
    expected.fit(_fitted_frame(feature_order), role="TRAIN")
    pd.testing.assert_frame_equal(
        loaded.preprocessor.transform(frame).values, expected.transform(frame).values
    )


def _fitted_frame(factor_order: tuple[str, ...]) -> pd.DataFrame:
    """The same frame _fitted_preprocessing fits on, so the two states are comparable."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({name: rng.standard_normal(64) for name in factor_order})


def test_a_bundle_with_a_hash_but_no_state_is_refused(tmp_path: Path) -> None:
    """This is exactly what the driver used to write, and it is not a servable artifact."""
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    seeds = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"
        torch.save({"model_state_dict": model.state_dict()}, path)
        seeds[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")
    store.write(
        "hash-only",
        model_card={
            "dataset_id": "dataset-1", "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED", "model_code_version": "residual-mlp-1",
        },
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=_F15,
        preprocessing={"state_hash": _PRE_F15_HASH},
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    with pytest.raises(ValueError, match="cannot be served"):
        load_bundle_for_evaluation(
            store, "hash-only", expected_dataset_id="dataset-1",
            expected_feature_order=_F15, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_a_state_tampered_after_the_bundle_was_written_is_refused(tmp_path: Path) -> None:
    """The restore recomputes the hash from what it rebuilt, so an edited payload cannot pass."""
    tampered = json.loads(json.dumps(_PRE_F15))
    per_factor = tampered["state_payload"]["state"]
    per_factor[_F15[0]]["median"] = per_factor[_F15[0]]["median"] + 1.0
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path, preprocessing_override=tampered
    )
    with pytest.raises(ValueError, match="does not match its recorded state hash"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=_PRE_F15_HASH,
            device=torch.device("cpu"),
        )


def test_a_state_fitted_on_other_factors_is_refused(tmp_path: Path) -> None:
    """One run's weights must not be served with another run's preprocessing."""
    foreign = _fitted_preprocessing(tuple(f"g{index}" for index in range(15)))
    foreign_hash = str(foreign["state_hash"])
    store, model_id, feature_order = _write_residual_bundle(
        tmp_path, preprocessing_override=foreign
    )
    with pytest.raises(ValueError, match="different factor order"):
        load_bundle_for_evaluation(
            store, model_id, expected_dataset_id="dataset-1",
            expected_feature_order=feature_order, expected_preprocessor_hash=foreign_hash,
            device=torch.device("cpu"),
        )
