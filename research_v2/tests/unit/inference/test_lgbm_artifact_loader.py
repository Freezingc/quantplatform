"""Tests for the LightGBM inference loader.

The property that matters most is the last one: loading a frozen bundle through this path and
scoring a frame must reproduce what the walk-forward chain scored, value for value. Everything
else is the allowlist and integrity discipline that makes loading it safe.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.lgbm_ranker import LGBMHyperparameters, LGBMRankerBaseline
from research_v2.baselines.lgbm_rankreg import LGBMRankRegBaseline
from research_v2.inference.lgbm_artifact_loader import (
    LGBM_MODEL_CODE_VERSIONS,
    load_lgbm_inference_artifact,
    score_lgbm_artifact,
)
from research_v2.models.base import ModelSignature
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

FACTORS = tuple(f"f{i:02d}" for i in range(4))
KEYS = ["signal_date", "ts_code"]
HP = LGBMHyperparameters(num_leaves=7, max_depth=3, learning_rate=0.1, min_child_samples=5,
                         feature_fraction=1.0, num_threads=2)
CONFIG = {"num_leaves": 7, "max_depth": 3, "learning_rate": 0.1, "min_child_samples": 5,
          "feature_fraction": 1.0, "lambda_l2": 0.0}


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(31)
    feats, labs = [], []
    for day in range(6):
        signal_date = date(2021, 3, 1) + timedelta(days=day)
        for i in range(40):
            signal = float(rng.standard_normal())
            feats.append({"signal_date": signal_date, "ts_code": f"S{i:03d}",
                          **{name: (signal if name == "f00" else float(rng.standard_normal()))
                             for name in FACTORS}})
            labs.append({"signal_date": signal_date, "ts_code": f"S{i:03d}",
                         "raw_return_5d": 0.9 * signal + 0.1 * float(rng.standard_normal()),
                         "label_valid": True})
    return pd.DataFrame(feats), pd.DataFrame(labs)


def _checkpoints(tmp_path: Path, carrier: str) -> dict[int, Path]:
    features, labels = _frames()
    paths: dict[int, Path] = {}
    for seed in REQUIRED_SEEDS:
        klass = LGBMRankRegBaseline if carrier == "rankreg" else LGBMRankerBaseline
        model = klass.fit(features.loc[:, KEYS + list(FACTORS)], labels,
                          factor_order=FACTORS, role="TRAIN", num_boost_round=8,
                          minimum_cross_section=30, hyperparameters=HP, seed=seed,
                          allow_nan=True)
        path = tmp_path / f"booster-{carrier}-{seed}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.booster.save_model(str(path))
        paths[seed] = path
    return paths


def _write_bundle(
    tmp_path: Path, *, model_id: str = "m0r-f1", family: str = "LGBM_RANKREG",
    carrier: str = "rankreg", code_version: str | None = None, config: dict | None = None,
    status: str = "WALK_FORWARD_PASSED", dataset_id: str = "ds:train",
    feature_order: tuple[str, ...] = FACTORS, preprocessor: str = "identity",
    signature: dict | None = None, drop_seed: int | None = None,
) -> ModelArtifactBundleStore:
    store = ModelArtifactBundleStore(tmp_path / "models")
    used_config = dict(config if config is not None else CONFIG)
    if signature is None:
        signature = dict(ModelSignature(
            model_family=family, raw_factor_count=len(feature_order),
            config_hash=hashlib.sha256(
                json.dumps(used_config, separators=(",", ":"), sort_keys=True).encode()
            ).hexdigest(),
        ).__dict__)
    checkpoints = _checkpoints(tmp_path, carrier)
    if drop_seed is not None:
        del checkpoints[drop_seed]
    features, _ = _frames()
    validation = features.loc[:, KEYS].copy()
    validation["percentile_rank"] = np.linspace(0.0, 1.0, len(validation))
    store.write(
        model_id,
        model_card={"family": family, "dataset_id": dataset_id, "model_status": status,
                    "model_code_version": code_version or LGBM_MODEL_CODE_VERSIONS[family]},
        config=dict(used_config, raw_factor_count=len(feature_order)),
        signature=signature,
        feature_order=feature_order,
        preprocessing={"state_hash": preprocessor},
        seed_checkpoints=checkpoints,
        validation_predictions=validation,
    )
    return store


def _load(store, model_id="m0r-f1", dataset_id="ds:train", order=FACTORS, **kw):
    return load_lgbm_inference_artifact(
        store, model_id, expected_dataset_id=dataset_id, expected_feature_order=order, **kw)


@pytest.fixture
def rankreg_store(tmp_path: Path) -> ModelArtifactBundleStore:
    return _write_bundle(tmp_path)


def test_loads_a_verified_bundle_with_all_three_seeds(rankreg_store) -> None:
    loaded = _load(rankreg_store)
    assert loaded.family == "LGBM_RANKREG"
    assert loaded.factor_order == FACTORS
    assert sorted(loaded.boosters) == sorted(REQUIRED_SEEDS)
    assert all(booster.num_trees() >= 1 for booster in loaded.boosters.values())
    assert loaded.preprocessing == {"state_hash": "identity"}
    audit = loaded.audit
    assert audit is not None
    assert audit.device == "cpu:lightgbm"
    assert audit.code_version == "lgbm-rankreg-1"
    assert len(audit.bundle_digest) == 64
    assert set(audit.artifact_hashes) >= {f"seed-{seed}.pt" for seed in REQUIRED_SEEDS}


def test_ranker_family_is_allowlisted_too(tmp_path: Path) -> None:
    """M0 had the same gap; a loader that served only M0R would make them incomparable."""
    store = _write_bundle(tmp_path, model_id="m0-f1", family="LGBM_RANKER", carrier="lambdarank")
    loaded = _load(store, model_id="m0-f1")
    assert loaded.family == "LGBM_RANKER"
    assert loaded.audit is not None and loaded.audit.code_version == "lgbm-ranker-1"


def test_scoring_matches_the_walk_forward_driver_value_for_value(
    rankreg_store, tmp_path: Path
) -> None:
    """The gates were computed on the driver's scores; inference must reproduce them exactly."""
    import importlib.util
    import sys

    scripts = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "driver_under_test", scripts / "run_real_candidate_training.py")
    assert spec is not None and spec.loader is not None
    driver = importlib.util.module_from_spec(spec)
    sys.modules["driver_under_test"] = driver
    spec.loader.exec_module(driver)

    loaded = _load(rankreg_store)
    features, _ = _frames()
    frame = features.loc[:, KEYS + list(FACTORS)]

    from_inference = score_lgbm_artifact(loaded, frame)
    from_driver = driver._m0_seed_mean_scores(
        {seed: driver._m0_load_booster(rankreg_store, "m0r-f1", seed, carrier="rankreg")
         for seed in REQUIRED_SEEDS},
        frame, FACTORS)
    merged = from_inference.merge(
        from_driver.rename(columns={"score": "driver"}), on=KEYS, validate="one_to_one")
    assert len(merged) == len(frame)
    assert float((merged["score"] - merged["driver"]).abs().max()) == 0.0


def test_scoring_is_invariant_to_row_order(rankreg_store) -> None:
    """Merging on business keys, not positionally, is what makes that true."""
    loaded = _load(rankreg_store)
    features, _ = _frames()
    frame = features.loc[:, KEYS + list(FACTORS)]
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    a = score_lgbm_artifact(loaded, frame)
    b = score_lgbm_artifact(loaded, shuffled)
    merged = a.merge(b.rename(columns={"score": "shuffled"}), on=KEYS, validate="one_to_one")
    assert float((merged["score"] - merged["shuffled"]).abs().max()) == 0.0


def test_scores_are_bounded_percentiles_and_never_degenerate(rankreg_store) -> None:
    """Each seed contributes an exact per-day percentile; the reported score is their mean.

    The mean of three per-day percentiles is bounded by (0, 1] but does NOT reach 1.0 unless all
    three seeds rank the same name first, so asserting a per-day max of exactly 1.0 here would
    be asserting something false -- measured maxima on this fixture are 0.9375 to 0.9875. What
    must hold is the bound, and that no day collapses to a constant.
    """
    loaded = _load(rankreg_store)
    features, _ = _frames()
    scores = score_lgbm_artifact(loaded, features.loc[:, KEYS + list(FACTORS)])
    assert float(scores["score"].min()) > 0.0
    assert float(scores["score"].max()) <= 1.0
    per_day = scores.groupby("signal_date")["score"]
    assert float(per_day.max().max()) <= 1.0
    assert bool((per_day.nunique() > 1).all()), "a day scored every name identically"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"dataset_id": "other:train"}, "dataset_id", id="wrong_dataset"),
        pytest.param({"preprocessor": "not-identity"}, "preprocessor", id="wrong_preprocessor"),
        pytest.param({"status": "DRAFT"}, "WALK_FORWARD_PASSED", id="not_promoted"),
        pytest.param({"code_version": "lgbm-rankreg-999"}, "model_code_version",
                     id="wrong_code_version"),
        pytest.param({"family": "RESIDUAL_MLP", "carrier": "rankreg",
                      "code_version": "residual-mlp-1"}, "not allowlisted", id="torch_family"),
        pytest.param({"signature": {"model_family": "LGBM_RANKREG", "raw_factor_count": 4,
                                    "config_hash": "0" * 64}}, "signature", id="bad_signature"),
        pytest.param({"family": "LGBM_RANKER", "carrier": "rankreg"}, "objective",
                     id="family_disagrees_with_booster"),
    ],
)
def test_load_refuses_an_inconsistent_bundle(tmp_path: Path, kwargs: dict, match: str) -> None:
    store = _write_bundle(tmp_path, **kwargs)
    with pytest.raises(ValueError, match=match):
        _load(store)


def test_a_bundle_missing_a_seed_cannot_be_written_at_all(tmp_path: Path) -> None:
    """The loader checks for a missing seed, but the store refuses to create that state first.

    Both guards are kept: the store's is the one that fires in practice, and the loader's is
    defence in depth for a bundle assembled by something other than the store.
    """
    with pytest.raises(ValueError, match="requires all required seed checkpoints"):
        _write_bundle(tmp_path, drop_seed=REQUIRED_SEEDS[1])


def test_load_refuses_a_wrong_feature_order(tmp_path: Path) -> None:
    store = _write_bundle(tmp_path)
    with pytest.raises(ValueError, match="feature_order"):
        _load(store, order=tuple(reversed(FACTORS)))


def test_load_propagates_the_inference_contract_check(tmp_path: Path) -> None:
    store = _write_bundle(tmp_path)
    with pytest.raises(ValueError):
        _load(store, expected_inference_contract={"factor_set_id": "not-in-the-card"})


def test_scoring_refuses_a_bad_frame(rankreg_store) -> None:
    loaded = _load(rankreg_store)
    features, _ = _frames()
    good = features.loc[:, KEYS + list(FACTORS)]

    with pytest.raises(ValueError, match="missing columns"):
        score_lgbm_artifact(loaded, good.drop(columns=["f03"]))
    duplicated = pd.concat([good, good.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate business keys"):
        score_lgbm_artifact(loaded, duplicated)
    infinite = good.copy()
    infinite.loc[0, "f01"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        score_lgbm_artifact(loaded, infinite)
