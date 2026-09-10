"""Tests for the M2 spawned search worker.

Two properties decide whether M2 can be searched in parallel at all, and they are the ones pinned
here. First, the worker must rebuild the SAME batches the parent would build, from identifiers
alone -- if it reconstructed a slightly different panel, the validation RankIC that selects the
winning config would be measuring a different model than the one the parent finally fits. Second,
the task payload must stay small: the reason this module exists rather than reusing
``cross_section_search`` is that a temporal batch holds a 20-day window per row and the validation
panel is train+validation concatenated, so carrying batches in the payload is not slow, it is
impossible. That is asserted as a byte budget rather than left as a comment.
"""

from __future__ import annotations

import math
import pickle
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
from research_v2.inference.torch_scoring import prepare_frame
from research_v2.training.parallel_search import SearchTask, run_parallel_search
from research_v2.training.temporal_batches import build_temporal_training_batches
from research_v2.training.temporal_search import (
    TemporalConfigTask,
    TemporalFoldSpec,
    run_temporal_config_task,
    temporal_fold_batches,
    temporal_model,
)

FACTOR_ORDER = ("f00", "f01")
# Thirty consecutive days inside one month: enough that the train half has dates with a full
# 15-day history (day 15 onward) while the validation half's windows can only be filled by
# reaching back into train.
TRAIN_DAYS = [date(2025, 1, day) for day in range(1, 21)]
VALIDATION_DAYS = [date(2025, 1, day) for day in range(21, 31)]
# Disjoint from both: the manifest validates business keys across the whole dataset, so a test
# split that reused the validation rows is refused before anything else can be checked.
TEST_DAYS = [date(2025, 2, day) for day in range(1, 6)]
SPLIT_DAY = VALIDATION_DAYS[0]
STOCKS = 32


def _frame(days: list[date], rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for day in days:
        for stock in range(STOCKS):
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": f"S{stock:03d}",
                    "raw_return_5d": float(rng.standard_normal()) * 0.01,
                    "label_valid": True,
                    "f00": float(rng.standard_normal()),
                    "f01": float(rng.standard_normal()),
                }
            )
    return pd.DataFrame(rows)


def _fold_frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    return {
        "train": _frame(TRAIN_DAYS, rng),
        "validation": _frame(VALIDATION_DAYS, rng),
        "test": _frame(TEST_DAYS, rng),
    }


def _published_fold(tmp_path: Path) -> tuple[TemporalFoldSpec, str, str]:
    """Publish a tiny fold the way the driver does, and return its spec plus the dataset ids."""
    frames = _fold_frames()
    preprocessor = FoldPreprocessor(factor_order=FACTOR_ORDER)
    preprocessor.fit(frames["train"].loc[:, list(FACTOR_ORDER)], role="TRAIN")
    metadata = {
        "fold_id": "F1",
        "factor_order": list(FACTOR_ORDER),
        "data_versions": {"label_daily": {"2025-01": "v1"}},
    }
    manifest = build_split_dataset_manifest(
        frames, business_keys=("signal_date", "ts_code"), metadata=metadata
    )
    store = DatasetArtifactStore(tmp_path / "datasets")
    store.write(manifest, frames, preprocessing={"assembler": "test"})
    spec = TemporalFoldSpec(
        workspace=tmp_path,
        fold_id="F1",
        dataset_id=manifest.dataset_id,
        factor_order=FACTOR_ORDER,
        preprocessing={
            "state_hash": preprocessor.state_hash,
            "state_payload": preprocessor.state_payload(),
        },
        torch_threads=1,
    )
    return spec, f"{manifest.dataset_id}:train", f"{manifest.dataset_id}:validation"


def _parent_recipe(spec: TemporalFoldSpec) -> dict[str, Any]:
    """What the driver's serial M2 loop builds, written out independently for comparison."""
    store = DatasetArtifactStore(spec.workspace / "datasets")
    train = store.load_split(spec.dataset_id, "train", expected_fold_id=spec.fold_id)
    validation = store.load_split(spec.dataset_id, "validation", expected_fold_id=spec.fold_id)
    preprocessor = FoldPreprocessor.restore(
        spec.preprocessing["state_payload"],
        expected_state_hash=str(spec.preprocessing["state_hash"]),
    )
    prepared_train = prepare_frame(train, preprocessor, spec.factor_order)
    prepared_validation = prepare_frame(validation, preprocessor, spec.factor_order)
    train_dates = set(pd.to_datetime(train["signal_date"]).dt.date)
    validation_dates = set(pd.to_datetime(validation["signal_date"]).dt.date)
    panel = pd.concat([prepared_train, prepared_validation], ignore_index=True)
    # Literals on purpose: this is the independent side of the comparison, so a drift in the
    # module's window contract has to break the equality rather than move both sides together.
    train_windows = build_temporal_training_batches(
        prepared_train,
        factor_order=spec.factor_order,
        history_length=20,
        minimum_valid_history_length=15,
    )
    panel_windows = build_temporal_training_batches(
        panel,
        factor_order=spec.factor_order,
        history_length=20,
        minimum_valid_history_length=15,
    )
    return {
        "train": {day: b for day, b in train_windows.items() if day in train_dates},
        "validation": {day: b for day, b in panel_windows.items() if day in validation_dates},
        "labels": validation,
    }


def test_the_worker_rebuilds_the_batches_the_parent_would_build(tmp_path: Path) -> None:
    """Same bytes, or the search selects on a metric the final fit does not share."""
    spec, _, _ = _published_fold(tmp_path)
    expected = _parent_recipe(spec)
    built = temporal_fold_batches(spec)
    assert set(built.train_batches) == set(expected["train"])
    assert set(built.validation_batches) == set(expected["validation"])
    assert built.train_batches, "the fixture must produce at least one eligible train date"
    for day, batch in built.train_batches.items():
        reference = expected["train"][day]
        assert torch.equal(batch.inputs.features, reference.inputs.features)
        assert torch.equal(batch.inputs.stock_mask, reference.inputs.stock_mask)
        assert batch.inputs.ts_codes == reference.inputs.ts_codes
        assert torch.equal(batch.target_rank, reference.target_rank)
    for day, batch in built.validation_batches.items():
        assert torch.equal(batch.inputs.features, expected["validation"][day].inputs.features)
    pd.testing.assert_frame_equal(built.validation_labels, expected["labels"])


def test_the_validation_panel_reaches_back_into_train(tmp_path: Path) -> None:
    """The first validation date has a window at all only if the panel includes train.

    Day 21 is the split's first day, so within the validation split alone it has zero days of
    history and could never reach the 15-day minimum. Its batch existing is the proof that the
    worker concatenated train before windowing, which is what the driver's serial loop does.
    """
    spec, _, _ = _published_fold(tmp_path)
    built = temporal_fold_batches(spec)
    assert min(built.validation_batches) == SPLIT_DAY
    assert SPLIT_DAY in built.validation_batches
    # And the train half is windowed inside train only. Day 15 is the first eligible date, which
    # pins the window semantics: the 15-day minimum counts the signal date itself (days 1-15), so
    # a change to what "history length" means moves this boundary and fails here.
    assert min(built.train_batches) == date(2025, 1, 15)
    assert max(built.train_batches) == date(2025, 1, 20)


def test_a_worker_builds_one_fold_once_however_many_configs_it_is_handed(tmp_path: Path) -> None:
    spec, _, _ = _published_fold(tmp_path)
    first = temporal_fold_batches(spec)
    assert temporal_fold_batches(spec) is first


def test_the_payload_carries_identifiers_and_not_gigabytes_of_batches(tmp_path: Path) -> None:
    """The whole reason M2 cannot reuse the static worker, expressed as a budget."""
    spec, train_id, validation_id = _published_fold(tmp_path)
    task = TemporalConfigTask(
        config_index=0,
        config_payload={"family": "GRU", "hidden_dim": 4, "dropout": 0.0, "learning_rate": 0.001},
        fold=spec,
        train_dataset_id=train_id,
        validation_dataset_id=validation_id,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_prefix="m2-f1-c0",
        max_epochs=1,
        early_stopping_patience=10,
    )
    assert len(pickle.dumps(task)) < 100_000


@pytest.mark.parametrize("family", ["GRU", "TCN"])
def test_the_model_factory_follows_the_config_choice(family: str) -> None:
    model = temporal_model({"family": family, "hidden_dim": 4, "dropout": 0.0}, FACTOR_ORDER)
    assert type(model).__name__ == f"Temporal{family}"


def test_the_model_factory_refuses_an_unknown_choice() -> None:
    with pytest.raises(ValueError, match="unsupported temporal family"):
        temporal_model({"family": "LSTM", "hidden_dim": 4, "dropout": 0.0}, FACTOR_ORDER)


def test_a_spawned_temporal_worker_returns_a_complete_three_seed_result(tmp_path: Path) -> None:
    """A package-level worker must survive spawn and rebuild its own batches in the child."""
    spec, train_id, validation_id = _published_fold(tmp_path)
    task = TemporalConfigTask(
        config_index=0,
        config_payload={"family": "GRU", "hidden_dim": 4, "dropout": 0.0, "learning_rate": 0.001},
        fold=spec,
        train_dataset_id=train_id,
        validation_dataset_id=validation_id,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_prefix="m2-f1-c0",
        max_epochs=1,
        early_stopping_patience=10,
    )
    results = run_parallel_search(
        (SearchTask(index=0, payload={"task": task}),),
        workers=1,
        run_task=run_temporal_config_task,
    )
    assert results[0][0] == 0
    result = results[0][1]
    assert result["config"] == dict(task.config_payload)
    assert set(result["seed_validation_rank_ic"]) == {20260801, 20260802, 20260803}
    assert math.isfinite(float(result["ensemble_validation_rank_ic"]))
    assert int(result["worst_seed"]) in {20260801, 20260802, 20260803}
    assert (tmp_path / "checkpoints").exists()
