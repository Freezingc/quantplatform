from __future__ import annotations

import math
import time
from collections.abc import Mapping
from datetime import date

import pandas as pd

import pytest

from research_v2.training.cross_section_search import (
    CrossSectionConfigTask,
    run_cross_section_config_task,
)
from research_v2.training.parallel_search import (
    SearchTask,
    capped_worker_count,
    run_parallel_search,
)
from research_v2.training.static_batches import build_static_training_batches


def _delayed_result(payload: Mapping[str, object]) -> dict[str, object]:
    """Top-level and importable so a spawned worker executes real scheduler work."""
    time.sleep(float(payload["delay_seconds"]))
    return {"value": int(payload["value"])}


def _raising_result(payload: Mapping[str, object]) -> dict[str, object]:
    raise RuntimeError(f"worker failed for {payload['value']}")


def test_parallel_search_restores_grid_order_after_reverse_completion() -> None:
    """Removing index sorting would select/tie-break based on scheduler timing."""
    tasks = (
        SearchTask(index=0, payload={"delay_seconds": 0.05, "value": 10}),
        SearchTask(index=1, payload={"delay_seconds": 0.0, "value": 20}),
    )

    results = run_parallel_search(tasks, workers=2, run_task=_delayed_result)

    assert results == [(0, {"value": 10}), (1, {"value": 20})]


def test_parallel_search_caps_requested_workers_to_the_available_tasks() -> None:
    """Removing the cap would start idle heavyweight Torch workers."""
    assert capped_worker_count(task_count=1, requested_workers=48) == 1


def test_parallel_search_propagates_a_worker_exception() -> None:
    """Swallowing a worker failure would let the parent choose from a partial grid."""
    with pytest.raises(RuntimeError, match="worker failed for 10"):
        run_parallel_search(
            (SearchTask(index=0, payload={"value": 10}),),
            workers=1,
            run_task=_raising_result,
        )


def test_spawned_cross_section_worker_returns_a_complete_three_seed_result(tmp_path) -> None:
    """A package-level worker must survive spawn, not merely work in the parent process."""
    def frame(day: date) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": [day] * 30,
                "ts_code": [f"S{index:03d}" for index in range(30)],
                "raw_return_5d": [float(index) for index in range(30)],
                "label_valid": [True] * 30,
                "f00": [float(index) for index in range(30)],
            }
        )

    train = frame(date(2020, 1, 2))
    validation = frame(date(2020, 1, 3))
    task = CrossSectionConfigTask(
        family="M1",
        config_index=0,
        config_payload={"hidden_dim": 8, "num_blocks": 1, "dropout": 0.0},
        factor_order=("f00",),
        train_batches=build_static_training_batches(train, factor_order=("f00",)),
        validation_batches=build_static_training_batches(validation, factor_order=("f00",)),
        validation_labels=validation,
        checkpoint_root=tmp_path / "checkpoints",
        checkpoint_prefix="m1-f1-c0",
        train_dataset_id="train",
        validation_dataset_id="validation",
        max_epochs=1,
        early_stopping_patience=10,
        torch_threads=1,
    )

    results = run_parallel_search(
        (SearchTask(index=0, payload={"task": task}),),
        workers=1,
        run_task=run_cross_section_config_task,
    )

    assert results[0][0] == 0
    result = results[0][1]
    assert result["config"] == task.config_payload
    assert set(result["seed_validation_rank_ic"]) == {20260801, 20260802, 20260803}
    assert math.isfinite(float(result["ensemble_validation_rank_ic"]))
