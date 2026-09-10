from __future__ import annotations

import time
from collections.abc import Mapping

import pytest

from research_v2.training.parallel_search import (
    SearchTask,
    capped_worker_count,
    run_parallel_search,
)


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
