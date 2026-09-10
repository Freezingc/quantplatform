"""Ordered spawned-process execution for independent configuration trials."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context


@dataclass(frozen=True)
class SearchTask:
    """One immutable config-search task, bound to its canonical grid index."""

    index: int
    payload: Mapping[str, object]


SearchTaskRunner = Callable[[Mapping[str, object]], dict[str, object]]


def capped_worker_count(*, task_count: int, requested_workers: int) -> int:
    """Return a valid worker count without starting idle heavyweight workers."""
    if task_count < 0:
        raise ValueError("task count cannot be negative")
    if isinstance(requested_workers, bool) or requested_workers < 1:
        raise ValueError("requested workers must be a positive integer")
    return min(task_count, requested_workers)


def run_parallel_search(
    tasks: Sequence[SearchTask],
    *,
    workers: int,
    run_task: SearchTaskRunner,
) -> list[tuple[int, dict[str, object]]]:
    """Run independent tasks in spawned processes and restore canonical index order.

    The parent receives every result before returning. A failed child is deliberately
    allowed to raise through ``future.result``: selecting from a partial grid is invalid.
    """
    task_list = tuple(tasks)
    if len({task.index for task in task_list}) != len(task_list):
        raise ValueError("parallel search task indexes must be unique")
    if any(task.index < 0 for task in task_list):
        raise ValueError("parallel search task indexes cannot be negative")
    worker_count = capped_worker_count(
        task_count=len(task_list), requested_workers=workers
    )
    if worker_count == 0:
        return []

    completed: list[tuple[int, dict[str, object]]] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(run_task, task.payload): task.index for task in task_list
        }
        for future in as_completed(futures):
            result = future.result()
            if not isinstance(result, dict):
                raise TypeError("parallel search worker must return a dictionary")
            completed.append((futures[future], result))
    return sorted(completed, key=lambda item: item[0])


__all__ = ["SearchTask", "SearchTaskRunner", "capped_worker_count", "run_parallel_search"]
