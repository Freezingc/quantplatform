"""Ordered spawned-process execution for independent configuration trials."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Literal

MEMORY_HEADROOM = 0.9
"""Fraction of MemAvailable a run may commit to workers.

Not 1.0: the parent process holds its own copy of the fold's frames and batches, the page cache is
doing real work for the dataset reads, and a host at 100% committed does not fail gracefully -- it
picks a victim.
"""


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


def granularity_makespans(
    *, config_count: int, workers: int, seeds_per_config: int
) -> tuple[int, int]:
    """Both candidate makespans in fit-times: ``(config-level, seed-level)``.

    A config-level task is ``seeds_per_config`` fits that one worker runs back to back and that
    cannot be split once dispatched; a seed-level task is one fit. Wall clock is therefore rounds
    times task length -- ``ceil(configs / workers) * seeds`` against ``ceil(configs * seeds /
    workers)`` -- and the two diverge whenever the configs do not divide evenly into the workers,
    which is exactly the case that "fill the workers" reasoning gets wrong.
    """
    _validate_granularity_inputs(
        config_count=config_count, workers=workers, seeds_per_config=seeds_per_config
    )
    config_rounds = math.ceil(config_count / workers)
    seed_rounds = math.ceil(config_count * seeds_per_config / workers)
    return config_rounds * seeds_per_config, seed_rounds


def search_granularity(
    *, config_count: int, workers: int, seeds_per_config: int
) -> Literal["config", "seed"]:
    """Pick the task granularity with the shorter makespan; ties go to the coarser one.

    The measured case is M2: 24 configurations, but 20 workers because F4's window panels put each
    worker at ~21.6 GiB and the host has ~497 GiB. Config-level runs two rounds of three fits --
    six fit-times, because the four configs left over still cost a whole round -- while seed-level
    runs four rounds of one fit. A third less wall clock on a multi-day run, from choosing the unit
    of work by arithmetic instead of by whether the workers look busy.

    Coarse wins ties: it dispatches a third as many tasks, returns scalars instead of per-seed
    prediction frames, and keeps one worker per configuration.
    """
    config_span, seed_span = granularity_makespans(
        config_count=config_count, workers=workers, seeds_per_config=seeds_per_config
    )
    return "config" if config_span <= seed_span else "seed"


def _validate_granularity_inputs(
    *, config_count: int, workers: int, seeds_per_config: int
) -> None:
    if config_count < 1:
        raise ValueError("config_count must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if seeds_per_config < 1:
        raise ValueError("seeds_per_config must be positive")


def available_memory_gb(path: Path = Path("/proc/meminfo")) -> float | None:
    """MemAvailable in gibibytes, or None when the host does not publish it.

    MemAvailable and not MemFree: on a host that has been reading hundreds of gigabytes of parquet,
    the reclaimable cache is memory a spawned worker can actually have. MemFree here is ~125 GiB
    against ~497 GiB available, so reading the wrong line would refuse runs that fit comfortably --
    a guard that cries wolf gets disabled, and then it guards nothing.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            kibibytes = int(fields[1])
        except ValueError:
            return None
        return kibibytes / (1024 * 1024)
    return None


def resident_memory_gb(path: Path = Path("/proc/self/status")) -> float | None:
    """This process's resident set in gibibytes, or None when the host does not publish it.

    The parent is not a spectator in a temporal run: it holds the prepared train, validation and
    test frames and all three batch sets, including the test panel that concatenates them, so it can
    be larger than any single worker. A budget that counted only the workers would pass 20 x 21.6
    GiB against a 447 GiB ceiling and then be killed for the 30 GiB nobody wrote down.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) < 2:
            return None
        try:
            kibibytes = int(fields[1])
        except ValueError:
            return None
        return kibibytes / (1024 * 1024)
    return None


@dataclass(frozen=True)
class WorkerMemoryBudget:
    """Whether N workers each holding G GiB fit beside the parent, and how many would."""

    workers: int
    per_worker_gb: float
    available_gb: float
    required_gb: float
    safe_workers: int
    fits: bool
    parent_gb: float = 0.0


def worker_memory_budget(
    *,
    workers: int,
    per_worker_gb: float,
    available_gb: float,
    parent_gb: float = 0.0,
    headroom: float = MEMORY_HEADROOM,
) -> WorkerMemoryBudget:
    """Arithmetic the run should do before it starts, not the kernel's OOM killer should do later.

    ``safe_workers`` is returned alongside the verdict because the operator's next question is
    always "then how many", and a refusal that does not answer it just gets retried by guesswork.
    """
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    if per_worker_gb <= 0:
        raise ValueError("per_worker_gb must be a positive number")
    if available_gb <= 0:
        raise ValueError("available_gb must be a positive number")
    if parent_gb < 0:
        raise ValueError("parent_gb cannot be negative")
    if not 0 < headroom <= 1:
        raise ValueError("headroom must be in (0, 1]")
    limit = float(available_gb) * headroom
    required = workers * float(per_worker_gb) + float(parent_gb)
    safe = max(0, int((limit - float(parent_gb)) // float(per_worker_gb)))
    return WorkerMemoryBudget(
        workers=workers,
        per_worker_gb=float(per_worker_gb),
        available_gb=float(available_gb),
        required_gb=required,
        safe_workers=safe,
        fits=required <= limit,
        parent_gb=float(parent_gb),
    )


def run_parallel_search(
    tasks: Sequence[SearchTask],
    *,
    workers: int,
    run_task: SearchTaskRunner,
    on_result: Callable[[int, dict[str, object]], None] | None = None,
) -> list[tuple[int, dict[str, object]]]:
    """Run independent tasks in spawned processes and restore canonical index order.

    The parent receives every result before returning. A failed child is deliberately
    allowed to raise through ``future.result``: selecting from a partial grid is invalid.

    ``on_result`` is called with each result as it lands, before any later failure is raised. That
    is not a convenience hook: the caller persists search records there, so a round that fails part
    way keeps every configuration that finished instead of losing the fold's whole search. Results
    collected only at the end are discarded by the raise, and on a 24-configuration temporal round
    that is twenty-odd hours of fits lost to one bad worker -- with nothing for ``--resume`` to
    adopt on the restart.
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
            if on_result is not None:
                on_result(futures[future], result)
    return sorted(completed, key=lambda item: item[0])


__all__ = [
    "MEMORY_HEADROOM",
    "SearchTask",
    "SearchTaskRunner",
    "WorkerMemoryBudget",
    "available_memory_gb",
    "capped_worker_count",
    "granularity_makespans",
    "resident_memory_gb",
    "run_parallel_search",
    "search_granularity",
    "worker_memory_budget",
]
