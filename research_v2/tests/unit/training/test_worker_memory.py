"""Tests for the worker-memory precondition.

The failure this prevents is the expensive kind: a worker that does not fit is not refused at
startup, it is OOM-killed hours into a multi-day run. Two choices here are load-bearing and get
their own tests -- reading MemAvailable rather than MemFree, because on a host with hundreds of
gigabytes of reclaimable parquet cache MemFree would refuse runs that fit comfortably, and computing
the largest safe worker count rather than only saying no, because the operator's next question is
always "then how many".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_v2.training.parallel_search import (
    available_memory_gb,
    worker_memory_budget,
)

MEMINFO = """MemTotal:       659450880 kB
MemFree:        131072000 kB
MemAvailable:   521878944 kB
Buffers:          1024000 kB
"""


def _meminfo(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "meminfo"
    path.write_text(text, encoding="utf-8")
    return path


def test_available_memory_reads_mem_available_not_mem_free(tmp_path: Path) -> None:
    """MemFree on this host is 125 GiB while 497 GiB is actually available to a new process."""
    path = _meminfo(tmp_path, MEMINFO)
    available = available_memory_gb(path)
    assert available is not None
    assert available == pytest.approx(521878944 / 1024 / 1024)
    assert available > 400.0
    # And it is demonstrably not the MemFree line, which is a quarter of the size.
    assert available != pytest.approx(131072000 / 1024 / 1024)


def test_available_memory_is_none_when_the_host_does_not_publish_it(tmp_path: Path) -> None:
    assert available_memory_gb(tmp_path / "absent") is None
    assert available_memory_gb(_meminfo(tmp_path, "MemTotal: 1 kB\n")) is None
    assert available_memory_gb(_meminfo(tmp_path, "MemAvailable:   not-a-number kB\n")) is None


def test_a_budget_that_fits_is_reported_as_fitting() -> None:
    budget = worker_memory_budget(workers=17, per_worker_gb=25.0, available_gb=497.0)
    assert budget.fits is True
    assert budget.required_gb == pytest.approx(425.0)
    assert budget.safe_workers == 17
    # One more worker is the same request that does not fit, so the boundary is the boundary.
    assert worker_memory_budget(workers=18, per_worker_gb=25.0, available_gb=497.0).fits is False


def test_a_budget_that_does_not_fit_says_how_many_would() -> None:
    """48 temporal workers on this host is an OOM kill; the message has to carry the alternative."""
    budget = worker_memory_budget(workers=48, per_worker_gb=25.0, available_gb=497.0)
    assert budget.fits is False
    assert budget.required_gb == pytest.approx(1200.0)
    # 497 * 0.9 = 447.3 GiB of usable headroom, so 17 workers fit and 18 do not.
    assert budget.safe_workers == 17
    assert worker_memory_budget(workers=17, per_worker_gb=25.0, available_gb=497.0).fits is True


def test_the_headroom_is_not_available_memory() -> None:
    """Filling the host to the last gibibyte leaves nothing for the parent or the page cache."""
    budget = worker_memory_budget(workers=10, per_worker_gb=10.0, available_gb=105.0)
    assert budget.fits is False
    assert budget.safe_workers == 9


@pytest.mark.parametrize(
    ("workers", "per_worker_gb", "available_gb"),
    [
        pytest.param(0, 10.0, 100.0, id="no_workers"),
        pytest.param(-1, 10.0, 100.0, id="negative_workers"),
        pytest.param(4, 0.0, 100.0, id="zero_footprint"),
        pytest.param(4, -1.0, 100.0, id="negative_footprint"),
        pytest.param(4, 10.0, 0.0, id="no_memory"),
    ],
)
def test_a_nonsensical_budget_is_refused_rather_than_computed(
    workers: int, per_worker_gb: float, available_gb: float
) -> None:
    with pytest.raises(ValueError):
        worker_memory_budget(
            workers=workers, per_worker_gb=per_worker_gb, available_gb=available_gb
        )
