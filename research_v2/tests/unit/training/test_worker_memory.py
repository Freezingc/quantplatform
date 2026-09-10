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
    resident_memory_gb,
    worker_memory_budget,
)

MEMINFO = """MemTotal:       659450880 kB
MemFree:        131072000 kB
MemAvailable:   521878944 kB
Buffers:          1024000 kB
"""

STATUS = """Name:	python
VmPeak:	31457280 kB
VmRSS:	31457280 kB
Threads:	52
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


def test_resident_memory_reads_this_process_vmrss(tmp_path: Path) -> None:
    path = tmp_path / "status"
    path.write_text(STATUS, encoding="utf-8")
    resident = resident_memory_gb(path)
    assert resident is not None
    assert resident == pytest.approx(30.0)
    assert resident_memory_gb(tmp_path / "absent") is None
    assert resident_memory_gb(_meminfo(tmp_path, "Name: python\n")) is None


def test_the_parent_footprint_counts_against_the_same_ceiling() -> None:
    """The M2 configuration this guard exists for: 20 workers at the largest fold's footprint.

    20 x 21.6 = 432 GiB fits under the 447.3 GiB of usable headroom on its own, which is exactly
    how a budget that forgot the parent gets written -- and the temporal parent holds the prepared
    train, validation and test frames plus all three batch sets, so it is not a rounding error.
    With a 30 GiB parent the request is 462 GiB and does not fit; nineteen workers do.
    """
    without_parent = worker_memory_budget(
        workers=20, per_worker_gb=21.6, available_gb=497.0
    )
    assert without_parent.fits is True
    with_parent = worker_memory_budget(
        workers=20, per_worker_gb=21.6, available_gb=497.0, parent_gb=30.0
    )
    assert with_parent.fits is False
    assert with_parent.required_gb == pytest.approx(462.0)
    assert with_parent.safe_workers == 19
    assert worker_memory_budget(
        workers=19, per_worker_gb=21.6, available_gb=497.0, parent_gb=30.0
    ).fits is True


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
