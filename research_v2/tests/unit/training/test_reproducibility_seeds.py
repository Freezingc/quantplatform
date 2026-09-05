from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from research_v2.training.seeds import (
    UNSEEDED,
    configure_reproducibility,
    describe_runtime_environment,
)


def _numpy_state() -> tuple[object, ...]:
    kind, state, *rest = np.random.get_state()
    return (kind, np.asarray(state).copy(), *rest)


def _same_numpy_state(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return (
        left[0] == right[0]
        and bool(np.array_equal(np.asarray(left[1]), np.asarray(right[1])))
        and left[2:] == right[2:]
    )


def test_describe_runtime_environment_observes_without_perturbing_anything() -> None:
    """A trust root has to self-describe, but describing must not change the result.

    ``configure_reproducibility`` seeds every RNG and flips torch into
    deterministic-algorithm mode, so calling it merely to report the runtime would
    perturb the very run being described -- and in the baseline walk-forward path,
    where the B2/B3 fits are seed-sensitive, it would change the reported metrics.
    """
    python_before = random.getstate()
    numpy_before = _numpy_state()
    torch_before = torch.initial_seed()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    threads_before = torch.get_num_threads()

    environment = describe_runtime_environment()

    assert random.getstate() == python_before
    assert _same_numpy_state(numpy_before, _numpy_state())
    assert torch.initial_seed() == torch_before
    assert torch.are_deterministic_algorithms_enabled() is deterministic_before
    assert torch.get_num_threads() == threads_before
    # The observation still has to return a usable record, not just avoid touching state.
    assert environment.seed == UNSEEDED
    assert environment.torch_intraop_threads == threads_before


def test_describe_runtime_environment_reports_the_live_numerics_affecting_fields() -> None:
    environment = describe_runtime_environment()

    assert environment.seed == UNSEEDED
    assert environment.torch_version == torch.__version__
    assert environment.torch_intraop_threads == torch.get_num_threads()
    assert environment.torch_interop_threads == torch.get_num_interop_threads()
    assert environment.torch_intraop_threads >= 1
    assert environment.cuda_available is torch.cuda.is_available()


def test_unseeded_sentinel_is_rejected_by_the_seeding_entry_point() -> None:
    """The observation-only marker must never be mistaken for a seeded run."""
    with pytest.raises(ValueError):
        configure_reproducibility(UNSEEDED)


def test_seed_configuration_replays_python_numpy_and_torch_streams() -> None:
    first_environment = configure_reproducibility(20260801)
    first = (random.random(), float(np.random.random()), float(torch.rand(())))
    second_environment = configure_reproducibility(20260801)
    second = (random.random(), float(np.random.random()), float(torch.rand(())))

    assert first == second
    assert first_environment.seed == second_environment.seed == 20260801
    assert first_environment.torch_version == torch.__version__
    assert torch.are_deterministic_algorithms_enabled()


def test_environment_records_the_torch_thread_configuration() -> None:
    """Intra-op thread count changes numerics, so it must be captured with the seed.

    Measured 2026-09-05 on the real M1 code path: identical seed, config and data
    produced three distinct checkpoint digests at intra-op threads 1 / 4 / 8, while
    nothing in the recorded evidence said which configuration produced a result.
    Recording it is the numerics-neutral half of closing that gap; pinning it is a
    separate, adjudicated change.
    """
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(3)
        environment = configure_reproducibility(20260801)
        assert environment.torch_intraop_threads == 3
        assert environment.torch_interop_threads == torch.get_num_interop_threads()
    finally:
        torch.set_num_threads(original)
    assert configure_reproducibility(20260801).torch_intraop_threads == original
