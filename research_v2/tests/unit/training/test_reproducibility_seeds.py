from __future__ import annotations

import random

import numpy as np
import torch

from research_v2.training.seeds import configure_reproducibility


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
