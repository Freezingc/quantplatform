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
