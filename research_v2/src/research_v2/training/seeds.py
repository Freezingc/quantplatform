"""Explicit reproducibility setup for Python, NumPy, and Torch training runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproducibilityEnvironment:
    seed: int
    torch_version: str
    cuda_available: bool
    cuda_device_names: tuple[str, ...]
    # Intra-op thread count is a numerics-affecting parameter: floating-point
    # reductions are not associative, so the same seed/config/data yields different
    # weights at different thread counts (measured: three distinct checkpoint
    # digests at 1/4/8 threads). It defaults to half the host core count, i.e. it is
    # a host property that was previously neither pinned nor recorded. Capturing it
    # here makes every derived artifact self-describing; pinning it is a separate,
    # adjudicated change so this addition cannot alter any result.
    torch_intraop_threads: int
    torch_interop_threads: int


def configure_reproducibility(seed: int) -> ReproducibilityEnvironment:
    """Seed every supported RNG and fail instead of silently allowing nondeterministic ops."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device_names = (
        tuple(torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count()))
        if cuda_available
        else ()
    )
    return ReproducibilityEnvironment(
        seed=seed,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_device_names=device_names,
        torch_intraop_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
    )
