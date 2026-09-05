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


UNSEEDED = -1
"""Marker for a runtime that was *observed* rather than configured.

``configure_reproducibility`` rejects negative seeds, so a ``ReproducibilityEnvironment``
carrying this value can never be mistaken for the record of a seeded run.
"""


def configure_reproducibility(seed: int) -> ReproducibilityEnvironment:
    """Seed every supported RNG and fail instead of silently allowing nondeterministic ops."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return _capture(seed)


def describe_runtime_environment() -> ReproducibilityEnvironment:
    """Report the numerics-affecting runtime while changing nothing.

    A trust root has to self-describe the runtime that produced it: the intra-op thread
    count changes checkpoint digests (floating-point reductions are not associative) and
    defaults to half the host core count, i.e. it is a host property rather than a
    declared one, and the device is numerics-affecting too. But :func:`configure_reproducibility`
    seeds every RNG and flips torch into deterministic-algorithm mode, so calling it just
    to describe the runtime would perturb the run being described -- and in the baseline
    walk-forward path, whose B2/B3 fits are seed-sensitive, it would change the reported
    metrics of an artifact that is under a bit-identical reproduction contract.

    This reads the same fields and touches no global state. The returned ``seed`` is
    :data:`UNSEEDED`, which the seeding entry point rejects, so an observed environment
    can never be read back as a seeded one.
    """
    return _capture(UNSEEDED)


def _capture(seed: int) -> ReproducibilityEnvironment:
    """Read the live runtime fields; pure observation, no global state changes."""
    cuda_available = torch.cuda.is_available()
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
