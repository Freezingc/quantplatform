"""The one place that names the torch model families, derived from the models themselves.

Four modules need to know which family names the chain can write into a bundle's model card: the
walk-forward evaluator's allowlist, the reproduction audit's scorer routing, the inference builder's
batch-shape dispatch, and the artifact loader's code-version map. Each was written as string
literals and three of the four drifted from what the models actually declare:

* the evaluator accepted ``{"M2", "TEMPORAL"}`` while ``run_full_m2`` writes
  ``f"TEMPORAL_{winning family}"``, so it accepted neither ``TEMPORAL_GRU`` nor ``TEMPORAL_TCN`` and
  would have refused EVERY M2 run at the evaluation step, after the training was already done;
* the inference builder's temporal set held only ``TEMPORAL_GRU``, so a TCN bundle could not be
  batched for scoring;
* the artifact loader's code-version map had no ``TEMPORAL_TCN`` and no factory branch for it, so a
  TCN bundle could not be loaded at all.

None of that was visible, because ``test_unknown_family_is_rejected`` used ``TEMPORAL_TCN`` as its
example of an unknown family and the loader's allowlist test compared one hand-written dict against
another -- both pinned the omission as if it were the contract.

Deriving the sets from the classes' own ``model_family`` attributes makes a rename propagate instead
of drift, and ``tests/unit/models/test_families.py`` pins every consumer to these sets so the next
omission is a test failure in seconds rather than a multi-day run that cannot finish.
"""

from __future__ import annotations

from collections.abc import Mapping

from research_v2.models.residual_mlp import ResidualMLP
from research_v2.models.set_attention import SetAttention
from research_v2.models.temporal import TemporalGRU, TemporalTCN

STATIC_TORCH_FAMILIES = frozenset({ResidualMLP.model_family, SetAttention.model_family})
"""Families whose score is a pure function of (bundle, frozen frame): one row per key, no window."""

TEMPORAL_TORCH_FAMILIES = frozenset({TemporalGRU.model_family, TemporalTCN.model_family})
"""Families that score a history window per row.

M2's card is named by the WINNING configuration rather than by the carrier, because the grid
searches both architectures, so this is a set of two and every consumer has to accept both.
"""

TORCH_FAMILIES = STATIC_TORCH_FAMILIES | TEMPORAL_TORCH_FAMILIES
"""Every family a torch carrier can write into a model card."""

TEMPORAL_CODE_VERSIONS = {
    family: f"temporal-{family.removeprefix('TEMPORAL_').lower()}-1"
    for family in sorted(TEMPORAL_TORCH_FAMILIES)
}
"""The ``model_code_version`` ``run_full_m2`` writes, derived rather than restated.

The driver builds it as ``f"temporal-{str(best_config['family']).lower()}-1"`` from the winning
config's ``GRU``/``TCN`` choice, which is the same string as stripping the ``TEMPORAL_`` prefix from
the class's own ``model_family``. Computing it here keeps the loader's expectation tied to the name
the models declare instead of to a second transcription of the driver's format string.
"""

CARRIER_CARD_FAMILIES: Mapping[str, frozenset[str]] = {
    "M1": frozenset({ResidualMLP.model_family}),
    "M2": TEMPORAL_TORCH_FAMILIES,
    "M3": frozenset({SetAttention.model_family}),
}
"""Which architecture names each torch carrier's bundles may record.

M2 maps to both temporal families because its card is named by the winning configuration; M1 and M3
each search one architecture, so theirs are singletons. The walk-forward evaluator builds its card
allowlist from this, which is what keeps it from accepting a spelling no writer produces -- the
failure that blocked every M2 run was an allowlist entry reading ``TEMPORAL``.
"""

__all__ = [
    "CARRIER_CARD_FAMILIES",
    "STATIC_TORCH_FAMILIES",
    "TEMPORAL_CODE_VERSIONS",
    "TEMPORAL_TORCH_FAMILIES",
    "TORCH_FAMILIES",
]
