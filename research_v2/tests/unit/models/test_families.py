"""Every module that routes by model family must agree with the model classes themselves.

Four modules each enumerated the torch family names as string literals: the walk-forward evaluator's
card allowlist, the reproduction audit's scorer routing, the inference builder's batch-shape
dispatch, and the artifact loader's code-version map. Three of the four had drifted, and the drift
stayed invisible because a test pinned one omission as intended behaviour --
``test_unknown_family_is_rejected`` used ``TEMPORAL_TCN`` as its example of an unknown family, while
the M2 grid searches ``family: [GRU, TCN]`` and ``TemporalTCN.model_family`` is ``TEMPORAL_TCN``.

The consequence was concrete, not stylistic. The evaluator accepted neither ``TEMPORAL_GRU`` nor
``TEMPORAL_TCN`` -- only ``TEMPORAL``, a spelling no writer produces -- so EVERY M2 run would train
for days, freeze and sign four bundles, and then die at the evaluation step on a family mismatch;
that is the same shape as the three blockers that cost both real M1 runs their verdicts. And a run
whose search picked TCN could not be batched or loaded for serving either, because the builder and
the loader both omitted it.

Deriving the sets from the classes makes a rename propagate instead of drift, and this file pins
every consumer to them, so the next omission is a test failure in seconds rather than a multi-day
run that cannot finish.
"""

from __future__ import annotations

from research_v2.evaluation import model_walk_forward
from research_v2.inference import artifact_loader, builder, reproduction
from research_v2.models import families
from research_v2.models.residual_mlp import ResidualMLP
from research_v2.models.set_attention import SetAttention
from research_v2.models.temporal import TemporalGRU, TemporalTCN

_CONFIGS = {
    ResidualMLP.model_family: {
        "raw_factor_count": 15,
        "hidden_dim": 64,
        "num_blocks": 1,
        "dropout": 0.0,
    },
    SetAttention.model_family: {
        "raw_factor_count": 15,
        "hidden_dim": 64,
        "num_heads": 4,
        "inducing_points": 8,
        "num_blocks": 1,
        "dropout": 0.0,
    },
    TemporalGRU.model_family: {
        "raw_factor_count": 15,
        "hidden_dim": 64,
        "dropout": 0.0,
    },
    TemporalTCN.model_family: {
        "raw_factor_count": 15,
        "hidden_dim": 64,
        "dropout": 0.0,
    },
}


def test_the_family_sets_come_from_the_model_classes() -> None:
    assert frozenset(
        {ResidualMLP.model_family, SetAttention.model_family}
    ) == families.STATIC_TORCH_FAMILIES
    assert frozenset(
        {TemporalGRU.model_family, TemporalTCN.model_family}
    ) == families.TEMPORAL_TORCH_FAMILIES
    assert families.TORCH_FAMILIES == (
        families.STATIC_TORCH_FAMILIES | families.TEMPORAL_TORCH_FAMILIES
    )
    assert not families.STATIC_TORCH_FAMILIES & families.TEMPORAL_TORCH_FAMILIES


def test_every_family_router_knows_every_torch_family() -> None:
    for family in families.STATIC_TORCH_FAMILIES:
        assert family in builder._CROSS_SECTIONAL_FAMILIES
        assert family in reproduction._STATIC_TORCH_FAMILIES
        carrier = "M1" if family == ResidualMLP.model_family else "M3"
        assert family in model_walk_forward._MODEL_CARD_FAMILIES[carrier]
    for family in families.TEMPORAL_TORCH_FAMILIES:
        assert family in builder._TEMPORAL_FAMILIES
        assert family in reproduction._TEMPORAL_TORCH_FAMILIES
        assert family in model_walk_forward._MODEL_CARD_FAMILIES["M2"]
    for family in families.TORCH_FAMILIES:
        assert family in artifact_loader._MODEL_CODE_VERSIONS


def test_every_allowlisted_family_has_a_factory_behind_it() -> None:
    """An allowlist entry with no constructor behind it is a refusal dressed as support.

    Iterated over the families the model classes declare rather than over the allowlist itself.
    Mutating the map to drop ``TEMPORAL_TCN`` showed why that matters: looping over the map simply
    stopped covering the missing family and this test still passed, so on its own it could never
    notice an omission -- only a wrong entry.
    """
    for family in sorted(families.TORCH_FAMILIES):
        code_version = artifact_loader._MODEL_CODE_VERSIONS[family]
        factory = artifact_loader._allowlisted_factory(
            {"family": family, "model_code_version": code_version}, _CONFIGS[family]
        )
        assert factory().model_family == family


def test_the_temporal_code_versions_match_what_the_driver_writes() -> None:
    """``run_full_m2`` writes ``f"temporal-{winning family lowercased}-1"`` into the card."""
    for family in families.TEMPORAL_TORCH_FAMILIES:
        written = f"temporal-{family.removeprefix('TEMPORAL_').lower()}-1"
        assert artifact_loader._MODEL_CODE_VERSIONS[family] == written
