"""Per-factor routing in the script that writes the frozen trust root, pinned.

Why this file exists: the id ``yeslab_206`` denotes two different factors in the
platform's two lineages. In the YesLab source lineage it is
``Bias_Vol_Demean_60_Calculator`` over ``factor_data5`` and the median repair carries its
own id ``yeslab_206_median_abs_v2``; in the frozen lineage that ``factor_daily`` stores,
``yeslab_206`` *is* the median repair. Eight scripts route around that collision through
one shared helper (``materialize_alpha206_median_repair``) and none of them was enforced by
an assertion, so a consumer that reached ``REPLAY_DEFINITIONS["yeslab_206"]`` and compared
the result against ``factor_daily`` would silently compute the wrong factor.

``materialize_frozen_factor_daily`` is the one consumer whose output *becomes* the trust
root, and before this file it had no test at all. These tests pin its routing and make the
id collision explicit rather than incidental: the divergence between
``REPLAY_DEFINITIONS["yeslab_206"]`` and what the frozen table stores is deliberate, so it
is asserted here instead of being left for the next reader to rediscover.

Routing is asserted by replacing the three callees with sentinels, so no database, market
cache or network is touched.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import audit_formula_replay_consistency as replay
import materialize_frozen_factor_daily as mfrozen
import pandas as pd
import pytest

# tests/unit/scripts/<this file> -> parents[3] is the research_v2 root
RESEARCH_V2 = Path(__file__).resolve().parents[3]
FROZEN_SET = RESEARCH_V2 / "artifacts" / "diagnostics" / "factor_set_frozen_v1.json"


def _frozen_factors() -> list[str]:
    if not FROZEN_SET.exists():
        pytest.skip(f"frozen set artifact not present: {FROZEN_SET}")
    payload = json.loads(FROZEN_SET.read_text(encoding="utf-8"))
    return [str(factor) for factor in payload["factors"]]


def _tiny_market() -> pd.DataFrame:
    """The smallest frame the 443 branch's pivots accept: two dates by two codes."""
    dates = [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")]
    codes = ["000001.SZ", "600000.SH"]
    rows = [
        {"trade_date": d, "ts_code": c, "amount": 1.0 + i, "pct_chg": 0.5 * i}
        for i, d in enumerate(dates)
        for c in codes
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[object]]:
    """Replace the three routing targets with sentinels and record which one fired."""
    seen: dict[str, list[object]] = {"median_206": [], "gp074_443": [], "replay": []}

    def median_sentinel(market, start, end):  # noqa: ANN001, ANN202
        seen["median_206"].append((start, end))
        return pd.DataFrame(columns=["trade_date", "ts_code", "replay_value"])

    def gp074_sentinel(wide, register_window, max_window):  # noqa: ANN001, ANN202
        seen["gp074_443"].append((register_window, max_window))
        return pd.DataFrame(
            {"A": [1.0, 2.0]}, index=pd.DatetimeIndex(
                ["2020-01-02", "2020-01-03"], name="trade_date"
            )
        )

    def replay_sentinel(session, definition, start, end, industry_provider=None):  # noqa: ANN001, ANN202
        seen["replay"].append(definition)
        return pd.DataFrame(columns=["trade_date", "ts_code", "replay_value"])

    monkeypatch.setattr(mfrozen, "materialize_alpha206_median_repair", median_sentinel)
    monkeypatch.setattr(mfrozen, "compute_gp074_variant", gp074_sentinel)
    monkeypatch.setattr(replay, "_replay_factor", replay_sentinel)
    return seen


def test_frozen_206_uses_the_median_repair_and_never_the_source_definition(
    calls: dict[str, list[object]],
) -> None:
    """The trust root stores the median repair under the id the source lineage means
    as the de-mean factor, so this route must not fall through to REPLAY_DEFINITIONS."""
    mfrozen.replay_factor_values(
        mfrozen.MEDIAN_206, session=None, market=_tiny_market(), provider=None,
        start=date(2020, 1, 2), end=date(2020, 1, 3),
    )
    assert len(calls["median_206"]) == 1
    assert calls["gp074_443"] == []
    assert calls["replay"] == [], (
        "yeslab_206 fell through to REPLAY_DEFINITIONS, which names "
        "Bias_Vol_Demean_60_Calculator -- the source-lineage factor, not the median "
        "repair the frozen table stores under the same id"
    )


def test_frozen_443_variant_uses_the_repair_formula_and_has_no_replay_definition(
    calls: dict[str, list[object]],
) -> None:
    """The 443 repair variant is admitted by decision D1=A and is not in
    REPLAY_DEFINITIONS at all, so a routing mistake here fails closed with a KeyError
    rather than silently replaying canonical 443 (graded D, 2017 collapse)."""
    assert mfrozen.REPAIR_VARIANT_443 not in replay.REPLAY_DEFINITIONS

    mfrozen.replay_factor_values(
        mfrozen.REPAIR_VARIANT_443, session=None, market=_tiny_market(), provider=None,
        start=date(2020, 1, 2), end=date(2020, 1, 3),
    )
    assert calls["gp074_443"] == [(30, 20)], (
        "the admitted variant is reg30_max20; different windows would be a different factor"
    )
    assert calls["median_206"] == []
    assert calls["replay"] == []


def test_every_other_frozen_factor_replays_through_its_own_definition(
    calls: dict[str, list[object]],
) -> None:
    """The two repair routes are exceptions; nothing else may take them, and every
    other member must reach the replay with the definition registered for its own id."""
    others = [
        fid for fid in _frozen_factors()
        if fid not in (mfrozen.MEDIAN_206, mfrozen.REPAIR_VARIANT_443)
    ]
    assert others, "the frozen set lost every replayable member"

    for fid in others:
        calls["replay"].clear()
        mfrozen.replay_factor_values(
            fid, session=None, market=_tiny_market(), provider=None,
            start=date(2020, 1, 2), end=date(2020, 1, 3),
        )
        assert calls["median_206"] == [] and calls["gp074_443"] == [], (
            f"{fid} took a repair route that belongs to 206 or the 443 variant"
        )
        assert len(calls["replay"]) == 1
        assert calls["replay"][0] is replay.REPLAY_DEFINITIONS[fid]


def test_pit_factors_are_the_only_ones_given_an_industry_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An industry provider changes the value of an industry-dependent factor, so which
    members receive one is part of the routing contract and not an incidental argument."""
    received: dict[str, object] = {}

    def replay_sentinel(session, definition, start, end, industry_provider=None):  # noqa: ANN001, ANN202
        received[definition.factor_id] = industry_provider
        return pd.DataFrame(columns=["trade_date", "ts_code", "replay_value"])

    monkeypatch.setattr(replay, "_replay_factor", replay_sentinel)
    sentinel_provider = object()
    for fid in _frozen_factors():
        if fid in (mfrozen.MEDIAN_206, mfrozen.REPAIR_VARIANT_443):
            continue
        mfrozen.replay_factor_values(
            fid, session=None, market=_tiny_market(), provider=sentinel_provider,
            start=date(2020, 1, 2), end=date(2020, 1, 3),
        )
    for fid, provider in received.items():
        if fid in mfrozen.PIT_FACTORS:
            assert provider is sentinel_provider, f"{fid} is PIT-dependent but got no provider"
        else:
            assert provider is None, f"{fid} is not in PIT_FACTORS but was given a provider"


def test_the_206_id_collision_is_deliberate_and_documented_on_both_sides() -> None:
    """Pin the divergence itself, so that removing either half is a visible change.

    The source-lineage definition and the admission-time alias are the two halves of one
    arrangement: ``REPLAY_DEFINITIONS["yeslab_206"]`` describes the YesLab factor, and
    ``grade_alias`` says the frozen id's grade comes from the repair variant's id. If
    either half changes, the frozen table and the grading evidence stop describing the
    same factor, and nothing else in the repository would notice.
    """
    definition = replay.REPLAY_DEFINITIONS[mfrozen.MEDIAN_206]
    assert definition.calculator_class == "Bias_Vol_Demean_60_Calculator"
    assert definition.store_factor == "factor_data5"
    assert "median" not in definition.calculator_class.lower(), (
        "REPLAY_DEFINITIONS now describes the median repair for yeslab_206; if the source "
        "lineage changed too, the frozen materializer's special case is redundant and the "
        "two lineages should be reconciled rather than left divergent"
    )

    admission = (RESEARCH_V2 / "scripts" / "apply_admission_decisions.py").read_text(
        encoding="utf-8"
    )
    assert '"yeslab_206": "yeslab_206_median_abs_v2"' in admission, (
        "the grade alias that makes the frozen id's A grade refer to the median repair "
        "variant is gone; without it the frozen yeslab_206 has no grade row at all"
    )
