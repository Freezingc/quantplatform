"""Unit tests for the operational factor materializer's pure assembly helpers."""

from __future__ import annotations

from datetime import date, datetime

import materialize_operational_factors as operational
import pandas as pd
import pytest


def _replay_frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": pd.to_datetime([row[0] for row in rows]),
            "ts_code": [row[1] for row in rows],
            "replay_value": [row[2] for row in rows],
        }
    )


def test_assemble_long_frame_keeps_only_universe_members_with_finite_values() -> None:
    replay = _replay_frame(
        [
            ("2026-01-05", "000001.SZ", 1.5),
            ("2026-01-05", "000002.SZ", float("nan")),
            ("2026-01-05", "000003.SZ", 2.5),
        ]
    )
    universe = {date(2026, 1, 5): {"000001.SZ", "000002.SZ"}}
    assembled = operational.assemble_long_frame(
        replay, universe, factor_id="yeslab_96", factor_version=1
    )
    assert assembled["ts_code"].tolist() == ["000001.SZ"]
    assert assembled["value"].tolist() == [1.5]
    assert assembled.columns.tolist() == operational.LONG_COLUMNS
    assert assembled["available_date"].tolist() == [date(2026, 1, 5)]
    assert assembled["signal_date"].tolist() == [date(2026, 1, 5)]


def test_assemble_long_frame_rejects_duplicate_business_keys() -> None:
    replay = _replay_frame(
        [("2026-01-05", "000001.SZ", 1.5), ("2026-01-05", "000001.SZ", 1.6)]
    )
    universe = {date(2026, 1, 5): {"000001.SZ"}}
    with pytest.raises(ValueError, match="duplicate"):
        operational.assemble_long_frame(
            replay, universe, factor_id="yeslab_96", factor_version=1
        )


def test_assemble_long_frame_aligns_a_timestamp_keyed_universe() -> None:
    """``load_official_universe`` keys its map by ``pd.Timestamp``, not ``datetime.date``.

    A date-typed lookup against Timestamp keys silently matches nothing, so every factor comes
    back with zero universe-aligned rows and the run reports success while writing an empty
    window. That is exactly what an unguarded ``.dt.date`` conversion produced.
    """
    replay = _replay_frame([("2026-01-05", "000001.SZ", 1.5)])
    universe = {pd.Timestamp("2026-01-05"): {"000001.SZ"}}
    assembled = operational.assemble_long_frame(
        replay, universe, factor_id="yeslab_96", factor_version=1
    )
    assert assembled["ts_code"].tolist() == ["000001.SZ"]


def test_window_open_dates_keeps_dates_inside_2026() -> None:
    """The frozen helper silently drops everything from 2026-01-01; this window must not."""
    raw = [date(2025, 12, 31), date(2026, 1, 5), date(2026, 1, 6), date(2026, 2, 2)]
    assert operational.window_open_dates(raw, date(2026, 1, 1), date(2026, 1, 31)) == (
        date(2026, 1, 5),
        date(2026, 1, 6),
    )


def test_window_open_dates_refuses_an_empty_window() -> None:
    with pytest.raises(ValueError, match="no open trading dates"):
        operational.window_open_dates([date(2025, 12, 31)], date(2026, 1, 1), date(2026, 1, 31))


def test_as_plain_date_truncates_every_datetime_shape() -> None:
    expected = date(2026, 1, 5)
    assert operational.as_plain_date(expected) == expected
    assert operational.as_plain_date(pd.Timestamp("2026-01-05")) == expected
    assert operational.as_plain_date(datetime(2026, 1, 5, 15, 30)) == expected


def test_require_frozen_factors_refuses_an_id_outside_the_v2_set() -> None:
    # yeslab_405 and yeslab_398 were removed by the v2 amendment; materializing them would
    # produce a factor set no frozen model consumes.
    with pytest.raises(ValueError, match="yeslab_405"):
        operational.require_frozen_factors(("yeslab_96", "yeslab_405"))


def _market_frame(volumes: dict[str, list[float]], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = [
        {"trade_date": day, "ts_code": code, "vol": volume}
        for code, series in volumes.items()
        for day, volume in zip(dates, series, strict=True)
    ]
    return pd.DataFrame(rows)


def test_replay_market_only_factor_computes_alpha206_inside_2026() -> None:
    """The frozen 206 repair refuses any window reaching 2026; the operational path must not.

    The formula itself is the shared one, so the values must stay the 60-day rolling median
    deviation of volume -- 61 days of volume 1..61 give a last-day deviation of 29.5/31.5.
    """
    dates = pd.bdate_range("2026-01-05", periods=61)
    market = _market_frame(
        {"000001.SZ": [float(i) for i in range(1, 62)], "000002.SZ": [10.0] * 61}, dates
    )
    values = operational.replay_market_only_factor(
        "yeslab_206", market, date(2026, 1, 5), date(2026, 3, 31)
    )
    assert set(values.columns) == {"trade_date", "ts_code", "replay_value"}
    finite = values.dropna(subset=["replay_value"])
    assert finite["trade_date"].max().date() == dates[-1].date()
    last = finite.loc[finite["trade_date"].eq(dates[-1])].set_index("ts_code")["replay_value"]
    assert last.loc["000001.SZ"] == pytest.approx(29.5 / 31.5)
    assert last.loc["000002.SZ"] == pytest.approx(0.0)


def test_replay_market_only_factor_rejects_an_unknown_factor_id() -> None:
    dates = pd.bdate_range("2026-01-05", periods=61)
    market = _market_frame({"000001.SZ": [1.0] * 61}, dates)
    with pytest.raises(ValueError, match="not a market-only"):
        operational.replay_market_only_factor(
            "yeslab_96", market, date(2026, 1, 5), date(2026, 3, 31)
        )


def _industry_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    members = pd.DataFrame(
        {
            "index_code": ["801010.SI"],
            "con_code": ["000001.SZ"],
            "in_date": ["20211213"],
            "out_date": [None],
        }
    )
    classifications = pd.DataFrame(
        {"index_code": ["801010.SI"], "industry_name": ["农林牧渔"], "level": ["L1"]}
    )
    return members, classifications


def test_base_industry_provider_refuses_2026_dates() -> None:
    """The shared provider's guard is the reason a separate operational provider exists."""
    from audit_tushare_pit_industry import PITIndustryProvider

    members, classifications = _industry_frames()
    provider = PITIndustryProvider(members, classifications)
    with pytest.raises(ValueError, match="2026 locked-test"):
        provider.get_industry_labels([date(2026, 3, 2)], ["000001.SZ"])


def test_operational_industry_provider_serves_2026_dates() -> None:
    members, classifications = _industry_frames()
    provider = operational.OperationalPITIndustryProvider(members, classifications)
    labels = provider.get_industry_labels([date(2026, 3, 2)], ["000001.SZ"])
    assert labels.loc[pd.Timestamp("2026-03-02"), "000001.SZ"] == "农林牧渔"


def test_operational_industry_provider_matches_the_base_before_the_boundary() -> None:
    """The override duplicates the shared snapshot body, so pin the two together."""
    from audit_tushare_pit_industry import PITIndustryProvider

    members, classifications = _industry_frames()
    base = PITIndustryProvider(members, classifications).snapshot(date(2025, 6, 2), ["000001.SZ"])
    override = operational.OperationalPITIndustryProvider(members, classifications).snapshot(
        date(2025, 6, 2), ["000001.SZ"]
    )
    pd.testing.assert_frame_equal(base, override)


def test_compare_against_reference_reports_agreement_and_disagreement() -> None:
    replay = pd.DataFrame(
        {
            "signal_date": [date(2025, 12, 30), date(2025, 12, 30), date(2025, 12, 31)],
            "ts_code": ["a", "b", "c"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    reference = pd.DataFrame(
        {
            "signal_date": [date(2025, 12, 30), date(2025, 12, 30), date(2025, 12, 31)],
            "ts_code": ["a", "b", "c"],
            "value": [1.0, 2.5, 3.0],
        }
    )
    summary = operational.compare_against_reference(replay, reference, tolerance=1e-9)
    assert summary["matched_rows"] == 3
    assert summary["replay_only_rows"] == 0
    assert summary["reference_only_rows"] == 0
    assert summary["within_tolerance_rows"] == 2
    assert summary["max_abs_diff"] == pytest.approx(0.5)


def test_compare_against_reference_uses_a_relative_tolerance() -> None:
    """An absolute-only tolerance is meaningless for a factor whose values are ~1e6.

    A 0.058 difference on a 2.6e6 value is float noise, but an absolute 1e-9 bound reported it as
    a disagreement -- which made a faithful factor look like a total mismatch.
    """
    replay = pd.DataFrame(
        {"signal_date": [date(2025, 12, 1)], "ts_code": ["a"], "value": [2.6196088e06]}
    )
    reference = pd.DataFrame(
        {"signal_date": [date(2025, 12, 1)], "ts_code": ["a"], "value": [2.6196090e06]}
    )
    summary = operational.compare_against_reference(
        replay, reference, tolerance=1e-9, relative_tolerance=1e-6
    )
    assert summary["max_abs_diff"] == pytest.approx(0.2, abs=0.05)
    assert summary["within_tolerance_rows"] == 1
