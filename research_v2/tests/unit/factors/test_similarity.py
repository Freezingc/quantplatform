from __future__ import annotations

import math
from datetime import date

import pandas as pd
import pytest

from research_v2.factors.similarity import compute_factor_similarity


def test_similarity_is_scope_bounded_and_uses_absolute_daily_spearman() -> None:
    rows: list[dict[str, object]] = []
    for signal_date in (date(2025, 1, 2), date(2025, 1, 3)):
        for ts_code, left, right in zip(
            ["000001.SZ", "000002.SZ", "000003.SZ"],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            strict=True,
        ):
            rows.extend(
                [
                    {
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "factor_id": "a",
                        "factor_version": 1,
                        "value": left,
                    },
                    {
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "factor_id": "b",
                        "factor_version": 1,
                        "value": right,
                    },
                ]
            )
    rows.extend(
        [
            {
                "signal_date": date(2025, 1, 6),
                "ts_code": "000001.SZ",
                "factor_id": "a",
                "factor_version": 1,
                "value": 1.0,
            },
            {
                "signal_date": date(2025, 1, 6),
                "ts_code": "000001.SZ",
                "factor_id": "b",
                "factor_version": 1,
                "value": 1.0,
            },
        ]
    )

    result = compute_factor_similarity(
        pd.DataFrame(rows),
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        minimum_common_members=3,
        minimum_valid_days=2,
    )

    assert len(result) == 1
    assert result.loc[0, "factor_id_left"] == "a"
    assert result.loc[0, "factor_id_right"] == "b"
    assert result.loc[0, "median_daily_abs_spearman"] == 1.0
    assert result.loc[0, "valid_days"] == 2
    assert result.loc[0, "status"] == "PASS"


# --- C7.2 / C7.5 branches the single test above never reached --------------------

_DAYS = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))


def _observations(
    present: dict[str, dict[date, int]], codes: tuple[str, ...]
) -> pd.DataFrame:
    """Long-form observations where ``present[factor][day]`` is how many codes have a value.

    Values increase with the code index, so any two factors present on the same codes are
    perfectly rank-correlated; the interesting variable is *which* codes are present.
    """
    rows: list[dict[str, object]] = []
    for factor_id, by_day in present.items():
        for day, count in by_day.items():
            for index, code in enumerate(codes[:count]):
                rows.append(
                    {
                        "signal_date": day,
                        "ts_code": code,
                        "factor_id": factor_id,
                        "factor_version": 1,
                        "value": float(index) + 1.0,
                    }
                )
    return pd.DataFrame(rows)


_CODES = tuple(f"{index:06d}.SZ" for index in range(40))


def test_a_pair_below_the_valid_day_floor_is_insufficient_and_carries_no_correlation() -> None:
    """C7.2: fewer than the required valid days means insufficient, not a weak edge.

    The distinction matters downstream: ``select_cluster_representatives`` only builds edges
    from PASS rows, so an INSUFFICIENT pair must not cluster. Reporting a median over one or
    two days as if it were a measured correlation is how a spurious duplicate gets merged.
    """
    frame = _observations({"a": {_DAYS[0]: 40}, "b": {_DAYS[0]: 40}}, _CODES)

    result = compute_factor_similarity(
        frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        minimum_common_members=30,
        minimum_valid_days=2,
    )

    assert result.loc[0, "valid_days"] == 1
    assert result.loc[0, "status"] == "INSUFFICIENT"


def test_misaligned_missingness_skips_the_day_rather_than_counting_it() -> None:
    """C7.5's misaligned-missingness case: a thin overlap day is skipped, not correlated.

    Two factors can each be well populated and still overlap on only a handful of names on a
    given day. Correlating over that handful would produce a number that looks measured and
    is not, so days under ``minimum_common_members`` are dropped from the sample entirely --
    which can push the pair below the valid-day floor and make it INSUFFICIENT.
    """
    frame = _observations(
        {
            "a": {day: 40 for day in _DAYS},
            # only 10 shared names on the second and third days
            "b": {_DAYS[0]: 40, _DAYS[1]: 10, _DAYS[2]: 10},
        },
        _CODES,
    )

    result = compute_factor_similarity(
        frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        minimum_common_members=30,
        minimum_valid_days=2,
    )

    assert result.loc[0, "valid_days"] == 1
    assert result.loc[0, "status"] == "INSUFFICIENT"


def test_a_day_where_one_factor_is_wholly_absent_is_not_a_valid_day() -> None:
    frame = _observations(
        {"a": {day: 40 for day in _DAYS}, "b": {_DAYS[0]: 40, _DAYS[1]: 40}},
        _CODES,
    )

    result = compute_factor_similarity(
        frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        minimum_common_members=30,
        minimum_valid_days=2,
    )

    assert result.loc[0, "valid_days"] == 2
    assert result.loc[0, "status"] == "PASS"


def _corrupt(frame: pd.DataFrame, factor_id: str, count: int, filler: float) -> pd.DataFrame:
    """Overwrite the values of the last ``count`` codes of ``factor_id`` with ``filler``."""
    out = frame.copy(deep=True)
    mask = out["factor_id"] == factor_id
    values = out.loc[mask, "value"].tolist()
    for index in range(len(values) - count, len(values)):
        values[index] = filler
    out.loc[mask, "value"] = values
    return out


@pytest.mark.parametrize("filler", [float("inf"), float("nan")], ids=["inf", "nan"])
def test_non_finite_values_cannot_prop_up_a_day_below_the_member_floor(
    filler: float,
) -> None:
    """A non-finite value is not an observation, so it must not count toward the floor.

    Eleven of the forty shared names are unusable, leaving 29 finite pairs -- one short of the
    floor of 30. Counting the filler anyway would report PASS over a cross-section a quarter
    thinner than the contract requires; for ``nan`` it would additionally report a correlation
    that was in fact computed on 29 names while claiming 40. Both must fall below the floor and
    leave the pair with no measured correlation at all.
    """
    frame = _corrupt(
        _observations({"a": {_DAYS[0]: 40}, "b": {_DAYS[0]: 40}}, _CODES), "b", 11, filler
    )

    result = compute_factor_similarity(
        frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        minimum_common_members=30,
        minimum_valid_days=1,
    )

    assert result.loc[0, "valid_days"] == 0
    assert result.loc[0, "status"] == "INSUFFICIENT"
    assert math.isnan(float(result.loc[0, "median_daily_abs_spearman"]))


def test_a_day_exactly_at_the_common_member_floor_still_counts() -> None:
    """Pins the floor as ``>=`` and not ``>``: ten unusable names leave exactly thirty."""
    frame = _corrupt(
        _observations({"a": {_DAYS[0]: 40}, "b": {_DAYS[0]: 40}}, _CODES),
        "b",
        10,
        float("inf"),
    )

    result = compute_factor_similarity(
        frame,
        start_date=_DAYS[0],
        end_date=_DAYS[-1],
        minimum_common_members=30,
        minimum_valid_days=1,
    )

    assert result.loc[0, "valid_days"] == 1
    assert result.loc[0, "status"] == "PASS"
