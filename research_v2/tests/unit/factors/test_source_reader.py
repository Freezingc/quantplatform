from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.contracts.factor import (
    AvailabilityPolicy,
    FactorDefinition,
    FactorType,
)
from research_v2.factors.source_reader import (
    SourceFactorSchemaError,
    normalize_technical_factor_source,
)


def _definition() -> FactorDefinition:
    return FactorDefinition(
        factor_id="yeslab_1",
        factor_version=1,
        display_name="alpha",
        source_database="dfs://YesLab",
        source_table="factor_data1",
        source_value_column="value",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="DAILY",
        expected_direction=0,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="UNKNOWN",
        owner="research",
    )


def test_source_reader_normalizes_only_the_registered_factor_name() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "beta"],
            "value": [1.5, 9.0],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
    )

    assert len(normalized) == 1
    assert normalized.loc[0, "factor_id"] == "yeslab_1"
    assert normalized.loc[0, "observation_date"] == date(2025, 1, 2)
    assert normalized.loc[0, "source_available_date"] == date(2025, 1, 2)
    assert normalized.loc[0, "raw_value"] == 1.5
    assert normalized.loc[0, "value_is_valid"]


def test_source_reader_marks_nonfinite_values_and_rejects_missing_contract_columns() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [np.inf],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
    )

    assert not normalized.loc[0, "value_is_valid"]
    assert normalized.loc[0, "invalid_reason"] == "NONFINITE_VALUE"
    with pytest.raises(SourceFactorSchemaError, match="factor_name"):
        normalize_technical_factor_source(
            source.drop(columns="factor_name"),
            _definition(),
            source_factor_name="alpha",
            source_watermark_id="watermark",
        )


def test_source_reader_can_use_bounded_preflight_row_ids_without_hashing() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "alpha"],
            "value": [1.5, 1.5],
        }
    )

    normalized = normalize_technical_factor_source(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
        include_source_row_id=False,
    )

    assert normalized["source_row_id"].tolist() == [
        "preflight-000000000000",
        "preflight-000000000001",
    ]


# --- C2.1 / C2.2 / C2.3 clauses that had no test --------------------------------

CANONICAL_COLUMNS = [
    "ts_code",
    "factor_id",
    "factor_version",
    "observation_date",
    "source_available_date",
    "raw_value",
    "value_is_valid",
    "invalid_reason",
    "source_row_id",
    "source_table",
    "source_watermark_id",
]


def _wide_source() -> pd.DataFrame:
    """Two months, two securities, and one of each awkward value.

    Carries an exact duplicate (2025-02-03 / 000001.SZ / 3.0 twice), an infinity, a NaN,
    and a value whose decimal form is lossy (0.1 + 0.2), so the tests below can tell the
    difference between preserving a value and re-deriving it.
    """
    return pd.DataFrame(
        {
            "ts_code": [
                "000001.SZ", "000002.SZ",
                "000001.SZ", "000002.SZ",
                "000001.SZ", "000001.SZ",
                "000002.SZ",
            ],
            "trade_date": [
                date(2025, 1, 2), date(2025, 1, 2),
                date(2025, 1, 3), date(2025, 1, 3),
                date(2025, 2, 3), date(2025, 2, 3),
                date(2025, 2, 4),
            ],
            "factor_name": ["alpha"] * 7,
            "value": [1.5, float("inf"), 0.1 + 0.2, float("nan"), 3.0, 3.0, -2.5],
        }
    )


def _normalize(source: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
    kwargs.setdefault("source_factor_name", "alpha")
    kwargs.setdefault("source_watermark_id", "watermark")
    return normalize_technical_factor_source(source, _definition(), **kwargs)  # type: ignore[arg-type]


def test_the_reader_never_alters_drops_or_invents_a_value() -> None:
    """C2.3: no winsorising, no scaling, no imputation, and raw precision preserved.

    This is the same property the deduplication tests pin one layer down, and for the same
    reason: the source reader is where a value enters the platform, so anything it changes
    is indistinguishable from what the source said. A clipped outlier or an imputed mean
    would propagate into every factor value, every baseline and every trust root, and no
    downstream check could tell.
    """
    source = _wide_source()

    normalized = _normalize(source)

    assert len(normalized) == len(source), "rows were dropped or invented"
    delivered: dict[tuple[object, object], list[object]] = {}
    for row in source.to_dict(orient="records"):
        delivered.setdefault((row["trade_date"], row["ts_code"]), []).append(row["value"])
    for row in normalized.to_dict(orient="records"):
        key = (row["observation_date"], row["ts_code"])
        value = row["raw_value"]
        candidates = delivered[key]
        assert any(_same_value(value, candidate) for candidate in candidates), (
            f"published {value!r} at {key}, which the source never delivered"
        )


def _same_value(published: object, delivered: object) -> bool:
    """Bit-identical, treating two NaNs as the same absence."""
    if pd.isna(published) or pd.isna(delivered):
        return bool(pd.isna(published) and pd.isna(delivered))
    return float(published).hex() == float(delivered).hex()  # type: ignore[arg-type]


def test_extreme_but_finite_values_are_passed_through_not_clipped() -> None:
    """A huge magnitude is a fact about the source, not an error to correct."""
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [date(2025, 1, 2)] * 2,
            "factor_name": ["alpha"] * 2,
            "value": [1e308, -1e308],
        }
    )

    normalized = _normalize(source)

    # Compared as a set: the reader sorts by (code, date, value), and asserting an order
    # here would test the sort rather than the preservation.
    assert set(normalized["raw_value"]) == {1e308, -1e308}
    assert normalized["value_is_valid"].all(), "finite extremes were marked invalid"
    assert normalized["invalid_reason"].isna().all()


def test_source_column_order_does_not_change_the_output() -> None:
    """C2.1: a wide extract whose columns arrive in a different order is the same extract."""
    source = _wide_source()

    reordered = source.loc[:, ["value", "factor_name", "trade_date", "ts_code"]]

    pd.testing.assert_frame_equal(_normalize(source), _normalize(reordered))


def test_extra_unregistered_columns_are_not_loaded() -> None:
    """C2.1: only the registered mapping is read, so a new source column cannot leak in.

    Reporting the unmapped column belongs to the catalog dry-run (C1.5), which does not
    exist yet; the reader's half of the clause is that the column has no effect.
    """
    source = _wide_source()

    polluted = source.assign(unregistered_extra=1.0, another_column="text")
    normalized = _normalize(source)

    pd.testing.assert_frame_equal(normalized, _normalize(polluted))
    assert list(normalized.columns) == CANONICAL_COLUMNS


def test_month_chunks_normalize_identically_to_a_single_read() -> None:
    """C2.2: chunk boundaries must not change the result, source_row_id included.

    The row id hashes (table, code, date, factor name, value, occurrence-within-group),
    never a position in the frame, so splitting on the documented axes -- month and factor
    group -- cannot change it. That is what makes a chunked fetch safe to publish: the
    identity of a row does not depend on how the extract happened to be batched. An
    arbitrary row split is a different matter and is not what the plan specifies.
    """
    source = _wide_source()
    whole = _normalize(source)

    months = sorted({(day.year, day.month) for day in source["trade_date"]})
    chunks = [
        source.loc[
            [(day.year, day.month) == month for day in source["trade_date"]]
        ]
        for month in months
    ]
    assert len(chunks) == 2 and all(len(chunk) for chunk in chunks)
    chunked = pd.concat([_normalize(chunk) for chunk in chunks], ignore_index=True)

    ordering = ["observation_date", "ts_code", "source_row_id"]
    pd.testing.assert_frame_equal(
        whole.sort_values(ordering, kind="mergesort", ignore_index=True),
        chunked.sort_values(ordering, kind="mergesort", ignore_index=True),
    )
    assert len(set(whole["source_row_id"])) == len(set(chunked["source_row_id"]))


def test_every_row_of_a_duplicate_group_carries_the_same_available_date() -> None:
    """The invariant that makes C3.1(a) moot, pinned so it cannot lapse silently.

    C3.1(a) records that IDENTICAL_COLLAPSE compares only ``raw_value``: when two rows share
    a business key and a value but differ in ``source_available_date``, the survivor's
    available date -- the point-in-time field that decides when the value may be used
    downstream -- is picked by the smallest ``source_row_id``, a provenance field with no
    semantic meaning. A census of the source found the collapse acting on between 3.5 and
    7.3 million keys in each of seven frozen factors, so the branch is not hypothetical.

    On the technical read path the hazard still cannot fire, and not by luck: the reader
    assigns ``source_available_date`` from ``observation_date``, and ``observation_date`` is
    one of the four business keys the deduplicator groups on, so every candidate in a group
    agrees on it by construction. There is nothing for the provenance ordering to choose
    between.

    That deserves a test rather than a sentence in an audit, because exactly one writer holds
    it up. The fundamental path derives its available date from ``ann_date`` instead
    (``availability.py:102-103``), which is a genuinely different value, and it neither is
    wired nor passes through this deduplicator today. If decisions B or C ever route it here,
    two revisions of one report period can land on one business key with different available
    dates, and C3.1(a) turns from "nothing to choose between" into "the PIT field is decided
    by row order". This is the tripwire for that moment: whoever breaks it is making the
    decision rather than inheriting it.

    Both assertions are load-bearing in opposite directions. The per-group count would also
    pass if the reader stamped every row with one constant date, so the equality against
    ``observation_date`` pins the mechanism; and the equality alone would keep passing if
    ``observation_date`` ever stopped being a business key, so the per-group count pins the
    consequence.
    """
    normalized = _normalize(_wide_source())
    business_keys = ["observation_date", "ts_code", "factor_id", "factor_version"]
    duplicated = normalized.loc[normalized.duplicated(business_keys, keep=False)]

    assert not duplicated.empty, "the fixture must contain a duplicate group to say anything"
    distinct_per_group = duplicated.groupby(business_keys, sort=True)[
        "source_available_date"
    ].nunique()
    assert set(distinct_per_group) == {1}, distinct_per_group.to_dict()
    assert (normalized["source_available_date"] == normalized["observation_date"]).all()
