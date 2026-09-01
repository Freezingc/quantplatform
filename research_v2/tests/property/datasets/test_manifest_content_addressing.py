"""Property evidence that dataset identity is logical-content-addressed (D9.4)."""

from __future__ import annotations

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from research_v2.datasets.manifest import build_dataset_manifest, build_split_dataset_manifest

_METADATA = {"fixture": "property", "version": 1}
_BUSINESS_KEYS = ("signal_date", "ts_code")


@st.composite
def keyed_frames(draw, min_rows: int = 2, max_rows: int = 40):
    row_count = draw(st.integers(min_value=min_rows, max_value=max_rows))
    days = draw(
        st.lists(
            st.dates(
                min_value=pd.Timestamp("2024-01-01").date(),
                max_value=pd.Timestamp("2024-12-31").date(),
            ),
            min_size=row_count,
            max_size=row_count,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=row_count,
            max_size=row_count,
        )
    )
    rows = [
        {"signal_date": day, "ts_code": f"s{index:02d}", "value": value}
        for index, (day, value) in enumerate(zip(days, values, strict=True))
    ]
    return pd.DataFrame(rows)


@settings(deadline=None, max_examples=40)
@given(frame=keyed_frames(), shuffle_seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_dataset_identity_is_invariant_to_row_order(frame, shuffle_seed) -> None:
    baseline = build_dataset_manifest(frame, business_keys=_BUSINESS_KEYS, metadata=_METADATA)
    shuffled = frame.sample(frac=1.0, random_state=shuffle_seed).reset_index(drop=True)
    permuted = build_dataset_manifest(shuffled, business_keys=_BUSINESS_KEYS, metadata=_METADATA)

    assert baseline.dataset_id == permuted.dataset_id
    assert baseline.content_hash == permuted.content_hash
    assert baseline.row_count == permuted.row_count


@settings(deadline=None, max_examples=40)
@given(frame=keyed_frames(), delta=st.floats(min_value=1e-6, max_value=1.0))
def test_dataset_identity_changes_with_any_value_change(frame, delta) -> None:
    baseline = build_dataset_manifest(frame, business_keys=_BUSINESS_KEYS, metadata=_METADATA)
    mutated = frame.copy()
    mutated.loc[0, "value"] = float(mutated.loc[0, "value"]) + delta
    changed = build_dataset_manifest(mutated, business_keys=_BUSINESS_KEYS, metadata=_METADATA)

    assert changed.dataset_id != baseline.dataset_id


@settings(deadline=None, max_examples=20)
@given(frame=keyed_frames(min_rows=4))
def test_split_assignment_is_part_of_dataset_identity(frame) -> None:
    midpoint = len(frame) // 2
    first_half = frame.iloc[:midpoint].reset_index(drop=True)
    second_half = frame.iloc[midpoint:].reset_index(drop=True)

    assignment_a = build_split_dataset_manifest(
        {"train": first_half, "validation": second_half},
        business_keys=_BUSINESS_KEYS,
        metadata=_METADATA,
    )
    assignment_b = build_split_dataset_manifest(
        {"train": second_half, "validation": first_half},
        business_keys=_BUSINESS_KEYS,
        metadata=_METADATA,
    )

    assert assignment_a.dataset_id != assignment_b.dataset_id
