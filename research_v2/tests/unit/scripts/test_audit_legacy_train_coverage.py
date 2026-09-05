"""Tests for the train-side coverage calibration audit.

G1-G6 are computed only from TEST-period evidence, so nothing in the promotion
contract measures how wide the cross-section the model was actually *fitted* on
is, nor how it compares with the cross-section it is *scored* on. The frozen-17
measurement (``verify_frozen_fold_coverage.py``) reports that asymmetry but
deliberately does not gate it, because no threshold had been measured. This
audit supplies the missing half: the same instrument applied to the legacy-14
frames that produced the already-accepted criterion-303 trust root.

The fixtures below are therefore calibration fixtures. Any G7 threshold must let
the legacy numbers pass -- a gate that retroactively fails the accepted baseline
would be invalidating the trust root, which is a separate decision.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "audit_legacy_train_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("audit_legacy_train_coverage", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_legacy_train_coverage"] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def test_factor_columns_selects_only_the_dropna_subset_in_order() -> None:
    columns = ["signal_date", "ts_code", "f00", "f01", "f10", "raw_return_5d", "label_valid"]

    assert M.factor_columns(columns) == ("f00", "f01", "f10")


def test_factor_columns_rejects_a_frame_without_features() -> None:
    with pytest.raises(ValueError, match="no f"):
        M.factor_columns(["signal_date", "ts_code", "raw_return_5d"])


def test_daily_widths_counts_rows_per_day_exactly_as_the_runner_dropna_does() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2020-01-02"] * 3 + ["2020-01-03"] * 2),
            "ts_code": ["a", "b", "c", "a", "b"],
            "f00": [1.0, 2.0, float("nan"), 3.0, 4.0],
            "f01": [1.0, 2.0, 3.0, 4.0, float("nan")],
        }
    )

    widths = M.daily_widths_after_dropna(frame, ("f00", "f01"))

    assert widths == {pd.Timestamp("2020-01-02"): 2, pd.Timestamp("2020-01-03"): 1}


def test_daily_widths_preserves_a_day_that_loses_every_row() -> None:
    """A fully dropped day must appear as width 0, not vanish from the calendar.

    Grouping the post-dropna frame would silently omit it, which is exactly the
    failure mode that let 2015's crash days disappear from the frozen train
    segment unnoticed.
    """
    frame = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2015-08-24"] * 2 + ["2015-08-25"] * 2),
            "ts_code": ["a", "b", "a", "b"],
            "f00": [float("nan"), float("nan"), 1.0, 2.0],
        }
    )

    widths = M.daily_widths_after_dropna(frame, ("f00",))

    assert widths[pd.Timestamp("2015-08-24")] == 0
    assert widths[pd.Timestamp("2015-08-25")] == 2


def test_years_consistency_accepts_exact_spans_and_the_test_role_spillover() -> None:
    assert M.years_consistent(
        [2014, 2015, 2016, 2017, 2018], (2014, 2015, 2016, 2017, 2018), "train"
    )
    assert M.years_consistent([2019], (2019,), "validation")
    assert M.years_consistent([2020], (2020,), "test")
    # Test-year signals legitimately exit into the following January.
    assert M.years_consistent([2020, 2021], (2020,), "test")


def test_years_consistency_rejects_spillover_for_non_test_roles_and_missing_years() -> None:
    assert not M.years_consistent([2019, 2020], (2019,), "validation")
    assert not M.years_consistent([2014, 2015], (2014, 2015, 2016, 2017, 2018), "train")
    assert not M.years_consistent([2020, 2022], (2020,), "test")


def test_resolve_cache_paths_fails_closed_on_a_missing_key() -> None:
    index = {"aa": Path("/tmp/aa.parquet"), "bb": Path("/tmp/bb.parquet")}
    wanted = (("F01", "train", "aa"), ("F01", "test", "bb"))

    resolved = M.resolve_cache_paths(wanted, index)

    assert resolved == {
        ("F01", "train"): Path("/tmp/aa.parquet"),
        ("F01", "test"): Path("/tmp/bb.parquet"),
    }

    with pytest.raises(ValueError, match="no cached frame"):
        M.resolve_cache_paths((("F02", "train", "cc"),), index)


def test_calibration_bounds_takes_the_worst_fold_not_the_average() -> None:
    """The gate threshold is set by the worst accepted fold, not a comfortable mean."""
    assessments = (
        M.SegmentCoverage("F01", "train", 1220, 0, 1200, 310),
        M.SegmentCoverage("F02", "train", 1219, 0, 1300, 205),
        M.SegmentCoverage("F01", "test", 243, 0, 2400, 1500),
        M.SegmentCoverage("F02", "test", 243, 0, 2600, 1600),
    )

    bounds = M.calibration_bounds(assessments)

    assert bounds.min_train_median_width == 1200
    assert bounds.worst_train_day_width == 205
    assert bounds.max_train_days_below_floor == 0
    assert bounds.min_train_test_ratio == pytest.approx(0.5, abs=1e-9)


def test_calibration_bounds_rejects_a_fold_without_both_roles() -> None:
    partial = (M.SegmentCoverage("F01", "train", 1220, 0, 1200, 310),)

    with pytest.raises(ValueError, match="train/test"):
        M.calibration_bounds(partial)
