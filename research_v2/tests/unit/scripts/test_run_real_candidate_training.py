"""Contract tests for the real-data candidate training driver (Task 2).

The driver must enforce the E-plan execution order (M1 -> M2 -> M3), refuse
to run without fold-matched baseline artifacts, and load those artifacts
fail-closed. Actual training wiring lands in Tasks 3-4.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import torch

from research_v2.training.static_batches import build_static_training_batches

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "run_real_candidate_training.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_real_candidate_training", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_real_candidate_training"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

PREDICTION_COLUMNS = ["signal_date", "ts_code", "score", "model_id", "fold_id", "dataset_id"]


def _write_fold_artifact(directory: Path, fold_id: str) -> None:
    frame = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2020-01-02")],
            "ts_code": ["000001.SZ"],
            "score": [0.5],
            "model_id": ["B2"],
            "fold_id": [fold_id],
            "dataset_id": [f"real-{fold_id.lower()}-v1-test"],
        }
    )
    frame.to_parquet(directory / f"{fold_id}.parquet", index=False)


def test_load_baseline_predictions_reads_all_folds(tmp_path: Path) -> None:
    for fold in MODULE.ORDINARY_FOLDS:
        _write_fold_artifact(tmp_path, fold)
    frames = MODULE.load_baseline_predictions(tmp_path)
    assert sorted(frames) == list(MODULE.ORDINARY_FOLDS)
    assert list(frames["F01"].columns) == PREDICTION_COLUMNS


def test_load_baseline_predictions_requires_every_fold(tmp_path: Path) -> None:
    for fold in MODULE.ORDINARY_FOLDS[:-1]:  # drop F06
        _write_fold_artifact(tmp_path, fold)
    with pytest.raises(ValueError, match="F06"):
        MODULE.load_baseline_predictions(tmp_path)


def test_family_order_guard_allows_m1() -> None:
    MODULE.validate_family_order("M1", prior_families_passed=())


def test_family_order_guard_blocks_m2_without_evidence() -> None:
    with pytest.raises(ValueError, match="M1"):
        MODULE.validate_family_order("M2", prior_families_passed=())


def test_family_order_guard_blocks_m3_without_m2() -> None:
    with pytest.raises(ValueError, match="M2"):
        MODULE.validate_family_order("M3", prior_families_passed=("M1",))


def test_family_order_guard_allows_m3_after_m1_m2() -> None:
    MODULE.validate_family_order("M3", prior_families_passed=("M1", "M2"))


def test_reject_unknown_family() -> None:
    with pytest.raises(ValueError, match="family"):
        MODULE.validate_family_order("M4", prior_families_passed=("M1", "M2"))


def test_parse_args_requires_baseline_artifacts_for_full_runs(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    args = MODULE.parse_args(
        ["--family", "M1", "--baseline-artifacts", str(tmp_path), "--report", str(report)]
    )
    assert args.family == "M1"
    assert args.report == report
    assert not args.smoke


def test_parse_args_smoke_mode_allows_missing_artifacts(tmp_path: Path) -> None:
    args = MODULE.parse_args(
        ["--family", "M1", "--smoke", "--report", str(tmp_path / "smoke.json")]
    )
    assert args.smoke is True
    assert args.baseline_artifacts is None


def test_m1_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 3.1: orchestration evidence only - synthetic gates are not model evidence."""
    report = MODULE.train_m1_smoke(tmp_path)
    assert report["family"] == "M1"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert all(status in {"PASS", "FAIL", "NOT_APPLICABLE"} for status in oos["gates"].values())
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)
    assert report["worst_seed"] in MODULE.REQUIRED_SEEDS


def test_m2_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 4.1: M2 temporal (GRU) orchestration evidence on synthetic frames."""
    report = MODULE.train_m2_smoke(tmp_path)
    assert report["family"] == "M2"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)


def test_m3_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """Task 4.3: M3 set-attention orchestration evidence on synthetic frames."""
    report = MODULE.train_m3_smoke(tmp_path)
    assert report["family"] == "M3"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)


def _fold_frames() -> dict[str, pd.DataFrame]:
    def _split(day, with_target: bool) -> pd.DataFrame:
        rows = []
        for index in range(32):
            row: dict[str, object] = {
                "signal_date": day,
                "ts_code": f"s{index:02d}",
                "label_valid": True,
                "f00": float(index) * 0.1,
                "f01": float(index) * -0.05,
            }
            if with_target:
                row["raw_return_5d"] = float(index) * 0.01
            rows.append(row)
        return pd.DataFrame(rows)

    return {
        "train": _split(pd.Timestamp("2020-01-02"), with_target=True),
        "validation": _split(pd.Timestamp("2021-01-04"), with_target=True),
        "test": _split(pd.Timestamp("2022-01-04"), with_target=False).assign(
            raw_return_5d=0.0
        ),
    }


def test_publish_fold_dataset_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    from research_v2.datasets.storage import DatasetArtifactStore

    store = DatasetArtifactStore(tmp_path / "datasets")
    frames = _fold_frames()
    versions = {"label_daily": {"2020-01": "v1"}}
    factor_order = ("f00", "f01")
    first = MODULE.publish_fold_dataset(store, "F01", frames, versions, factor_order)
    second = MODULE.publish_fold_dataset(store, "F01", frames, versions, factor_order)
    assert first == second
    assert first  # non-empty content hash id


def test_reconstruct_baseline_record_round_trip() -> None:
    entry = {
        "fold_id": "F01",
        "selected_baseline_id": "B2",
        "validation_scores": [
            {
                "baseline_id": "B2",
                "mean_rank_ic": 0.02,
                "valid_day_count": 240,
                "invalid_day_count": 1,
            }
        ],
        "unselected_reasons": [["B0", "LOWER_VALIDATION_RANKIC"]],
    }
    record = MODULE.reconstruct_baseline_record(entry)
    assert record.fold_id == "F01"
    assert record.selected_baseline_id == "B2"
    assert record.validation_scores[0].invalid_day_count == 1
    assert record.unselected_reasons == (("B0", "LOWER_VALIDATION_RANKIC"),)


def test_issue_baseline_selection_evidence_signs_and_verifies(tmp_path: Path) -> None:
    from research_v2.datasets.storage import DatasetArtifactStore
    from research_v2.evaluation.selection_evidence import (
        SelectionEvidenceStore,
        SelectionEvidenceVerifier,
    )

    store = DatasetArtifactStore(tmp_path / "datasets")
    frames = _fold_frames()
    versions = {"label_daily": {"2020-01": "v1"}}
    dataset_id = MODULE.publish_fold_dataset(
        store, "F01", frames, versions, ("f00", "f01")
    )
    selection_store = SelectionEvidenceStore(tmp_path / "selections", signing_key=b"unit-key")
    baseline_report = {
        "selections": [
            {
                "fold_id": "F01",
                "selected_baseline_id": "B2",
                "validation_scores": [
                    {
                        "baseline_id": baseline_id,
                        "mean_rank_ic": 0.02 if baseline_id == "B2" else 0.01,
                        "valid_day_count": 240,
                        "invalid_day_count": 2,
                    }
                    for baseline_id in ("B0", "B1", "B2", "B3")
                ],
                "unselected_reasons": [
                    ["B0", "LOWER_VALIDATION_RANKIC"],
                    ["B1", "LOWER_VALIDATION_RANKIC"],
                    ["B3", "LOWER_VALIDATION_RANKIC"],
                ],
            }
        ]
    }
    evidence_ids = MODULE.issue_baseline_selection_evidence(
        baseline_report, {"F01": dataset_id}, selection_store
    )
    assert "F01" in evidence_ids
    verifier = SelectionEvidenceVerifier(tmp_path / "selections", verification_key=b"unit-key")
    loaded = verifier.load_baseline_selection(evidence_ids["F01"])
    assert loaded.fold_id == "F01"
    assert loaded.test_dataset_id == dataset_id
    assert loaded.selected_baseline_id == "B2"


def test_config_grids_respect_e7_budget() -> None:
    """E7: at most 30 configs per family search."""
    assert len(MODULE.m1_config_grid()) == 27 <= 30
    assert len(MODULE.m2_config_grid()) == 24 <= 30
    assert len(MODULE.m3_config_grid()) == 16 <= 30
    m2_families = {config["family"] for config in MODULE.m2_config_grid()}
    assert m2_families == {"GRU", "TCN"}


def _passed_family_report(family: str, *, mode: str = "full", all_passed: bool = True) -> dict:
    return {
        "family": family,
        "mode": mode,
        "folds": [],
        "oos": {"all_passed": all_passed, "gates": {}},
    }


def test_prior_families_from_reports_accepts_passed_full_run(tmp_path: Path) -> None:
    path = tmp_path / "m1_report.json"
    path.write_text(json.dumps(_passed_family_report("M1")), encoding="utf-8")
    assert MODULE.prior_families_from_reports([f"M1={path}"]) == ("M1",)


def test_prior_families_from_reports_rejects_failed_gates(tmp_path: Path) -> None:
    path = tmp_path / "m1_report.json"
    path.write_text(
        json.dumps(_passed_family_report("M1", all_passed=False)), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="all walk-forward gates"):
        MODULE.prior_families_from_reports([f"M1={path}"])


def test_prior_families_from_reports_rejects_smoke_mode(tmp_path: Path) -> None:
    path = tmp_path / "m1_report.json"
    path.write_text(
        json.dumps(_passed_family_report("M1", mode="smoke")), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="not a full run"):
        MODULE.prior_families_from_reports([f"M1={path}"])


def test_prior_families_from_reports_rejects_family_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "m1_report.json"
    path.write_text(json.dumps(_passed_family_report("M1")), encoding="utf-8")
    with pytest.raises(SystemExit, match="family mismatch"):
        MODULE.prior_families_from_reports([f"M2={path}"])


def test_prior_families_from_reports_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="does not exist"):
        MODULE.prior_families_from_reports([f"M1={tmp_path / 'absent.json'}"])


def test_prior_families_from_reports_rejects_bad_format(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="FAMILY=PATH"):
        MODULE.prior_families_from_reports(["M1"])


def test_prior_families_from_reports_rejects_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="unknown family"):
        MODULE.prior_families_from_reports([f"M4={tmp_path / 'x.json'}"])


def test_parse_args_collects_prior_reports(tmp_path: Path) -> None:
    args = MODULE.parse_args(
        [
            "--family",
            "M2",
            "--report",
            str(tmp_path / "m2.json"),
            "--prior-report",
            f"M1={tmp_path / 'm1.json'}",
        ]
    )
    assert args.prior_report == [f"M1={tmp_path / 'm1.json'}"]


# --- padding-mask regression -------------------------------------------------
#
# Every smoke fixture in this file gives each signal date the same number of
# stocks, so ``build_static_training_batches`` never pads and the padding bug
# below is invisible. Real segments are ragged: the F01/2019 validation segment
# runs 2,366 to 2,469 stocks per day, so every narrower day is padded to 2,469
# with empty security keys. The 2026-09-05 full M1 run trained for seven hours
# fifty-eight minutes and then died on exactly this:
#   ValueError: seed predictions contain duplicate business keys

RAGGED_FACTOR_ORDER = ("f00", "f01")
RAGGED_WIDTHS = {date(2019, 1, 2): 40, date(2019, 1, 3): 31}


def _ragged_batches():
    """Two dates of unequal width, so the narrower one carries nine padding slots."""
    rows: list[dict[str, object]] = []
    for day, count in RAGGED_WIDTHS.items():
        for index in range(count):
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": f"{index:06d}.SZ",
                    "raw_return_5d": float(index) / 100.0,
                    "label_valid": True,
                    "f00": float(index) + 1.0,
                    "f01": -1.0,
                }
            )
    return build_static_training_batches(
        pd.DataFrame(rows), factor_order=RAGGED_FACTOR_ORDER
    )


class _StubScorer:
    """Return ``[date, stock]`` scores with padding masked to zero, as every model does."""

    def __init__(self, sign: float = 1.0) -> None:
        self._sign = sign

    def eval(self) -> _StubScorer:
        return self

    def __call__(self, batch) -> torch.Tensor:
        scores = batch.features[:, :, 0] * self._sign
        return torch.Tensor.masked_fill(scores, ~batch.stock_mask, 0.0)


def test_ragged_fixture_really_is_padded() -> None:
    """Guard the guard: if this stops padding, the two tests below prove nothing."""
    batches = _ragged_batches()
    narrow = batches[date(2019, 1, 3)].inputs

    assert narrow.ts_codes[0].count("") == 9
    assert int(narrow.stock_mask[0].sum()) == 31


def test_predict_validation_percentiles_drops_padding_and_ranks_within_the_date() -> None:
    batches = _ragged_batches()

    result = MODULE._predict_validation_percentiles(_StubScorer(), batches)

    keys = ["signal_date", "ts_code"]
    assert not result.duplicated(keys).any(), "padding slots leaked out as duplicate keys"
    assert "" not in set(result["ts_code"]), "padding slots leaked out as empty ts_codes"
    assert len(result) == sum(RAGGED_WIDTHS.values())
    narrow = result.loc[result["signal_date"] == date(2019, 1, 3), "percentile_rank"]
    # Ranks must be within-date over the 31 valid stocks, not over the padded width
    # of 40. Ranking across padding shifts every real stock by a date-varying
    # amount, which is what corrupted the validation RankIC that selects the config.
    assert sorted(narrow) == pytest.approx([step / 31 for step in range(1, 32)])


def test_mean_seed_percentile_scores_averages_on_business_keys_not_positions() -> None:
    """Seeds order their own rows by score, so averaging must join on keys.

    Two scorers with opposite signs reverse each other's ordering. Key-aligned
    averaging gives every stock the same mean, (n+1)/2n; positional averaging
    would instead produce a ramp from 1/n to 1.

    That discriminating power is borrowed, not intrinsic: the two orderings differ
    only because ``predictions_to_frame`` sorts each date by descending score, which
    is pinned separately in ``tests/unit/models/test_model_batch_contracts.py``.
    Probed -- delete that sort and this test passes whether the merge is key-aligned
    or positional, i.e. it silently stops testing anything.
    """
    batch = _ragged_batches()[date(2019, 1, 3)]

    result = MODULE._mean_seed_percentile_scores(
        [_StubScorer(1.0), _StubScorer(-1.0)], batch
    )

    keys = ["signal_date", "ts_code"]
    assert not result.duplicated(keys).any()
    assert "" not in set(result["ts_code"])
    assert len(result) == 31
    assert list(result["score"]) == pytest.approx([32 / 62] * 31)


def test_percentile_frame_is_date_ordered_whatever_order_the_mapping_iterates() -> None:
    """``Mapping`` promises no iteration order, so the driver has to impose one.

    Nothing downstream would notice today: both batch builders iterate their dates
    through ``sorted(...)``, ``_temporal_panel_batches`` only filters that dict, and
    ``FrozenPredictionStore._normalize_scores`` re-sorts on business keys before the
    artifact is hashed. What is pinned here is the documented contract itself -- the
    frame comes out date-ordered because ``_percentile_frame`` says so, not because
    a caller happened to hand in a sorted dict.
    """
    batches = _ragged_batches()
    unordered = {
        date(2019, 1, 3): batches[date(2019, 1, 3)],
        date(2019, 1, 2): batches[date(2019, 1, 2)],
    }
    # Guard the guard: the mapping really does iterate out of date order.
    assert list(unordered) != sorted(unordered)

    result = MODULE._percentile_frame(_StubScorer(), unordered)

    dates = list(result["signal_date"])
    assert dates == sorted(dates), "dates were concatenated in mapping order"
    assert dates.count(date(2019, 1, 2)) == RAGGED_WIDTHS[date(2019, 1, 2)]
    assert dates.count(date(2019, 1, 3)) == RAGGED_WIDTHS[date(2019, 1, 3)]


class _ModeRecordingScorer(_StubScorer):
    """Track ``training`` the way ``nn.Module`` does, and record the mode at call time."""

    def __init__(self, sign: float = 1.0, training: bool = True) -> None:
        super().__init__(sign)
        self.training = training
        self.modes_at_call: list[bool] = []

    def eval(self) -> _ModeRecordingScorer:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _ModeRecordingScorer:
        self.training = mode
        return self

    def __call__(self, batch) -> torch.Tensor:
        self.modes_at_call.append(self.training)
        return super().__call__(batch)


def test_percentile_frame_scores_in_eval_mode_when_handed_a_train_mode_model() -> None:
    """The driver's only ``model.eval()`` is load-bearing, and it is the unpinned half.

    ``_percentile_frame``'s two callers pass models straight out of the training loop,
    and ``training/torch_loop.py:55`` leaves them in train mode, so this function has
    to switch mode itself. ``_mean_seed_percentile_scores`` correctly does not: every
    ``_concat_date_scores`` site passes ``loaded.models`` from ``load_inference_artifact``,
    and that loader already calls ``model.eval()`` -- pinned on its own side by
    ``tests/unit/inference/test_artifact_loader.py``'s
    ``all(not candidate.training for candidate in loaded.models)``.

    The asymmetry is exactly right, which is why it needs a pin: M2 and M3 both use
    ``nn.Dropout(0.1)`` (``models/temporal.py``, ``models/set_attention.py``), so
    deleting this one call would score the frozen TEST predictions fed to G1-G6 with
    dropout active and make them run-to-run stochastic. Correct-but-unpinned can be
    removed by any refactor that looks harmless.
    """
    scorer = _ModeRecordingScorer(training=True)
    # Guard the guard: the model really does arrive in train mode, as the loop leaves it.
    assert scorer.training is True

    MODULE._percentile_frame(scorer, _ragged_batches())

    assert scorer.modes_at_call == [False, False], (
        "scores were produced while the model was still in training mode"
    )
