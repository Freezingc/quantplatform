"""Contract tests for the real-data candidate training driver (Task 2).

The driver must enforce the E-plan execution order (M1 -> M2 -> M3), refuse
to run without fold-matched baseline artifacts, and load those artifacts
fail-closed. Actual training wiring lands in Tasks 3-4.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore
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
    assert list(frames["F1"].columns) == PREDICTION_COLUMNS


def test_load_baseline_predictions_requires_every_fold(tmp_path: Path) -> None:
    for fold in MODULE.ORDINARY_FOLDS[:-1]:  # drop F4
        _write_fold_artifact(tmp_path, fold)
    with pytest.raises(ValueError, match="F4"):
        MODULE.load_baseline_predictions(tmp_path)


def test_family_order_guard_allows_m1() -> None:
    MODULE.validate_family_order("M1", prior_families_passed=())


def test_family_order_guard_allows_m0() -> None:
    """M0 is the ML-first entry point and has no E-plan prerequisites."""
    MODULE.validate_family_order("M0", prior_families_passed=())


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


def test_m0_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """M0 (LightGBM LambdaRank candidate) orchestration evidence on synthetic frames."""
    report = MODULE.train_m0_smoke(tmp_path)
    assert report["family"] == "M0"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    assert oos["valid_day_count"] == 8
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert all(status in {"PASS", "FAIL", "NOT_APPLICABLE"} for status in oos["gates"].values())
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)
    assert report["worst_seed"] in MODULE.REQUIRED_SEEDS


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
    assert len(MODULE.m0_config_grid()) == 24 <= 30
    assert len(MODULE.m0_refined_config_grid()) == 24 <= 30
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


def test_family_order_guard_allows_m0r() -> None:
    """M0R is an alternative carrier for the same slot, not a later E-plan stage."""
    MODULE.validate_family_order("M0R", prior_families_passed=())


def test_m0r_is_a_registered_family_with_m0_prerequisites() -> None:
    assert "M0R" in MODULE.MODEL_FAMILIES
    assert MODULE.FAMILY_ORDER_PREREQUISITES["M0R"] == ()
    # the E-plan order for the torch families is untouched
    assert MODULE.FAMILY_ORDER_PREREQUISITES["M2"] == ("M1",)
    assert MODULE.FAMILY_ORDER_PREREQUISITES["M3"] == ("M1", "M2")


def test_m0r_spec_leaves_the_m0_contract_untouched() -> None:
    """M0's family name, code version and ids must not move, or its evidence is invalidated."""
    m0, m0r = MODULE.M0_SPEC, MODULE.M0R_SPEC
    assert (m0.family, m0.family_name, m0.model_code_version) == (
        "M0", "LGBM_RANKER", "lgbm-ranker-1",
    )
    assert m0.model_id_prefix == "m0" and m0.carrier == "lambdarank"
    assert m0.fit is MODULE._m0_fit
    assert (m0r.family, m0r.family_name, m0r.model_code_version) == (
        "M0R", "LGBM_RANKREG", "lgbm-rankreg-1",
    )
    assert m0r.model_id_prefix == "m0r" and m0r.carrier == "rankreg"
    assert m0r.fit is MODULE._m0r_fit
    # distinct prefixes are what keep the two families' bundles from colliding in one workspace
    assert m0.model_id_prefix != m0r.model_id_prefix


def test_full_run_wrappers_dispatch_their_own_spec() -> None:
    seen: list[object] = []

    def _record(args: object, spec: object) -> dict[str, object]:
        seen.append(spec)
        return {"family": spec.family}  # type: ignore[attr-defined]

    original = MODULE._run_full_lightgbm_family
    MODULE._run_full_lightgbm_family = _record
    try:
        assert MODULE.run_full_m0(None)["family"] == "M0"
        assert MODULE.run_full_m0r(None)["family"] == "M0R"
    finally:
        MODULE._run_full_lightgbm_family = original
    assert [spec.family for spec in seen] == ["M0", "M0R"]
    assert [spec.carrier for spec in seen] == ["lambdarank", "rankreg"]


def test_m0r_smoke_pipeline_end_to_end(tmp_path: Path) -> None:
    """M0R (rank-regression carrier) orchestration evidence on the same synthetic frames."""
    report = MODULE.train_m0r_smoke(tmp_path)
    assert report["family"] == "M0R"
    assert report["mode"] == "smoke"
    oos = report["oos"]
    # >= 7, not == 8 like the M0 smoke: this fixture is 40 trivially separable names a day,
    # and regressing the per-day percentile rank fits it so closely that many names share a
    # leaf, leaving 9-14 distinct scores per day where LambdaRank leaves 27-34. One day lands
    # on 9 and trips the score-health floor of unique_score_count >= 10, so that day is
    # correctly reported unevaluable. It is a degeneracy of the 40-name synthetic fixture, not
    # of the carrier: on the real frozen-v2 folds (2400-3800 names a day) M0R scored G6
    # 0.9752, the same 945/969 valid days as M0, and passed all six gates. The bound stays
    # here so a collapse to zero days still fails loudly.
    assert oos["valid_day_count"] >= 7
    assert set(oos["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert oos["baseline_mean_rank_ic"] is not None
    assert set(report["seed_validation_rank_ic"]) == set(MODULE.REQUIRED_SEEDS)
    assert report["worst_seed"] in MODULE.REQUIRED_SEEDS
    assert "never model passing evidence" in report["note"]


def test_m0r_bundle_round_trips_as_the_rankreg_carrier(tmp_path: Path) -> None:
    """The bundle a promotion run reads back must reconstruct the right baseline type."""
    MODULE.train_m0r_smoke(tmp_path)
    store = MODULE.ModelArtifactBundleStore(tmp_path / "models")
    card = json.loads((store.bundle_path("smoke-m0r") / "model_card.json").read_text())
    assert card["family"] == "LGBM_RANKREG"
    assert card["model_code_version"] == "lgbm-rankreg-1"
    for seed in MODULE.REQUIRED_SEEDS:
        loaded = MODULE._m0_load_booster(store, "smoke-m0r", seed, carrier="rankreg")
        assert isinstance(loaded, MODULE.LGBMRankRegBaseline)
        # the six sparse frozen-v2 factors are ~50% NaN, so a reloaded booster that rejected
        # NaN would raise on the first TEST row
        assert loaded.allow_nan is True


def test_m0_bundle_still_round_trips_as_the_ranker(tmp_path: Path) -> None:
    """Adding M0R must not change what an M0 bundle reconstructs into."""
    MODULE.train_m0_smoke(tmp_path)
    store = MODULE.ModelArtifactBundleStore(tmp_path / "models")
    card = json.loads((store.bundle_path("smoke-m0") / "model_card.json").read_text())
    assert card["family"] == "LGBM_RANKER"
    assert card["model_code_version"] == "lgbm-ranker-1"
    loaded = MODULE._m0_load_booster(store, "smoke-m0", MODULE.REQUIRED_SEEDS[0])
    assert isinstance(loaded, MODULE.LGBMRankerBaseline)
    assert loaded.allow_nan is True


def test_m0r_grid_is_pinned_to_the_single_measured_config() -> None:
    """M0R searches nothing on purpose: its +0.00915 was measured with no validation search.

    A 24-config search is what makes M0's +0.004202 an optimistic maximum over draws selected
    on one validation year per fold. Pinning the grid keeps the production number free of that
    bias and comparable to the measurement that motivated the carrier.
    """
    grid = MODULE.m0r_config_grid()
    assert grid == [MODULE.M0R_MEASURED_CONFIG]
    assert len(grid) == 1
    assert grid[0] == {
        "num_leaves": 15, "max_depth": 4, "learning_rate": 0.05,
        "min_child_samples": 200, "feature_fraction": 0.85, "lambda_l2": 0.0,
    }
    # --grid is an M0-only knob; it must not move M0R's space
    for choice in ("base", "refined"):
        args = argparse.Namespace(grid=choice, max_configs=None)
        assert MODULE.M0R_SPEC.grid(args, None) == grid
        assert MODULE.M0R_SPEC.grid(args, 1) == grid


def test_m0_grid_still_honours_the_grid_flag() -> None:
    """Wiring M0R through the shared runner must not change M0's two 24-config grids."""
    base = MODULE.M0_SPEC.grid(argparse.Namespace(grid="base"), None)
    refined = MODULE.M0_SPEC.grid(argparse.Namespace(grid="refined"), None)
    assert base == MODULE.m0_config_grid()
    assert refined == MODULE.m0_refined_config_grid()
    assert len(base) == 24 == len(refined)
    assert not any("lambda_l2" in config for config in base)
    assert all("lambda_l2" in config for config in refined)
    # both stay inside the E7 budget of 30 trials per fold
    assert max(len(base), len(refined), len(MODULE.m0r_config_grid())) <= 30


def test_every_runnable_family_is_admitted_by_the_walk_forward_evaluator() -> None:
    """A family the driver trains but the evaluator rejects fails only after the whole run.

    This is a real failure, not a hypothetical: M0R was added to the driver's family list
    while research_v2.evaluation.model_walk_forward kept its own copy without it, so a
    completed four-fold run (~46 min of training, all bundles and signed evidence written)
    died at the final evaluate_model_walk_forward step on "requires one supported model
    family". The driver now imports that tuple instead of redeclaring it, and this pins the
    rest of the admission path -- the E-plan prerequisite table and the model-card family
    names the evaluator matches against each bundle.
    """
    from research_v2.evaluation.model_walk_forward import (
        _MODEL_CARD_FAMILIES,
        MODEL_FAMILIES,
    )

    assert MODULE.MODEL_FAMILIES is MODEL_FAMILIES, "the driver redeclared the admission list"
    for family in MODULE.MODEL_FAMILIES:
        assert family in MODULE.FAMILY_ORDER_PREREQUISITES
        assert family in _MODEL_CARD_FAMILIES
    for spec in (MODULE.M0_SPEC, MODULE.M0R_SPEC):
        assert spec.family in MODEL_FAMILIES
        assert spec.family_name in _MODEL_CARD_FAMILIES[spec.family], (
            f"{spec.family} bundles are written with family={spec.family_name}, which the "
            "evaluator would reject as not matching the frozen model selection"
        )


def test_torch_seed_helper_seeds_before_constructing_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Seeding must precede model construction, or the first seed is irreproducible.

    run_validation_governed_training seeds, but it runs after the factory, so the initial weights
    came from whatever the global RNG held on entry: for the first seed that is the process's
    entropy-seeded state, for later seeds the state the previous iteration left behind. Measured on
    the M1 smoke before the fix -- seeds 20260802 and 20260803 were identical across four runs
    (0.828893 / 0.830957) while 20260801 ranged over 0.7289-0.8527. One irreproducible seed of
    three is enough to make the ensemble, the config selected on it, and every gate number
    downstream irreproducible. After the fix all four family smokes are byte-identical run to run.
    """
    from types import SimpleNamespace

    from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
    from research_v2.training.artifacts import CheckpointStore

    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        MODULE, "configure_reproducibility", lambda seed: events.append(("seed", seed))
    )

    def fake_run(*args: object, **kwargs: object) -> object:
        events.append(("train", kwargs["seed"]))
        return SimpleNamespace(checkpoint_ids=["ckpt-0"])

    monkeypatch.setattr(MODULE, "run_validation_governed_training", fake_run)

    def factory(config: object, order: tuple[str, ...]) -> object:
        events.append(("construct", None))
        return ResidualMLP(
            ResidualMLPConfig(raw_factor_count=len(order), hidden_dim=4, num_blocks=1)
        )

    MODULE._train_cross_section_seed(
        {},
        ("f0", "f1"),
        {},
        {},
        seed=20260802,
        checkpoint_store=CheckpointStore(tmp_path / "checkpoints"),
        checkpoint_prefix="prefix",
        train_dataset_id="ds:train",
        validation_dataset_id="ds:validation",
        max_epochs=1,
        model_factory=factory,
    )
    assert [kind for kind, _ in events] == ["seed", "construct", "train"]
    assert events[0] == ("seed", 20260802)
    assert events[1] == ("train", 20260802) or events[2] == ("train", 20260802)


# --- search resume: the loop had no fold- or config-level resume, so any interruption restarted ---


def _smoke_args(*extra: str) -> argparse.Namespace:
    return MODULE.parse_args(["--family", "M1", "--smoke", "--report", "/tmp/r.json", *extra])


def test_resume_is_opt_in_and_defaults_to_off() -> None:
    """Adopting must never be the default, or a fresh run could inherit another run's search."""
    assert _smoke_args().resume is False
    assert _smoke_args("--resume").resume is True


def test_parallel_workers_require_explicit_torch_thread_count() -> None:
    """Parallel workers without a declared numerical runtime must be rejected."""
    args = _smoke_args("--parallel-workers", "2")

    with pytest.raises(SystemExit, match="torch-threads"):
        MODULE._parallel_execution(args)


def test_ordered_search_results_restore_grid_order_after_parallel_completion() -> None:
    """Selecting directly from completion order would make ties timing-dependent."""
    first = {"config": {"hidden_dim": 64}, "ensemble_validation_rank_ic": 0.1}
    second = {"config": {"hidden_dim": 128}, "ensemble_validation_rank_ic": 0.1}

    ordered = MODULE._ordered_search_results(
        [{"hidden_dim": 64}, {"hidden_dim": 128}],
        {1: second, 0: first},
    )

    assert ordered == [first, second]


def test_the_resume_handle_is_bound_to_this_fold_and_this_run(tmp_path: Path) -> None:
    handle = MODULE._search_resume(
        _smoke_args("--resume"),
        tmp_path,
        fold_id="F2",
        family="RESIDUAL_MLP",
        model_code_version="residual-mlp-1",
        train_dataset_id="ds-2:train",
        validation_dataset_id="ds-2:validation",
        factor_order=("f00", "f01"),
        runtime={"torch_version": "2.4.1"},
    )
    assert handle.record_path(0, {"hidden_dim": 64}).parent == tmp_path / "search" / "F2"
    adopted = handle.adopt(0, {"hidden_dim": 64}, budget={"max_epochs": 100})
    assert adopted.record is None


def test_the_resume_handle_logs_refusals_to_the_run_transcript(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A transcript that does not say why a config was trained again is unauditable."""
    handle = MODULE._search_resume(
        _smoke_args("--resume"),
        tmp_path,
        fold_id="F1",
        family="RESIDUAL_MLP",
        model_code_version="residual-mlp-1",
        train_dataset_id="ds:train",
        validation_dataset_id="ds:validation",
        factor_order=("f00",),
        runtime={},
    )
    handle.adopt(0, {"hidden_dim": 64}, budget={"max_epochs": 100})
    assert "not resumable" in capsys.readouterr().out


def test_an_ordinary_run_does_not_gain_a_line_per_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --resume the transcript must be byte-for-byte what it was before this existed."""
    handle = MODULE._search_resume(
        _smoke_args(),
        tmp_path,
        fold_id="F1",
        family="RESIDUAL_MLP",
        model_code_version="residual-mlp-1",
        train_dataset_id="ds:train",
        validation_dataset_id="ds:validation",
        factor_order=("f00",),
        runtime={},
    )
    assert handle.adopt(0, {"hidden_dim": 64}, budget={"max_epochs": 100}).record is None
    assert capsys.readouterr().out == ""


def test_the_torch_budget_binds_the_stopping_rule_the_trainer_uses() -> None:
    """One constant, referenced by both the fit and the record, so the binding cannot drift."""
    budget = MODULE._torch_search_budget(100)
    assert budget == {
        "max_epochs": 100,
        "early_stopping_patience": MODULE.TORCH_EARLY_STOPPING_PATIENCE,
    }
    assert MODULE._torch_search_budget(50) != budget


def test_the_lightgbm_budget_carries_defaults_the_config_never_names() -> None:
    """The reason to bind effective hyperparameters: a config hash cannot see a library default."""
    config = {
        "num_leaves": 31,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_samples": 200,
    }
    budget = MODULE._lightgbm_search_budget(config)
    assert budget["num_boost_round"] == MODULE.M0_NUM_BOOST_ROUND
    assert budget["early_stopping_patience"] == MODULE.M0_EARLY_STOPPING_PATIENCE
    assert budget["minimum_cross_section"] == 30
    hyperparameters = budget["hyperparameters"]
    assert hyperparameters["num_threads"] == MODULE.M0_NUM_THREADS
    for defaulted in ("bagging_fraction", "bagging_freq", "lambda_l2", "feature_fraction"):
        assert defaulted not in config
        assert defaulted in hyperparameters


def test_all_three_search_loops_consult_the_resume_handle() -> None:
    """Wiring check: a handle built but never asked would leave the risk exactly where it was."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.count("resume = _search_resume(") == 3
    # Three serial loops, one M1/M3 parallel preflight, and one fold adoption rebuild.
    assert source.count("resume.adopt(config_index, config_payload") == 5
    assert source.count("resume.persist(config_index, config_payload, result") == 3
    assert "early_stopping_patience=10" not in source


# --- fold-level adoption: the half of resume that crosses a fold boundary ---

FACTOR_ORDER_2 = ("f00", "f01")
CONFIG_0: dict[str, Any] = {
    "hidden_dim": 64,
    "num_blocks": 1,
    "dropout": 0.0,
    "learning_rate": 0.003,
}
BUDGET_100: dict[str, Any] = {"max_epochs": 100}


def _completed_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    signed_digest: str | None = None,
    write_evidence: bool = True,
    write_records: bool = True,
    indexed_kinds: tuple[str, ...] = ("candidate", "best_simple"),
) -> dict[str, Any]:
    """Build a REAL completed fold: verified bundle, signed evidence, indexed predictions, records.

    Nothing here is a stub of the thing being checked -- the bundle is written through the same
    immutable store the driver uses, the evidence is signed and loaded back through its verifier,
    and the search records go through the same persistence the search loop uses. Only the
    prediction index is faked, because it is a library function with its own tests and faking it is
    what makes "one of the two artifacts is missing" expressible at all.
    """
    from research_v2.evaluation.selection_evidence import SelectionEvidenceStore

    dataset_id = "ds-1"
    model_id = "m1-f1"
    configs = [CONFIG_0]
    seed_files: dict[int, Path] = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"
        path.write_bytes(b"weights")
        seed_files[seed] = path
    model_store = ModelArtifactBundleStore(tmp_path / "models")
    model_store.write(
        model_id,
        model_card={
            "dataset_id": dataset_id,
            "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": "residual-mlp-1",
        },
        config=dict(CONFIG_0, raw_factor_count=len(FACTOR_ORDER_2)),
        signature={"model_family": "RESIDUAL_MLP", "raw_factor_count": 2, "config_hash": "c"},
        feature_order=FACTOR_ORDER_2,
        preprocessing={"state_hash": "identity"},
        seed_checkpoints=seed_files,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    selection_store = SelectionEvidenceStore(tmp_path / "selections", signing_key=b"unit-key")
    baseline_evidence_id = ""
    model_evidence_id = ""
    if write_evidence:
        baseline_entry = {
            "fold_id": "F1",
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
                [baseline_id, "LOWER_VALIDATION_RANKIC"]
                for baseline_id in ("B0", "B1", "B3")
            ],
        }
        baseline = selection_store.write_baseline_selection(
            record=MODULE.reconstruct_baseline_record(baseline_entry),
            test_dataset_id=dataset_id,
        )
        baseline_evidence_id = baseline.evidence_id
        model_evidence = selection_store.write_model_selection(
            fold_id="F1",
            model_family="M1",
            model_artifact_id=model_id,
            model_bundle_digest=(
                model_store.bundle_digest(model_id)
                if signed_digest is None
                else signed_digest
            ),
            training_dataset_id=f"{dataset_id}:train",
            validation_dataset_id=f"{dataset_id}:validation",
            test_dataset_id=dataset_id,
            baseline_selection_id=baseline_evidence_id,
            validation_rank_ic=0.1057,
        )
        model_evidence_id = model_evidence.evidence_id
    monkeypatch.setattr(
        MODULE,
        "_prediction_ids_by_fold_and_kind",
        lambda workspace: {("F1", kind): f"pred-{kind}" for kind in indexed_kinds},
    )
    resume = MODULE._search_resume(
        _smoke_args("--resume"),
        tmp_path,
        fold_id="F1",
        family="RESIDUAL_MLP",
        model_code_version="residual-mlp-1",
        train_dataset_id=f"{dataset_id}:train",
        validation_dataset_id=f"{dataset_id}:validation",
        factor_order=FACTOR_ORDER_2,
        runtime={"torch_version": "2.4.1"},
    )
    if write_records:
        for index, config in enumerate(configs):
            resume.persist(
                index,
                config,
                {
                    "config": config,
                    "ensemble_validation_rank_ic": 0.1057,
                    "seed_validation_rank_ic": {seed: 0.1 for seed in REQUIRED_SEEDS},
                    "worst_seed": sorted(REQUIRED_SEEDS)[0],
                },
                budget=BUDGET_100,
            )
    return {
        "workspace": tmp_path,
        "model_store": model_store,
        "selection_store": selection_store,
        "resume": resume,
        "configs": configs,
        "dataset_id": dataset_id,
        "model_id": model_id,
        "baseline_evidence_id": baseline_evidence_id,
        "model_evidence_id": model_evidence_id,
    }


def _adoption(state: dict[str, Any], **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "fold_id": "F1",
        "model_id": state["model_id"],
        "dataset_id": state["dataset_id"],
        "train_dataset_id": f"{state['dataset_id']}:train",
        "validation_dataset_id": f"{state['dataset_id']}:validation",
        "factor_order": FACTOR_ORDER_2,
        "configs": state["configs"],
        "budget_for": lambda _config: BUDGET_100,
    }
    kwargs.update(overrides)
    return MODULE._fold_adoption(
        state["workspace"],
        state["model_store"],
        state["selection_store"],
        state["resume"],
        **kwargs,
    )


def test_a_completed_fold_is_adopted_with_its_report_entry_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _completed_fold(tmp_path, monkeypatch)
    adopted = _adoption(state).fold
    assert adopted is not None
    entry = adopted.report_entry
    assert entry["fold_id"] == "F1"
    assert entry["model_id"] == "m1-f1"
    assert entry["dataset_id"] == "ds-1"
    # best_config comes from the winning record, so it is the payload the fresh search would have
    # selected -- not a copy read back out of the bundle.
    assert entry["best_config"] == CONFIG_0
    assert len(entry["search_results"]) == 1
    assert entry["baseline_selection_id"] == state["baseline_evidence_id"]
    assert entry["model_selection_id"] == state["model_evidence_id"]
    assert adopted.oos_fold.model_selection_id == state["model_evidence_id"]
    assert adopted.oos_fold.candidate_prediction_id == "pred-candidate"
    assert adopted.oos_fold.best_simple_prediction_id == "pred-best_simple"


def test_adoption_refuses_a_bundle_that_no_longer_re_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The re-hash is the first check: a byte changed on disk ends the question immediately."""
    state = _completed_fold(tmp_path, monkeypatch)
    seed = tmp_path / "models" / "m1-f1" / "seed-20260801.pt"
    seed.write_bytes(seed.read_bytes() + b"x")
    adoption = _adoption(state)
    assert adoption.fold is None
    assert "does not re-verify" in adoption.reason


def test_adoption_refuses_a_bundle_from_another_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _completed_fold(tmp_path, monkeypatch)
    adoption = _adoption(state, dataset_id="ds-2")
    assert adoption.fold is None
    assert "dataset_id" in adoption.reason


def test_adoption_refuses_a_bundle_fitted_on_another_factor_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _completed_fold(tmp_path, monkeypatch)
    adoption = _adoption(state, factor_order=("f00",))
    assert adoption.fold is None
    assert "factor order" in adoption.reason


def test_adoption_refuses_evidence_that_binds_a_different_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The signed digest is what ties the evidence to THESE bytes, so it has to be compared."""
    state = _completed_fold(tmp_path, monkeypatch, signed_digest="0" * 64)
    adoption = _adoption(state)
    assert adoption.fold is None
    assert "bundle digest" in adoption.reason


def test_adoption_refuses_a_fold_whose_selection_evidence_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _completed_fold(tmp_path, monkeypatch, write_evidence=False)
    adoption = _adoption(state)
    assert adoption.fold is None
    assert "selection evidence" in adoption.reason


@pytest.mark.parametrize(
    "indexed",
    [("candidate",), ("best_simple",), ()],
    ids=["no_best_simple", "no_candidate", "none"],
)
def test_adoption_refuses_a_fold_missing_a_signed_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indexed: tuple[str, ...]
) -> None:
    state = _completed_fold(tmp_path, monkeypatch, indexed_kinds=indexed)
    adoption = _adoption(state)
    assert adoption.fold is None
    assert "prediction artifacts" in adoption.reason


def test_adoption_refuses_a_fold_whose_search_cannot_be_reconstructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without every config's record there is no search_results, so there is no report entry."""
    state = _completed_fold(tmp_path, monkeypatch, write_records=False)
    adoption = _adoption(state)
    assert adoption.fold is None
    assert "search record" in adoption.reason


def test_a_fold_with_no_bundle_is_neither_adopted_nor_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary case: nothing frozen yet, so the fold trains. Adoption must not fire."""
    state = _completed_fold(tmp_path, monkeypatch)
    empty = ModelArtifactBundleStore(tmp_path / "other-models")
    assert (
        MODULE._adopt_completed_fold(
            _smoke_args("--resume"),
            tmp_path,
            empty,
            state["selection_store"],
            state["resume"],
            fold_id="F1",
            model_id="m1-f1",
            dataset_id="ds-1",
            train_dataset_id="ds-1:train",
            validation_dataset_id="ds-1:validation",
            factor_order=FACTOR_ORDER_2,
            configs=state["configs"],
            budget_for=lambda _config: BUDGET_100,
        )
        is None
    )


def test_an_unadoptable_completed_fold_is_refused_before_any_fitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing up front is the whole point: the alternative is hours of fitting then a crash."""
    state = _completed_fold(tmp_path, monkeypatch, signed_digest="0" * 64)
    with pytest.raises(SystemExit) as exit_info:
        MODULE._adopt_completed_fold(
            _smoke_args("--resume"),
            tmp_path,
            state["model_store"],
            state["selection_store"],
            state["resume"],
            fold_id="F1",
            model_id="m1-f1",
            dataset_id="ds-1",
            train_dataset_id="ds-1:train",
            validation_dataset_id="ds-1:validation",
            factor_order=FACTOR_ORDER_2,
            configs=state["configs"],
            budget_for=lambda _config: BUDGET_100,
        )
    message = str(exit_info.value)
    assert "bundle digest" in message
    assert "immutable" in message


def test_a_completed_fold_without_resume_is_refused_with_the_way_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _completed_fold(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exit_info:
        MODULE._adopt_completed_fold(
            _smoke_args(),
            tmp_path,
            state["model_store"],
            state["selection_store"],
            state["resume"],
            fold_id="F1",
            model_id="m1-f1",
            dataset_id="ds-1",
            train_dataset_id="ds-1:train",
            validation_dataset_id="ds-1:validation",
            factor_order=FACTOR_ORDER_2,
            configs=state["configs"],
            budget_for=lambda _config: BUDGET_100,
        )
    assert "--resume" in str(exit_info.value)


def test_every_search_loop_adopts_completed_folds_before_fitting() -> None:
    """Wiring: a helper no loop calls leaves the trap exactly where it was."""
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert source.count("adopted_fold = _adopt_completed_fold(") == 3
    assert source.count("fold_reports.append(adopted_fold.report_entry)") == 3
    assert source.count('model_id = f"{spec.model_id_prefix}-{fold_id.lower()}"') == 1
    assert source.count('model_id = f"{family.lower()}-{fold_id.lower()}"') == 1
    assert source.count('model_id = f"m2-{fold_id.lower()}"') == 1
