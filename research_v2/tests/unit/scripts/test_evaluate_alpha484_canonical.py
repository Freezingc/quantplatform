"""Tests for the isolated canonical Alpha484 admission evaluator."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import evaluate_alpha484_canonical as evaluation
import pandas as pd
import pytest


def test_candidate_identity_is_isolated_from_historical_factor() -> None:
    assert evaluation.CANDIDATE_ID == "yeslab_484_tushare_shsz_v1"
    assert evaluation.SOURCE_FACTOR_ID == "yeslab_484"
    assert evaluation.CANDIDATE_ID != evaluation.SOURCE_FACTOR_ID


def test_parse_args_rejects_the_locked_test_boundary() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/out"])


def test_validate_full_market_dates_rejects_a_missing_open_date() -> None:
    market = pd.DataFrame({"trade_date": [pd.Timestamp("2025-01-02")]})

    with pytest.raises(ValueError, match="full SH/SZ"):
        evaluation.validate_full_market_dates(
            market, (date(2025, 1, 2), date(2025, 1, 3))
        )


def test_summarize_coverage_rejects_a_month_below_seventy_percent() -> None:
    labels = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"] * 10 + ["2025-02-03"] * 10),
            "ts_code": [f"0000{index:02d}.SZ" for index in range(10)] * 2,
        }
    )
    values = labels.iloc[:16].rename(columns={"signal_date": "trade_date"})
    values["replay_value"] = 1.0

    result = evaluation.summarize_coverage(values, labels)

    assert result["status"] == "FAIL"
    assert result["monthly_min_ratio"] == pytest.approx(0.60)


def test_freeze_training_direction_uses_only_training_rank_ic() -> None:
    training = pd.DataFrame(
        {
            "signal_date": pd.date_range("2020-01-02", periods=3),
            "rank_ic": [-0.1, -0.2, 0.0],
        }
    )

    assert evaluation.freeze_training_direction(training) == -1


def test_build_review_never_makes_candidate_a_model_input() -> None:
    review = evaluation.build_review(
        training={
            "signed_mean_rank_ic": 0.04,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "annual_gate_years": 5,
        },
        validation={
            "signed_mean_rank_ic": 0.03,
            "signed_icir": 0.55,
            "directional_day_share": 0.60,
        },
        coverage={"status": "PASS"},
        train_similarity={"status": "PASS"},
        validation_similarity={"status": "PASS"},
    )

    assert review["status"] == "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
    assert review["model_eligible"] is False
    assert review["formal_factor_set_changed"] is False
    assert review["locked_test_read"] is False


def test_run_evaluation_writes_non_publishing_review(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    formal_ids = tuple(f"yeslab_{value}" for value in range(1, 10))

    def fake_window(
        client: object, *, start: date, end: date, cache_dir: Path
    ) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
        del client, end, cache_dir
        years = range(2020, 2025) if start.year == 2020 else (2025,)
        dates = [
            timestamp
            for year in years
            for timestamp in (pd.Timestamp(f"{year}-01-02"), pd.Timestamp(f"{year}-01-03"))
        ]
        daily = pd.DataFrame(
            {
                "signal_date": dates,
                "rank_ic": [-0.10, -0.20] * len(tuple(years)),
                "sample_count": [100] * len(dates),
            }
        )
        pairs = pd.DataFrame(
            {
                "factor_id_left": [evaluation.CANDIDATE_ID] * len(formal_ids),
                "factor_id_right": formal_ids,
                "median_abs_spearman": [0.20] * len(formal_ids),
                "max_abs_spearman": [0.30] * len(formal_ids),
                "valid_day_count": [1212 if start.year == 2020 else 243] * len(formal_ids),
            }
        )
        return daily, {"status": "PASS"}, pairs

    monkeypatch.setattr(evaluation, "FORMAL_FACTOR_IDS", formal_ids)
    monkeypatch.setattr(evaluation, "_evaluate_window", fake_window)
    args = evaluation.parse_args(["--output", str(tmp_path)])

    review = evaluation.run_evaluation(args, client=object())

    assert review["status"] == "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
    assert review["model_eligible"] is False
    written = json.loads((tmp_path / "admission_review.json").read_text())
    assert written["formal_factor_set_changed"] is False
    assert len(json.loads((tmp_path / "similarity_summary.json").read_text())) == 18


def test_parse_args_accepts_staged_checkpoint_options(tmp_path: Path) -> None:
    args = evaluation.parse_args(
        [
            "--output",
            str(tmp_path / "out"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--phase",
            "train",
            "--resume",
            "--log-file",
            str(tmp_path / "run.log"),
        ]
    )

    assert args.checkpoint_dir == tmp_path / "checkpoints"
    assert args.phase == "train"
    assert args.resume is True
    assert args.log_file == tmp_path / "run.log"


def test_checkpoint_fingerprint_rejects_changed_inputs(tmp_path: Path) -> None:
    fingerprint = evaluation.build_input_fingerprint(
        start=date(2020, 1, 2),
        end=date(2024, 12, 31),
        cache_manifest={"files": ["daily_20200102.json"]},
        code_commit="abc123",
    )
    path = tmp_path / "manifest.json"
    evaluation._write_checkpoint_state(
        path,
        {
            "schema_version": evaluation.CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "status": "candidate_complete",
        },
    )

    loaded = evaluation._load_checkpoint_state(path)
    assert loaded["fingerprint"] == fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        evaluation._assert_checkpoint_fingerprint(
            loaded,
            {**fingerprint, "code_commit": "changed"},
        )


def test_checkpoint_fingerprint_allows_cache_growth_but_not_mutation() -> None:
    state = {
        "fingerprint": evaluation.build_input_fingerprint(
            start=date(2020, 1, 2),
            end=date(2024, 12, 31),
            cache_manifest={"files": [{"name": "daily_20200102.json", "size": 10}]},
            code_commit="abc123",
        )
    }
    grown = evaluation.build_input_fingerprint(
        start=date(2020, 1, 2),
        end=date(2024, 12, 31),
        cache_manifest={
            "files": [
                {"name": "daily_20200102.json", "size": 10},
                {"name": "daily_20200103.json", "size": 11},
            ]
        },
        code_commit="abc123",
    )
    evaluation._assert_checkpoint_fingerprint(state, grown)

    mutated = {**grown, "cache_manifest": {"files": [{"name": "daily_20200102.json", "size": 12}]}}
    with pytest.raises(ValueError, match="fingerprint"):
        evaluation._assert_checkpoint_fingerprint(state, mutated)


def test_incremental_similarity_skips_completed_months_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    candidate = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02", "2025-02-03"]),
            "ts_code": ["000001.SZ", "000001.SZ"],
            "replay_value": [1.0, 2.0],
        }
    )
    formal = {
        "yeslab_380": pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2025-01-02", "2025-02-03"]),
                "ts_code": ["000001.SZ", "000001.SZ"],
                "replay_value": [3.0, 4.0],
            }
        )
    }
    loads: list[tuple[date, date]] = []
    calls: list[set[str]] = []

    monkeypatch.setattr(
        evaluation,
        "FORMAL_FACTOR_IDS",
        ("yeslab_380",),
    )

    def fake_load_formal(
        client: object, *, start: date, end: date
    ) -> dict[str, pd.DataFrame]:
        del client
        loads.append((start, end))
        return {
            factor_id: frame.loc[
                frame["trade_date"].dt.date.between(start, end)
            ].copy()
            for factor_id, frame in formal.items()
        }

    def fake_similarity(
        frames: dict[str, pd.DataFrame], *, include_daily_values: bool = False
    ) -> pd.DataFrame:
        del include_daily_values
        calls.append(set(frames))
        return pd.DataFrame(
            {
                "factor_id_left": [evaluation.CANDIDATE_ID],
                "factor_id_right": ["yeslab_380"],
                "median_abs_spearman": [0.2],
                "max_abs_spearman": [0.3],
                "valid_day_count": [1],
            }
        )

    monkeypatch.setattr(evaluation, "_load_formal_values", fake_load_formal)
    monkeypatch.setattr(evaluation, "compute_pair_daily_spearman", fake_similarity)

    first = evaluation._compute_incremental_similarity(
        object(),
        candidate,
        start=date(2025, 1, 1),
        end=date(2025, 2, 28),
        checkpoint_dir=tmp_path,
        resume=False,
    )
    second = evaluation._compute_incremental_similarity(
        object(),
        candidate,
        start=date(2025, 1, 1),
        end=date(2025, 2, 28),
        checkpoint_dir=tmp_path,
        resume=True,
    )

    assert len(first) == len(second) == 2
    assert len(loads) == 2
    assert calls == [{evaluation.CANDIDATE_ID, "yeslab_380"}] * 2


def test_collapse_similarity_chunks_preserves_valid_day_count() -> None:
    chunks = pd.DataFrame(
        {
            "factor_id_left": [evaluation.CANDIDATE_ID, evaluation.CANDIDATE_ID],
            "factor_id_right": ["yeslab_380", "yeslab_380"],
            "median_abs_spearman": [0.2, 0.4],
            "max_abs_spearman": [0.3, 0.5],
            "valid_day_count": [2, 1],
            "daily_spearman": [[0.2, -0.3], [0.5]],
        }
    )

    result = evaluation._collapse_similarity_chunks(chunks)

    assert result.to_dict("records") == [
        {
            "factor_id_left": evaluation.CANDIDATE_ID,
            "factor_id_right": "yeslab_380",
            "median_abs_spearman": pytest.approx(0.3),
            "max_abs_spearman": pytest.approx(0.5),
            "valid_day_count": 3,
        }
    ]


def test_run_evaluation_phase_train_writes_only_train_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_window(
        client: object,
        *,
        start: date,
        end: date,
        cache_dir: Path,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
        del client, end, cache_dir, checkpoint_dir, resume
        return (
            pd.DataFrame(
                {
                    "signal_date": pd.to_datetime([f"{start.year}-01-02"]),
                    "rank_ic": [0.1],
                    "sample_count": [100],
                }
            ),
            {"status": "PASS"},
            pd.DataFrame(
                {
                    "factor_id_left": [evaluation.CANDIDATE_ID],
                    "factor_id_right": ["yeslab_380"],
                    "median_abs_spearman": [0.1],
                    "max_abs_spearman": [0.2],
                    "valid_day_count": [1000],
                }
            ),
        )

    calls: list[tuple[date, date]] = []

    def tracked_window(*args: object, **kwargs: object):
        calls.append((kwargs["start"], kwargs["end"]))
        return fake_window(*args, **kwargs)

    monkeypatch.setattr(evaluation, "_evaluate_window", tracked_window)
    args = evaluation.parse_args(
        ["--output", str(tmp_path), "--phase", "train"]
    )

    review = evaluation.run_evaluation(args, client=object())

    assert review["phase"] == "train"
    assert calls == [(evaluation.TRAIN_START, evaluation.TRAIN_END)]
    assert (tmp_path / "daily_rank_ic.csv").is_file()
    assert not (tmp_path / "similarity_summary.json").is_file()


def test_main_records_failure_and_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(evaluation, "run_evaluation", fail)
    log_file = tmp_path / "run.log"

    result = evaluation.main(
        ["--output", str(tmp_path / "out"), "--log-file", str(log_file)]
    )

    assert result == 1
    assert "run_failed" in log_file.read_text(encoding="utf-8")
    assert "synthetic failure" in capsys.readouterr().err
