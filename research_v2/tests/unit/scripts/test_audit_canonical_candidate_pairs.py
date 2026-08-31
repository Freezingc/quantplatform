"""Tests for the canonical candidate-pair comparison contract."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import audit_canonical_candidate_pairs as audit
import pandas as pd
import pytest


def _write_checkpoint_integrity(root: Path) -> None:
    phases: dict[str, object] = {}
    for phase in ("train", "validation"):
        directory = root / phase
        values_path = directory / "candidate_values.pkl"
        manifest_path = directory / "manifest.json"
        values = pd.read_pickle(values_path)
        manifest = json.loads(manifest_path.read_text())
        producer_code_commit = manifest["fingerprint"]["code_commit"]
        dates = sorted(pd.to_datetime(values["trade_date"]).dt.date.astype(str).unique())
        numeric = pd.to_numeric(values["replay_value"], errors="coerce")
        finite = numeric.replace([float("inf"), float("-inf")], pd.NA).notna()
        phases[phase] = {
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "producer_code_commit": producer_code_commit,
            "fingerprint_sha256": audit._json_digest(manifest["fingerprint"]),
            "cache_manifest_sha256": audit._json_digest(
                manifest["fingerprint"]["cache_manifest"]
            ),
            "values_sha256": hashlib.sha256(values_path.read_bytes()).hexdigest(),
            "row_count": len(values),
            "date_count": values["trade_date"].nunique(),
            "code_count": values["ts_code"].nunique(),
            "finite_value_count": int(finite.sum()),
            "min_trade_date": dates[0],
            "max_trade_date": dates[-1],
            "trade_dates_sha256": hashlib.sha256("\n".join(dates).encode()).hexdigest(),
        }
    (root / "integrity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "candidate_id": "yeslab_484_tushare_shsz_v1",
                "source_factor_id": "yeslab_484",
                "market_scope": "SH/SZ",
                "label": "forward_open_5d:v1",
                "phases": phases,
            }
        ),
        encoding="utf-8",
    )


def test_registry_has_exactly_six_versioned_candidates() -> None:
    assert [item.candidate_id for item in audit.CANDIDATES] == [
        "yeslab_114_tushare_shsz_v2",
        "yeslab_381_tushare_shsz_v1",
        "yeslab_405_tushare_shsz_v1",
        "yeslab_425_tushare_shsz_v2",
        "yeslab_484_tushare_shsz_v1",
        "yeslab_206_median_abs_v2",
    ]


def test_parser_defaults_to_the_fixed_comparison_window() -> None:
    args = audit.parse_args(["--output", "/tmp/x"])

    assert args.start == date(2020, 1, 2)
    assert args.end == date(2025, 12, 31)


def test_parser_requires_an_output_location() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])


def test_parser_rejects_locked_test_range() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args(
            ["--start", "2025-01-01", "--end", "2026-01-01", "--output", "/tmp/x"]
        )


def test_classify_pair_requires_complete_phase_days() -> None:
    assert audit.classify_pair(0.20, 1212, 0.20, 242) == "INCOMPLETE"


def test_classify_pair_marks_a_pair_below_monitor_threshold_as_unique() -> None:
    assert audit.classify_pair(0.79, 1212, 0.20, 243) == "PAIR_UNIQUE"


def test_classify_pair_marks_a_pair_at_monitor_threshold() -> None:
    assert audit.classify_pair(0.80, 1212, 0.20, 243) == "ELEVATED_PAIR_REDUNDANCY_MONITOR"


def test_classify_pair_marks_a_pair_at_redundancy_threshold() -> None:
    assert audit.classify_pair(0.20, 1212, 0.90, 243) == "REDUNDANT"


def test_alpha206_median_repair_uses_rolling_median() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=60).repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 60,
            "vol": [1.0, 2.0] * 59 + [100.0, 2.0],
        }
    )

    result = audit.materialize_alpha206_median_repair(
        market, pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-01")
    )

    assert list(result.columns) == ["trade_date", "ts_code", "replay_value"]
    assert result.loc[result.ts_code == "000001.SZ", "replay_value"].iloc[0] == pytest.approx(
        99.0
    )


def test_summarize_phase_pairs_rejects_missing_edge() -> None:
    with pytest.raises(ValueError, match="incomplete candidate pair matrix"):
        audit.summarize_phase_pairs(
            {"yeslab_114_tushare_shsz_v2": pd.DataFrame()},
            1,
            expected_dates=[pd.Timestamp("2025-01-02")],
        )


def test_summarize_phase_pairs_returns_each_unordered_candidate_pair_once() -> None:
    keys = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")] * 30,
            "ts_code": [f"{index:06d}.SZ" for index in range(30)],
        }
    )
    replay_frames = {
        candidate.candidate_id: keys.assign(replay_value=range(30))
        for candidate in audit.CANDIDATES
    }

    result = audit.summarize_phase_pairs(
        replay_frames,
        expected_days=1,
        expected_dates=[pd.Timestamp("2025-01-02")],
    )

    assert len(result) == 15
    assert not result.duplicated(["factor_id_left", "factor_id_right"]).any()
    assert set(result["valid_day_count"]) == {1}
    assert set(result["status"]) == {"COMPLETE"}


def test_summarize_phase_pairs_rejects_a_substituted_date() -> None:
    expected_date = pd.Timestamp("2025-01-02")
    actual_date = pd.Timestamp("2025-01-03")
    codes = [f"{index:06d}.SZ" for index in range(30)]
    replay_frames = {
        candidate.candidate_id: pd.DataFrame(
            {
                "trade_date": [actual_date] * len(codes),
                "ts_code": codes,
                "replay_value": range(len(codes)),
            }
        )
        for candidate in audit.CANDIDATES
    }

    with pytest.raises(ValueError, match="expected date set mismatch"):
        audit.summarize_phase_pairs(
            replay_frames,
            expected_days=1,
            expected_dates=[expected_date],
        )


def test_load_comparison_keys_normalizes_the_v2_member_label_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audit,
        "_load_validation_inputs",
        lambda _client, _start, _end: pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2025-01-03")],
                "ts_code": ["000001.SZ"],
                "raw_return": [0.02],
            }
        ),
    )

    result = audit.load_comparison_keys(
        object(), date(2025, 1, 2), date(2025, 1, 31)
    )

    assert result.to_dict(orient="records") == [
        {"trade_date": pd.Timestamp("2025-01-03"), "ts_code": "000001.SZ"}
    ]


def test_comparison_keys_accepts_the_normalized_internal_date_column() -> None:
    result = audit._comparison_keys(
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-01-03")],
                "ts_code": ["000001.SZ"],
            }
        )
    )

    assert result.to_dict(orient="records") == [
        {"trade_date": pd.Timestamp("2025-01-03"), "ts_code": "000001.SZ"}
    ]


def test_month_key_restriction_preserves_rolling_window_nonfinite_values() -> None:
    result = audit._restrict_replay_to_month_keys(
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp("2025-01-03")],
                "ts_code": ["000001.SZ"],
                "replay_value": [float("nan")],
            }
        ),
        pd.DataFrame(
            {
                "signal_date": [pd.Timestamp("2025-01-03")],
                "ts_code": ["000001.SZ"],
            }
        ),
        pd.Timestamp("2025-01-03"),
        pd.Timestamp("2025-01-03"),
    )

    assert pd.isna(result.loc[0, "replay_value"])


def test_load_alpha484_checkpoint_rejects_self_consistent_sparse_replay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "alpha484"
    for phase, start, end in (
        ("train", audit.TRAIN_START, audit.TRAIN_END),
        ("validation", audit.VALIDATION_START, audit.VALIDATION_END),
    ):
        directory = root / phase
        directory.mkdir(parents=True)
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(start)],
                "ts_code": ["000001.SZ"],
                "replay_value": [0.1],
            }
        ).to_pickle(directory / "candidate_values.pkl")
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "fingerprint": {
                        "candidate_id": "yeslab_484_tushare_shsz_v1",
                        "source_factor_id": "yeslab_484",
                        "market_scope": "SH/SZ",
                        "label": "forward_open_5d:v1",
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "code_commit": "fixture",
                        "cache_manifest": {"files": [{"name": "fixture", "size": 1}]},
                    },
                }
            )
        )

    _write_checkpoint_integrity(root)

    with pytest.raises(ValueError, match="canonical coverage"):
        audit.load_alpha484_checkpoint_values(root)


def test_load_alpha484_checkpoint_rejects_tampered_values(tmp_path: Path) -> None:
    root = tmp_path / "alpha484"
    for phase, start, end in (
        ("train", audit.TRAIN_START, audit.TRAIN_END),
        ("validation", audit.VALIDATION_START, audit.VALIDATION_END),
    ):
        directory = root / phase
        directory.mkdir(parents=True)
        values = pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(start), pd.Timestamp(end)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "replay_value": [0.1, 0.2],
            }
        )
        values_path = directory / "candidate_values.pkl"
        values.to_pickle(values_path)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "complete",
                    "fingerprint": {
                        "candidate_id": "yeslab_484_tushare_shsz_v1",
                        "source_factor_id": "yeslab_484",
                        "market_scope": "SH/SZ",
                        "label": "forward_open_5d:v1",
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "code_commit": "bf551ff",
                        "cache_manifest": {"files": [{"name": "fixture", "size": 1}]},
                    },
                }
            )
        )

    _write_checkpoint_integrity(root)
    (root / "train" / "candidate_values.pkl").write_bytes(
        (root / "train" / "candidate_values.pkl").read_bytes() + b"tampered"
    )
    with pytest.raises(ValueError, match="integrity"):
        audit.load_alpha484_checkpoint_values(root)


def test_normalize_replay_frame_rejects_bj_values() -> None:
    with pytest.raises(ValueError, match="SH/SZ"):
        audit._normalize_replay_frame(
            pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp("2025-01-02")],
                    "ts_code": ["830001.BJ"],
                    "replay_value": [0.1],
                }
            )
        )


def test_build_audit_payload_classifies_only_complete_train_and_validation_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audit, "TRAIN_EXPECTED_DAYS", 1)
    monkeypatch.setattr(audit, "VALIDATION_EXPECTED_DAYS", 1)
    codes = [f"{index:06d}.SZ" for index in range(30)]

    def frames_for(trade_date: str, values: range) -> dict[str, pd.DataFrame]:
        return {
            candidate.candidate_id: pd.DataFrame(
                {
                    "trade_date": [pd.Timestamp(trade_date)] * len(codes),
                    "ts_code": codes,
                    "replay_value": list(values),
                }
            )
            for candidate in audit.CANDIDATES
        }

    payload = audit.build_audit_payload(
        frames_for("2024-12-31", range(30)),
        frames_for("2025-01-02", range(29, -1, -1)),
        train_expected_dates=[pd.Timestamp("2024-12-31")],
        validation_expected_dates=[pd.Timestamp("2025-01-02")],
    )

    assert payload["decision"] == "COMPARISON_ONLY_NO_ADMISSION"
    assert payload["database_writes"] is False
    assert payload["locked_test_read"] is False
    assert payload["label"] == "forward_open_5d:v1 raw five-trading-day adjusted-open return"
    assert payload["comparison_scope"] == {
        "output_keys": "V2 is_member=true and label_valid=true",
        "formula_input_market": "Tushare daily SH/SZ full panel",
        "excluded_exchanges": ["BJ"],
    }
    assert payload["thresholds"] == {
        "pair_monitor_median_abs_spearman": 0.8,
        "pair_redundancy_median_abs_spearman": 0.9,
    }
    assert len(payload["train_pairs"]) == 15
    assert len(payload["validation_pairs"]) == 15
    assert {row["status"] for row in payload["pair_classifications"]} == {"REDUNDANT"}


def test_run_audit_writes_only_complete_comparison_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit, "TRAIN_EXPECTED_DAYS", 1)
    monkeypatch.setattr(audit, "VALIDATION_EXPECTED_DAYS", 1)
    codes = [f"{index:06d}.SZ" for index in range(30)]
    keys = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-12-31")] * len(codes)
            + [pd.Timestamp("2025-01-02")] * len(codes),
            "ts_code": codes * 2,
        }
    )
    monkeypatch.setattr(audit, "load_comparison_keys", lambda *_args: keys)
    monkeypatch.setattr(
        audit,
        "load_alpha484_checkpoint_values",
        lambda _root: keys.assign(replay_value=range(len(keys))),
    )
    monkeypatch.setattr(
        audit,
        "load_candidate_phase_market",
        lambda *_args, **_kwargs: pd.DataFrame(
            {"trade_date": pd.to_datetime(["2024-12-31", "2025-01-02"])}
        ),
    )

    def fake_materialize(
        _client: object, _market: pd.DataFrame, comparison_keys: pd.DataFrame, **kwargs: object
    ) -> dict[str, pd.DataFrame]:
        selected = comparison_keys.loc[
            comparison_keys["trade_date"].between(
                pd.Timestamp(kwargs["phase_start"]), pd.Timestamp(kwargs["phase_end"])
            )
        ].copy()
        return {
            candidate.candidate_id: selected.assign(replay_value=range(len(selected)))
            for candidate in audit.CANDIDATES
        }

    monkeypatch.setattr(audit, "materialize_candidate_phase", fake_materialize)
    output = tmp_path / "pair_similarity"
    args = audit.parse_args(
        ["--start", "2024-12-31", "--end", "2025-01-02", "--output", str(output)]
    )

    payload = audit.run_audit(args, client=object())

    assert payload["decision"] == "COMPARISON_ONLY_NO_ADMISSION"
    assert json.loads(output.with_suffix(".json").read_text())["model_eligible"] is False
    assert len(pd.read_csv(output / "train_pairs.csv")) == 15
    assert len(pd.read_csv(output / "validation_pairs.csv")) == 15


def test_run_audit_materializes_each_phase_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audit, "TRAIN_EXPECTED_DAYS", 1)
    monkeypatch.setattr(audit, "VALIDATION_EXPECTED_DAYS", 1)
    codes = [f"{index:06d}.SZ" for index in range(30)]
    keys = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-12-31")] * len(codes)
            + [pd.Timestamp("2025-01-02")] * len(codes),
            "ts_code": codes * 2,
        }
    )
    calls: list[tuple[date, date]] = []
    monkeypatch.setattr(audit, "load_comparison_keys", lambda *_args: keys)
    monkeypatch.setattr(
        audit,
        "load_alpha484_checkpoint_values",
        lambda _root: keys.assign(replay_value=range(len(keys))),
    )

    def fake_load_phase(
        _client: object, *, phase_start: date, phase_end: date, **_kwargs: object
    ) -> pd.DataFrame:
        calls.append((phase_start, phase_end))
        expected = "2024-12-31" if phase_start.year == 2020 else "2025-01-02"
        return pd.DataFrame({"trade_date": pd.to_datetime([expected])})

    def fake_materialize_phase(
        _client: object,
        _market: pd.DataFrame,
        comparison_keys: pd.DataFrame,
        *,
        phase_start: date,
        phase_end: date,
        **_kwargs: object,
    ) -> dict[str, pd.DataFrame]:
        selected = comparison_keys.loc[
            comparison_keys["trade_date"].between(
                pd.Timestamp(phase_start), pd.Timestamp(phase_end)
            )
        ]
        return {
            candidate.candidate_id: selected.assign(replay_value=range(len(selected)))
            for candidate in audit.CANDIDATES
        }

    monkeypatch.setattr(audit, "load_candidate_phase_market", fake_load_phase)
    monkeypatch.setattr(audit, "materialize_candidate_phase", fake_materialize_phase)
    args = audit.parse_args(
        [
            "--start",
            "2020-01-02",
            "--end",
            "2025-12-31",
            "--output",
            str(tmp_path / "pair_similarity"),
        ]
    )

    audit.run_audit(args, client=object())

    assert calls == [
        (audit.TRAIN_START, audit.TRAIN_END),
        (audit.VALIDATION_START, audit.VALIDATION_END),
    ]
