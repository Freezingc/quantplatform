from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import refresh_factor_quality_latest as refresh


class RecordingSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self) -> None:
        self.uploads: list[tuple[str, pd.DataFrame]] = []
        self.executed: list[str] = []
        self.queries: list[str] = []

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self.uploads.append((name, frame.copy(deep=True)))

    def execute(self, script: str) -> None:
        self.executed.append(script)

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "count(distinct" in script:
            raise RuntimeError("Cannot nest aggregate function")
        return pd.DataFrame(
            {
                "factor_id": [f"yeslab_{index}" for index in range(1, 492)],
                "model_eligible": [index <= 9 for index in range(1, 492)],
            }
        )


@pytest.fixture
def valid_frame(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    frame = pd.DataFrame({"factor_id": ["yeslab_1"]})
    monkeypatch.setattr(refresh, "validate_quality_latest_frame", lambda value, **_: value)
    return frame


def test_refresh_dry_run_does_not_upload_or_execute(
    valid_frame: pd.DataFrame,
) -> None:
    session = RecordingSession()

    assert refresh.refresh_latest(session, valid_frame, publish=False) == "dry-run"
    assert session.uploads == []
    assert session.executed == []
    assert session.queries == []


def test_refresh_publish_verifies_invariants_from_raw_rows(
    valid_frame: pd.DataFrame,
) -> None:
    session = RecordingSession()

    assert refresh.refresh_latest(session, valid_frame, publish=True) == "published"

    script = "\n".join(session.executed)
    assert session.uploads[0][0] == "factorQualityLatestStage"
    assert "factor_quality_latest" in script
    assert "factor_quality_snapshot" not in script
    assert "delete from loadTable" in script
    assert "tableInsert" in script
    assert "factor_quality_latest" in session.queries[0]


def test_load_formal_scoreboard_merges_metrics_and_monitoring(tmp_path: Path) -> None:
    scoreboard_path = tmp_path / "scoreboard.json"
    train_path = tmp_path / "train.json"
    validation_path = tmp_path / "validation.json"
    train_similarity_path = tmp_path / "train_similarity.json"
    validation_similarity_path = tmp_path / "validation_similarity.json"
    scoreboard_path.write_text(
        json.dumps(
            {
                "factor_ids": ["yeslab_1"],
                "monitoring_factor_ids": ["yeslab_1"],
                "rows": [
                    {
                        "factor_id": "yeslab_1",
                        "admission_status": "PASS_WITH_MONITORING",
                        "annual_abs_icir_pass_years": 4,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    train_path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "factor_id": "yeslab_1",
                        "direction": -1,
                        "valid_day_count": 1212,
                        "signed_mean_rank_ic": 0.04,
                        "icir": -0.60,
                        "directional_day_share": 0.65,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        json.dumps(
            {
                "summaries": [
                    {
                        "factor_id": "yeslab_1",
                        "valid_day_count": 243,
                        "signed_mean_rank_ic": 0.03,
                        "icir": -0.55,
                        "directional_day_share": 0.60,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    train_similarity_path.write_text("[]", encoding="utf-8")
    validation_similarity_path.write_text("[]", encoding="utf-8")

    result = refresh.load_formal_scoreboard(
        scoreboard_path=scoreboard_path,
        train_summary_path=train_path,
        validation_summary_path=validation_path,
        train_similarity_path=train_similarity_path,
        validation_similarity_path=validation_similarity_path,
    )

    assert result["rows"] == [
        {
            "factor_id": "yeslab_1",
            "status": "PASS_WITH_MONITORING",
            "monitoring_required": True,
            "reason_code": "FORMAL_NINE_SCOREBOARD",
            "frozen_direction": -1,
            "train_valid_days": 1212,
            "train_signed_rank_ic": 0.04,
            "train_icir": 0.60,
            "train_directional_share": 0.65,
            "annual_gate_years": 4,
            "train_max_similarity": 0.0,
            "validation_valid_days": 243,
            "validation_signed_rank_ic": 0.03,
            "validation_icir": 0.55,
            "validation_directional_share": 0.60,
            "validation_max_similarity": 0.0,
            "evidence_path": str(scoreboard_path),
        }
    ]


def test_metric_parser_rejects_infinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        refresh._number({"metric": float("inf")}, "metric")


def test_parser_keeps_publish_and_model_eligible_paths_separate() -> None:
    args = refresh.parse_args(
        [
            "--effective-date",
            "2026-08-14",
            "--evidence-commit",
            "a" * 40,
            "--alpha484-review",
            "/tmp/alpha484.json",
        ]
    )

    assert args.publish is False
    assert args.model_eligible is False
    with pytest.raises(SystemExit):
        refresh.parse_args(
            [
                "--model-eligible",
                "--publish",
            ]
        )


def test_build_frame_from_artifacts_is_local_and_deterministic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog_path = tmp_path / "catalog.yaml"
    alpha95_path = tmp_path / "alpha95.json"
    alpha484_path = tmp_path / "alpha484.json"
    catalog_path.write_text("- factor_id: yeslab_1\n", encoding="utf-8")
    alpha95_path.write_text('{"candidates": {}}', encoding="utf-8")
    alpha484_path.write_text('{"status": "REJECT_CANONICAL_GATE"}', encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(refresh, "load_formal_scoreboard", lambda **_: {"rows": []})

    def fake_build(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame({"factor_id": ["yeslab_1"]})

    monkeypatch.setattr(refresh, "build_quality_latest_frame", fake_build)

    result = refresh.build_frame_from_artifacts(
        catalog_path=catalog_path,
        scoreboard_path=tmp_path / "scoreboard.json",
        train_summary_path=tmp_path / "train.json",
        validation_summary_path=tmp_path / "validation.json",
        train_similarity_path=tmp_path / "train_similarity.json",
        validation_similarity_path=tmp_path / "validation_similarity.json",
        alpha95_path=alpha95_path,
        alpha484_path=alpha484_path,
        effective_date=pd.Timestamp("2026-08-14").date(),
        evidence_commit="a" * 40,
    )

    assert result["factor_id"].tolist() == ["yeslab_1"]
    assert captured["catalog"].to_dict("records") == [{"factor_id": "yeslab_1"}]
    assert captured["alpha95"] == {"candidates": {}}
    assert captured["alpha484"] == {"status": "REJECT_CANONICAL_GATE"}


def test_main_dry_run_never_connects_to_dolphindb(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frame = pd.DataFrame(
        {
            "factor_id": ["yeslab_1", "yeslab_2"],
            "model_eligible": [True, False],
        }
    )
    monkeypatch.setattr(refresh, "build_frame_from_artifacts", lambda **_: frame)

    def fake_refresh(session: object, received: pd.DataFrame, *, publish: bool) -> str:
        assert session is None
        assert received is frame
        assert publish is False
        return "dry-run"

    monkeypatch.setattr(refresh, "refresh_latest", fake_refresh)
    monkeypatch.setattr(
        refresh,
        "_connect_v2",
        lambda: (_ for _ in ()).throw(AssertionError("dry-run must not connect")),
        raising=False,
    )

    assert (
        refresh.main(
            [
                "--effective-date",
                "2026-08-14",
                "--evidence-commit",
                "a" * 40,
                "--alpha484-review",
                "/tmp/alpha484.json",
            ]
        )
        == 0
    )
    rendered = json.loads(capsys.readouterr().out)
    assert rendered == {"eligible_count": 1, "row_count": 2, "status": "dry-run"}
