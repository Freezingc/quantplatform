from __future__ import annotations

import os
import re
from datetime import date

import pandas as pd
import pytest

from research_v2.db.publish import PublishValidationError, V2BatchPublisher
from research_v2.inference.publisher import build_prediction_batch
from research_v2.quality.engine import CheckResult, CheckStatus, Severity
from research_v2.runs.context import RunContext

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run inference integration tests",
    ),
]

_SIGNAL_DATE = date(2026, 8, 31)
_MODEL_ID = "model-1"
_LOGICAL_PARTITION = f"{_SIGNAL_DATE.isoformat()}|{_MODEL_ID}|1"
_CODES = tuple(f"{index:06d}.SZ" for index in range(1, 4))
_QUALITY_PASS = [CheckResult("score_health", Severity.BLOCKER, CheckStatus.PASS)]


class VisibilitySession:
    """Small V2Test double whose pointer is the only visibility boundary."""

    target_database = "dfs://QuantResearchV2Test"

    def __init__(self) -> None:
        self.uploads: dict[str, pd.DataFrame] = {}
        self.rows: dict[tuple[str, str], list[pd.DataFrame]] = {}
        self.pointers: dict[tuple[str, str], str] = {}
        self.scripts: list[str] = []
        self.fail_pointer_once = False

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self.uploads[name] = frame.copy(deep=True)

    def execute(self, script: str) -> object:
        normalized = " ".join(script.split())
        self.scripts.append(normalized)
        if "published_version" in normalized:
            if self.fail_pointer_once:
                self.fail_pointer_once = False
                raise RuntimeError("publish interrupted before visibility pointer")
            object_name = re.search(r"\[`([^]]+)\] as object_name", normalized)
            logical_partition = re.search(r"\['([^']*)'\] as logical_partition", normalized)
            data_version = re.search(r"\['([^']*)'\] as data_version", normalized)
            assert object_name and logical_partition and data_version
            self.pointers[(object_name.group(1), logical_partition.group(1))] = (
                data_version.group(1)
            )
            return None

        stage = re.search(r"\}\((stage_[0-9a-f]+)\)", normalized)
        table = re.search(r"loadTable\('[^']+', '([^']+)'\)", normalized)
        assert stage and table
        frame = self.uploads[stage.group(1)]
        data_version = str(frame["data_version"].iloc[0])
        self.rows.setdefault((table.group(1), data_version), []).append(frame.copy(deep=True))
        return None

    def query(self, script: str) -> object:
        normalized = " ".join(script.split())
        if normalized.startswith("schema("):
            return pd.DataFrame(columns=["name", "typeString"])
        if "count(*) as row_count" in normalized:
            table = re.search(r"loadTable\('[^']+', '([^']+)'\)", normalized)
            data_version = re.search(r"data_version='([^']*)'", normalized)
            assert table and data_version
            count = sum(
                len(frame)
                for (object_name, version), frames in self.rows.items()
                if object_name == table.group(1) and version == data_version.group(1)
                for frame in frames
            )
            return pd.DataFrame({"row_count": [count]})
        if "published_version" in normalized:
            object_name = re.search(r"object_name=`([A-Za-z0-9_]+)", normalized)
            logical_partition = re.search(r"logical_partition='([^']*)'", normalized)
            assert object_name and logical_partition
            version = self.pointers.get((object_name.group(1), logical_partition.group(1)))
            if version is None:
                return pd.DataFrame(columns=["data_version"])
            return pd.DataFrame({"data_version": [version]})
        raise AssertionError(f"unexpected V2Test query: {normalized}")

    def visible(self, signal_date: date) -> pd.DataFrame:
        version = self.pointers.get(("prediction_daily", _LOGICAL_PARTITION))
        if version is None:
            return pd.DataFrame()
        frames = self.rows.get(("prediction_daily", version), [])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _batch(data_version: str, run_id: str) -> object:
    ensemble = pd.DataFrame(
        {
            "signal_date": [_SIGNAL_DATE] * len(_CODES),
            "ts_code": _CODES,
            "ensemble_score": [0.1, 0.5, 0.9],
        }
    )
    return build_prediction_batch(
        ensemble,
        model_id=_MODEL_ID,
        model_version=1,
        universe_id="universe-v2test",
        universe_version=1,
        inference_manifest_id="manifest-v1",
        source_watermark_id="watermark-v1",
        context=RunContext(run_id=run_id, data_version=data_version),
    )


def test_publish_interruption_keeps_partial_rows_invisible_and_recovery_is_idempotent() -> None:
    session = VisibilitySession()
    publisher = V2BatchPublisher(session)

    publisher.publish(
        _batch("data-v1", "00000000-0000-0000-0000-000000000001"),
        _QUALITY_PASS,
    )
    baseline = session.visible(_SIGNAL_DATE)
    assert len(baseline) == len(_CODES)

    session.fail_pointer_once = True
    with pytest.raises(RuntimeError, match="visibility pointer"):
        publisher.publish(
            _batch("data-v2-orphan", "00000000-0000-0000-0000-000000000002"),
            _QUALITY_PASS,
        )

    pd.testing.assert_frame_equal(session.visible(_SIGNAL_DATE), baseline)
    assert session.pointers[("prediction_daily", _LOGICAL_PARTITION)] == "data-v1"
    assert len(session.rows[("prediction_daily", "data-v2-orphan")]) == 1

    with pytest.raises(PublishValidationError, match="orphan rows"):
        publisher.publish(
            _batch("data-v2-orphan", "00000000-0000-0000-0000-000000000002"),
            _QUALITY_PASS,
        )
    pd.testing.assert_frame_equal(session.visible(_SIGNAL_DATE), baseline)

    recovered = _batch("data-v2-retry", "00000000-0000-0000-0000-000000000003")
    publisher.publish(recovered, _QUALITY_PASS)
    visible_after_recovery = session.visible(_SIGNAL_DATE)
    assert len(visible_after_recovery) == len(_CODES)
    assert set(visible_after_recovery["data_version"]) == {"data-v2-retry"}
    uploads_after_recovery = len(session.uploads)

    # A same-day rerun with the same content-addressed version is a no-op.
    publisher.publish(
        _batch("data-v2-retry", "00000000-0000-0000-0000-000000000003"),
        _QUALITY_PASS,
    )
    assert len(session.uploads) == uploads_after_recovery
    pd.testing.assert_frame_equal(session.visible(_SIGNAL_DATE), visible_after_recovery)
