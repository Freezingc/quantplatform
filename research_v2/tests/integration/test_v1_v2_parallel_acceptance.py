"""V1/V2 parallel non-interference acceptance test (plan task F10.4).

Proves, around one complete in-process V2Test daily inference flow, that V1
(dfs://YesLab) stays strictly read-only:

1. The approved V1 schema fingerprints and the sentinel V1 output hashes are
   byte-identical before and after the flow.
2. The V1 connection boundary is replaced with a fail-closed recorder that
   serves only the two approved read shapes (``schema(...)`` introspection and
   ``select ...`` queries) and fails the test on any other, mutation-class
   call; the flow incurs zero V1 mutation calls.
3. The V2 compatibility exporter writes only inside its own isolated export
   tree (``<v2 root>/exports/v2/...``), verifies through its manifest, and
   refuses any destination under a declared V1 root.

No DolphinDB server or network access is involved: the V1 store is a frozen
on-disk snapshot under ``tmp_path`` and every seam is exercised in-process.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.datasets.builder import FactorInput
from research_v2.export.compatibility import (
    CompatibilityExport,
    UnsafeExportPath,
    export_compatibility_predictions,
    verify_compatibility_export,
)
from research_v2.inference.builder import build_label_free_inference_frame
from research_v2.inference.daily_pipeline import DailyInferencePipeline, DailyPipelineStatus
from research_v2.inference.manifest import build_inference_manifest
from research_v2.monitoring.score_health import evaluate_score_health
from research_v2.monitoring.watermarks import (
    InferenceWatermark,
    WatermarkStatus,
    check_inference_watermarks,
)
from research_v2.sources.yeslab import YesLabAdapter

pytestmark = pytest.mark.integration

_DATA_ROOT = Path(__file__).parent / "data"
_SIGNAL_DATE = date(2026, 1, 2)
_WINDOW_START = date(2025, 12, 26)
_MODEL_ID = "sha256:" + "0f" * 32
_UNIVERSE_ID = "universe:v2test:2026"
_FACTOR_SET_ID = "factor-set:v2test:2026"
_PREPROCESSOR_HASH = "sha256:" + "1" * 64
_SPEC_BYTES = (
    json.dumps(
        {
            "algorithm": "frozen-percentile-rank-ensemble",
            "feature_names": ["close_px", "trade_vol"],
            "model_id": _MODEL_ID,
            "model_version": 1,
        },
        sort_keys=True,
    )
    + "\n"
).encode("utf-8")

# Approved, frozen V1 fingerprints and sentinel hashes. Any drift in the V1
# snapshot fixtures or in the flow's view of V1 fails the acceptance test.
_APPROVED_V1_SCHEMA_FINGERPRINTS = {
    "stock_factor_daily": "8cce052722faad480aeb2e248077b6e9c9ad0a3e40addd84bccc7acab25f33e8",
    "trade_calendar": "4ac6e18222f5f9de5b4d2190bc9d9ee4a606fdd877b84cd5fb6287fbe7b9e028",
}
_APPROVED_V1_SENTINEL_HASHES = {
    "outputs/predictions_20251231.csv": (
        "bf23c6df141efcd87b8a120ee92cdd26e765c615d320b7261f7a59f2cf441d26"
    ),
    "outputs/settlement_20251231.txt": (
        "bc6ca5c5ade9ca444c006b14ea4885843af32d33d68089dd634a8c8ed8576c9c"
    ),
}

_INTEGRATION_GATE = pytest.mark.skipif(
    os.environ.get("QRV2_INTEGRATION") != "1",
    reason="set QRV2_INTEGRATION=1 to run V1/V2 parallel acceptance tests",
)


class V1MutationGuardError(AssertionError):
    """Raised the moment a mutation-class call reaches the V1 boundary."""


class RecordingYesLabSession:
    """Fail-closed recorder standing in for the dfs://YesLab connection.

    Only the two read shapes the V2 sources are allowed to issue are served:
    ``schema(...)`` introspection and ``select ...`` queries. Anything else is
    recorded as a mutation attempt and fails the test immediately.
    """

    def __init__(self, v1_root: Path) -> None:
        self._v1_root = v1_root
        self.scripts: list[str] = []
        self.mutation_attempts: list[str] = []

    def query(self, script: str) -> object:
        normalized = " ".join(script.split())
        self.scripts.append(normalized)
        if normalized.startswith(("schema(", "select ")):
            return self._serve_read(normalized)
        self.mutation_attempts.append(normalized)
        raise V1MutationGuardError(f"V1 mutation-class call blocked: {normalized}")

    def _serve_read(self, script: str) -> pd.DataFrame:
        table = self._table_name(script)
        suffix = ".schema.csv" if script.startswith("schema(") else ".csv"
        return pd.read_csv(self._v1_root / "tables" / f"{table}{suffix}")

    @staticmethod
    def _table_name(script: str) -> str:
        marker = "loadTable('dfs://YesLab', '"
        start = script.index(marker) + len(marker)
        return script[start : script.index("'", start)]


@dataclass(frozen=True)
class InferenceInputs:
    frame: pd.DataFrame
    manifest_id: str


def _install_v1_snapshot(tmp_path: Path) -> Path:
    v1_root = tmp_path / "v1" / "YesLab"
    shutil.copytree(_DATA_ROOT / "v1_snapshot", v1_root)
    return v1_root


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v1_schema_fingerprints(v1_root: Path) -> dict[str, str]:
    return {
        schema_path.name.removesuffix(".schema.csv"): _sha256(schema_path)
        for schema_path in sorted((v1_root / "tables").glob("*.schema.csv"))
    }


def _v1_sentinel_hashes(v1_root: Path) -> dict[str, str]:
    return {
        output.relative_to(v1_root).as_posix(): _sha256(output)
        for output in sorted((v1_root / "outputs").rglob("*"))
        if output.is_file()
    }


def _snapshot_tree(root: Path) -> dict[str, str]:
    """Record every file hash and symlink target under ``root`` (follows links)."""
    entries: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        current = Path(dirpath)
        for name in [*filenames, *dirnames]:
            path = current / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                entries[relative] = "symlink:" + os.readlink(path)
            elif path.is_file():
                entries[relative] = _sha256(path)
    return entries


def _percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True)


@_INTEGRATION_GATE
def test_full_v2test_daily_flow_leaves_v1_unchanged_and_writes_only_inside_v2_tree(
    tmp_path: Path,
) -> None:
    v1_root = _install_v1_snapshot(tmp_path)
    v2_root = tmp_path / "v2test"
    artifacts_root = v2_root / "artifacts"
    artifacts_root.mkdir(parents=True)
    spec_path = artifacts_root / "model_spec.json"
    spec_path.write_bytes(_SPEC_BYTES)
    approved_spec_digest = _sha256(spec_path)
    before = _snapshot_tree(tmp_path)

    fingerprints_before = _v1_schema_fingerprints(v1_root)
    sentinels_before = _v1_sentinel_hashes(v1_root)
    assert fingerprints_before == _APPROVED_V1_SCHEMA_FINGERPRINTS
    assert sentinels_before == _APPROVED_V1_SENTINEL_HASHES

    session = RecordingYesLabSession(v1_root)
    adapter = YesLabAdapter(session)

    # Watermark phase: read-only source extracts feed the label-free gate.
    market_batch = adapter.fetch_market(_WINDOW_START, _SIGNAL_DATE)
    calendar_batch = adapter.fetch_trading_calendar("SSE", _WINDOW_START, _SIGNAL_DATE)
    calendar_dates = pd.to_datetime(calendar_batch.frame["cal_date"], errors="raise").dt.date
    is_trading_day = bool(
        calendar_batch.frame.loc[
            (calendar_dates == _SIGNAL_DATE)
            & (calendar_batch.frame["is_open"].astype(int) == 1)
        ].shape[0]
        == 1
    )
    watermarks = {
        "yeslab.stock_factor_daily": InferenceWatermark(
            dependency="yeslab.stock_factor_daily",
            max_signal_date=market_batch.max_observation_date,
            data_version=market_batch.watermark_id,
            published=market_batch.source_row_count > 0,
            has_blocker=False,
        ),
        "yeslab.trade_calendar": InferenceWatermark(
            dependency="yeslab.trade_calendar",
            max_signal_date=calendar_batch.max_observation_date,
            data_version=calendar_batch.watermark_id,
            published=calendar_batch.source_row_count > 0,
            has_blocker=False,
        ),
    }
    watermark_report = check_inference_watermarks(
        _SIGNAL_DATE,
        is_trading_day=is_trading_day,
        required_dependencies=frozenset(watermarks),
        watermarks=watermarks,
    )
    assert watermark_report.status is WatermarkStatus.READY
    assert watermark_report.blockers == ()

    factor_members = (
        FactorInput(factor_id="alpha.close", factor_version=1, feature_name="close_px"),
        FactorInput(factor_id="alpha.volume", factor_version=1, feature_name="trade_vol"),
    )
    build_market_fingerprints: list[str] = []

    def build_batch() -> object:
        batch = adapter.fetch_market(_WINDOW_START, _SIGNAL_DATE)
        build_market_fingerprints.append(batch.schema_fingerprint)
        market = batch.frame
        observation_dates = pd.to_datetime(market["trade_date"], errors="raise").dt.date
        universe = pd.DataFrame(
            {
                "signal_date": [_SIGNAL_DATE] * len(market),
                "ts_code": market["ts_code"],
                "is_member": [True] * len(market),
            }
        )
        factors = pd.concat(
            [
                pd.DataFrame(
                    {
                        "signal_date": observation_dates,
                        "ts_code": market["ts_code"],
                        "factor_id": "alpha.close",
                        "factor_version": 1,
                        "value": market["close"].astype(float),
                        "available_date": observation_dates,
                    }
                ),
                pd.DataFrame(
                    {
                        "signal_date": observation_dates,
                        "ts_code": market["ts_code"],
                        "factor_id": "alpha.volume",
                        "factor_version": 1,
                        "value": market["vol"].astype(float),
                        "available_date": observation_dates,
                    }
                ),
            ],
            ignore_index=True,
        )
        frame = build_label_free_inference_frame(
            universe,
            factors,
            factor_members=factor_members,
            signal_date=_SIGNAL_DATE,
        )
        manifest = build_inference_manifest(
            frame,
            model_id=_MODEL_ID,
            universe_id=_UNIVERSE_ID,
            factor_set_id=_FACTOR_SET_ID,
            preprocessor_hash=_PREPROCESSOR_HASH,
            input_versions={
                "yeslab.stock_factor_daily": market_batch.watermark_id,
                "yeslab.trade_calendar": calendar_batch.watermark_id,
            },
        )
        assert not manifest.contains_labels
        return InferenceInputs(frame=frame, manifest_id=manifest.manifest_id)

    def load_artifact() -> object:
        if _sha256(spec_path) != approved_spec_digest:
            raise ValueError("model artifact changed during the daily flow")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        if (
            not isinstance(spec, dict)
            or spec.get("model_id") != _MODEL_ID
            or spec.get("model_version") != 1
        ):
            raise ValueError("model artifact identity does not match the approved pointer")
        return spec

    def predict(batch: object, artifact: object) -> object:
        assert isinstance(batch, InferenceInputs)
        frame = batch.frame
        ensemble_score = _percentile_rank(
            frame["close_px"].astype(float)
        ) + 0.001 * _percentile_rank(frame["trade_vol"].astype(float))
        return pd.DataFrame(
            {
                "signal_date": frame["signal_date"],
                "ts_code": frame["ts_code"],
                "ensemble_score": ensemble_score,
            }
        )

    def score_health(prediction: object) -> bool:
        assert isinstance(prediction, pd.DataFrame)
        report = evaluate_score_health(
            prediction,
            expected_key_count=market_batch.source_row_count,
        )
        return not report.blocking_reasons

    exports: list[CompatibilityExport] = []

    def publish(prediction: object) -> None:
        assert isinstance(prediction, pd.DataFrame)
        export = export_compatibility_predictions(
            prediction.rename(columns={"ensemble_score": "score"}),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=_SIGNAL_DATE,
            source_prediction_data_version=market_batch.watermark_id,
            artifact_root=v2_root,
            legacy_output_roots=(v1_root,),
        )
        if not verify_compatibility_export(export.manifest_path):
            raise ValueError("compatibility export failed manifest verification")
        exports.append(export)

    result = DailyInferencePipeline().run(
        _SIGNAL_DATE,
        watermark_report=watermark_report,
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )
    assert result.status is DailyPipelineStatus.SUCCESS
    assert result.failed_step is None and result.error_message is None
    assert len(exports) == 1

    # Assertion 1: approved V1 schema fingerprints and sentinel hashes unchanged.
    assert _v1_schema_fingerprints(v1_root) == fingerprints_before
    assert _v1_sentinel_hashes(v1_root) == sentinels_before
    # The flow saw the same V1 schema at watermark time and at build time.
    assert build_market_fingerprints == [market_batch.schema_fingerprint]

    # Assertion 2: zero V1 mutation calls; only approved read shapes were issued.
    assert session.mutation_attempts == []
    assert len(session.scripts) == 6
    assert any(script.startswith("schema(") for script in session.scripts)
    assert any(script.startswith("select ") for script in session.scripts)

    # Assertion 3: every write stayed inside the isolated V2 export tree.
    after = _snapshot_tree(tmp_path)
    changed_or_removed = {
        name for name, digest in before.items() if after.get(name) != digest
    }
    assert changed_or_removed == set()
    added = sorted(set(after) - set(before))
    assert added
    export_prefix = (v2_root / "exports").relative_to(tmp_path).as_posix() + "/"
    assert all(name.startswith(export_prefix) for name in added)
    export = exports[0]
    assert export.csv_path.is_relative_to(v2_root / "exports" / "v2")
    assert export.pkl_path.is_relative_to(v2_root / "exports" / "v2")
    assert verify_compatibility_export(export.manifest_path)


@_INTEGRATION_GATE
def test_compatibility_export_refuses_v1_destinations_and_requires_declared_v1_roots(
    tmp_path: Path,
) -> None:
    v1_root = _install_v1_snapshot(tmp_path)
    before = _snapshot_tree(tmp_path)
    predictions = pd.DataFrame(
        {
            "signal_date": [_SIGNAL_DATE, _SIGNAL_DATE],
            "ts_code": ["000001.SZ", "600000.SH"],
            "score": [0.7, 0.3],
        }
    )
    with pytest.raises(UnsafeExportPath):
        export_compatibility_predictions(
            predictions,
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=_SIGNAL_DATE,
            source_prediction_data_version="prediction-v1",
            artifact_root=v1_root / "outputs",
            legacy_output_roots=(v1_root,),
        )
    with pytest.raises(UnsafeExportPath, match="explicit V1 output roots"):
        export_compatibility_predictions(
            predictions,
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=_SIGNAL_DATE,
            source_prediction_data_version="prediction-v1",
            artifact_root=tmp_path / "v2-undeclared",
            legacy_output_roots=(),
        )
    assert _snapshot_tree(tmp_path) == before
    assert _v1_schema_fingerprints(v1_root) == _APPROVED_V1_SCHEMA_FINGERPRINTS
    assert _v1_sentinel_hashes(v1_root) == _APPROVED_V1_SENTINEL_HASHES


@_INTEGRATION_GATE
def test_v1_connection_guard_fails_closed_on_any_non_read_call(tmp_path: Path) -> None:
    v1_root = _install_v1_snapshot(tmp_path)
    session = RecordingYesLabSession(v1_root)
    v1_before_reads = _snapshot_tree(v1_root)
    mutation_scripts = (
        "tableInsert(loadTable('dfs://YesLab', 'stock_factor_daily'), newRows)",
        "update loadTable('dfs://YesLab', 'stock_factor_daily') set close = 0",
        "delete from loadTable('dfs://YesLab', 'stock_factor_daily') where ts_code = `600519.SH",
        "drop table dfs://YesLab.stock_factor_daily",
        "create table dfs://YesLab.shadow_copy as select * from stock_factor_daily",
        "append!(dfs://YesLab.stock_factor_daily, chunk)",
        "save dfs://YesLab.stock_factor_daily",
    )
    for script in mutation_scripts:
        with pytest.raises(V1MutationGuardError):
            session.query(script)
    assert session.mutation_attempts == list(mutation_scripts)
    frame = session.query(
        "select close from loadTable('dfs://YesLab', 'stock_factor_daily') order by ts_code"
    )
    assert isinstance(frame, pd.DataFrame)
    assert not frame.empty
    assert _snapshot_tree(v1_root) == v1_before_reads
    assert _v1_schema_fingerprints(v1_root) == _APPROVED_V1_SCHEMA_FINGERPRINTS
