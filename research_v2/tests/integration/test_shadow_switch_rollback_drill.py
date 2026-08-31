"""F10.5 shadow switch/rollback drill: automated end-to-end proof in the V2Test seam.

Executes the incident sequence from plan task F10.5 entirely in-process:

    source delay -> recovery backfill -> 60-day shadow evidence ->
    shadow pointer switch -> score collapse injection -> pointer rollback ->
    post-rollback recovery

The drill follows the runbooks it is recorded in:

- ``runbooks/daily-inference.md`` (watermark gate, failure classification, reruns)
- ``runbooks/source-delay.md`` (late source handling, recovery and backfill)
- ``runbooks/model-rollback.md`` (restore-target verification, pointer rollback)

The hash of the drill log (``runbooks/drill-shadow-switch-rollback.md``) is bound
into the candidates' shadow deployment approvals as ``v2test_drill_hash``, the
rollback runbook hash as ``rollback_plan_hash``, and the daily inference runbook
hash as ``inference_contract_hash``. Calendar time is compressed: each model's
60-day shadow window is represented by its signed qualification evidence.

No DolphinDB server or network access is involved; every seam runs against
``tmp_path`` with fixed clocks and deterministic content.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from research_v2.datasets.builder import FactorInput
from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.inference.builder import build_label_free_inference_frame
from research_v2.inference.daily_pipeline import DailyInferencePipeline, DailyPipelineStatus
from research_v2.inference.manifest import InferenceManifest, build_inference_manifest
from research_v2.monitoring.score_health import ScoreHealthReport, evaluate_score_health
from research_v2.monitoring.shadow import (
    ShadowDeploymentApproval,
    ShadowDeploymentApprovalStore,
    ShadowEvidence,
    ShadowEvidenceStore,
    ShadowEvidenceVerifier,
    ShadowQualification,
    ShadowQualificationStatus,
)
from research_v2.monitoring.watermarks import (
    InferenceWatermark,
    WatermarkReport,
    WatermarkStatus,
    check_inference_watermarks,
)
from research_v2.registry.active_pointer import (
    ActiveModelPointerStore,
    PointerEligibilityGuard,
    PointerEventKind,
    PointerSwitchEvidence,
)
from research_v2.registry.model_registry import ModelRegistry, ModelStatus

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run the shadow switch/rollback drill",
    ),
]

_APPROVAL_KEY = b"f10-drill-approval-key"
_EVIDENCE_KEY = b"f10-drill-evidence-key"

_MODEL_A = "sha256:" + "a" * 64  # incumbent serving the V2Test shadow pointer
_MODEL_B = "sha256:" + "b" * 64  # candidate promoted after its 60-day shadow window
_CONSUMER = "paper-sim"
_ENVIRONMENT = "shadow"
_BASELINE_DATE = date(2026, 2, 27)
_DELAYED_DATE = date(2026, 3, 2)
_COLLAPSE_DATE = date(2026, 3, 3)
_UNIVERSE_ID = "universe:v2test:2026"
_FACTOR_SET_ID = "factor-set:v2test:2026"
_PREPROCESSOR_HASH = "sha256:" + "9" * 64
_TS_CODES = tuple(f"{code:06d}.SZ" for code in range(1, 13))
_MARKET_DEP = "yeslab.stock_factor_daily"
_UNIVERSE_DEP = "yeslab.universe_daily"
_FACTOR_MEMBERS = (
    FactorInput(factor_id="alpha.close", factor_version=1, feature_name="close_px"),
    FactorInput(factor_id="alpha.volume", factor_version=1, feature_name="trade_vol"),
)

_BASELINE_RUN_ID = "00000000-0000-0000-0000-000000000100"
_DELAY_RUN_ID = "00000000-0000-0000-0000-000000000101"
_RECOVERY_RUN_ID = "00000000-0000-0000-0000-000000000102"
_SWITCH_RUN_ID = "00000000-0000-0000-0000-000000000103"
_COLLAPSE_RUN_ID = "00000000-0000-0000-0000-000000000104"
_ROLLBACK_RUN_ID = "00000000-0000-0000-0000-000000000105"
_POST_ROLLBACK_RUN_ID = "00000000-0000-0000-0000-000000000106"

_RUNBOOK_DIR = Path(__file__).resolve().parents[2] / "runbooks"


@pytest.fixture(autouse=True)
def _drill_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "f10-drill-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", "f10-drill-evidence-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "f10-drill-shadow-deployer")


class _DrillClock:
    """Fixed, strictly increasing timestamps so the audit history is deterministic."""

    def __init__(self, *times: datetime) -> None:
        self._times = list(times)

    def __call__(self) -> datetime:
        if not self._times:
            raise AssertionError("drill clock exhausted")
        return self._times.pop(0)


class _PublishedSurface:
    """Two-phase publish: stage under a run-scoped directory, then move into place."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def publish(
        self,
        prediction: pd.DataFrame,
        *,
        signal_date: date,
        run_id: str,
        model_id: str,
        manifest: InferenceManifest,
    ) -> None:
        day_dir = self._root / signal_date.isoformat()
        if day_dir.exists():
            raise FileExistsError(f"prediction already published for {signal_date}")
        staging = self._root / f".staging-{run_id}"
        staging.mkdir(parents=True)
        content = prediction.to_csv(index=False).encode("utf-8")
        (staging / "predictions.csv").write_bytes(content)
        provenance = {
            "content_hash": hashlib.sha256(content).hexdigest(),
            "manifest_id": manifest.manifest_id,
            "model_id": model_id,
            "row_count": int(len(prediction)),
            "run_id": run_id,
            "signal_date": signal_date.isoformat(),
        }
        (staging / "provenance.json").write_text(json.dumps(provenance, sort_keys=True) + "\n")
        staging.replace(day_dir)


@dataclass
class _DrillWorld:
    registry: ModelRegistry
    store: ActiveModelPointerStore
    surface: _PublishedSurface
    artifacts_root: Path
    published_root: Path
    digests: dict[str, str]


def _runbook_hash(name: str) -> str:
    return hashlib.sha256((_RUNBOOK_DIR / name).read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_tree(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            path = Path(dirpath) / name
            entries[path.relative_to(root).as_posix()] = _sha256(path)
    return entries


def _gate_evidence() -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in range(2020, 2026)},
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(6)
        ),
    )


def _qualification_dates(start: date) -> tuple[date, ...]:
    return tuple(start + timedelta(days=index) for index in range(60))


def _qualification(model_id: str, dates: tuple[date, ...]) -> ShadowQualification:
    return ShadowQualification(
        model_id=model_id,
        trading_calendar_id="7" * 64,
        valid_day_count=len(dates),
        status=ShadowQualificationStatus.READY_FOR_REVIEW,
        max_consecutive_psi_errors=0,
        valid_signal_dates=dates,
        invalid_signal_dates=(),
        valid_report_ids=tuple(f"{index:064x}" for index in range(len(dates))),
        valid_runtime_day_ids=tuple(f"{index + 100:064x}" for index in range(len(dates))),
        invalid_runtime_day_ids=(),
    )


def _write_artifact(root: Path, model_id: str, weights: Mapping[str, float]) -> str:
    bundle = {"feature_weights": dict(weights), "model_id": model_id, "model_version": 1}
    payload = (json.dumps(bundle, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    (root / f"{digest}.json").write_bytes(payload)
    return digest


def _load_artifact(artifacts_root: Path, digests: Mapping[str, str], model_id: str) -> object:
    digest = digests[model_id]
    path = artifacts_root / f"{digest}.json"
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("model artifact bundle failed hash verification")
    bundle = json.loads(payload)
    if bundle["model_id"] != model_id or bundle["model_version"] != 1:
        raise ValueError("model artifact bundle identity mismatch")
    return bundle


def _predict(frame: pd.DataFrame, artifact: object) -> pd.DataFrame:
    weights: Mapping[str, object] = {}
    if isinstance(artifact, dict):
        raw = artifact.get("feature_weights")
        weights = raw if isinstance(raw, dict) else {}
    score = pd.Series(0.0, index=frame.index)
    for feature, weight in sorted(weights.items()):
        score = score + float(weight) * frame[feature].rank(method="average", pct=True)
    total = sum(abs(float(weight)) for weight in weights.values())
    if total > 0:
        score = score / total
    prediction = frame.loc[:, ["signal_date", "ts_code"]].copy()
    prediction["ensemble_score"] = score.to_numpy()
    return prediction


def _market_inputs(signal_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    universe = pd.DataFrame(
        {
            "signal_date": [signal_date] * len(_TS_CODES),
            "ts_code": list(_TS_CODES),
            "is_member": [True] * len(_TS_CODES),
        }
    )
    rows: list[dict[str, object]] = []
    for index, ts_code in enumerate(_TS_CODES):
        for factor_id, value in (
            ("alpha.close", 10.0 + 1.5 * index),
            ("alpha.volume", 1000.0 * (index + 1)),
        ):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": ts_code,
                    "factor_id": factor_id,
                    "factor_version": 1,
                    "value": value,
                    "available_date": signal_date,
                }
            )
    return universe, pd.DataFrame(rows)


def _watermark_report(
    signal_date: date, watermarks: Mapping[str, InferenceWatermark]
) -> WatermarkReport:
    return check_inference_watermarks(
        signal_date,
        is_trading_day=True,
        required_dependencies=frozenset(watermarks),
        watermarks=watermarks,
    )


def _current_watermarks(signal_date: date, stamp: str) -> dict[str, InferenceWatermark]:
    return {
        _MARKET_DEP: InferenceWatermark(_MARKET_DEP, signal_date, f"m-{stamp}", True, False),
        _UNIVERSE_DEP: InferenceWatermark(_UNIVERSE_DEP, signal_date, f"u-{stamp}", True, False),
    }


def _run_daily(
    world: _DrillWorld,
    signal_date: date,
    *,
    run_id: str,
    watermarks: Mapping[str, InferenceWatermark],
) -> tuple[object, dict[str, int], list[ScoreHealthReport]]:
    """Run one immutable DAG attempt; every callback does real work and is counted."""
    calls = {"build_batch": 0, "load_artifact": 0, "predict": 0, "score_health": 0, "publish": 0}
    health_reports: list[ScoreHealthReport] = []
    manifests: list[InferenceManifest] = []

    def build_batch() -> object:
        calls["build_batch"] += 1
        universe, factors = _market_inputs(signal_date)
        frame = build_label_free_inference_frame(
            universe,
            factors,
            factor_members=_FACTOR_MEMBERS,
            signal_date=signal_date,
        )
        pointer = world.store.get(_CONSUMER, _ENVIRONMENT)
        assert pointer is not None
        manifest = build_inference_manifest(
            frame,
            model_id=pointer.model_id,
            universe_id=_UNIVERSE_ID,
            factor_set_id=_FACTOR_SET_ID,
            preprocessor_hash=_PREPROCESSOR_HASH,
            input_versions={name: mark.data_version for name, mark in watermarks.items()},
        )
        assert not manifest.contains_labels
        manifests.append(manifest)
        return frame

    def load_artifact() -> object:
        calls["load_artifact"] += 1
        pointer = world.store.get(_CONSUMER, _ENVIRONMENT)
        assert pointer is not None
        return _load_artifact(world.artifacts_root, world.digests, pointer.model_id)

    def predict(batch: object, artifact: object) -> object:
        calls["predict"] += 1
        assert isinstance(batch, pd.DataFrame)
        return _predict(batch, artifact)

    def score_health(prediction: object) -> bool:
        calls["score_health"] += 1
        assert isinstance(prediction, pd.DataFrame)
        report = evaluate_score_health(prediction, expected_key_count=len(_TS_CODES))
        health_reports.append(report)
        return not report.blocking_reasons

    def publish(prediction: object) -> None:
        calls["publish"] += 1
        assert isinstance(prediction, pd.DataFrame)
        pointer = world.store.get(_CONSUMER, _ENVIRONMENT)
        assert pointer is not None
        world.surface.publish(
            prediction,
            signal_date=signal_date,
            run_id=run_id,
            model_id=pointer.model_id,
            manifest=manifests[-1],
        )

    result = DailyInferencePipeline().run(
        signal_date,
        watermark_report=_watermark_report(signal_date, watermarks),
        build_batch=build_batch,
        load_artifact=load_artifact,
        predict=predict,
        score_health=score_health,
        publish=publish,
    )
    return result, calls, health_reports


def _deploy_to_shadow(
    world: _DrillWorld,
    approvals_root: Path,
    evidence_root: Path,
    model_id: str,
    *,
    shadow_dates: tuple[date, ...],
) -> tuple[ShadowDeploymentApproval, ShadowEvidence]:
    """Full lifecycle: draft -> walk-forward -> signed approval -> SHADOW, then evidence."""
    world.registry.register_draft(model_id, actor="trainer")
    world.registry.promote_walk_forward(model_id, actor="reviewer", evidence=_gate_evidence())
    approval = ShadowDeploymentApprovalStore(approvals_root, signing_key=_APPROVAL_KEY).issue(
        model_id=model_id,
        actor="risk-approver",
        walk_forward_evidence_hash=world.registry.history(model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash=world.digests[model_id],
        inference_contract_hash=_runbook_hash("daily-inference.md"),
        v2test_drill_hash=_runbook_hash("drill-shadow-switch-rollback.md"),
        rollback_plan_hash=_runbook_hash("model-rollback.md"),
    )
    record = world.registry.enter_shadow(
        model_id, approval=approval, approval_root=approvals_root
    )
    assert record.status is ModelStatus.SHADOW
    evidence = ShadowEvidenceStore(evidence_root, signing_key=_EVIDENCE_KEY).issue(
        _qualification(model_id, shadow_dates)
    )
    return approval, evidence


def _shadow_pointer_evidence(
    approvals_root: Path,
    approval: ShadowDeploymentApproval,
    evidence_root: Path,
    evidence: ShadowEvidence,
) -> PointerSwitchEvidence:
    return PointerSwitchEvidence(
        shadow_approval=approval,
        shadow_approval_root=approvals_root,
        shadow_evidence=evidence,
        shadow_evidence_root=evidence_root,
    )


def test_shadow_switch_rollback_drill(tmp_path: Path) -> None:
    approvals_root = tmp_path / "shadow-approvals"
    evidence_root = tmp_path / "shadow-evidence"
    artifacts_root = tmp_path / "artifacts"
    artifacts_root.mkdir()
    published_root = tmp_path / "published"

    digest_a = _write_artifact(artifacts_root, _MODEL_A, {"close_px": 0.6, "trade_vol": 0.4})
    # Defect injected for the drill: candidate B's bundle ranks every stock identically.
    digest_b = _write_artifact(artifacts_root, _MODEL_B, {"close_px": 0.0, "trade_vol": 0.0})
    digests = {_MODEL_A: digest_a, _MODEL_B: digest_b}
    clock = _DrillClock(
        datetime(2026, 3, 2, 1, 0, tzinfo=UTC),  # pre-drill incumbent install
        datetime(2026, 3, 2, 12, 0, tzinfo=UTC),  # candidate pointer switch
        datetime(2026, 3, 3, 12, 30, tzinfo=UTC),  # incident rollback
    )
    registry = ModelRegistry()
    world = _DrillWorld(
        registry=registry,
        store=ActiveModelPointerStore(
            eligibility=PointerEligibilityGuard(registry, bundle_digest=digests.__getitem__),
            clock=clock,
        ),
        surface=_PublishedSurface(published_root),
        artifacts_root=artifacts_root,
        published_root=published_root,
        digests=digests,
    )

    # Pre-drill state: incumbent A completed its own 60-day shadow window and holds
    # the V2Test shadow pointer; the last healthy baseline day published under A.
    approval_a, evidence_a = _deploy_to_shadow(
        world,
        approvals_root,
        evidence_root,
        _MODEL_A,
        shadow_dates=_qualification_dates(date(2025, 10, 1)),
    )
    install = world.store.compare_and_swap(
        consumer_id=_CONSUMER,
        environment=_ENVIRONMENT,
        expected_current=None,
        model_id=_MODEL_A,
        model_version=1,
        actor="f10-drill-shadow-deployer",
        reason="pre-drill baseline: incumbent A serves V2Test shadow",
        evidence=_shadow_pointer_evidence(approvals_root, approval_a, evidence_root, evidence_a),
        run_id=_BASELINE_RUN_ID,
    )
    assert install is not None and install.pointer.pointer_version == 1
    incumbent_pointer = install.pointer
    baseline_result, baseline_calls, _ = _run_daily(
        world,
        _BASELINE_DATE,
        run_id=_BASELINE_RUN_ID,
        watermarks=_current_watermarks(_BASELINE_DATE, "20260227-1"),
    )
    assert baseline_result.status is DailyPipelineStatus.SUCCESS
    assert all(count == 1 for count in baseline_calls.values())
    baseline_surface = _snapshot_tree(published_root)
    assert f"{_BASELINE_DATE.isoformat()}/predictions.csv" in baseline_surface

    # Step 1 (source-delay.md: detection and late-source handling;
    # daily-inference.md step 3): the 19:30 attempt finds the market feed STALE and
    # the universe feed UNPUBLISHED; the gate blocks before any step runs.
    delayed_watermarks = {
        _MARKET_DEP: InferenceWatermark(_MARKET_DEP, _BASELINE_DATE, "m-20260227-1", True, False),
        _UNIVERSE_DEP: InferenceWatermark(
            _UNIVERSE_DEP, _DELAYED_DATE, "u-20260302-1", False, False
        ),
    }
    delayed_report = _watermark_report(_DELAYED_DATE, delayed_watermarks)
    assert delayed_report.status is WatermarkStatus.BLOCKED
    assert delayed_report.blockers == (
        f"STALE:{_MARKET_DEP}",
        f"UNPUBLISHED:{_UNIVERSE_DEP}",
    )
    surface_before_delay = _snapshot_tree(published_root)
    delayed_result, delayed_calls, delayed_health = _run_daily(
        world, _DELAYED_DATE, run_id=_DELAY_RUN_ID, watermarks=delayed_watermarks
    )
    assert delayed_result.status is DailyPipelineStatus.FAILED
    assert delayed_result.failed_step == "watermark"
    assert delayed_result.error_message == f"STALE:{_MARKET_DEP}; UNPUBLISHED:{_UNIVERSE_DEP}"
    assert delayed_calls == {
        "build_batch": 0,
        "load_artifact": 0,
        "predict": 0,
        "score_health": 0,
        "publish": 0,
    }
    assert delayed_health == []
    assert _snapshot_tree(published_root) == surface_before_delay  # nothing published
    assert not (published_root / _DELAYED_DATE.isoformat()).exists()

    # Step 2 (source-delay.md: recovery and backfill): the source publishes a clean
    # watermark; the SAME signal date is rerun as a NEW run with a NEW run id.
    assert _RECOVERY_RUN_ID != _DELAY_RUN_ID
    recovered_watermarks = _current_watermarks(_DELAYED_DATE, "20260302-1")
    recovered_report = _watermark_report(_DELAYED_DATE, recovered_watermarks)
    assert recovered_report.status is WatermarkStatus.READY
    assert recovered_report.blockers == ()
    recovery_result, recovery_calls, recovery_health = _run_daily(
        world, _DELAYED_DATE, run_id=_RECOVERY_RUN_ID, watermarks=recovered_watermarks
    )
    assert recovery_result.status is DailyPipelineStatus.SUCCESS
    assert recovery_result.failed_step is None and recovery_result.error_message is None
    assert all(count == 1 for count in recovery_calls.values())
    assert recovery_health and recovery_health[0].blocking_reasons == ()
    recovery_day = published_root / _DELAYED_DATE.isoformat()
    provenance = json.loads((recovery_day / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["run_id"] == _RECOVERY_RUN_ID  # fresh run, never copied predictions
    assert provenance["model_id"] == _MODEL_A
    assert provenance["row_count"] == len(_TS_CODES)
    assert provenance["content_hash"] == _sha256(recovery_day / "predictions.csv")

    # Step 3 (60-day shadow evidence): candidate B's signed qualification evidence —
    # 60 valid days, READY_FOR_REVIEW, zero consecutive PSI errors.
    approval_b, evidence_b = _deploy_to_shadow(
        world,
        approvals_root,
        evidence_root,
        _MODEL_B,
        shadow_dates=_qualification_dates(date(2026, 1, 1)),
    )
    assert registry.get(_MODEL_B).status is ModelStatus.SHADOW
    assert evidence_b.payload["model_id"] == _MODEL_B
    assert evidence_b.payload["valid_day_count"] == 60
    assert evidence_b.payload["max_consecutive_psi_errors"] == 0
    assert (
        evidence_b.payload["qualification_status"]
        == ShadowQualificationStatus.READY_FOR_REVIEW
    )
    ShadowEvidenceVerifier.from_environment(evidence_root).verify(evidence_b)

    # Step 4 (daily-inference.md pointer plan checks; model-rollback.md
    # preconditions): atomic compare-and-swap of the shadow pointer from A to B.
    surface_before_switch = _snapshot_tree(published_root)
    switch = world.store.compare_and_swap(
        consumer_id=_CONSUMER,
        environment=_ENVIRONMENT,
        expected_current=incumbent_pointer,
        model_id=_MODEL_B,
        model_version=1,
        actor="f10-drill-shadow-deployer",
        reason="F10.5 drill: deploy candidate B to V2Test shadow",
        evidence=_shadow_pointer_evidence(approvals_root, approval_b, evidence_root, evidence_b),
        run_id=_SWITCH_RUN_ID,
    )
    assert switch is not None
    assert switch.pointer.model_id == _MODEL_B
    assert switch.pointer.pointer_version == incumbent_pointer.pointer_version + 1 == 2
    assert switch.event.kind is PointerEventKind.SWITCH
    assert switch.event.shadow_approval_id == approval_b.approval_id
    assert switch.event.shadow_evidence_id == evidence_b.evidence_id
    assert switch.event.reverts_event_id is None
    assert world.store.get(_CONSUMER, _ENVIRONMENT) == switch.pointer
    # A stale compare-and-swap observed against the pre-switch pointer is rejected.
    assert (
        world.store.compare_and_swap(
            consumer_id=_CONSUMER,
            environment=_ENVIRONMENT,
            expected_current=incumbent_pointer,
            model_id=_MODEL_B,
            model_version=1,
            actor="f10-drill-shadow-deployer",
            reason="F10.5 drill: stale duplicate switch must be rejected",
            evidence=_shadow_pointer_evidence(
                approvals_root, approval_b, evidence_root, evidence_b
            ),
            run_id=_SWITCH_RUN_ID,
        )
        is None
    )
    # Pointer moves never publish.
    assert _snapshot_tree(published_root) == surface_before_switch

    # Step 5 (daily-inference.md: score_health failure classification): B's
    # degenerate bundle collapses every score; the gate rejects and nothing publishes.
    collapse_watermarks = _current_watermarks(_COLLAPSE_DATE, "20260303-1")
    surface_before_collapse = _snapshot_tree(published_root)
    collapse_result, collapse_calls, collapse_health = _run_daily(
        world, _COLLAPSE_DATE, run_id=_COLLAPSE_RUN_ID, watermarks=collapse_watermarks
    )
    assert collapse_result.status is DailyPipelineStatus.FAILED
    assert collapse_result.failed_step == "score_health"
    assert collapse_result.error_message == "score health gate rejected prediction"
    assert collapse_calls["publish"] == 0  # the gate fired before any publish
    assert len(collapse_health) == 1
    collapse_report = collapse_health[0]
    assert "SCORE_STD_COLLAPSE" in collapse_report.blocking_reasons
    assert "INSUFFICIENT_UNIQUE_SCORES" in collapse_report.blocking_reasons
    assert collapse_report.score_std == 0.0
    assert collapse_report.unique_score_count == 1
    assert _snapshot_tree(published_root) == surface_before_collapse  # nothing published
    assert not (published_root / _COLLAPSE_DATE.isoformat()).exists()

    # Step 6 (model-rollback.md): verify the restore target's artifact first, refuse
    # a rollback whose evidence chain cannot verify it, then roll back exactly the
    # SWITCH event that installed B.
    _load_artifact(world.artifacts_root, digests, _MODEL_A)  # restore target verifies
    rollback_evidence = _shadow_pointer_evidence(
        approvals_root, approval_a, evidence_root, evidence_a
    )
    del digests[_MODEL_A]
    with pytest.raises(ValueError, match="artifact"):
        world.store.rollback(
            consumer_id=_CONSUMER,
            environment=_ENVIRONMENT,
            expected_current=switch.pointer,
            event_id=switch.event.event_id,
            actor="incident-operator",
            reason="F10.5 drill: rollback with unverifiable A artifact must be refused",
            evidence=rollback_evidence,
            run_id=_ROLLBACK_RUN_ID,
        )
    digests[_MODEL_A] = digest_a
    surface_before_rollback = _snapshot_tree(published_root)
    rollback = world.store.rollback(
        consumer_id=_CONSUMER,
        environment=_ENVIRONMENT,
        expected_current=switch.pointer,
        event_id=switch.event.event_id,
        actor="incident-operator",
        reason="F10.5 drill: B score-health collapse; restore incumbent A",
        evidence=rollback_evidence,
        run_id=_ROLLBACK_RUN_ID,
    )
    assert rollback is not None
    assert rollback.pointer.model_id == _MODEL_A
    assert rollback.pointer.pointer_version == switch.pointer.pointer_version + 1 == 3
    assert rollback.event.kind is PointerEventKind.ROLLBACK
    assert rollback.event.reverts_event_id == switch.event.event_id
    assert world.store.get(_CONSUMER, _ENVIRONMENT) == rollback.pointer
    # A stale rollback observed against the pre-rollback pointer is rejected.
    assert (
        world.store.rollback(
            consumer_id=_CONSUMER,
            environment=_ENVIRONMENT,
            expected_current=switch.pointer,
            event_id=switch.event.event_id,
            actor="incident-operator",
            reason="F10.5 drill: stale duplicate rollback must be rejected",
            evidence=rollback_evidence,
            run_id=_ROLLBACK_RUN_ID,
        )
        is None
    )
    # Rollback moves pointers; it never publishes.
    assert _snapshot_tree(published_root) == surface_before_rollback

    # Step 7 (model-rollback.md step 4): the next daily run picks up the restored
    # model automatically and passes the score health gate.
    post_result, post_calls, post_health = _run_daily(
        world, _COLLAPSE_DATE, run_id=_POST_ROLLBACK_RUN_ID, watermarks=collapse_watermarks
    )
    assert post_result.status is DailyPipelineStatus.SUCCESS
    assert post_result.failed_step is None and post_result.error_message is None
    assert all(count == 1 for count in post_calls.values())
    assert post_health and post_health[0].blocking_reasons == ()
    assert post_health[0].unique_score_count == len(_TS_CODES)
    post_day = published_root / _COLLAPSE_DATE.isoformat()
    post_provenance = json.loads((post_day / "provenance.json").read_text(encoding="utf-8"))
    assert post_provenance["run_id"] == _POST_ROLLBACK_RUN_ID
    assert post_provenance["model_id"] == _MODEL_A
    assert post_provenance["content_hash"] == _sha256(post_day / "predictions.csv")

    # Audit reconciliation (model-rollback.md step 3 and audit recovery): the pointer
    # history is append-only with exactly the three events, and the registry recorded
    # both models' full lifecycle.
    history = world.store.history(_CONSUMER, _ENVIRONMENT)
    assert [event.kind for event in history] == [
        PointerEventKind.SWITCH,
        PointerEventKind.SWITCH,
        PointerEventKind.ROLLBACK,
    ]
    assert [event.new_pointer.pointer_version for event in history] == [1, 2, 3]
    assert history[0].new_pointer.model_id == _MODEL_A
    assert history[1].new_pointer.model_id == _MODEL_B
    assert history[2].new_pointer.model_id == _MODEL_A
    assert [event.run_id for event in history] == [
        _BASELINE_RUN_ID,
        _SWITCH_RUN_ID,
        _ROLLBACK_RUN_ID,
    ]
    assert [event.actor for event in history] == [
        "f10-drill-shadow-deployer",
        "f10-drill-shadow-deployer",
        "incident-operator",
    ]
    assert [transition.new_status for transition in registry.history(_MODEL_B)] == [
        ModelStatus.DRAFT,
        ModelStatus.WALK_FORWARD_PASSED,
        ModelStatus.SHADOW,
    ]
    # The published surface holds exactly the three successful days.
    assert sorted(published_root.iterdir()) == [
        published_root / _BASELINE_DATE.isoformat(),
        published_root / _DELAYED_DATE.isoformat(),
        published_root / _COLLAPSE_DATE.isoformat(),
    ]
