# Drill Log: Shadow Switch / Rollback (F10.5)

- Drill id: F10.5-V2TEST-2026-08-29
- Plan reference: `docs/superpowers/plans/2026-08-01-quant-research-v2-f-inference-monitoring-migration-plan.md`,
  Task F10.5
- Automated drill: `research_v2/tests/integration/test_shadow_switch_rollback_drill.py`
- Date executed: 2026-08-29
- Status: PASSED (automated drill green; tabletop CLI checks observed fail-closed)

## Purpose

Prove end to end, in the V2Test seam, that the incident sequence works exactly as the
runbooks describe: a late source never publishes, recovery backfills the original
signal date as a NEW run, a candidate with 60 days of clean shadow evidence can take
the shadow pointer through a signed approval chain, a score collapse is caught by the
score-health gate before publish, and the pointer rolls back atomically to the verified
restore target. Every step asserts both the expected result and that nothing was
published where publishing is forbidden.

Sequence: source delay -> recovery backfill -> 60-day shadow evidence -> shadow
pointer switch -> inject score collapse -> rollback -> post-rollback recovery.

## Environment

- Target environment: **V2Test only** (`dfs://QuantResearchV2Test`; consumer
  `paper-sim`, environment `shadow`). No production system, no DolphinDB server, no
  network: the automated drill runs in-process against `tmp_path` with fixed clocks;
  the CLI checks run against the unconfigured-backend, fail-closed CLI path.
- Python: `/data/zhangyudong/.conda/envs/quantyd/bin/python`; run tests from
  `research_v2/`.
- Trust-root variables referenced by name only (values never recorded):
  `QRV2_SHADOW_APPROVAL_HMAC_KEY`, `QRV2_SHADOW_EVIDENCE_HMAC_KEY`,
  `QRV2_SHADOW_DEPLOYER_ID` (set by the test fixture), `QRV2_ARTIFACT_ROOT`
  (tabletop sandbox only).
- Roles exercised: `research-owner`, `data-owner`, `platform-operator` (acting as
  `incident-operator` during rollback), `risk-approver` (approval signer in the
  automated drill).

## Sign-off

| Role | Responsibility in this drill | Name | Date | Sign-off |
| --- | --- | --- | --- | --- |
| Research (research-owner) | Shadow evidence criteria, score-collapse definition, escalation | | | |
| Data (data-owner) | Watermark blockers, recovery backfill correctness | | | |
| Operations (platform-operator / incident-operator) | Pointer switch, rollback, audit reconciliation | | | |

All three roles must sign before this drill is considered complete. Drill findings do
not authorize any production change.

## Commands

Automated drill (exact invocations, from `research_v2/`):

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  tests/integration/test_shadow_switch_rollback_drill.py -q
QRV2_INTEGRATION=1 /data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest \
  tests/integration/test_shadow_switch_rollback_drill.py -q
/data/zhangyudong/.conda/envs/quantyd/bin/python -m ruff check \
  tests/integration/test_shadow_switch_rollback_drill.py
```

Expected: the first command skips (exit 0, `1 skipped`) unless `QRV2_INTEGRATION=1`;
the second passes (exit 0, `1 passed`); ruff reports `All checks passed!`.

Tabletop CLI checks executed by the operators (read-only / fail-closed; observed
outputs below). The drill sandbox used `QRV2_ARTIFACT_ROOT=/tmp/f10-drill-2026-08-29/artifacts`
with a purpose-built restore-target artifact
(`e818ecca23c4c2a398a6c5e46510de973bada41b96529fd1b9379e5177ffa8f6`):

```bash
qr-v2 dataset inspect --dataset-id e818ecca23c4c2a398a6c5e46510de973bada41b96529fd1b9379e5177ffa8f6
qr-v2 dataset verify --dataset-id e818ecca23c4c2a398a6c5e46510de973bada41b96529fd1b9379e5177ffa8f6
qr-v2 registry pointer plan --consumer paper-sim --model-id sha256:<64-hex>
qr-v2 materialize market --start 2026-03-02 --end 2026-03-02 --dry-run
qr-v2 materialize universe --start 2026-03-02 --end 2026-03-02 --dry-run
qr-v2 registry pointer rollback --consumer paper-sim --environment shadow \
  --event-id <switch-event-uuid> --expected-current sha256:<64-hex> \
  --expected-version 1 --expected-pointer-version 2 \
  --reason "F10.5 drill: B score-health collapse; restore incumbent A" \
  --evidence <sha256-64-hex> --run-id <uuid>
```

## Drill steps

Automated steps 1-7 run inside the single pytest drill; per-step durations were
measured by an instrumented replay of the same seams and data (harness total 83.3 ms).
"Surface unchanged" means the hash snapshot of the entire published tree is
byte-identical before and after the step.

| # | Step (role) | Runbook reference | Command / seam | Expected | Observed | Duration |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Preconditions: verify restore-target artifact and read the pointer plan (operations) | `model-rollback.md` Preconditions | `qr-v2 dataset inspect/verify`; `qr-v2 registry pointer plan` | manifest JSON; `{"dataset_id": ..., "valid": true}` exit 0; plan JSON `checks` exit 0 | manifest printed; `valid: true` exit 0; plan printed with `BACKEND_UNAVAILABLE` checks, exit 0 | 3.6 s / 3.6 s / 3.6 s |
| 1 | Source delay: market feed STALE, universe feed UNPUBLISHED at 19:30 (data) | `source-delay.md` Detection/Late Source; `daily-inference.md` step 3 | automated drill step 1: `check_inference_watermarks` + `DailyInferencePipeline.run` | report BLOCKED with `STALE:yeslab.stock_factor_daily`, `UNPUBLISHED:yeslab.universe_daily`; run FAILED at `watermark`; zero downstream callbacks; published surface unchanged | PASS: FAILED at `watermark`, blocker list exact, no callback ran, no file created for 2026-03-02, surface byte-identical | 0.0 ms (gate aborts before any downstream work) |
| 2 | Recovery backfill: clean watermark, SAME signal date, NEW run id (data, operations) | `source-delay.md` Recovery and Backfill | automated drill step 2; operator dry-runs `qr-v2 materialize market/universe --dry-run` | watermarks READY; run SUCCESS; prediction published for 2026-03-02 bound to the recovery run id; content hash matches the published file | PASS: SUCCESS, provenance `run_id` = recovery run id, model A, 12 rows, content hash verified; dry-runs printed plans with `target_database: dfs://QuantResearchV2Test`, exit 0 | 18.3 ms (drill) + 3.6 s / 3.6 s (dry-runs) |
| 3 | 60-day shadow evidence for candidate B (research) | plan F10.5; shadow qualification seam (`monitoring/shadow.py`) | automated drill step 3: `ShadowEvidenceStore.issue` + `ShadowEvidenceVerifier` | qualification READY_FOR_REVIEW, 60 valid days, 0 consecutive PSI errors; signed evidence verifies | PASS: payload shows `valid_day_count: 60`, `max_consecutive_psi_errors: 0`, `READY_FOR_REVIEW`; verifier accepted the issued record | 2.2 ms |
| 4 | Shadow pointer switch A -> B (operations, risk-approver signature) | `daily-inference.md` pointer checks; `model-rollback.md` Preconditions | automated drill step 4: registry lifecycle + signed approval + `compare_and_swap` | model B reaches SHADOW via verified approval (drill-log hash bound as `v2test_drill_hash`); CAS switch succeeds, pointer_version 1 -> 2; stale duplicate CAS rejected; no publish | PASS: SWITCH event recorded with approval id and evidence id; duplicate CAS with stale `expected_current` returned None; surface byte-identical | 0.6 ms |
| 5 | Inject score collapse under B (research injects; gate detects) | `daily-inference.md` Failure Classification (`score_health`) | automated drill step 5: next daily run with B's degenerate artifact | run FAILED at `score_health`; gate reports `SCORE_STD_COLLAPSE` and `INSUFFICIENT_UNIQUE_SCORES`; publish never called; nothing published for 2026-03-03 | PASS: FAILED at `score_health` ("score health gate rejected prediction"), `score_std == 0.0`, `unique_score_count == 1`, surface byte-identical | 16.7 ms |
| 6 | Rollback B -> A (operations as incident-operator) | `model-rollback.md` Pointer Rollback | automated drill step 6: artifact re-verification + `store.rollback`; tabletop `qr-v2 registry pointer rollback` | rollback with an unverifiable restore-target artifact refused; valid rollback reverts exactly the SWITCH event, pointer_version 2 -> 3; stale duplicate rollback rejected; CLI without backend validates then refuses (exit 2); no publish | PASS: damaged-artifact rollback raised artifact verification error; rollback restored A with `reverts_event_id` = switch event; duplicate rollback returned None; CLI printed "registry pointer backend is not configured; refusing to rollback", exit 2; surface byte-identical | 0.7 ms (drill) + 3.4 s (CLI) |
| 7 | Post-rollback recovery: next run picks up restored model A (operations) | `model-rollback.md` step 4 | automated drill step 7 | rerun of 2026-03-03 SUCCESS under A; score health clean; fresh content bound to the post-rollback run id | PASS: SUCCESS, no blocking reasons, `unique_score_count == 12`, provenance bound to post-rollback run id and model A | 17.6 ms |
| 8 | Audit reconciliation (operations) | `model-rollback.md` step 3; Audit Recovery | automated drill assertions on pointer/registry history | pointer history exactly SWITCH(A), SWITCH(B), ROLLBACK(A) with versions 1/2/3 and correct actors/run ids; registry history DRAFT -> WALK_FORWARD_PASSED -> SHADOW; published surface holds exactly the three successful days | PASS: all audit assertions held | included above |

Automated drill totals: `QRV2_INTEGRATION=1` pytest run `1 passed`, exit 0 (≈2-3 s
wall time including collection); without the flag `1 skipped`, exit 0; ruff `All
checks passed!`.

## Evidence binding

The candidate's shadow deployment approval is signed over, among other inputs:

- `v2test_drill_hash` = SHA-256 of this drill log
  (`runbooks/drill-shadow-switch-rollback.md`),
- `rollback_plan_hash` = SHA-256 of `runbooks/model-rollback.md`,
- `inference_contract_hash` = SHA-256 of `runbooks/daily-inference.md`.

The automated drill computes these hashes from the files at run time, so any edit to
this drill log changes the approval issued during the drill; the report for F10.5
records the hash values observed at verification time.

## Improvement items

1. Per-phase duration emission: the automated drill asserts behavior but does not log
   wall-clock per phase; durations here came from an instrumented replay harness.
   Candidate follow-up: optional timing log lines gated behind the integration flag.
2. CLI startup dominates tabletop step latency (≈3.5 s per command, interpreter
   startup). Irrelevant to incident speed today, but noted for future CLI batching.
3. The CLI `pointer rollback` path has no `--dry-run`; request validation happens in
   the command and the backend refusal is the fail-closed stop. Keep this documented
   in `model-rollback.md` (it is), and revisit if a pointer backend is configured.
4. Extend the drill to a second consumer/environment pair once a production consumer
   exists; today only `paper-sim`/`shadow` is exercised.
5. Consider adding a PSI-breach variant of step 5 (score collapse via the PSI monitor
   rather than the score-health gate) once shadow PSI accounting is wired into the
   daily report flow.

## Production drill rule

This drill is valid for **V2Test only**. Any drill, switch, or rollback against a
production environment requires separate, explicit production-change approval, the
production approval trust root (`QRV2_PRODUCTION_APPROVAL_HMAC_KEY`), and
`--confirm-production` on pointer commands, scheduled in an approved window. Nothing
in this log authorizes production pointer mutations.

## Related runbooks

- `daily-inference.md` — daily DAG, watermark gate, failure classification, reruns
- `source-delay.md` — late sources, schema drift, duplicate conflicts, backfill
- `model-rollback.md` — pointer rollback and infrastructure failure handling
- `disaster-recovery.md` — backup/restore (V2Test snapshots only in drills)
- `locked-test-2026.md` — the only path that may touch 2026 labels (never in this drill)
