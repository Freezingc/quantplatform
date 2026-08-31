# Daily Inference Runbook

## Scope

This runbook governs the fail-closed daily inference DAG for the V2 platform: watermark
gate, batch build, artifact load, prediction, score health, and atomic publish. It applies
to every trading day in `shadow` and `production` environments. It does not authorize
model changes, pointer switches, or label access; those follow `model-rollback.md`,
`locked-test-2026.md`, and the registry procedures.

## Owner and Roles

The `platform-operator` owns this runbook end to end: trigger, checks, retries, and
classification. The `data-owner` resolves source watermark and DQ blockers
(`source-delay.md`). The `risk-approver` owns drift review and any model change decision.
The `research-owner` is the final escalation for a missed SLO.

## Schedule and SLO

- Trigger: 19:30 Asia/Shanghai on trading days. The scheduler only triggers the DAG;
  business correctness is decided by the calendar and watermark guards.
- SLO: prediction publish completes by 20:00 Asia/Shanghai.
- Watermark retries: when the gate is BLOCKED, the DAG is re-triggered at 19:35, 19:45,
  and 20:00 (three retries). Each attempt re-captures all source watermarks from scratch.
- After the 20:00 attempt, the run is FAILED, an alert is recorded, and nothing is
  published. Never publish stale or partial data to meet the SLO.
- Non-trading days: the DAG returns `NOOP_NON_TRADING_DAY`. No publish, no alert, no
  manual action.

## DAG and Statuses

Execution order: watermark gate -> build_batch -> load_artifact -> predict ->
score_health -> end-watermark snapshot -> publish. Any failure returns `FAILED` with
`failed_step` set; the pipeline never skips ahead. A `BLOCKED` watermark gate aborts
before any step runs. The end-watermark snapshot is captured after score health and
must be `READY` with the same dependency fingerprint as the start snapshot; otherwise
publish is skipped and the run fails with `failed_step: watermark_end`.
When the start report contains a fingerprint, the caller must provide the end snapshot
factory; omitting it fails at `watermark` before any downstream callback runs.

Watermark blocker reasons: `MISSING:<dep>`, `UNPUBLISHED:<dep>`, `MISSING_VERSION:<dep>`,
`STALE:<dep>`, `DQ_BLOCKER:<dep>`, `IDENTITY_MISMATCH:<dep>`. Inference dependencies must
never include a label dependency; the gate rejects them.

## Execution

1. Before 19:30, confirm the active model pointer target for the environment. Read-only:

   ```bash
   qr-v2 registry pointer plan --consumer paper-sim --model-id sha256:<64-hex>
   ```

   Expected: JSON `checks` list (artifact, compatibility_export, latest_prediction,
   model_status, schema) and exit code 0. Backend status reads `BACKEND_UNAVAILABLE`
   until a pointer backend is configured; this is a read-only check and changes nothing.
   For production, replace the consumer with the configured production consumer id.

2. At 19:30 the scheduler triggers the DAG for today's signal date. Observe the pipeline
   result: `status`, `failed_step`, `error_message`.

3. If the run fails with `failed_step: watermark` (gate `BLOCKED`), re-trigger at
   19:35, 19:45, and 20:00. If any dependency watermark stays
   `UNPUBLISHED`/`STALE`/`DQ_BLOCKER`, follow `source-delay.md`. Never bypass the gate.

4. Success checks (all must hold):

   - Pipeline result is `SUCCESS` with `failed_step` empty.
   - The shadow daily report records `pipeline_status: SUCCESS` and
     `prediction_published: true`, with `latency_seconds` within `latency_slo_seconds`.
   - The notification audit shows no unresolved `ERROR` alert for this signal date
     (watermark, publish, score, drift, coverage keys).
   - `performance_status` in the daily report reads `LOCKED_NOT_COMPUTED`. Any other
     value is an incident: 2026 performance is never computed outside the locked test.

5. If a step failed, classify it (table below), handle per section, and rerun only after
   the root cause is fixed (Manual Rerun).

## Failure Classification

| failed_step | Meaning | First action |
| --- | --- | --- |
| `watermark` | Source watermark BLOCKED (see blocker reasons above) | `source-delay.md`; retry schedule 19:35/19:45/20:00 |
| `build_batch` | Label-free batch build failed | Escalate to `data-owner`; inspect source watermarks and contracts |
| `load_artifact` | Active model artifact missing or unreadable | Verify the artifact stores (dataset artifacts live under `QRV2_ARTIFACT_ROOT`); `model-rollback.md` if corrupt |
| `predict` | Inference compute failed (e.g., GPU unavailable) | Restore compute, then Manual Rerun; `model-rollback.md` GPU section |
| `score_health` | Score health gate rejected the prediction | Do not publish or bypass; investigate drift/score health with `risk-approver` |
| `watermark_end` | A dependency changed or failed its end-of-run snapshot check | Do not publish; investigate source mutation, then rerun with a fresh start snapshot |
| `publish` | Two-phase publish failed | Nothing became visible; fix DB/staging, then Manual Rerun |

A `score_health` rejection is a protection, not a nuisance: the prediction was not
published and must not be forced out.

## Manual Rerun

Preconditions: root cause fixed; same signal date; the day is still a trading day.
A rerun is a new run (new run id) that re-captures watermarks and re-executes the full
DAG; already successful content is never republished as a substitute, and a failed day
stays empty until a full success.

1. If a data step failed, repair the V2 data first. Dry-run the plan and check the
   target environment before any mutation:

   ```bash
   qr-v2 materialize market --start 2026-08-28 --end 2026-08-28 --dry-run
   qr-v2 materialize universe --start 2026-08-28 --end 2026-08-28 --dry-run
   ```

   Expected: plan JSON with `target_database: dfs://QuantResearchV2Test` and exit 0.
   Confirm the target is the V2Test database, never a V1 database. Without `--dry-run`
   the same command prints `runtime backend is not configured; refusing to materialize`
   and exits 2 until the runtime backend is configured; that refusal is expected
   fail-closed behavior, not an error to work around.

2. Verify any dataset artifact you depend on (read-only):

   ```bash
   qr-v2 dataset inspect --dataset-id <dataset-id>
   qr-v2 dataset verify --dataset-id <dataset-id>
   ```

   Expected: manifest JSON; `{"dataset_id": ..., "valid": true}` with exit 0. A
   `valid: false` (exit 1) means the artifact is corrupt: stop and follow
   `model-rollback.md`.

3. Re-trigger the DAG for the original signal date through the same scheduler/operator
   integration that runs the 19:30 schedule. The merged CLI does not expose a dedicated
   `inference` subcommand; do not invent an alternative trigger path.

4. After the rerun, repeat the success checks in Execution step 4.

## Forbidden Operations

- Never publish stale, partial, or previous-day predictions as today's output.
- Never bypass or loosen the watermark gate, score health gate, or publish validation.
- Never add a label dependency to inference watermarks.
- Never manually edit pointer, registry, report, or notification audit records; they are
  immutable, and mismatches are incidents.
- Never run pointer mutations without the dry-run/plan checks in this runbook; production
  pointer changes additionally require `--confirm-production` and signed approvals.
- Never put secret values in commands, tickets, or reports; reference environment
  variable names only (`QRV2_*`).

## Report Locations

- Notification audit root (AuditNotifier): `delivered/`, `pending/`, `records/`,
  `locks/`. Alerts are deduplicated by alert key, signal date, and severity; recoveries
  are recorded as `RESOLVED`.
- Shadow daily report JSON: `pipeline_status`, `prediction_published`, `latency_seconds`
  vs `latency_slo_seconds`, `universe_size`, `feature_coverage`, `score_health`,
  `seed_agreement`, `drift_alert_keys`, `psi_error_features`, and
  `performance_status: LOCKED_NOT_COMPUTED`.
- Pipeline result returned to the scheduler: `signal_date`, `status`, `failed_step`,
  `error_message`.

## Escalation Path

1. `platform-operator` classifies the failure and exhausts the retry schedule.
2. Watermark/source/DQ blockers -> `data-owner` (`source-delay.md`).
3. Compute/DB/disk/artifact failures -> `platform-operator` lead with
   `model-rollback.md` procedures.
4. PSI `ERROR` on three consecutive valid days -> drift review; no new promotions or
   pointer switches until a signed drift review approval is issued; the active model
   stays unchanged.
5. Missed 20:00 SLO -> notify consumers that no prediction exists for the day, then
   escalate to `research-owner` and `risk-approver`; record the incident with the failed
   step, blocker list, and run id.
