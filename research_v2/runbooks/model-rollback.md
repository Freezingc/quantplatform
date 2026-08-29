# Model Rollback Runbook

## Scope

This runbook covers active model pointer rollback and the infrastructure failures that
trigger it: artifact verification failure, V2 database unavailability, GPU unavailability,
disk exhaustion, staging cleanup, and audit recovery. Owner: `platform-operator`
(`incident-operator` role during an incident); `risk-approver` approves any model change
decision. Pointer changes are atomic compare-and-swap events with append-only audit
history; nothing in this runbook edits those records.

## Preconditions

1. Identify the environment (`shadow` or `production`) and consumer id.
2. Read the current pointer state (read-only):

   ```bash
   qr-v2 registry pointer plan --consumer paper-sim --model-id sha256:<current-64-hex>
   ```

   Expected: JSON `checks` list, exit 0. Until a pointer backend is configured the
   checks read `BACKEND_UNAVAILABLE`; that refusal is expected fail-closed behavior.
3. From the pointer event history, note: the SWITCH `event_id` (UUID) that installed the
   current model, the current model id and model version, and the current pointer
   version. A rollback reverts exactly one current SWITCH event to its previous pointer;
   a switch with no previous pointer is not rollback-eligible.
4. Verify the restore target's artifact before attempting rollback (read-only):

   ```bash
   qr-v2 dataset verify --dataset-id <restore-target-dataset-id>
   ```

   Expected: `{"dataset_id": ..., "valid": true}` with exit 0.

## Pointer Rollback

1. Build the rollback request. `--expected-current` is the model the pointer must
   currently hold (the model being rolled back); `--expected-version` is its model
   version; `--expected-pointer-version` is the current pointer version; `--evidence` is
   a SHA-256 digest; `--run-id` is a UUID; the signature evidence must verify for the
   restore-target model and its artifact bundle.

   ```bash
   qr-v2 registry pointer rollback \
     --consumer paper-sim \
     --environment shadow \
     --event-id <switch-event-uuid> \
     --expected-current sha256:<current-model-64-hex> \
     --expected-version 1 \
     --expected-pointer-version <n> \
     --reason "<incident reference and cause>" \
     --evidence <sha256-64-hex> \
     --run-id <uuid>
   ```

   Behavior today: the CLI validates the full request, then prints
   `registry pointer backend is not configured; refusing to rollback` and exits 2 until
   a pointer backend is configured. Unlike `pointer switch`, `rollback` has no
   `--dry-run` flag; request validation happens inside the command itself, and backend
   configuration is required before any live rollback.
2. Production rollback adds `--environment production --confirm-production` and requires
   a signed production deployment approval bound to the restore-target model and its
   artifact (`QRV2_PRODUCTION_APPROVAL_HMAC_KEY` set; never write the key value
   anywhere). Shadow rollback requires the signed shadow deployment/evidence chain
   (`QRV2_SHADOW_APPROVAL_HMAC_KEY`, `QRV2_SHADOW_EVIDENCE_HMAC_KEY`).
3. After a successful backend-level rollback, confirm: a ROLLBACK event exists in the
   history with `reverts_event_id` equal to the reverted SWITCH event, the pointer now
   holds the restore-target model, and the pointer version incremented by one.
4. The next daily run picks up the restored model automatically. Watch the next run's
   score health and drift status before closing the incident.

## Artifact Verification Failure

Symptoms: `load_artifact` step fails, `dataset verify` returns `valid: false`, or a
rollback attempt is refused because the restore target's artifact digest cannot be
verified.

1. Stop. Never point any environment at an artifact that fails hash verification.
2. Identify which digest failed (dataset artifact shards vs model artifact bundle) with
   the read-only verify commands above.
3. Restore the artifact from the verified backup (`disaster-recovery.md`; V2Test
   snapshot or artifact backup under `QRV2_ARTIFACT_ROOT`), then re-verify hashes.
4. Only after verification passes, retry the rerun or rollback. If the restore target
   itself cannot be verified, escalate to `risk-approver`: choose a different eligible
   model through the normal switch procedure instead of forcing the rollback.

## V2 Database Temporarily Unavailable

1. Fail-closed is the intended behavior: publish fails, the step is `FAILED(publish)` or
   an earlier DB-dependent step, and nothing becomes visible. No prediction is partial.
2. Do not retry indefinitely; follow the daily retry schedule, then stop and page the
   database owner. Record run ids and error messages.
3. Pointer operations are refused while the backend is unavailable; do not attempt to
   apply pointer state by hand.
4. After the database recovers, rerun the daily DAG for the same signal date (new run).
   Visibility is staged: only a fully validated batch becomes visible, and the failed
   publish leaves no visible partial data behind.

## GPU Unavailable

1. The `predict` step fails; the run is `FAILED(predict)`. Nothing is published and the
   active pointer is unchanged.
2. Restore GPU capacity (or replace the node) with the infrastructure owner. There is no
   fallback mode to enable and no degraded prediction path; do not improvise one.
3. Rerun the daily DAG for the same signal date after recovery. If the outage extends
   past the 20:00 SLO, follow `daily-inference.md` escalation and notify consumers that
   no prediction exists for the day.

## Disk Full

1. Symptoms: publish, artifact write, or report write fails with I/O errors. Treat any
   partially written staging object as untrusted.
2. Free space only from V2-owned temporary/staging objects whose publish state is
   terminal (`REJECTED` or fully `PUBLISHED` and superseded). Check state before
   deleting anything.
3. Never delete or truncate audit, registry, pointer, approval, or evidence files to
   free space. If disk pressure threatens those roots, escalate immediately and stop
   new runs.
4. After freeing space, verify artifacts (read-only `dataset verify` / `export verify`),
   then rerun the failed day.

## Staging Cleanup

1. The two-phase publisher cleans staged objects itself after a terminal publish outcome;
   manual cleanup is the exception, not routine.
2. When manual cleanup is required, list staged objects and their publish state first;
   delete only objects in terminal state for superseded logical partitions. Never delete
   a staged object whose state is `CREATED`/`VALIDATED` for the current signal date
   while a run may still be active.
3. Never clean staging by dropping or truncating published target tables.
4. Record what was removed (object name, logical partition, state) in the incident.

## Audit Recovery

1. Notification audit: after a crash or restart, drain pending events; they replay with
   their original idempotency keys, so no alert duplicates and none lost. Pending
   entries without a matching delivered record are still outstanding — treat them as
   active alerts until delivered.
2. Pointer/registry history is append-only and content-addressed. If a record fails
   signature or hash verification, stop and declare an incident: preserve the files,
   restore the affected root from the verified backup (`disaster-recovery.md`), and
   re-verify. Never hand-edit, reorder, or delete audit records.
3. After recovery, reconcile: the latest pointer event matches the live pointer, the
   notification audit shows no orphan pending entries, and the next daily run completes
   the success checks in `daily-inference.md`.

## Forbidden Operations

- Never roll back by editing pointer state, registry records, or configuration files.
- Never roll back to a model whose artifact or signature evidence fails verification.
- Never run a production rollback without `--confirm-production` and the signed
  production approval for the restore target.
- Never retry a refused rollback with altered expected versions to "make it match";
  re-read the live pointer state first.
- Never delete audit, approval, or evidence files under any operational pressure.
