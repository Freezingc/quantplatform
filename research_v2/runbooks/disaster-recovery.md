# Disaster Recovery Runbook

## Scope

This runbook defines backup, restore, and drill procedures for the V2 platform.
Drills operate ONLY on V2Test snapshots (the `dfs://QuantResearchV2Test` database and
the V2-owned file roots). Any production disaster recovery requires signed approval and
a separately scheduled window; it is never performed as part of a drill. Owners:
`platform-operator` executes; `risk-approver` approves drill plans;
`production-risk-approver` approves any production restore.

## Protected Assets

V2-owned roots to back up (never V1 paths; V1 tables are read-only sources):

- Dataset artifact root (`QRV2_ARTIFACT_ROOT`, default `artifacts/datasets`), with
  immutable manifests and shard hashes.
- Compatibility export root (CLI plan target `artifacts/exports/v2`), with manifests
  and per-file SHA-256 hashes; exports never write outside the configured V2-owned
  roots.
- Registry and pointer state: pointer event history, model registry history, and the
  signed approval/evidence roots (shadow deployment approvals, shadow evidence,
  production approvals, drift review approvals). These are content-addressed, signed,
  append-only records.
- Notification audit root (`delivered/`, `pending/`, `records/`, `locks/`).
- Configuration under `research_v2/configs/` (monitoring policy, contracts, exports).

All records are verified by hash; a backup that cannot be hash-verified is not a backup.

## Backup Discipline

1. Backups are read-only snapshots: never mutate, re-sign, or repack signed records to
   make them fit a backup format.
2. Each backup carries its own manifest (contents, hashes, snapshot time, source
   environment label `V2Test`).
3. Retain backups of signed roots even when models are retired; audit history is
   append-only and must remain reconstructible.

## Restore Drill (V2Test Snapshots Only)

1. Announce the drill window and confirm the snapshot is labeled V2Test. Never load a
   snapshot of unknown provenance.
2. Restore the snapshot into an isolated V2Test environment. The restore target must be
   the V2Test database and scratch file roots only; check the target environment before
   starting and never point a drill at production paths or consumers.
3. Verify restored content with read-only commands:

   ```bash
   qr-v2 dataset inspect --dataset-id <dataset-id>
   qr-v2 dataset verify --dataset-id <dataset-id>
   ```

   Expected: manifest JSON; `{"dataset_id": ..., "valid": true}` with exit 0.

   ```bash
   qr-v2 export verify --manifest <restored-manifest-path>
   ```

   Expected: `{"manifest": ..., "valid": true}` with exit 0; exit 1 on hash mismatch,
   exit 2 on a malformed manifest.

   ```bash
   qr-v2 registry pointer plan --consumer paper-sim --model-id sha256:<64-hex>
   ```

   Expected: JSON `checks` list, exit 0. Backend checks read `BACKEND_UNAVAILABLE`
   until a pointer backend is configured; that is expected fail-closed behavior, not a
   drill failure.

4. Success criteria: every artifact, export, and manifest hash verifies; pointer and
   registry histories match the snapshot manifests; the drill wrote nothing outside the
   isolated V2Test environment.
5. Record drill evidence: date, snapshot manifest hashes, verification command outputs,
   anomalies, and the drill evidence hash. V2Test drill evidence feeds later deployment
   approvals, so the record must be exact.

## Production Disaster Recovery

1. Production DR is an incident, not a drill. Declare the incident and freeze changes:
   no pointer switches, promotions, or reruns until the plan is approved.
2. Obtain signed approval from the `production-risk-approver`
   (`QRV2_PRODUCTION_APPROVAL_HMAC_KEY`; reference the variable name only, never the
   value). The approval binds the restore to specific backup manifests and hashes.
3. Schedule a separate maintenance window. Production restore is never executed inline
   during the incident, never inside a drill window, and never without the approval
   above.
4. During the window: restore the approved backups into the production environment,
   re-run the hash verifications from the drill section, reconcile pointer and registry
   history, and drain the notification audit pending queue.
5. Before returning to service: run one full daily cycle per `daily-inference.md` and
   pass all success checks; notify consumers of the outage window and any missing
   prediction days. Missing days stay missing; never backfill them with stale data.
6. File the incident report: cause, approval reference, restored manifest hashes,
   verification outputs, and any data days lost.

## Forbidden Operations

- Never run a drill against production data, pointers, or consumers.
- Never restore a production environment without signed approval and a scheduled window.
- Never edit, re-sign, or reorder signed records (registry, pointer, approval, evidence,
  audit) during backup or restore.
- Never restore from a backup whose manifest hashes do not verify.
- Never fabricate missing production prediction days after a disaster; gaps are recorded
  and disclosed.
