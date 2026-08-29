# Source Delay and DQ Failure Runbook

## Scope

This runbook governs late sources, schema drift, and duplicate conflicts on the daily
inference path. The watermark gate is fail-closed: a late or blocked source is never
published, and the pipeline produces no output rather than partial output. Owner:
`data-owner`, with `platform-operator` operating the retries and reruns.

## Detection

The daily DAG fails at the watermark gate with `failed_step: watermark` and blocker
reasons:

- `UNPUBLISHED:<dep>` — the source has not published its watermark for the signal date.
- `STALE:<dep>` — the source watermark is behind the signal date.
- `MISSING:<dep>` / `MISSING_VERSION:<dep>` — no watermark record or version.
- `DQ_BLOCKER:<dep>` — the source carries an unresolved data-quality blocker.

Read the blocker list from the pipeline result and the notification audit
(`watermark:*` alert keys). A late source shows `UNPUBLISHED`/`STALE` on the 19:30
attempt and the 19:35/19:45/20:00 retries (`daily-inference.md`).

## Late Source Handling

1. A late source is never published. Do not release the day's predictions from any
   other data, and do not publish a partial universe.
2. Never copy yesterday's predictions as today's output. A day without a complete,
   gate-passing run stays empty; consumers must see the gap, not stale data.
3. Run the standard retry schedule (19:35, 19:45, 20:00). Each retry re-captures all
   watermarks; nothing is queued behind the blocked source.
4. Contact the source owner and record the incident (source, dependency name, blocker
   reason, signal date, run id).
5. Use the Tushare adapter strictly to confirm differences between the expected and the
   delayed feed. It is a secondary reconciliation source with cached responses
   (`QRV2_TUSHARE_TOKEN`; never store the token value in tickets or runbooks). Tushare
   output is never ingested as primary data and never used to fabricate the missing day.
6. If the source recovers before the 20:00 attempt, the next retry proceeds normally.
   If it recovers after the SLO, run Recovery and Backfill below.

## Schema Drift Handling

Source extracts are bound to a schema fingerprint; the contract defines the expected
columns. When the observed schema no longer matches the contract:

1. Stop the pipeline for the affected dependency. Do not ingest the drifted extract and
   do not patch rows to make them fit the old contract.
2. Diff the observed schema (column names and types) against the contract and record
   the delta in the incident.
3. Update the source contract through the normal contract change process with the
   `data-owner`'s review. A schema drift incident always ends in a contract update or a
   confirmed upstream revert; it never ends in silent tolerance.
4. Re-extract and re-verify against the updated contract, then rerun the daily DAG for
   the original signal date (Recovery and Backfill).

## Duplicate Conflict Handling

1. Publish validation rejects duplicate business keys; a conflicting batch is never made
   visible. Treat the rejection as quarantine: the conflicting rows stay out of the
   published tables.
2. Do not merge, overwrite, or deduplicate the conflict by hand in the database.
3. Identify the conflicting extract (source, table, business keys, row counts) and the
   watermark that admitted it; record both in the incident.
4. Resolve at the source (upstream correction or contract-level decision), then re-run
   the affected step so validation admits exactly one consistent row per business key.

## Recovery and Backfill

After the source publishes a clean watermark, backfill runs as a NEW run with the
original signal date:

1. Confirm the watermark for the dependency now covers the original signal date
   (published, current version, no DQ blocker).
2. If materialization steps must be redone for the missed range, dry-run first and check
   the target environment:

   ```bash
   qr-v2 materialize market --start <signal-date> --end <signal-date> --dry-run
   qr-v2 materialize universe --start <signal-date> --end <signal-date> --dry-run
   ```

   Expected: plan JSON with `target_database: dfs://QuantResearchV2Test` and exit 0.
   Without `--dry-run` the command exits 2 with `runtime backend is not configured;
   refusing to materialize` until the runtime backend is configured; expected fail-closed
   behavior.

3. Re-trigger the daily DAG as a new run (new run id) for the original signal date.
   Never reuse the failed run id, and never backfill with a substituted later date.
4. Success criteria: pipeline `SUCCESS`, prediction published for the original signal
   date, notification audit shows the `RESOLVED` record for the alert, and the shadow
   daily report for that date shows `prediction_published: true`. A successful backfill
   can count once for that signal date in shadow qualification.

## Forbidden Operations

- Never publish a late, partial, or substituted prediction set.
- Never copy or roll forward yesterday's predictions.
- Never ingest Tushare data as primary data or to fill a missing day.
- Never hand-edit watermarks, contracts, staged batches, or published tables.
- Never silence or deduplicate a DQ blocker alert without resolving the blocker.

## Evidence Locations

- Notification audit root: `watermark:*` alerts, their `DEDUPLICATED` repeats, and the
  final `RESOLVED` record.
- Pipeline results for the failed attempts and the successful backfill run (status,
  `failed_step`, blocker list, run ids).
- Incident record: source, dependency, blocker reasons, schema delta (if any), contract
  change reference, and the backfill run id.
