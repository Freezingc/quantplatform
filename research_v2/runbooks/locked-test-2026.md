# 2026 Locked-Test Runbook

## Scope

This runbook governs the only permitted labeled evaluation for the FINAL/TEST 2026 segment.
It applies to the frozen model, factor set, preprocessing artifact, label definition, and
evaluation code recorded in `LockedTestPreregistration`. It does not authorize tuning,
factor selection, retraining, backtesting variations, or per-day/per-security result access.

## Role Separation

`research-owner` submits the unlock request. `risk-approver` validates the signed SHADOW
evidence and approves it. `locked-test-evaluator` runs the isolated aggregate evaluation.
`production-risk-approver` independently approves a passed result. The requester cannot be
the unlock approver; the evaluator cannot be the production approver.

## Preconditions

Before requesting unlock, record an immutable preregistration containing:

- model ID, lineage ID, and exactly three seed artifact hashes;
- FINAL train and validation manifest hashes, factor-set ID, preprocessor hash, 2026 universe
  ID, label-definition hash, and evaluation-code hash;
- historical G1-G6 evidence requirement, command, and the required operator roles.

The registry must show `SHADOW`. The signed `shadow_evidence.json` must verify, identify the
same model, have `READY_FOR_REVIEW` status, and have fewer than three consecutive PSI errors.

The market-calendar service must show the sixth trading day after the final 2026 trading day.
The label watermark for that exact exit date must be published, adjusted-open stable,
reconciled, and DQ-passed. Calendar-day substitutions are not permitted.

## Execution

1. The `research-owner` submits a request with the issued preregistration and the verified
   maturity watermark. The authority records a signed immutable request.
2. The `risk-approver` verifies the request and SHADOW evidence. Approval changes the registry
   from `SHADOW` to `LOCKED_TEST_APPROVED` and issues one expiring bearer capability bound to
   the preregistration hash, model, lineage, and watermark.
3. The isolated evaluator records a durable recovery marker, then consumes the capability before
   opening any labeled FINAL/TEST input. Consumption records `LOCKED_EXPOSED` for the lineage
   before the evaluation adapter runs.
4. The evaluator may construct only the frozen FINAL/TEST artifact and return final aggregate
   RankIC, HAC t-statistic, baseline delta, valid-day count, and score-health pass ratio. Do not
   emit interim, daily, per-security, label, prediction, or PnL data.
5. A passing result requires RankIC at least `0.030`, HAC t-statistic at least `2.000`, delta
   RankIC at least `0.005`, and score-health pass ratio at least `0.95`. The registry becomes
   `LOCKED_TEST_PASSED`; any failed threshold becomes `LOCKED_TEST_FAILED`.
6. The `production-risk-approver` verifies the signed passed result, binds the approval to the
   exact artifact bundle hash, and moves the model to `PRODUCTION`. A production pointer switch
   must include this signed approval and match its model and artifact hash exactly.

## Failure Handling

- A request before maturity, with a dirty watermark, an altered preregistration, the same
  requester and approver, expired capability, or invalid signed evidence is denied and audited.
- A consumed capability is never retried. A second call is denied and audited.
- An evaluator exception, or a restart finding an unfinished recovery marker, records
  `LOCKED_TEST_FAILED`; do not rerun it or issue a replacement capability.
- Any label read marks the full lineage `LOCKED_EXPOSED`. Descendants adjusted after that read
  cannot claim 2026 as a locked out-of-sample test and must wait for a later fully mature year.
- Do not manually edit registry, request, capability, exposure, result, or approval files.
  They are content-addressed, signed, immutable records. Treat signature or record mismatch as
  an incident and preserve the files for audit.

## Evidence Locations

The configured authority root retains `requests/`, `capabilities/`, `consumed/`, `exposure/`,
and `audit/`. The evaluator root retains `results/`. The production approval root retains
`.issued/`. Retain all three roots with the model artifact bundle and the preregistration root.
