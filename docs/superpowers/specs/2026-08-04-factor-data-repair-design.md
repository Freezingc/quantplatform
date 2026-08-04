# Factor Data Repair Overlay Design

Date: 2026-08-04  
Status: Approved design, implementation pending  
Scope: 47 source-qualified factor candidates, TRAIN 2020-01–2024-12 first, 2025 VALIDATION second

## Goal

Repair data-quality defects in the 47 factor candidates that already passed the
source-level audit without modifying any existing YesLab source table or
changing the active six-factor contract. The repaired result must be
reproducible, point-in-time safe, auditable, and eligible for a fresh RankIC
quality recheck.

The repair phase does not train a model, publish a new active factor version,
or read the 2026 locked-test data.

## Non-goals

- Do not rewrite or delete rows in `dfs://YesLab`.
- Do not change the schema of existing V2 tables.
- Do not change factor formulas merely to improve RankIC.
- Do not use validation or locked-test statistics to decide how TRAIN data is repaired.
- Do not automatically replace the frozen six-factor set.
- Do not treat a negative direction as a data defect.

## Current context

The current code already contains narrow helpers in
`research_v2/src/research_v2/factors/repair.py` for adding missing market rows,
reconstructing the approved `GP_022` and `GP_064` candidates, and filling only
non-finite source values. The new repair overlay generalizes the audit and
publication contract around these helpers; it does not weaken their
no-overwrite behavior.

The 47-factor scope is selected from the existing source-quality combined
scope. Candidate IDs are read from the quality snapshot rather than copied
into a second hard-coded registry. Every run records the resolved candidate
IDs and their source definition hashes.

## Design alternatives

### A. Read-only audit plus a new repair overlay (selected)

Read YesLab into bounded factor/month chunks, audit business keys and finite
values, resolve only evidence-backed defects, and materialize a new V2 repair
batch. The original source remains immutable and the repaired batch can be
compared, rechecked, or abandoned independently.

### B. Runtime-only repair

Deduplicate and fill values whenever RankIC or a dataset is requested. This has
low storage cost but produces run-dependent results, repeats expensive work,
and makes an audit trail difficult to reproduce.

### C. Full formula rebuild

Recompute every factor in the 47-factor scope from the calculator scripts. This
could recover missing values but combines data repair with formula changes and
has a much larger operational and correctness surface.

The overlay is selected because it preserves provenance, limits writes to new
objects, and separates repair evidence from formula redesign.

## Components

### 1. Read-only source auditor

Add a pure, DataFrame-level audit contract around the existing normalizers. For
each factor/month input it reports:

- expected trading-date coverage and observed source-date coverage;
- total rows, exact duplicate rows, duplicate business keys, and conflicting
  finite values;
- finite, non-finite, and missing-value counts;
- candidate overlap count, fillable count, remaining non-finite count, and
  overlap mismatch tolerances;
- source table, source factor name, formula/source definition hash, and the
  source watermark used for the read.

The source business key is `(trade_date, ts_code, factor_name)`. Exact
duplicate rows may be collapsed only when all non-key values are identical.
Multiple finite values for one business key are always quarantined. A null or
non-finite value never wins over a finite value.

### 2. Repair resolver

Repair candidates are resolved in this order:

1. a deterministic secondary market extract for missing market keys;
2. an approved calculator reconstruction for a known factor gap;
3. a source re-read with corrected date/table filtering;
4. no automatic repair when the evidence cannot prove the value and its
   availability date.

The resolver may add a missing key or fill a non-finite value, but it must not
overwrite an existing finite source value. A candidate is admissible only when
all of the following hold:

- candidate keys are unique;
- candidate values are finite;
- the candidate source and calculation definition are recorded;
- overlapping finite source values match within the declared absolute and
  relative tolerances;
- the candidate's observation date is not later than the signal date;
- all required market inputs are available without future information.

For technical factors, `available_date == observation_date == signal_date`.
Any repair that changes the factor's lag or availability rule is a new factor
version and is outside this data-only repair phase.

### 3. Repair materializer

Materialize one bounded factor/month repair batch through the existing
`build_technical_factor_batches` path. The output is a new
`factor_observation_clean` data version; it is never merged in place with the
old version. Non-admissible records go to a new repair quarantine object and
remain unavailable to RankIC evaluation.

The batch carries the existing lineage columns generated by the materializer,
plus a repair status in the repair audit record. The source watermark includes
the source table, factor, date range, repair method, candidate hash, and repair
rules version.

### 4. Repair audit persistence

Create new V2 tables rather than altering existing tables:

`factor_repair_audit` stores one summary row per factor/month/repair run with
factor ID/version, source table, partition, repair method, source and candidate
row counts, duplicate/conflict counts, fill counts, mismatch maxima, PIT
checks, decision status, rules version, and lineage identity.

`factor_repair_quarantine` stores bounded conflict or rejected-candidate rows
with the business key, factor, reason, source/candidate values, and lineage
identity. It is an audit object, not an input table.

The new tables are append-only. A latest repair manifest may point to the
current approved repair batch, but no existing table schema or historical row
is modified.

## Processing flow

1. Resolve the 47 factor IDs from the existing source-quality scope.
2. Resolve metadata and definition hashes from the read-only YesLab catalog.
3. Stream TRAIN months in bounded chunks; never retain the complete five-year
   raw panel in memory.
4. Run source-key, finite-value, coverage, duplicate/conflict, and PIT audits.
5. Apply only evidence-backed repairs and write rejected records to repair
   quarantine.
6. Publish a new repair batch only if all blocker checks pass for the complete
   requested partition.
7. Re-run source/PIT coverage and daily RankIC on the repaired TRAIN batch.
8. Repeat the same procedure for 2025 VALIDATION, with repair decisions fixed
   from the data-repair rules and TRAIN evidence.
9. Produce a comparison report: before repair, repaired, unresolved, and
   rejected counts plus quality changes.
10. Leave the active factor pointer and model registry unchanged.

## Blocking checks

The repair batch cannot become eligible when any blocker fails:

- source business keys are not unique after exact-duplicate collapse;
- any conflicting finite value remains unresolved;
- candidate overlap mismatch exceeds the declared tolerance;
- candidate keys or values are non-unique/non-finite;
- a technical observation violates same-day availability;
- a requested factor/month has no source evidence;
- the repaired clean batch contains duplicate production keys;
- the repair watermark or definition hash is missing;
- the published batch mixes partitions, factor definitions, or data versions.

Warnings such as a reduced but still above-threshold finite ratio are recorded
and do not bypass the existing factor-quality admission rules.

## Quality recheck and decision states

Repair status and factor-quality status remain separate:

- `REPAIR_AUDIT_PASS`
- `REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE`
- `REPAIR_BLOCKED`
- `REPAIRED_PENDING_QUALITY_RECHECK`

Only repaired factors with `REPAIR_AUDIT_PASS` may proceed to the existing
TRAIN/VALIDATION RankIC screen. They must still satisfy the current gates:

- absolute mean RankIC at least `0.02`;
- absolute ICIR at least `0.50`;
- at least 60 valid days;
- directional-day share at least `0.55`;
- fixed TRAIN direction remains valid in 2025;
- 5-lag HAC Benjamini–Hochberg q-value at most `0.10`;
- source coverage, PIT, and full-day similarity checks remain valid.

A repaired factor that still fails RankIC remains a predictive failure; repair
does not override the predictive gate.

## Testing strategy

Tests are written before implementation and cover pure functions first:

- exact duplicate collapse preserves one value;
- conflicting duplicate keys are quarantined and block publication;
- finite source values are never overwritten;
- non-finite source values are filled only from finite, matching candidates;
- candidate overlap mismatches block repair;
- candidate dates later than signal dates block repair;
- missing source evidence is reported without fabricating rows;
- mixed factor/month data versions cannot publish atomically;
- repair audit and quarantine rows contain deterministic lineage fields;
- a repaired batch can be re-evaluated by the existing source/PIT and RankIC
  evaluators without changing their contracts.

Integration tests use in-memory session doubles and the V2 test database only;
they do not read or write the 2026 locked-test scope.

## Acceptance criteria

The first implementation phase is complete only when:

1. The 47-factor TRAIN repair audit has a persisted status for every requested
   factor/month.
2. No existing YesLab or V2 source table schema was changed.
3. Every published repair batch passes all blocker checks and has a reproducible
   manifest.
4. All unresolved rows are visible in repair quarantine rather than silently
   dropped or imputed.
5. Repaired TRAIN/VALIDATION quality results are reported separately from the
   original results.
6. No model training starts and no 2026 locked-test data is read.

