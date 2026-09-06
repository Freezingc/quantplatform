# Factor golden sample: the eight cases C8.3 requires

Regenerate with `python scripts/compute_factor_golden_expected.py`. The expectations are
produced by the hand-written reference in that script, which imports nothing from
`research_v2.factors`, so the golden test compares the production pipeline against an
independent derivation rather than against a snapshot of itself.

Universe: 50 securities x 60 signal dates
(2025-01-02 .. 2025-10-10, two blocks of 30 business days).
The gap between the blocks is deliberate: a contiguous 60-day span cannot reach the
180-day staleness bound, so the stale case would otherwise require
lowering a threshold to fit the fixture.

Technical source rows: 18000. Expected clean: 17900.
Expected quarantine: 25.

| # | case | where | expected effect |
|---|---|---|---|
| 1 | daily technical missing | `alpha_dense`/`alpha_copy`/`alpha_mirror`, securities 10-14, day indices 20-24 | those cells are absent from clean; they are **not** forward-filled from the previous day |
| 2 | fundamental announced on a Friday | ann_date 2025-01-03 | available_date 2025-01-06, the next open day |
| 3 | 180-day stale | September block | those signal dates publish **nothing** for yeslab_910, because the latest version exceeds 180 days |
| 4 | revision announcement | ann_date 2025-01-17, revision `r2` | signal dates from its available date onward see `r2`; earlier dates keep `r1` |
| 5 | exact duplicate | `alpha_dup`, securities 0-4, day indices 0-4 | one published row, `dedup_resolution=IDENTICAL_COLLAPSE` |
| 6 | single-nonnull | `alpha_single`, securities 20-24, day indices 10-14 | the finite value is published, `dedup_resolution=SINGLE_NONNULL` |
| 7 | conflict | `alpha_conflict`, securities 30-34, day indices 30-34 | the whole group is quarantined as `CONFLICT_QUARANTINE` and nothing is published; no side is chosen |
| 8 | identical and sign-opposite factors | `alpha_copy` = `alpha_dense`, `alpha_mirror` = -`alpha_dense` | both reach |Spearman| of 1 against `alpha_dense` (the mirror measures 0.9999999999999999, since a rank correlation over 50 names accumulates float error), so no threshold the pipeline would use can separate either from its origin |

## Expected counts

- clean `dedup_resolution`: {'UNIQUE': 17850, 'IDENTICAL_COLLAPSE': 25, 'SINGLE_NONNULL': 25}
- quarantine `reason`: {'CONFLICT_QUARANTINE': 25}
- fundamental `source_revision_id`: {'r2': 900, 'r1': 500}
- fundamental `stale_days`: min 0, max 23

Columns compared by the golden test are the business-meaningful ones only. `created_at` is a
wall-clock timestamp and this path's `record_hash` derives from it, so neither is reproducible and
both are excluded.

**Correction (2026-09-07): this paragraph used to justify that exclusion by claiming it worked
"exactly as the B plan's market golden test excludes its own non-deterministic fields". There is no
such precedent.** `tests/golden/test_pit_market.py` exercises `clean_market_daily`, which runs one
layer *below* lineage stamping, so its frame has no `created_at`, `run_id`, `data_version` or
`record_hash` to exclude -- it excludes nothing, and `rg` for those four names in that file returns
zero hits.

The relevant comparison is with the market *materializer*, which sits at the same layer as
`build_technical_factor_batches` and reaches the opposite result: `materializers/market.py:23`
defines `_RUN_LINEAGE_COLUMNS = {"run_id", "data_version", "created_at"}` and `_hash_record`
(line 75-83) hashes the row *without* them, so the market `record_hash` is wall-clock and run-id
independent. That is pinned by
`test_market_batch_record_hashes_are_wall_clock_independent` and used as a real gate by
`scripts/rehearse_full_range_pipeline.py:137`, which rebuilds the market batch and asserts
`market_stable_rebuild`. `research_v2.factors.materializer._record_hash` hashes every column
including those three, which is why this path's hash is not reproducible.

So the exclusion above is correct for this path but it is a **difference from the market path, not
a shared convention** -- and the factor side is the one that is non-deterministic. The consequence
is that the rebuild-determinism guarantee the rehearsal asserts for market has no factor
counterpart, and cannot have one without changing what `record_hash` means here. That is recorded
as decision N in `docs/handoffs/2026-09-06-c-plan-audit-decision-list.md`; nothing about the hash
is changed by this note.
