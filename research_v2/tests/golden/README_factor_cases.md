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

Columns compared by the golden test are the business-meaningful ones only. `created_at` is a wall-clock timestamp and `record_hash` derives from it, so neither is reproducible and both are excluded, exactly as the B plan's market golden test excludes its own non-deterministic fields.
