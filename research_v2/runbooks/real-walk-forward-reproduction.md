# Real-Data Walk-Forward Reproduction (Criterion 303)

- Scope: reproduce the F01-F06 out-of-sample B0-B3 walk-forward evaluation that
  completion criterion 303 requires ("2014-2025 F01-F06 OOS 评价可复现，B0-B3 齐全").
- Driver: `research_v2/scripts/run_real_walk_forward.py`
- Verdict renderer: `research_v2/scripts/summarize_real_walk_forward.py`
- Factor source: `--factor-source legacy|frozen`, **default `legacy`**. Everything in
  this runbook up to the "Frozen-set baseline" section describes the legacy path
  (`factor_observation_clean` + the 14 `WALK_FORWARD_FACTOR_IDS`), which is the
  criterion-303 evidence and stays bit-identical when the flag is omitted.
- Applies to: `dfs://QuantResearchV2Test` (V2Test seam). The same procedure applies
  to `dfs://QuantResearchV2` once production materialization exists, by changing
  `TARGET_DATABASE` in the driver.

## Purpose

Demonstrate that the full-range point-in-time tables (market_daily_clean ->
tradability_daily -> universe_daily -> label_daily -> factor_observation_clean)
produce a reproducible OOS evaluation for the six annual test folds 2020-2025, with
all four approved baselines (B0-B3) fitted and scored in every fold, and with the
G1-G6 gates evaluated on the concatenated OOS predictions.

## Prerequisites

1. Full-range PIT materialization present in the target database:
   `market_daily_clean`, `tradability_daily`, `universe_daily`, `label_daily`
   covering 2014-01-02 onward, and `factor_observation_clean` covering the TRAIN
   years (2014-2019) for every admitted factor (see
   `docs/reports/2026-09-01-train-years-recomputed-backfill-completion.md`).
2. `published_version` rows exist for `universe_daily`, `factor_observation_clean`
   and `label_daily`; the driver reads the current version per logical partition
   and fails closed when a partition has no published version.
3. The admitted factor set is encoded in `WALK_FORWARD_FACTOR_IDS` inside the
   driver. Update that tuple only through the factor admission process; factors
   whose replay verification failed must stay excluded.
4. Read access to `dfs://YesLab` `trade_calendar` (SSE open days, read-only).

## Environment

- Python: `/data/zhangyudong/.conda/envs/quantyd/bin/python`; run from `research_v2/`.
- DolphinDB connection via environment variables (defaults shown):
  `QRV2_DDB_HOST`, `QRV2_DDB_PORT=8848`, `QRV2_DDB_USERNAME`, `QRV2_DDB_PASSWORD`.
- The run is CPU-bound (ElasticNet grid and LightGBM grid on millions of rows);
  it uses all cores and roughly 10-16 GB RSS. Expect hours, not minutes: the
  reference 2026-09-02 six-fold run exceeded 10 hours wall time.
  Measured 2026-09-05: a full six-fold re-run with a warm fold cache took
  **14 h 29 m** and reproduced the criterion-303 evidence exactly.
- `ConvergenceWarning` and `LinAlgWarning` lines from sklearn/scipy are expected
  output of the B2 grid on raw factor scales; they are diagnostics, not failures.

## Reproduction steps

```bash
cd research_v2
nohup /data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/run_real_walk_forward.py \
  --folds F01,F02,F03,F04,F05,F06 \
  --report /tmp/real_walk_forward_report.json \
  > /tmp/walk_forward.log 2>&1 &
```

All six folds must be requested together: the OOS evaluation and the G1-G6 gates
are defined only on the complete ordinary fold set F01-F06 (the driver enforces
this via `_require_ordinary_fold_set`).

Progress signals while the run executes:

- Log lines `assembling datasets for <fold>` then
  `<fold>: train=... validation=... test=...` mark dataset assembly.
- The line `running walk-forward baselines` marks the start of the training phase,
  which prints nothing further until completion; a slowly growing count of
  sklearn warnings in the log is the expected heartbeat.
- Completion is signalled by the JSON report appearing and the final log line
  `walk-forward complete; report=...`.

## Verdict rendering

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/summarize_real_walk_forward.py \
  --report /tmp/real_walk_forward_report.json \
  --out docs/reports/<date>-real-walk-forward-criterion-303.md
```

The renderer checks four completeness conditions and prints the verdict:

- all G1-G6 gates satisfied (`all_passed`; G4 is NOT_APPLICABLE on the baseline
  path by construction);
- F01-F06 all present in the selection records;
- B0-B3 all fitted and validation-scored in every fold ("B0-B3 齐全");
- yearly mean RankIC present for every year 2020-2025.

Archive the rendered markdown under `docs/reports/` together with the raw JSON
(or a copy of it) as the criterion-303 evidence pair. The report JSON carries a
`data_versions` snapshot (per logical partition) identifying exactly which
published versions were consumed; if a report predates that field, capture the
`published_version` rows externally during the run and archive them alongside
(see `docs/reports/2026-09-02-real-walk-forward-data-versions.json`).

## Re-run consistency and candidate pre-gates

A baseline re-run must reproduce the archived criterion-303 evidence before it
may replace the trust root consumed by candidate training:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/verify_baseline_rerun_consistency.py \
  --report /data/zhangyudong/qrv2_research_artifacts/baseline_oos_report.json \
  --reference docs/reports/2026-09-04-real-walk-forward-criterion-303.json
```

Consistency alone is not sufficient for candidate consumption: the M1-M3
driver additionally requires the extended selection fields
(`invalid_day_count`, `unselected_reasons`), the fold-matched baseline
prediction artifacts, and the data-version snapshot. The schema gate checks
all of it fail-closed, including a `reconstruct_baseline_record` round-trip:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/verify_baseline_report_candidate_schema.py \
  --report /data/zhangyudong/qrv2_research_artifacts/baseline_oos_report.json
```

Only after both checks pass may `run_real_candidate_training.py --family M1`
start (the E-plan order guard then chains M2/M3 through `--prior-report`).

## Frozen-set baseline (the new trust root)

The frozen factor set (17 ids, `factor_set_frozen_v1.json`, frozen 2026-09-03) is the
admitted set for formal model training. Its factor values live in `factor_daily`
(isolated canonical replay), **not** in `factor_observation_clean`. Running the
baseline on it produces the trust root that formal candidates must be measured
against; criterion-303 remains the *legacy 14-factor* evidence and is not superseded
as a pipeline-reproducibility artifact.

### Prerequisites (both are fail-closed, and one fails silently — read this)

1. `factor_daily` must be **materialized for every year the folds need**. The approved
   folds (`datasets/splits.py::annual_folds`) start training in **2014**
   (F01 train = 2014-2018 ... F06 train = 2019-2023), so 2020-2025 alone is **not**
   enough. Verify coverage:

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python \
     scripts/verify_factor_daily_backfill.py --years 2014..2025
   ```

   Expected: every year reports `factors=17`, one `data_version`, `duplicates=0`,
   and `PASS`. The verifier recomputes each year's expected `data_version` from the
   materializer's window convention, so a wrong or mixed window fails loudly.
3. **Coverage gate** — G1-G6 do not measure how much of each day's cross-section
   survives assembly, so run this before accepting a frozen baseline as a trust root:

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/verify_frozen_fold_coverage.py
   ```

   It reports, per fold and role, the daily width of the complete-factor cross-section
   against the `minimum_cross_section` floor of 30, plus the train/test median-width
   ratio. It fails a **decision role** (`train` or `validation` — the roles whose
   cross-section determines a modelling decision; `test` only scores and is already
   governed by the ≥30-per-day validity rule that G6 reads) on two calibrated criteria:

   1. **days below the floor**, allowance **1**. A day-share threshold is deliberately
      not the gate, because 11 lost crash days inside a five-year train segment is a
      99.1% valid share and would pass the same 0.95 threshold G6 uses. The allowance is
      1 rather than 0 because the *accepted* criterion-303 baseline itself loses exactly
      one train day in F01–F04, and that day's width is 0: `yeslab_387` has no rows at
      all on **2017-10-09** in `factor_observation_clean` while the other 13 factors are
      normal (2,645–3,177 rows). A full-history scan of 2014–2025 — 2,919 trading days ×
      14 factors, **40,866 (factor, date) cells — finds exactly one absent cell**, so it
      is a single missing observation at source, not PIT `available_date` gating (that
      would suppress every factor equally) and not a holiday-calendar effect (the
      2016/2018/2019/2020 post-holiday first days are all complete). Repair that cell and
      the allowance should return to 0.
   2. **median cross-section width ≥ 1000** for a decision role.

   **Both constants are calibrated, not chosen.** A threshold taken only from the
   defective side is a verdict written in advance, not a gate: it has to admit the
   baseline already accepted, otherwise adopting it retroactively invalidates the trust
   root. `audit_legacy_train_coverage.py` measures the same quantity on the assembled
   frames criterion-303 actually consumed (`/tmp/walk_forward_cache_v1`, located by
   recomputing `fold_cache_key` from the currently published versions — no re-query, no
   re-assembly):

   | criterion | legacy-14 (accepted) | frozen-17 | separates? |
   |---|---|---|---|
   | train median cross-section | 1402 / 1615 / 1949 / 2464 / 2656 / 2794 | 310 / 343 / 419 / 513 / 600 / 673 | **yes**, 4.2–4.8× per fold, no overlap |
   | train days below floor | 1 / 1 / 1 / 1 / 0 / 0 | 15 / 15 / 4 / 0 / 0 / 0 | yes, but legacy is not 0 |
   | train/test median ratio | 0.53 / 0.58 / 0.60 / 0.69 / 0.70 / 0.74 | 0.52 / 0.49 / 0.60 / 0.66 / 0.77 / 0.74 | **no** — overlaps |

   The ratio is therefore **reported and deliberately not gated**: ≈0.5 is a property of
   the walk-forward design (a five-year train window dominated by early years, scored
   against one recent year over a universe that grew from ~2,700 to ~4,300 names), not of
   the frozen factor set, and any ratio threshold low enough to fail frozen-17 would also
   fail the accepted baseline. Re-verify the calibration any time a constant changes:

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python \
     scripts/audit_legacy_train_coverage.py --sources legacy --apply-coverage-gate
   ```

   Expected: `gate admits 18/18 measured legacy segments` and no `CALIBRATION INVALID`
   line. That runs `verify_frozen_fold_coverage.assess_coverage` — the same function, the
   same default constants — over the reference widths, so a constant that would reject
   the accepted baseline is caught before it is used to block a run. Measured 2026-09-06:
   **18/18 admitted**, while the frozen side fails **12 of 18** segments (all 6 train and
   all 6 validation; the 6 test segments pass, correctly).

   Why it exists: six of the frozen 17 compute `base ** exponent` where the base can be
   negative (`pct_chg`, or `tanh` of a 20-day change in volume/turnover), which is
   undefined for falling stocks. Their coverage therefore equals the market's advancing
   breadth — on 2015-08-24 only 2 of 1,466 liquid-universe stocks had all 17 factors,
   and on the two circuit-breaker days 2016-01-04/01-07 only 3 and 4. All six TEST years
   still pass G6 at 100% valid days, so **without this gate the defect is invisible**:
   a frozen baseline would clear G1-G6 and become the trust root while the model had
   never seen a crash day, fitted on a cross-section 35-56% narrower than the one it is
   scored on.

   The gate also detects incomplete materialization: a year with no rows shows up as
   ~244 days at width 0 for every fold whose train window includes it.
4. **Admission consistency** — confirm no admitted factor carries an unwaived blocking
   screening failure before building a trust root on the set:

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/verify_admission_consistency.py
   ```

   It cross-checks the contract's factor list against `factor_quality_snapshot` and fails
   closed on any non-PASS status in a `STRUCTURAL_COVERAGE` scope that has no explicit
   `--override <factor_id>`. Declaring a conflict in the contract is not a waiver; only a
   named override is, and an override matching nothing is reported so waivers do not rot.

   As of 2026-09-06 this reports FAIL on the frozen v1 set: seven admitted factors carry
   blocking structural-coverage failures in two distinct modes. Five (`yeslab_372/384/389/
   390/406`) plus the never-screened `yeslab_375` have low per-day coverage intrinsic to
   their formulas (`base ** exponent` with a sign-capable base). Two (`yeslab_398/464`)
   failed on missing periods in the YesLab source but measure 1212/1212 days and 60/60
   months on `factor_daily`, so the sanctioned replay source healed them and they need a
   re-screen rather than removal. Read a FAIL here as "the set needs an adjudicated
   decision", not as a broken tool.

   Note the screen ran against the YesLab source tables while this pipeline consumes
   `factor_daily`; that mismatch is why it both missed a real defect and reported two
   that do not apply. Re-screening against `factor_daily` is the correct basis.
5. **Per-factor coverage screen on the consumed source** — the authoritative per-factor
   evidence, and the fix for the source mismatch noted in step 4:

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/screen_factor_daily_coverage.py \
     --start 2014-01-01 --end 2019-12-31     # the early folds' train span
   /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/screen_factor_daily_coverage.py \
     --start 2020-01-01 --end 2024-12-31     # the historical screen's window, for comparison
   ```

   It recomputes the C6.2 metric set (day ratio, minimum monthly day ratio, median / p05 /
   minimum per-day coverage, duplicate business keys) with the C6.3 denominator — same-day
   `universe_daily` members, never "stocks appearing in the factor table" — and reuses the
   historical reason-code vocabulary so output is directly comparable.

   As of 2026-09-06 both windows give **PASS 11 / ELIGIBILITY_FAIL 6**, failing exactly
   `yeslab_372, 375, 384, 389, 390, 406` (medians 0.37-0.50 against the 0.70 gate, minima
   down to 0.0025 on crash days), while all 17 show day ratio and minimum monthly day ratio
   of exactly 1.0000 with zero duplicate keys. Relative to the historical YesLab-based
   screen this adds `yeslab_375` (never screened before) and clears `yeslab_398` /
   `yeslab_464` (their source-side missing periods do not exist in `factor_daily`).

**Single command** — `preflight_frozen_baseline.py` runs all of the above in dependency
order and refuses to clear Task 3.1 unless each passes or carries a reasoned waiver:

```bash
/data/zhangyudong/.conda/envs/quantyd/bin/python scripts/preflight_frozen_baseline.py
# only with an accountable decision reference:
/data/zhangyudong/.conda/envs/quantyd/bin/python scripts/preflight_frozen_baseline.py \
  --waive admission_consistency --waive factor_coverage_screen --waive fold_coverage \
  --waiver-reason "<decision doc + date + which option was chosen>"
```

Order is deliberate: data completeness first, because a missing year makes every later
number meaningless, and the run stops there rather than reporting on incomplete data.
Then governance, then per-factor coverage on the consumed source, then the per-fold joint
cross-section. A waiver must name a real gate and carry a non-empty reason; a waiver on a
gate that passes is reported as stale so waivers do not accumulate and silently mask a
later failure of the same gate.

**How the three gates differ** — they are complementary, not redundant:

| Script | Unit of analysis | Question it answers |
|---|---|---|
| `screen_factor_daily_coverage.py` | one factor, over a window | Is this factor's own coverage admissible under C6.2 on the source we consume? |
| `verify_frozen_fold_coverage.py` | one (fold, role) segment, per day | Will the *joint* all-factor cross-section be wide enough to train and score on? |
| `verify_admission_consistency.py` | the admitted set vs its screening history | Was anything with a recorded blocking failure admitted without a waiver? |
| `audit_legacy_train_coverage.py` | the accepted baseline's own assembled frames | Are the coverage gate's constants calibrated so it admits what it was calibrated on? |
2. `published_version` must carry **144 monthly partitions** for `factor_daily`
   (2014-01 .. 2025-12). Register with the registration-only publisher (it never
   writes factor rows, and re-running is idempotent):

   ```bash
   /data/zhangyudong/.conda/envs/quantyd/bin/python \
     scripts/publish_factor_daily_versions.py --months 2014-01..2025-12          # dry run
   /data/zhangyudong/.conda/envs/quantyd/bin/python \
     scripts/publish_factor_daily_versions.py --months 2014-01..2025-12 --execute
   ```

   **Silent-failure warning:** assembly filters rows through `filter_current`, which
   *drops* any signal date whose month has no registered version. A missing year
   therefore yields an empty or partial TRAIN frame with **no error**. Always confirm
   the assembly log line `<fold>: train=... validation=... test=...` shows plausible
   magnitudes (F01 train is on the order of 2 million rows) before trusting a run.

### Path isolation is by subdirectory, not by filename

`save_baseline_oos_artifacts` writes to `args.report.parent / "baseline_oos"` — the
directory **name is hardcoded**. So a frozen report placed beside the legacy one
would overwrite `qrv2_research_artifacts/baseline_oos/F01..F06.parquet`, which the
legacy M2/M3 consume via `--baseline-artifacts`, silently feeding frozen baseline
predictions to legacy candidates and corrupting their G4 delta. Put the frozen report
in its own subdirectory:

```bash
cd research_v2
mkdir -p /data/zhangyudong/qrv2_research_artifacts/frozen
setsid nohup /data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/run_real_walk_forward.py \
  --factor-source frozen \
  --folds F01,F02,F03,F04,F05,F06 \
  --report /data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos_report.json \
  > /tmp/frozen_baseline.log 2>&1 &
```

Artifacts then land in `/data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos/`,
disjoint from the legacy chain by construction. The same rule applies to frozen
candidate workspaces: keep them under `frozen/` (e.g. `frozen/m1`,
`frozen/m1_report.json`).

The first log line states the resolved source and must read
`factor source: frozen (17 factors, tables=universe_daily,factor_daily,label_daily)`.
On the legacy path it must read `legacy (14 factors, ...factor_observation_clean...)`.
That line is the cheap check that the intended source was actually used.

### Verdict, consistency and candidate gate for the frozen root

```bash
# Task 3.2 verdict (a new name; criterion-303 keeps its own)
/data/zhangyudong/.conda/envs/quantyd/bin/python scripts/summarize_real_walk_forward.py \
  --report /data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos_report.json \
  --out docs/reports/<date>-frozen-set-baseline-walk-forward.md

# Task 4.1: for FROZEN re-runs the reference is the frozen baseline report, not 303
/data/zhangyudong/.conda/envs/quantyd/bin/python scripts/verify_baseline_rerun_consistency.py \
  --report <frozen re-run report> \
  --reference /data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos_report.json

# Task 3.3 candidate-consumption schema gate (same gate as legacy; same schema)
/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/verify_baseline_report_candidate_schema.py \
  --report /data/zhangyudong/qrv2_research_artifacts/frozen/baseline_oos_report.json
```

Frozen candidates then start with
`run_real_candidate_training.py --family M1 --factor-source frozen
--baseline-artifacts .../frozen/baseline_oos --baseline-report .../frozen/baseline_oos_report.json
--workspace .../frozen/m1 --report .../frozen/m1_report.json`.
**The `--factor-source` of a candidate run must match the baseline report it consumes**;
a frozen candidate measured against a legacy baseline (or vice versa) makes G4
meaningless because the two are not fold-matched on the same factor set.

### Known reproducibility caveat on re-materialization

Materialized `factor_daily` rows are pinned by `data_version` once written, so
downstream assembly is deterministic. **Re-materializing** the same window is not
bit-reproducible at the margin: the materializer keeps a row only when
`np.isfinite(value)` holds, and for GP-style factors with long replay lookbacks
(e.g. `yeslab_406` / GP_037, `replay_lookback_days=180`) a near-zero denominator can
flip a marginal cell between finite and infinite depending on BLAS reduction order,
i.e. on host load. Observed once: a year stored one row more than its own earlier
dry-run predicted, with the other 16 factors exact, the `data_version` exact, and
zero duplicate keys. The next year showed the same effect in **both** directions at
once (`yeslab_372` +1 row, `yeslab_389` -1 row) with the year total **identical** to
its prediction — i.e. marginal cells flip finiteness symmetrically with no systematic
bias, rather than rows being gained or lost. Both years' stored `data_version` values
equalled the values predicted from the window convention before the write.
`verify_factor_daily_backfill.py` therefore keeps the version,
factor-set and duplicate-key gates exact and compares row counts within
`+/-max(4, 0.002%)`, printing any delta rather than swallowing it.

## Failure modes

| Symptom | Meaning | Response |
|---|---|---|
| `PermissionError` / missing published version | A partition lacks a published version row | Fail closed by design; publish or backfill the missing partition first |
| DolphinDB memory errors during extraction | Segment too large for one query | Re-run after lowering extraction chunk size in the dataset builder; do not silently sample |
| Fold set rejected | Not all of F01-F06 requested | Request the full ordinary fold set; single-fold runs are not a criterion-303 evidence |
| Frozen path: a fold's `train=` count is 0 or far below ~2-3 million | A train year is not materialized or not registered in `factor_daily`; `filter_current` drops unregistered months silently | Stop. Run `verify_factor_daily_backfill.py --years 2014..2025` and re-check the 144 `published_version` partitions; never evaluate on a partial train segment |
| Frozen path assembles full-size frames but the baseline trains on far fewer rows | This driver drops rows unless **every** factor is finite (`dropna(subset=feature_order)`, lines 177-179) while `run_real_candidate_training.py` keeps NaN. The frozen 17 contain six half-covered factors (`yeslab_372/375/384/389/390/406`, ~50-66% of the liquid universe), so all-17 survival is ~12-35% versus ~80-95% for the legacy 14 | Stop before treating the run as a trust root. Measure survival first: count `(signal_date, ts_code)` pairs having all N factors versus all pairs. A baseline fitted on a selected subsample is not comparable to candidates scored on the full cross-section, and it makes the G4 delta both non-fold-matched and too easy to pass |
| Frozen path produced results but G4 looks meaningless | Candidate `--factor-source` did not match the baseline report's factor set | Re-run with matching sources; a cross-source delta is not fold-matched evidence |
| Any gate FAIL | Baseline OOS evidence insufficient | Record the gate table in the report; do not claim criterion 303; investigate before any candidate evaluation |

## Interpretation rules

- The gates are evaluated on raw statistics without rounding; the report JSON is
  the source of truth, the markdown is a rendering.
- This baseline walk-forward is the reference run only. A complex candidate earns
  `WALK_FORWARD_PASSED` by additionally satisfying G4 (delta RankIC >= 0.005
  against this baseline path) through the E-plan candidate evaluation flow.
