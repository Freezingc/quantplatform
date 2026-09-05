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
