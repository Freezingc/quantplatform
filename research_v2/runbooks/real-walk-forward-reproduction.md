# Real-Data Walk-Forward Reproduction (Criterion 303)

- Scope: reproduce the F01-F06 out-of-sample B0-B3 walk-forward evaluation that
  completion criterion 303 requires ("2014-2025 F01-F06 OOS 评价可复现，B0-B3 齐全").
- Driver: `research_v2/scripts/run_real_walk_forward.py`
- Verdict renderer: `research_v2/scripts/summarize_real_walk_forward.py`
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

## Failure modes

| Symptom | Meaning | Response |
|---|---|---|
| `PermissionError` / missing published version | A partition lacks a published version row | Fail closed by design; publish or backfill the missing partition first |
| DolphinDB memory errors during extraction | Segment too large for one query | Re-run after lowering extraction chunk size in the dataset builder; do not silently sample |
| Fold set rejected | Not all of F01-F06 requested | Request the full ordinary fold set; single-fold runs are not a criterion-303 evidence |
| Any gate FAIL | Baseline OOS evidence insufficient | Record the gate table in the report; do not claim criterion 303; investigate before any candidate evaluation |

## Interpretation rules

- The gates are evaluated on raw statistics without rounding; the report JSON is
  the source of truth, the markdown is a rendering.
- This baseline walk-forward is the reference run only. A complex candidate earns
  `WALK_FORWARD_PASSED` by additionally satisfying G4 (delta RankIC >= 0.005
  against this baseline path) through the E-plan candidate evaluation flow.
