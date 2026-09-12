# qrv2 reproduction artifacts

Frozen evidence for the paper's reproduction appendix: the M0R candidate
walk-forward chain (`m0r_v2`) plus the on-chain / standalone baseline
predictions it is compared against (`baseline_v2`, `baseline_oos`).

Purpose: let a third party recompute the reported RankIC / ICIR / HAC-t numbers
from the published score frames, and settle the `+0.0022` rebuild-anchor gap
without re-running training.

- Source directory: `/data/zhangyudong/qrv2_research_artifacts/`
- Copied: 2026-09-12
- Size: 126 MB, 67 files (+ this README and `MANIFEST.sha256`)
- Integrity: `sha256sum -c MANIFEST.sha256` (all 67 files verified PASS)

## What is included / excluded

| Included | Excluded (deliberately) |
|---|---|
| `m0r_v2/predictions/**` — 8 score frames + manifests | `m0r_v2/datasets/` — 1.4 GB materialized train/val/test parquet |
| `m0r_v2/models/**` — cards, configs, feature order, signatures, checksums, validation predictions | `m0r_v2/checkpoints/**` — `booster.txt` LGBM weight dumps |
| `m0r_v2/report.json`, `verdict.md`, `promotion_evidence.json` | `m0r_v2/models/*/seed-*.pt` — trained weights |
| `m0r_v2/selections/**` — baseline & model selection evidence records | `secrets/` — HMAC signing key (never publishable) |
| `baseline_v2/**`, `baseline_oos/**` — baseline score frames + reports | all other run directories (`m0_*`, `m1_*`, `archive_rerun2`) |

Consequence: `models/*/checksums.sha256` lists `seed-*.pt` lines that cannot be
re-verified from this branch. The remaining lines (config / feature_order /
model_card / preprocessing / signature / validation_predictions) do verify.
`manifest.json → signature` is an HMAC over the payload; the key is not
published, so only `manifest.json → sha256` is independently checkable
(all 8 verified PASS).

## Layout

```
m0r_v2/
  report.json                 # per-fold best_config, search_results, OOS block, G1-G6 gates
  verdict.md                  # human-readable walk-forward report (WALK_FORWARD_PASSED)
  promotion_evidence.json     # promotion evidence bundle
  predictions/<prediction_id>/
      manifest.json           # kind, fold_id, model_id, dataset_id, selection_id, sha256, signature
      scores.parquet          # signal_date:date, ts_code:str, score:float64
  models/m0r-f{1..4}/
      model_card.json config.json feature_order.json preprocessing.json
      signature.json checksums.sha256
      validation_predictions.parquet   # signal_date, ts_code, percentile_rank
  selections/{baseline,model}/<evidence_id>.json/record.json
baseline_v2/
  report.json                 # folds, data_versions, selections, oos, environment, assembly_coverage
  report_gates_fixed.json     # OOS block re-evaluated with derived required_years (2022-2025)
  baseline_oos/F{1..4}.parquet  # signal_date, ts_code, score, model_id, fold_id, dataset_id
baseline_oos/
  F{01..06}.parquet           # DIFFERENT run — see warning below
```

## The 8 prediction frames (m0r_v2)

`kind=candidate` are the M0R ensemble scores; `kind=best_simple` are the
per-fold selected baseline (B2 for F1-F3, B3 for F4).

| kind | fold | model | rows | days | codes | test window |
|---|---|---|---|---|---|---|
| best_simple | F1 | B2 | 747,461 | 242 | 4,003 | 2022-01-04..2022-12-30 |
| best_simple | F2 | B2 | 820,829 | 242 | 4,652 | 2023-01-03..2023-12-29 |
| best_simple | F3 | B2 | 924,499 | 242 | 4,875 | 2024-01-02..2024-12-31 |
| best_simple | F4 | B3 | 925,174 | 243 | 4,906 | 2025-01-02..2025-12-31 |
| candidate | F1 | m0r-f1 | 804,575 | 242 | 4,342 | 2022-01-04..2022-12-30 |
| candidate | F2 | m0r-f2 | 882,426 | 242 | 4,716 | 2023-01-03..2023-12-29 |
| candidate | F3 | m0r-f3 | 932,167 | 242 | 4,883 | 2024-01-02..2024-12-31 |
| candidate | F4 | m0r-f4 | 938,594 | 243 | 4,912 | 2025-01-02..2025-12-31 |

Note the candidate frames carry more rows than the baseline frames for the same
fold: the baseline frames hold post-`dropna` *complete* rows (they equal
`assembly_coverage.folds.*.test.complete_rows`), the candidate frames hold the
pre-`dropna` assembled rows (they equal `...test.rows`). Join on
`(signal_date, ts_code)` and restrict to the baseline key set before computing
RankIC, or the two sides are not comparable.

## WARNING — `baseline_oos/` is not interchangeable with `baseline_v2/`

`baseline_oos/F01..F06.parquet` is a **separate 6-fold walk-forward over
2020-2025** with different dataset ids (`real-f0N-v1-test`), not the 4-fold
2022-2025 chain that `m0r_v2` is gated against. Overlapping years do **not**
agree:

| year | `baseline_oos` | `baseline_v2` | row delta |
|---|---|---|---|
| 2022 | F03, 787,930 rows, `real-f03-v1-test` | F1, 747,461 rows, `real-f1-v1-test` | +40,469 |
| 2023 | F04, 870,331 rows, `real-f04-v1-test` | F2, 820,829 rows, `real-f2-v1-test` | +49,502 |
| 2024 | F05, 923,404 rows, `real-f05-v1-test` | F3, 924,499 rows, `real-f3-v1-test` | -1,095 |
| 2025 | F06, 922,724 rows, `real-f06-v1-test` | F4, 925,174 rows, `real-f4-v1-test` | -2,450 |

For anything that must line up with `m0r_v2/report.json`, use
`baseline_v2/baseline_oos/F1..F4.parquet`. `baseline_oos/` is kept only as the
wider 2020-2025 baseline context.

## Verified: the two baseline copies are numerically identical

`baseline_v2/baseline_oos/F{1..4}.parquet` vs the `kind=best_simple` frames in
`m0r_v2/predictions/`:

- outer join on `(signal_date, ts_code)`: 0 unmatched rows on either side
- `max |score_baseline - score_chain| = 0.000e+00` across all 3,417,963 rows
- file-level sha256 differs only because the `baseline_v2` copies carry three
  extra metadata columns (`model_id`, `fold_id`, `dataset_id`)

So the standalone baseline rerun and the baseline embedded in the production
chain are the same numbers. Any anchor gap is a *reduction* difference, not a
scoring difference.

## The `+0.0022` anchor gap: it is a valid-day-count mismatch

Both reports describe the same 2022-2025 test window, but disagree on how many
days enter the mean:

| field | `baseline_v2/report.json` | `m0r_v2/report.json` | delta |
|---|---|---|---|
| `valid_day_count` | **969** | **945** | +24 |
| `baseline_mean_rank_ic` | 0.0870555454 | 0.0852268663 | +0.0018286791 |
| `mean_rank_ic` | 0.0870555454 | 0.0943759614 | -0.0073204160 |
| `icir` | 0.7507448095 | 0.7681728838 | -0.0174280744 |
| `hac_t_statistic` | 12.7498273623 | 12.9054487122 | -0.1556213498 |

Per-year baseline means differ in both directions (2022 -0.0119, 2023 +0.0051, 2024 -0.0139, 2025 -0.0065, all as `baseline_v2` minus `m0r_v2`), which rules out a uniform offset and points at different day sets.

`baseline_v2/report.json → assembly_coverage.folds` explains where both numbers
come from. Summing the per-fold `days` field:

| split | F1 | F2 | F3 | F4 | total |
|---|---|---|---|---|---|
| `train.days` | 968 | 967 | 966 | 964 | 3,865 |
| `validation.days` | 237 | 236 | 236 | 236 | **945** |
| `test.days` | 242 | 242 | 242 | 243 | **969** |

`969` is the true test day count and matches the distinct `signal_date` values
actually present in the published frames. `945` is the **validation** day count.
`m0r_v2/report.json` reports `valid_day_count = 945` inside an OOS block whose
`yearly_mean_rank_ic` is keyed on test years 2022-2025 — i.e. the day count
looks like it was taken from the validation stage while the RankIC values are
test-stage.

Prime suspect for the appendix: the M0R OOS evaluator applies a validation-stage
day mask (or an equivalent 5-7 day-per-fold forward-label-horizon drop) when
reducing test-stage scores, dropping 24 of the 969 test days. Check the day
filter in the OOS reduction path; the score frames themselves are consistent.

Reproduce the counts:

```python
import json, pandas as pd, glob, os
r = json.load(open("baseline_v2/report.json"))
ac = r["assembly_coverage"]["folds"]
for s in ("train", "validation", "test"):
    print(s, sum(ac[f][s]["days"] for f in ac))     # 3865 / 945 / 969
for f in ("F1", "F2", "F3", "F4"):
    df = pd.read_parquet(f"baseline_v2/baseline_oos/{f}.parquet")
    print(f, df.signal_date.nunique(), df.signal_date.min(), df.signal_date.max())
```

## Environment of record

From `baseline_v2/report.json → environment`:

- factor source `frozen_v2_baseline`, 9 factor ids (`yeslab_96/206/357/381/425/464/471/484`, `yeslab_443_reg30_max20_v1`)
- Python 3.11.13, numpy 2.3.3, pandas 2.3.3, pyarrow 21.0.0, scikit-learn 1.7.2, lightgbm 4.6.0, torch 2.6.0+cu124
- 104 CPU host, 4x NVIDIA GeForce RTX 4090, `torch_intraop_threads = 52`
- data versions: `factor_daily` 144 partitions (2014-01..2025-12), `label_daily`
  and `universe_daily` 151 partitions (2014-01..2026-07), all partition hashes
  recorded in `report.json → data_versions`

`omp_num_threads` was `null` (unset). If you need bit-identical training rather
than bit-identical scores, pin it; the score frames here do not depend on it.

## Reported verdict (m0r_v2)

`WALK_FORWARD_PASSED` — G1..G6 all PASS, computed over the 945-day set:
mean RankIC 0.09437596, ICIR 0.76817288, HAC-t 12.905449, baseline mean RankIC
0.08522687, delta +0.00914910. Per-fold best config was identical across F1-F4
(`num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=200,
feature_fraction=0.85, lambda_l2=0.0`), searched over a single candidate.

`baseline_v2/report.json → oos.gate_note` records that the run-time gates used a
stale 2020-2025 `required_years` default and logged G3/G5 as FAIL with
`observed=None`; `report_gates_fixed.json` is the re-evaluation under the derived
F1-F4 -> 2022-2025 contract, where G4 is `NOT_APPLICABLE` (a standalone baseline
has no delta against itself) and all other gates PASS.
