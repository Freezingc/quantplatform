# Real-Data B0-B3 Walk-Forward Report (Criterion 303)

OOS evaluation over F01-F06 (test years 2020-2025) built from the full-range PIT tables with the admitted factor set.

## Overall OOS statistics

| Metric | Value |
|---|---|
| Valid day count | 1455 |
| Mean RankIC | 0.0864 |
| ICIR | 0.6344 |
| HAC t-statistic | 13.092 |
| Baseline mean RankIC | 0.0864 |
| Delta RankIC | 0.0000 |

## Gate verdicts (G1-G6)

| Gate | Definition | Observed | Threshold | Status |
|---|---|---|---|---|
| G1 | OOS mean RankIC >= 0.030 | 0.0864 | 0.030 | PASS |
| G2 | HAC t-statistic >= 2.000 | 13.0920 | 2.000 | PASS |
| G3 | positive-year share >= 0.70 (2020-2025) | 1.0000 | 0.700 | PASS |
| G4 | delta RankIC vs baseline >= 0.005 (N/A for baseline path) | - | - | NOT_APPLICABLE |
| G5 | max single-year positive contribution share <= 0.50 | 0.2024 | 0.500 | PASS |
| G6 | valid score-day ratio >= 0.95 | 1.0000 | 0.950 | PASS |

## Per-fold validation selection (B0-B3)

| Fold | B0 | B1 | B2 | B3 | Winner |
|---|---|---|---|---|---|
| F01 | 0.0724 | -0.0749 | 0.0946 | **0.0989** | B3 |
| F02 | 0.0610 | -0.0529 | 0.0606 | **0.0669** | B3 |
| F03 | 0.0545 | -0.0701 | **0.0658** | 0.0616 | B2 |
| F04 | 0.0814 | -0.0900 | **0.0919** | 0.0917 | B2 |
| F05 | 0.0744 | -0.1008 | **0.0913** | 0.0802 | B2 |
| F06 | 0.0860 | -0.0886 | 0.1014 | **0.1111** | B3 |

## Yearly OOS mean RankIC

| Year | Mean RankIC |
|---|---|
| 2020 | 0.0693 |
| 2021 | 0.0659 |
| 2022 | 0.0949 |
| 2023 | 0.0853 |
| 2024 | 0.1050 |
| 2025 | 0.0984 |

## Data versions (published_version snapshot)

| Table | Partitions |
|---|---|
| factor_observation_clean | 151 partitions (2014-01..2026-07) |
| label_daily | 151 partitions (2014-01..2026-07) |
| universe_daily | 151 partitions (2014-01..2026-07) |

Full per-partition version mapping is stored in the report JSON under `data_versions`.

## Criterion 303 verdict

- All G1-G6 gates satisfied: yes
- F01-F06 all present: yes
- B0-B3 all fitted and scored per fold: yes
- Yearly RankIC complete for 2020-2025: yes

**Criterion 303: SATISFIED**
