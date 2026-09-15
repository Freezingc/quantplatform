# M2 F1: grid, standing, and dataset identity

Grid: **24 configs**, axes family=['GRU', 'TCN'], hidden_dim=['128', '64'], dropout=['0.0', '0.1', '0.2'], learning_rate=['0.0003', '0.001'].
Ensemble validation RankIC spans **0.047942 – 0.059239** (median 0.055455).

## Ranking

| rank | cfg | family | hidden | dropout | lr | ens. val RankIC | worst seed |
|---|---|---|---|---|---|---|---|
| 1 | c16 | TCN | 64 | 0.2 | 0.0003 | 0.059239 | 20260801 |
| 2 | c14 | TCN | 64 | 0.1 | 0.0003 | 0.058012 | 20260801 |
| 3 | c0 | GRU | 64 | 0.0 | 0.0003 | 0.057965 | 20260801 |
| 4 | c12 | TCN | 64 | 0.0 | 0.0003 | 0.057062 | 20260801 |
| 5 | c2 | GRU | 64 | 0.1 | 0.0003 | 0.056426 | 20260801 |
| 6 | c20 | TCN | 128 | 0.1 | 0.0003 | 0.056265 | 20260801 |
| 7 | c17 | TCN | 64 | 0.2 | 0.001 | 0.056118 | 20260801 |
| 8 | c22 | TCN | 128 | 0.2 | 0.0003 | 0.056108 | 20260801 |
| 9 | c4 | GRU | 64 | 0.2 | 0.0003 | 0.056015 | 20260801 |
| 10 | c8 | GRU | 128 | 0.1 | 0.0003 | 0.055869 | 20260801 |
| 11 | c13 | TCN | 64 | 0.0 | 0.001 | 0.055746 | 20260801 |
| 12 | c18 | TCN | 128 | 0.0 | 0.0003 | 0.055491 | 20260801 |
| 13 | c10 | GRU | 128 | 0.2 | 0.0003 | 0.055419 | 20260801 |
| 14 | c15 | TCN | 64 | 0.1 | 0.001 | 0.055234 | 20260801 |
| 15 | c5 | GRU | 64 | 0.2 | 0.001 | 0.054692 | 20260801 |
| 16 | c3 | GRU | 64 | 0.1 | 0.001 | 0.054504 | 20260801 |
| 17 | c6 | GRU | 128 | 0.0 | 0.0003 | 0.053920 | 20260801 |
| 18 | c1 | GRU | 64 | 0.0 | 0.001 | 0.053212 | 20260801 |
| 19 | c21 | TCN | 128 | 0.1 | 0.001 | 0.053165 | 20260801 |
| 20 | c23 | TCN | 128 | 0.2 | 0.001 | 0.052608 | 20260801 |
| 21 | c19 | TCN | 128 | 0.0 | 0.001 | 0.051169 | 20260801 |
| 22 | c7 | GRU | 128 | 0.0 | 0.001 | 0.050464 | 20260801 |
| 23 | c11 | GRU | 128 | 0.2 | 0.001 | 0.049371 | 20260801 |
| 24 | c9 | GRU | 128 | 0.1 | 0.001 | 0.047942 | 20260801 |

## Axis tests

- **learning_rate**: median 0.0003=0.056187 vs 0.001=0.053189 (Δ=+0.002998), Mann-Whitney U=132.0, p=0.0006, n=12/12, ranges overlap.
- **hidden_dim**: median 128=0.053543 vs 64=0.056066 (Δ=-0.002524), Mann-Whitney U=27.0, p=0.0102, n=12/12, ranges overlap.
- **family**: median GRU=0.054598 vs TCN=0.055927 (Δ=-0.001329), Mann-Whitney U=47.0, p=0.1572, n=12/12, ranges overlap.

## Seed structure

Worst-seed census over 24 configs: **{'20260801': 24}**.
- seed 20260801: median 0.044986, range 0.028420–0.049566, worst in 24/24 configs.
- seed 20260802: median 0.050626, range 0.042189–0.056576, worst in 0/24 configs.
- seed 20260803: median 0.063446, range 0.056726–0.067430, worst in 0/24 configs.
Within-config seed spread (max−min of the three seeds): median **0.019451**, range 0.014755–0.029967.

## Standing on the shared F1 validation split

M2 F1 = **0.059239**, position **7 of 8**.

| label | validation RankIC | M2 minus this |
|---|---|---|
| M0R | 0.067948 | -0.008710 |
| M0R-replicate | 0.067948 | -0.008710 |
| baseline_B2 | 0.062711 | -0.003472 |
| M0-refined | 0.061705 | -0.002466 |
| M0-base | 0.061705 | -0.002466 |
| M1-full | 0.060023 | -0.000784 |
| M1-carrier | 0.056058 | +0.003181 |

Baseline valid-day count 237; M2's own validation parquet has 237 distinct signal dates (2021-01-04 … 2021-12-23), match = **True**.

## Does F1 validation predict the OOS verdict?

Over 5 distinct prior carriers: Spearman rho = **+0.1539** (p=0.8048), Pearson r = +0.4382 (p=0.4605). M0R-replicate excluded as a bit-identical duplicate of M0R; M2 excluded, no OOS yet.

| label | F1 validation | OOS mean RankIC | gates |
|---|---|---|---|
| M0R | 0.067948 | 0.094376 | G1G2G3G4G5G6 pass / none fail |
| M1-full | 0.060023 | 0.093054 | G1G2G3G4G5G6 pass / none fail |
| M1-carrier | 0.056058 | 0.091071 | G1G2G3G4G5G6 pass / none fail |
| M0-refined | 0.061705 | 0.089429 | G1G2G3G5G6 pass / G4 fail |
| M0-base | 0.061705 | 0.088867 | G1G2G3G5G6 pass / G4 fail |

## Is M2's seed sensitivity unusual? (F1, every carrier)

| carrier | configs | within-config seed spread (median) | between-config ensemble range | spread / range | worst-seed census |
|---|---|---|---|---|---|
| M0-base | 24 | 0.012155 | 0.017941 | 0.677 | {'20260803': 14, '20260802': 4, '20260801': 6} |
| M0-refined | 24 | 0.009955 | 0.017941 | 0.555 | {'20260803': 21, '20260801': 3} |
| M0R | 1 | 0.000247 | 0.000000 | - | {'20260802': 1} |
| M0R-replicate | 1 | 0.000247 | 0.000000 | - | {'20260802': 1} |
| M1-carrier | 1 | 0.014376 | 0.000000 | - | {'20260801': 1} |
| M1-full | 27 | 0.027117 | 0.015753 | 1.721 | {'20260801': 22, '20260802': 5} |
| M2 | 24 | 0.019451 | 0.011297 | 1.722 | {'20260801': 24} |

## Would dropping the systematically worst seed close the gap?

Per-seed medians over the whole grid: 20260801=0.044986, 20260802=0.050626, 20260803=0.063446.
Seed effect span **0.018460** vs between-config ensemble range **0.011297**.
Ranking instead by the median of the two non-worst seeds still selects **c16** at **0.061853**, i.e. below the baseline's 0.062711.
Configs clearing the baseline: **0/24** on the three-seed ensemble, **0/24** on the two non-worst seeds, and per single seed {'20260801': 0, '20260802': 0, '20260803': 15}.
_Caveat: per-seed ICs are not the ensemble IC: the ensemble is scored on averaged predictions, so every two-seed figure below is an indicative counterfactual, not what a retrained two-seed ensemble would actually score._

Baseline source agreement (M2's own record vs the frozen report): **{'F2': True, 'F1': True}**.
M2 validation-parquet day count vs the baseline's valid_day_count: **True**.

## Dataset identity

All carriers agree on every fold's dataset id: **{'F1': True, 'F2': True, 'F3': True, 'F4': True}**.
M2 dataset dirs on disk: ['226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538', 'eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234']
Mapping to known folds: {'226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538': ['F1'], 'eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234': ['F2']}
M2 baseline selection id(s): {'F1': '6fe33eda88430e5ee90e8d975c16f36c8d0de1a44a70eda594b12b82ff5678b9'}

## One coincidence, checked

M0R's F2 validation RankIC vs the baseline's F3 validation RankIC look equal at 5 dp: M0R F2 = 0.09870757822362144, baseline F3 (B2) = 0.09870781290204374, |Δ| = 2.347e-07, same dataset = False → **COINCIDENCE_NOT_MISALIGNMENT**.

## Not done here

No OOS number for M2 exists yet, so nothing here says whether M2 passes the walk-forward; no database query was run; no weight was loaded; nothing was written into any artifact directory; the parquet was read in place for a day count only.
