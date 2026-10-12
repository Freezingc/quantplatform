# Operational extension: scoring and backtest past the published range (2026-10-12)

## Why an extension was needed

The models' output is a **within-date percentile rank**, so scoring a date needs that date's
cross-section. The published pipeline stops at **2026-07-31** (`universe_daily`,
`tradability_daily`, `market_daily_clean` all end there) while Tushare runs on, so dates after that
had factors but no rankable set. The factor *inputs* also need history: the binding lookback is
**180 trading days** (eight GP factors), and the replay fetches from ~1 year before the signal date.

The models themselves need **no** history — M0R and M1 are same-day cross-sectional — and the
execution convention is **signal T -> buy at T+1 open -> sell at T+6 open**, so a signal is computed
after T's close and traded at the next open. Buying on a Monday uses the previous Friday's signal.

## What was built

| step | script | guard |
|---|---|---|
| rebuild `cn_liquid_a` membership | `research_v2/scripts/materialize_operational_universe.py` | refuses to write unless it reproduces the published universe on an overlapping window (default: exact) |
| replay the 15 frozen v2 factors | `research_v2/scripts/materialize_operational_factors.py --universe-parquet` | window honoured exactly; never writes `factor_daily` |
| score fixed dates | `research_v2/scripts/score_operational_dates.py --universe-from parquet` | requires the `OPERATIONAL_UNGATED` marker; no promotion claim |
| backtest | `research_v2/scripts/backtest_operational_scores.py` | drops signals whose exit has not priced |

## Fidelity checks (both passed)

1. **Universe rebuild.** Against the published universe on 2026-07-20..07-31: **10/10 dates exact**,
   Jaccard agreement 1.0, max member-count difference 0. The first attempt reached only 0.9992
   because collapsing the security master to the newest row per board erased intervals still in
   force, excluding 600768.SH and 688309.SH as `OUTSIDE_BOARD`.

2. **Forward returns vs published labels.** Recomputed `adjusted_open(T+6)/adjusted_open(T+1)-1`
   against `label_daily` for 2026-07-01..07-23: **66,631 of 66,656 defined rows exact (99.96%)**.
   The 25 remaining rows (10 codes) are a **Tushare restatement**: e.g. 000908.SZ had a corporate
   action, and the adjustment factors for the pre-event days were revised after the platform's
   snapshot froze them; the two agree exactly from 2026-07-06 onward. 23,553 further rows have no
   published label at all because their exit date falls past the ingested market data.

## Backtest, full 2026 window — signal dates 2026-01-05 .. 2026-09-24 (178 dates, 35 cycles)

Equal-weight books, 15bp round trip, holding buffer `n_exit`, trend overlay on 932000.CSI as the
policy defines. Benchmark = the same engine holding every rankable name (~3,900), which **lost
3.51%** over the window with a **-19.90%** maximum drawdown.

| book | total | excess vs benchmark | max DD | ann. net | vol | Sharpe | turnover | mean names |
|---|---|---|---|---|---|---|---|---|
| 20:100 | **+3.37%** | **+6.88%** | -9.07% | 5.45% | 13.3% | 0.41 | 0.234 | 20 |
| 100:400 | +3.12% | +6.63% | -13.54% | 5.33% | 14.9% | 0.36 | 0.149 | 100 |
| 30:150 | +3.02% | +6.53% | -9.61% | 5.02% | 13.6% | 0.37 | 0.203 | 30 |
| 50:250 | +1.80% | +5.31% | -12.12% | 3.45% | 14.2% | 0.24 | 0.165 | 50 |
| 10:50 | +0.50% | +4.01% | -10.24% | 1.61% | 13.8% | 0.12 | 0.281 | 10 |

### Findings (full window)

* Every book was **positive while the average stock fell 3.5%**, and every book roughly **halved the
  benchmark's drawdown** (-9% to -13.5% against -19.9%). Selection added value net of costs.
* **20:100 and 30:150** are the best pair of excess and drawdown; 10:50 is the weakest on excess and
  the most expensive to run, so the earlier "best 10-30 names" reading is only half right — the
  narrowest book is not the best one.
* Sharpe 0.41 at 35 cycles is modest and still noisy, but it is the right order of magnitude for a
  five-day rebalancing strategy on a falling market.

## The same test on a shorter window — and why it is reported as a warning

## Backtest result — signal dates 2026-08-03 .. 2026-09-24 (7 cycles)

Equal-weight books, 15bp round trip, holding buffer `n_exit`, trend overlay on 932000.CSI as the
policy defines. Benchmark = the same engine holding every rankable name (~3,900).

| book | total | benchmark | excess | max DD | turnover | cycle win rate |
|---|---|---|---|---|---|---|
| 10:50 | **-0.31%** | +4.46% | -4.77% | -2.18% | 0.234 | 0.43 |
| 20:100 | +1.12% | +4.46% | -3.35% | -1.65% | 0.219 | 0.57 |
| 30:150 | +1.63% | +4.46% | -2.84% | -1.82% | 0.196 | 0.71 |
| 50:250 | +1.94% | +4.46% | -2.53% | -1.65% | 0.158 | 0.71 |
| 100:400 | +2.59% | +4.46% | **-1.87%** | -1.55% | 0.143 | 0.71 |

### Findings (and the sample-size lesson)

* On this window **every** configuration underperformed the equal-weight universe, and the top-10
  book lost money outright — the **opposite** of the full-window result above.
* The two windows disagree because seven cycles is not a sample: the sign of the excess flips, which
  is exactly the failure mode of reading a short backtest. The full-window numbers are the ones to
  use; this section is kept so the mistake is not repeated.
* Cycle returns move together across all books (one clearly negative cycle in the sixth), i.e. the
  books are dominated by market/style beta rather than by stock selection — consistent with the
  earlier drawdown attribution.
* **The annualised figures in the sidecar are not results.** Seven cycles carry no information about
  annual Sharpe.

* The trend overlay was **fully invested for the whole August-September window**: pinning
  `--trend-floor 1.0` reproduced the same totals to the digit, so the shortfall there was selection,
  not the overlay parking the book in cash.

### Caveats

* These signal dates are the **locked FINAL test segment**. This is an operational check, not a
  walk-forward result, and it must not be used to select, tune or retrain anything: once a locked
  performance number informs a decision, that segment is no longer out of sample.
* Neither bundle carries G1-G6 gate evidence (`OPERATIONAL_UNGATED`).

## Artifacts

Outside the repository, in `/data/zhangyudong/qrv2_operational/`:

* `universe_2026ext.parquet` (+ `.meta.json` with the validation summary)
* `factors_2026ext.parquet` (+ `.meta.json`)
* `scores_2026ext.parquet` (+ `.meta.json`) — 45 signal dates, 176,943 rows
* `backtest_2026ext.json` — full policy summaries and per-cycle returns
* `backtest_2026ext_nooverlay.json` — the same window with the overlay pinned off (identical)
* `backtest_2026full.json` — the full 2026 window (the result to use)
* `scores_2026full.parquet`, `universe_2026full.parquet` — the combined inputs for that run
* earlier: `fidelity_2025Q4.*` (factor fidelity), `factors_2026H1.*`, `scores_2026H1.*`
