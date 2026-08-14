# Alpha95 Relative Amount Volatility Design

## Goal

Evaluate an isolated repair candidate for `yeslab_163` (Alpha95). The stored
formula is `STD(amount, 20)`, whose raw monetary scale contributes to its
three-of-five training-year stability result and high similarity to
`yeslab_431`. The repair keeps the same economic signal: variability of traded
amount relative to a stock's own recent amount level.

## Candidate Formulas

For each `N` in `{10, 20, 40}`, compute:

```
relative_amount_volatility_N = STD(amount, N) / MA(amount, N)
```

The current day's amount is permitted because the existing factor has
same-day-close availability. Rolling computations use only data on or before
the signal date, use `min_periods=N`, and use sample standard deviation
(`ddof=1`). Non-finite denominators and zero denominators are excluded.

Candidates are research-only IDs:

- `yeslab_163_relative_amount_volatility_10_v1`
- `yeslab_163_relative_amount_volatility_20_v1`
- `yeslab_163_relative_amount_volatility_40_v1`

They never overwrite, backfill, or mix with historical `yeslab_163` values.

## Data And Evaluation

Use the canonical SH/SZ universe, exclude BJ, use Tushare daily amount, and
keep the raw five-trading-day forward adjusted-open return label.

Training selection is limited to 2020-01-02 through 2024-12-31. Freeze the
direction from this window. A candidate must satisfy:

- signed mean RankIC greater than 0.02;
- absolute ICIR at least 0.50;
- directional day share at least 0.55;
- annual double-gate passes in at least four of five years; and
- median absolute daily Spearman below 0.90 against every formal factor and
  every simultaneously evaluated Alpha95 variant.

If more than one candidate passes, select the highest training signed ICIR;
break a tie with lower maximum similarity to the formal nine.

Validate the selected candidate once on 2025. It must satisfy the same daily
quality gate and remain below 0.90 similarity to every formal factor. The 2026
locked test is never read.

## Outputs And Safety

Write only local diagnostic artifacts: candidate daily RankIC, quality
summaries, pairwise similarity, and an admission-review record. The existing
YesLab tables and the formal nine are unchanged. No V2 publication, database
write, model-input change, or model training is authorized.

## Verification

Unit tests cover formula arithmetic, zero-denominator handling, no future data
use, candidate selection, and rejection on a failed gate. The full evaluation
must show complete 2020-2024 and 2025 day coverage before an admission review
status is written.
