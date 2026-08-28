# Canonical Candidate Pair Similarity Design

## Goal

Produce a read-only, reproducible train/validation similarity matrix for the
six current factor candidates. The matrix must distinguish a factor's
individual admission evidence from comparison-only portfolio evidence.

## Problem

Each candidate has passed its own quality review on an isolated canonical
path, but most candidate-to-candidate pairs have not been measured. The old
review matrix therefore correctly prevents joint admission, but cannot explain
which additional pairs are redundant. Directly joining the old artifacts is
invalid because their historical calculation scopes differ.

## Fixed Comparison Contract

The comparison evaluator uses one fixed V2 daily SH/SZ membership and
label-valid output-key panel for every candidate. It replays all formulas on
the canonical full SH/SZ Tushare market scope, then applies that same V2 key
panel to every output before measuring absolute daily cross-sectional Spearman
correlations on common finite keys. This preserves formula-defined
cross-sectional inputs while removing output-panel differences.

The comparison panel is not a new factor version and does not replace the
individual canonical evidence for any candidate. In particular, it does not
change the Alpha484 admission artifact or replay the historical Alpha206
mean-based formula.

| Candidate ID | Comparison Formula Source |
| --- | --- |
| `yeslab_114_tushare_shsz_v2` | `TerriStdCalculator` replay definition |
| `yeslab_381_tushare_shsz_v1` | `GP_Factor_012_Calculator` replay definition |
| `yeslab_405_tushare_shsz_v1` | `GP_Factor_036_Calculator` replay definition |
| `yeslab_425_tushare_shsz_v2` | `GP_Factor_056_Calculator` replay definition |
| `yeslab_484_tushare_shsz_v1` | `HerdingCSADStd21Calculator` replay definition |
| `yeslab_206_median_abs_v2` | `ABS(vol - rolling_median(vol, 60)) / rolling_median(vol, 60)` |

## Inputs and Bounds

- Training window: `2020-01-02` through `2024-12-31`.
- Validation window: `2025-01-02` through `2025-12-31`.
- All input keys must be V2 `is_member` and `label_valid`, with `.SH` and
  `.SZ` codes only. BJ is excluded.
- Market inputs are local Tushare `daily`, `daily_basic`, and adjusted fields
  only when a replay definition requires them.
- The evaluator reads enough pre-window market history for the longest
  calculator replay lookback and executes once for each fixed train or
  validation phase. This avoids repeatedly recomputing overlapping rolling
  windows without changing formula inputs or outputs.
- The evaluator never reads dates on or after `2026-01-01`, writes no
  DolphinDB table, and never publishes factor values.

## Components

`research_v2/scripts/audit_canonical_candidate_pairs.py` owns the full
read-only command. It will:

1. Validate the fixed candidate registry, comparison dates, and locked-test
   boundary.
2. Read the complete V2 comparison key panel and derive its fixed SH/SZ code
   union.
3. Materialize each fixed phase from the Tushare cache over the canonical
   full SH/SZ code union, retaining warmup history but emitting values only
   for that phase’s valid V2 keys.
4. Replay the four non-Alpha484 source calculators with
   `audit_formula_replay_consistency._replay_factor`; materialize Alpha206's
   median repair through the existing `compute_variants` helper. Alpha484
   uses only the completed, fingerprint-verified Tushare canonical replay
   checkpoint because recomputing its 120-day all-stock correlation matrix in
   each repeated window would change no values while multiplying its cost.
5. Calculate all 15 candidate-pair similarity edges for training and
   validation, fail closed on an incomplete edge, and write a compact JSON
   summary plus pair CSV files.

The evaluator reuses `compute_pair_daily_spearman` from
`evaluate_tushare_scope_validation`. It does not modify that shared helper.
Rolling-window non-finite candidate values remain keyed to the V2 panel and
are excluded only at pairwise correlation time.  Each pair must still retain
at least 30 common finite values on every one of the required 1212/243 dates;
missing business keys, candidates, pair edges, or required dates fail closed.

## Output Contract

The output JSON records the comparison-only status, candidate registry, fixed
scope, train and validation pair summaries, per-edge valid-day counts, the
`0.90` redundancy threshold, and explicit no-admission/no-database/no-2026
flags. A pair is:

- `PAIR_UNIQUE` when both phase medians are below `0.90` and all required
  dates are present.
- `ELEVATED_PAIR_REDUNDANCY_MONITOR` when either phase median is at least
  `0.80` but below `0.90`.
- `REDUNDANT` when either phase median is at least `0.90`.
- `INCOMPLETE` when either phase lacks its full expected daily count.

The existing candidate decision matrix may link this artifact, but no status
can become model eligible from this audit.

## Failure Handling

The command fails before writing a final summary when a candidate definition,
Tushare market field, V2 business key, required pair edge, or required date is
missing.  A rolling-window NaN alone is not a missing key: it is handled by
the pairwise common-finite rule above.  Phase-local intermediate state remains
outside database tables and is not an admission artifact until both train and
validation summaries are complete.

## Verification

Unit tests cover the candidate registry, locked-test parser guard, conversion
of the Alpha206 median panel, complete-pair validation, and pair classification
boundaries. The real read-only run must produce exactly 15 train and 15
validation edges, each with 1212 and 243 valid days respectively, before its
result is cited in candidate governance.
