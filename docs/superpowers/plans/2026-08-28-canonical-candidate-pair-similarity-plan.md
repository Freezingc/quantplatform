# Canonical Candidate Pair Similarity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calculate the 15 train and 15 validation candidate-to-candidate
similarity edges on one read-only canonical SH/SZ comparison panel.

**Architecture:** A single evaluator builds one phase-wide Tushare replay
panel for the fixed six-candidate registry, restricts output to V2
member/valid-label keys, and aggregates daily absolute Spearman correlations
per phase. The result is a comparison-only artifact: no factor values are
published and no candidate status is changed.

**Tech Stack:** Python 3.11, pandas, numpy, local Tushare cache, DolphinDB
read sessions, pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-canonical-candidate-pair-similarity-design.md`

## Global Constraints

- Read only V2 and Tushare sources; perform no DolphinDB writes.
- Filter and validate labels as `label_id=forward_open_5d`,
  `label_version=1` before building comparison keys.
- Reject every date range that reaches `2026-01-01`.
- Use only `.SH` and `.SZ` V2 member plus valid-label business keys.
- Recompute candidate values; never use historical YesLab values as candidate
  input.
- The output must say `NO_AUTOMATIC_ADMISSION`, keep every candidate
  `model_eligible=false`, and not alter the formal nine.
- A final result requires all 15 pair edges and exactly 1212/243 valid days in
  train/validation; otherwise fail closed.

---

### Task 1: Define the comparison contract and pure pair classification

**Files:**
- Create: `research_v2/scripts/audit_canonical_candidate_pairs.py`
- Create: `research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py`

**Interfaces:**
- Produces `CANDIDATES: tuple[CandidateSpec, ...]` with the exact six
  versioned candidate IDs from the spec.
- Produces `parse_args(argv) -> argparse.Namespace` and rejects a range ending
  on or after `LOCKED_TEST_START`.
- Produces `classify_pair(median_train, days_train, median_validation,
  days_validation) -> str`.

- [x] **Step 1: Write failing registry and boundary tests**

```python
def test_registry_has_exactly_six_versioned_candidates() -> None:
    assert [item.candidate_id for item in audit.CANDIDATES] == [
        "yeslab_114_tushare_shsz_v2",
        "yeslab_381_tushare_shsz_v1",
        "yeslab_405_tushare_shsz_v1",
        "yeslab_425_tushare_shsz_v2",
        "yeslab_484_tushare_shsz_v1",
        "yeslab_206_median_abs_v2",
    ]


def test_parser_rejects_locked_test_range() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args(["--start", "2025-01-01", "--end", "2026-01-01", "--output", "/tmp/x"])


def test_classify_pair_requires_complete_phase_days() -> None:
    assert audit.classify_pair(0.20, 1212, 0.20, 242) == "INCOMPLETE"
```

- [x] **Step 2: Run the new tests and verify they fail because the module is missing**

Run:

```bash
PYTHONPATH=research_v2/scripts:research_v2/src /data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest -q research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py
```

Expected: `ModuleNotFoundError` for `audit_canonical_candidate_pairs`.

- [x] **Step 3: Implement the minimal registry, parser, and classifier**

```python
@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    source_factor_id: str
    is_alpha206_median_repair: bool = False


def classify_pair(train: float, train_days: int, validation: float, validation_days: int) -> str:
    if train_days != 1212 or validation_days != 243:
        return "INCOMPLETE"
    maximum = max(train, validation)
    if maximum >= 0.90:
        return "REDUNDANT"
    if maximum >= 0.80:
        return "ELEVATED_PAIR_REDUNDANCY_MONITOR"
    return "PAIR_UNIQUE"
```

The parser must require `--output`, default to the train and validation
windows in the spec, and reject 2026.

- [x] **Step 4: Run the focused test file and verify it passes**

Run the command from Step 2. Expected: all registry, parser, and classifier
tests pass.

- [x] **Step 5: Commit the pure comparison contract**

```bash
git add research_v2/scripts/audit_canonical_candidate_pairs.py research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py
git commit -m "feat(quality): define canonical candidate pair contract"
```

### Task 2: Materialize the shared comparison panel and summarize pair edges

**Files:**
- Modify: `research_v2/scripts/audit_canonical_candidate_pairs.py`
- Modify: `research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py`

**Interfaces:**
- Consumes `CandidateSpec`, a Tushare market frame, and V2 comparison keys.
- Produces frames with exactly `trade_date`, `ts_code`, and `replay_value`.
- Produces `summarize_phase_pairs(replay_frames, expected_days) -> DataFrame`
  with one row per unordered candidate pair and no duplicate pair keys.

- [x] **Step 1: Write failing Alpha206 and complete-edge tests**

```python
def test_alpha206_median_repair_uses_rolling_median() -> None:
    market = pd.DataFrame({
        "trade_date": pd.date_range("2025-01-01", periods=60).repeat(2),
        "ts_code": ["000001.SZ", "600000.SH"] * 60,
        "vol": [1.0, 2.0] * 59 + [100.0, 2.0],
    })
    result = audit.materialize_alpha206_median_repair(market, pd.Timestamp("2025-03-25"), pd.Timestamp("2025-03-25"))
    assert result.loc[result.ts_code == "000001.SZ", "replay_value"].iloc[0] == pytest.approx(99.0)


def test_summarize_phase_pairs_rejects_missing_edge() -> None:
    with pytest.raises(ValueError, match="incomplete candidate pair matrix"):
        audit.summarize_phase_pairs({"yeslab_114_tushare_shsz_v2": pd.DataFrame()}, 1)
```

- [x] **Step 2: Run the focused tests and verify failures**

Run the command from Task 1 Step 2. Expected: missing materializer and
summary functions fail the new tests.

- [x] **Step 3: Implement phase materialization and complete-matrix validation**

Use `_load_validation_inputs`, `TushareMarketSession`,
`_calculator_market_start`, and `replay._replay_factor` for the source
calculator candidates. Recompute each fixed train or validation phase once on
the full SH/SZ Tushare market input, rather than recomputing overlapping
monthly windows. Use `compute_variants(market)["median_abs"]` only for
`yeslab_206_median_abs_v2`, and convert its wide panel to the replay-frame
schema. Alpha484 may load only the completed fingerprint-verified Tushare
canonical replay checkpoint with its adjacent content/date integrity file.
Restrict every replay frame to the phase’s V2 comparison keys after
calculation, reject duplicate keys, and call
`compute_pair_daily_spearman` with the six fixed candidate IDs.

For each phase, require all 15 unordered pair keys, the exact expected trading
date set, and the phase's exact expected day count before classifying. Store
median and maximum absolute Spearman, valid-day count, and status for every
edge.

- [x] **Step 4: Run the focused test file and verify it passes**

Run the command from Task 1 Step 2. Expected: all pure materialization and
matrix-completeness tests pass.

- [x] **Step 5: Commit the comparison materializer**

```bash
git add research_v2/scripts/audit_canonical_candidate_pairs.py research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py
git commit -m "feat(quality): materialize canonical candidate pair audit"
```

### Task 3: Execute the read-only audit and link its evidence

**Files:**
- Modify: `research_v2/scripts/audit_canonical_candidate_pairs.py`
- Modify: `research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py`
- Create: `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1.json`
- Create: `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1/train_pairs.csv`
- Create: `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1/validation_pairs.csv`
- Require: `research_v2/artifacts/checkpoints/alpha484_canonical_v1/integrity.json`
- Modify: `research_v2/artifacts/diagnostics/tushare_shsz_candidate_admission_review_matrix_v2.json`
- Modify: `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json`
- Modify: `docs/handoffs/2026-08-19-factor-quality-handoff.md`

- [x] **Step 1: Run the full pre-2026 audit**

Before the live run, add the tested command entry point that materializes one
full SH/SZ Tushare panel per phase, separates training and validation replay
frames, validates their complete pair matrices, and writes the JSON plus pair
CSV outputs only after both phases succeed. The entry point must retain the
existing locked-test guard and must not write to DolphinDB.

```bash
set -a
source .env
set +a
PYTHONPATH=research_v2/scripts:research_v2/src /data/zhangyudong/.conda/envs/quantyd/bin/python research_v2/scripts/audit_canonical_candidate_pairs.py --start 2020-01-02 --end 2025-12-31 --output research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1
```

Expected: 15 train and 15 validation pair rows, 1212 and 243 valid days per
edge, and `database_writes=false`, `locked_test_read=false`.

- [x] **Step 2: Validate the output contract**

```bash
jq -e '(.train_pairs | length == 15) and (.validation_pairs | length == 15) and (.database_writes == false) and (.locked_test_read == false) and (.decision == "COMPARISON_ONLY_NO_ADMISSION")' research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1.json
```

Expected: `true`.

- [x] **Step 3: Link only the comparison evidence**

Add per-edge results to the matrix's `combination_constraints` or a new
`comparison_only_pair_evidence` field. Preserve each candidate's existing
`model_eligible=false` and do not alter individual review statuses. Add the
artifact path to the quality coverage ledger and handoff.

- [x] **Step 4: Run full regression and final artifact checks**

```bash
PYTHONPATH=research_v2/scripts:research_v2/src /data/zhangyudong/.conda/envs/quantyd/bin/python -m pytest -q
git diff --check
```

Expected: all tests pass and no whitespace errors.

- [x] **Step 5: Commit the audited evidence**

```bash
git add research_v2/scripts/audit_canonical_candidate_pairs.py research_v2/tests/unit/scripts/test_audit_canonical_candidate_pairs.py docs/handoffs/2026-08-19-factor-quality-handoff.md research_v2/artifacts/diagnostics/tushare_shsz_candidate_admission_review_matrix_v2.json research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json
git add -f research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1.json research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1
git commit -m "docs(quality): audit canonical candidate pair similarity"
```

## Plan Self-Review

- Spec coverage: Tasks 1-2 implement the registry, common V2 comparison panel,
  Alpha206 repair isolation, fail-closed edge validation, and locked-test
  boundary. Task 3 runs and links the only permitted read-only evidence.
- Scope: no database migration, V2 schema change, factor publication, formal
  set mutation, or model training appears in any task.
- Type consistency: every materializer produces the three-column replay frame
  consumed by `compute_pair_daily_spearman`; pair summaries are consumed only
  by the JSON output and comparison-only matrix link.
