# Evaluation Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make ordinary OOS evaluation enforce the approved G1-G6 score-health and coverage rules, while exposing deterministic daily and yearly summary evidence.

**Architecture:** Keep the existing pandas-based RankIC and frozen OOS evaluator APIs compatible where possible. Extend the daily RankIC result with score-health evidence, derive yearly summaries from valid daily rows, and pass those values into a fail-closed gate evaluator. FINAL/2026 access remains a separate capability; the ordinary evaluator rejects that period.

**Tech Stack:** Python 3.11, pandas, NumPy, statsmodels, pytest.

---

### Task 0: Preserve the Dataset package in Git

**Files:**
- Modify: `.gitignore`
- Add: `research_v2/configs/datasets/default.yaml`
- Add: `research_v2/src/research_v2/datasets/`
- Add: `research_v2/tests/unit/datasets/`

- [x] **Step 1: Verify the clean checkout cannot import Dataset modules**

The unanchored `datasets/` ignore rule hid the existing Dataset implementation from Git worktrees, causing downstream training imports to fail in a clean checkout.

- [x] **Step 2: Anchor the ignore rule to repository-root data directories**

Change `datasets/` to `/datasets/`, then include the existing Dataset source, configuration, and unit contracts so a clone contains the package required by `training/static_batches.py`.

- [x] **Step 3: Run Dataset tests and verify the package is portable**

Run:

```bash
conda run -n quantyd python -m pytest research_v2/tests/unit/datasets -q
```

Observed: 21 tests passed in the isolated worktree after the files were included.

### Task 1: Lock score-health gate behavior

**Files:**
- Modify: `research_v2/src/research_v2/evaluation/gates.py`
- Modify: `research_v2/tests/unit/evaluation/test_gates.py`

- [x] **Step 1: Add failing tests for all three G6 requirements**

Extend `GateEvidence` test fixtures with per-day score coverage, finite unique-score counts, and score standard deviations. Add cases proving that G6 fails when any one of these is below its threshold, and passes only when at least 95% of valid test days satisfy all three conditions.

- [x] **Step 2: Run the focused gate tests and verify the expected failure**

Run:

```bash
conda run -n quantyd python -m pytest research_v2/tests/unit/evaluation/test_gates.py -q
```

Expected: the new G6 assertions fail because the current evidence model only counts `good_score_day_count`.

- [x] **Step 3: Implement the minimal evidence and G6 calculation**

Add immutable per-day score-health fields to `GateEvidence`, validate their lengths against `valid_test_day_count`, and require each good day to satisfy finite coverage >= 0.90, unique finite scores >= 10, and score standard deviation > 1e-6. Return the ratio as G6's observed value and fail closed for missing or malformed evidence.

- [x] **Step 4: Run the focused gate tests and verify they pass**

Run the command from Step 2. Expected: all gate tests pass.

### Task 2: Produce deterministic score-health and yearly summaries

**Files:**
- Modify: `research_v2/src/research_v2/evaluation/rank_ic.py`
- Modify: `research_v2/src/research_v2/evaluation/model_evaluator.py`
- Modify: `research_v2/tests/unit/evaluation/test_rank_ic.py`
- Modify: `research_v2/tests/unit/evaluation/test_frozen_oos_evaluator.py`

- [x] **Step 1: Add failing tests for daily score-health columns and yearly summaries**

Assert that each valid signal date reports `eligible_count`, `valid_label_count`, `coverage`, `prediction_unique_count`, and `prediction_std`; assert that a date with collapsed scores is retained for diagnostics but is not counted as a valid RankIC day. Add yearly summary output with valid day counts and mean RankIC values.

- [x] **Step 2: Run the focused evaluation tests and verify the expected failure**

Run:

```bash
conda run -n quantyd python -m pytest research_v2/tests/unit/evaluation/test_rank_ic.py research_v2/tests/unit/evaluation/test_frozen_oos_evaluator.py -q
```

Expected: the new column and yearly-summary assertions fail against the current three-column daily frame and private yearly helper.

- [x] **Step 3: Implement the minimal daily and yearly evidence assembly**

Compute health from the prediction/label join without filling missing dates with zero. Keep invalid/undersized/collapsed days out of RankIC metrics while preserving their diagnostics in a separate health frame or explicit validity flag. Pass valid-day score-health arrays into `GateEvidence` and expose deterministic yearly summary records from the frozen evaluator.

- [x] **Step 4: Run the focused evaluation tests and verify they pass**

Run the command from Step 2. Expected: all focused evaluation tests pass.

### Task 3: Regression verification and review

**Files:**
- Modify: `research_v2/src/research_v2/materializers/oos.py`
- Add: `research_v2/migrations/013_oos_score_health.dos`
- Modify: `research_v2/tests/unit/materializers/test_oos_materializer.py`
- Modify: `research_v2/tests/unit/db/test_migrations.py`
- Modify: `docs/superpowers/plans/2026-08-27-evaluation-contract-hardening.md`

- [x] **Step 1: Persist score-health evidence without changing historical migration checksums**

Map the optional daily diagnostic columns into `model_oos_rank_ic_daily`, using NULL for legacy callers that only provide the historical three-column result. Add migration `013_oos_score_health.dos` with `addColumn` so migration 007 remains immutable and deployed V2 databases can receive the new fields.

- [x] **Step 2: Run all evaluation, baseline, materializer, and migration unit tests**

Run:

```bash
conda run -n quantyd python -m pytest research_v2/tests/unit/evaluation research_v2/tests/unit/baselines research_v2/tests/unit/materializers/test_oos_materializer.py research_v2/tests/unit/db/test_migrations.py -q
```

Expected: exit code 0 with no failures.

- [x] **Step 3: Run Ruff on the changed Python modules**

Run:

```bash
conda run -n quantyd python -m ruff check research_v2/src/research_v2/evaluation/gates.py research_v2/src/research_v2/evaluation/rank_ic.py research_v2/src/research_v2/evaluation/model_evaluator.py research_v2/tests/unit/evaluation/test_gates.py research_v2/tests/unit/evaluation/test_rank_ic.py research_v2/tests/unit/evaluation/test_frozen_oos_evaluator.py
```

Expected: no diagnostics.

- [x] **Step 4: Review the diff and commit the isolated slice**

Run `git diff --check` and `git status --short`, confirm no Alpha484 files changed, then commit:

```bash
git add .gitignore docs/superpowers/plans/2026-08-27-evaluation-contract-hardening.md research_v2/configs/datasets research_v2/src/research_v2/datasets research_v2/src/research_v2/evaluation research_v2/src/research_v2/materializers/oos.py research_v2/migrations/013_oos_score_health.dos research_v2/tests/unit/datasets research_v2/tests/unit/evaluation research_v2/tests/unit/materializers/test_oos_materializer.py research_v2/tests/unit/db/test_migrations.py
git commit -m "feat(v2): enforce evaluation score health gates"
```

### Task 4: Close review-identified boundary gaps

**Files:**
- Modify: `research_v2/src/research_v2/evaluation/gates.py`
- Modify: `research_v2/src/research_v2/evaluation/rank_ic.py`
- Modify: `research_v2/src/research_v2/evaluation/model_evaluator.py`
- Modify: `research_v2/src/research_v2/datasets/builder.py`
- Modify: `research_v2/src/research_v2/datasets/storage.py`
- Modify: related unit tests

- [x] Make malformed score-health evidence fail closed without raising runtime type errors.
- [x] Normalize intraday timestamps before business-key matching, duplicate detection, and label-day counting.
- [x] Validate split roles at runtime and keep ordinary TEST/Dataset reads outside the locked 2026 period.
- [x] Bind writes and verification to the recomputed dataset and split manifest identity, not only shard hashes.
- [x] Verify the focused boundary tests, full V2 test suite, Ruff, mypy, and diff hygiene.

Observed verification: 648 tests passed with three pre-existing warnings; the changed Python files passed Ruff, Ruff format, and mypy.
