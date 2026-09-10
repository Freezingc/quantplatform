# Candidate Search Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent M1/M3 validation configuration searches in spawned processes while preserving exact selection and evidence contracts.

**Architecture:** A reusable process scheduler accepts index-addressed work and returns index-addressed results in canonical order. The candidate driver uses it only for M1/M3 after explicitly pinning Torch threads; serial search, final fits, artifacts, and OOS evaluation remain unchanged.

**Tech Stack:** Python 3.11, `concurrent.futures.ProcessPoolExecutor`, PyTorch CPU, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-candidate-search-parallelism-design.md`

## Global Constraints

- Scope is M1/M3 cross-sectional config search only.
- `--parallel-workers` defaults to `1`; this legacy path does not configure Torch threads.
- `--parallel-workers > 1` requires a positive `--torch-threads` value and uses a spawned context.
- `SearchResume.persist` is the sole durable config-result writer.
- Report ordering is increasing config index, never worker completion order.
- Final fits, bundle/evidence/prediction writes, and G1-G6 aggregation stay in the parent.

---

### Task 1: Declare the execution contract

**Files:**

- Modify: `research_v2/src/research_v2/training/seeds.py`
- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/training/test_reproducibility_seeds.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `configure_torch_threads(thread_count: int) -> None` rejects zero and configures both Torch thread pools.
- `_parallel_execution(args) -> ParallelExecution` parses/validates explicit opt-in settings.

- [x] Write failing tests that a parallel run without `--torch-threads` is rejected and a zero thread count is invalid.
- [x] Run the two targeted test files and confirm the APIs are absent.
- [x] Add the CLI flags, validation helper, and thread setter.
- [x] Re-run the targeted tests (`70 passed`) and commit with `feat(training): declare parallel torch execution settings` (`05d3922`).

### Task 2: Add an ordered spawned scheduler

**Files:**

- Create: `research_v2/src/research_v2/training/parallel_search.py`
- Test: `research_v2/tests/unit/training/test_parallel_search.py`

**Interfaces:**

- `SearchTask(index: int, payload: Mapping[str, object])` binds every result to one grid position.
- `run_parallel_search(tasks, workers, run_task) -> list[tuple[int, dict[str, object]]]` uses `multiprocessing.get_context("spawn")`, caps workers to task count, propagates failures, and returns index order.

- [x] Write failing tests for reverse completion ordering, worker-cap calculation, and worker exception propagation.
- [x] Run `python -m pytest tests/unit/training/test_parallel_search.py -q` and confirm the missing-module failure.
- [x] Implement the scheduler with `ProcessPoolExecutor` and a spawned context.
- [x] Re-run the scheduler tests (`3 passed`) and commit with `feat(training): add ordered spawned config scheduler` (`0f77b95`).

### Task 3: Route M1/M3 search trials through the scheduler

**Files:**

- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `CrossSectionConfigTask` contains all three-seed search inputs and a config/seed-unique checkpoint prefix.
- `run_cross_section_config_task(payload) -> dict[str, object]` is package-level and writes no bundle, selection evidence, prediction, or report.
- `_ordered_search_results(configs, completed) -> list[dict[str, Any]]` restores grid order before best-config selection.

- [x] Write a failing test that reverse-order scheduler results become increasing-index search results.
- [x] Run that test and confirm the ordering helper is absent.
- [x] Extract the existing three-seed config body into the worker and call the scheduler only for M1/M3 when workers exceed one. Preserve the serial loop for one worker.
- [x] Run the complete candidate-driver suite (`69 passed`, including a real spawned three-seed worker) and commit with `feat(training): parallelize cross-sectional config search` (`77e8401`).

### Task 4: Verify the supported surface

**Files:**

- Modify: this plan to check completed steps and record exact commands/results.

- [x] Run `python -m pytest tests/unit/training/test_reproducibility_seeds.py tests/unit/training/test_parallel_search.py tests/unit/training/test_search_resume.py tests/unit/scripts/test_run_real_candidate_training.py -q` (`108 passed`).
- [x] Run `python -m pytest tests/unit/evaluation -q` (`72 passed`; two pre-existing pandas FutureWarnings).
- [x] Run `ruff check src tests scripts/run_real_candidate_training.py` (`All checks passed!`).
- [x] Run `git diff --check`, record exact results, and commit with `docs(training): record parallel search verification`.

### Task 5: Route the M2 temporal search through the same scheduler (2026-09-10 increment)

**Why:** Tasks 1-4 left M2 serial because of its temporal-batch contract. M2 is also the family
that cannot afford to stay serial -- measured ~8.6 h per fit and ~161 days for the family, so it is
the one search that will never finish without concurrency. The blocker was never the scheduler
(index-addressed, family-agnostic); it was the payload. A temporal batch holds a 20-day window per
row and the validation panel is train+validation concatenated, so per-task batches would be
gigabytes each.

**Files:**

- Add: `research_v2/src/research_v2/training/search_metrics.py`, `research_v2/src/research_v2/training/temporal_search.py`, `research_v2/tests/unit/training/test_temporal_search.py`.
- Modify: `research_v2/src/research_v2/training/cross_section_search.py`, `research_v2/scripts/run_real_candidate_training.py`, `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`, the design spec.

**Interfaces:**

- `search_metrics.validation_rank_ic(predictions, labels) -> float` -- the number the winning config is selected by, in one place. It had two copies (the driver's serial loops and the M1/M3 worker); three would have been three chances for "the model the search selected" to mean something different from "the model the parent fits".
- `temporal_search.TemporalFoldSpec / TemporalFoldBatches / TemporalConfigTask`, `temporal_fold_batches(spec)` (cached per worker), `temporal_panel_batches`, `temporal_model`, `run_temporal_config_task(payload)`.
- `HISTORY_LENGTH = 20` / `MINIMUM_VALID_HISTORY_LENGTH = 15` -- the window contract, imported by the driver's serial `_temporal_panel_batches` too.
- `_temporal_search_budget(max_epochs)` -- M2's record budget: fit length **plus the window**, because a 20-day and a 10-day window are different models and neither is in the config payload.

**Steps:**

- [x] Write failing tests: the worker rebuilds bitwise the batches the parent would build (against a real published dataset artifact), the validation window reaches back into train (day 21, the split's first day, has a batch only if the panel was concatenated -- and day 15 as the first eligible train date pins that the 15-day minimum counts the signal date itself), one fold is built once per worker, the task payload stays under a 100 KB budget, GRU/TCN follow the config, and a real spawned worker returns a complete three-seed result.
- [x] Run them and confirm the missing-module failure.
- [x] Implement `search_metrics` and `temporal_search`; point `cross_section_search` at the shared metric and delete its private copy.
- [x] Route the M2 search through `run_parallel_search` when `--parallel-workers > 1`, adopting and persisting search records exactly as the M1/M3 parallel path does, and restore grid order through the existing `_ordered_search_results` before the existing `max` selection.
- [x] Bind the window contract into M2's budget and make the driver's serial panel builder read the same constants, so serial loop, spawned worker and records cannot disagree about the window.
- [x] Allow M2 in `_parallel_execution`; keep M0/M0R refused, with the reason in the message.
- [x] Mutation-check: hardcoding the driver's window back to a literal, and dropping `history_length` from the budget, each kill exactly the test that pins them.
- [x] Full gate: `ruff check src tests` clean, `mypy src` **169 files** clean, `pytest tests -q` **1997 passed / 45 skipped**.

**Still open (not done by this task):** seed-level task granularity. With `--max-configs 1` -- which
is what the zero-tuning comparison runs use, and what M0R's +0.009149 was measured at -- there is
one config per fold, so `capped_worker_count` returns 1 and the parallel path gives no speedup at
all. The three seeds of a fit group are independent, so splitting the task granularity to
(config, seed) is what makes a 1-config run 3x faster; for M2's 24-config grid the config-level
granularity already saturates. Per-task payloads stay small either way once batches are rebuilt
worker-side, which is what this task established.
