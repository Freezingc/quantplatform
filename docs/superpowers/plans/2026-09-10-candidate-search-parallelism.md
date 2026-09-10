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

- [ ] Write failing tests that a parallel run without `--torch-threads` is rejected and a zero thread count is invalid.
- [ ] Run the two targeted test files and confirm the APIs are absent.
- [ ] Add the CLI flags, validation helper, and thread setter.
- [ ] Re-run the targeted tests and commit with `feat(training): declare parallel torch execution settings`.

### Task 2: Add an ordered spawned scheduler

**Files:**

- Create: `research_v2/src/research_v2/training/parallel_search.py`
- Test: `research_v2/tests/unit/training/test_parallel_search.py`

**Interfaces:**

- `SearchTask(index: int, payload: Mapping[str, object])` binds every result to one grid position.
- `run_parallel_search(tasks, workers, run_task) -> list[tuple[int, dict[str, object]]]` uses `multiprocessing.get_context("spawn")`, caps workers to task count, propagates failures, and returns index order.

- [ ] Write failing tests for reverse completion ordering, worker-cap calculation, and worker exception propagation.
- [ ] Run `python -m pytest tests/unit/training/test_parallel_search.py -q` and confirm the missing-module failure.
- [ ] Implement the scheduler with `ProcessPoolExecutor` and a spawned context.
- [ ] Re-run the scheduler tests and commit with `feat(training): add ordered spawned config scheduler`.

### Task 3: Route M1/M3 search trials through the scheduler

**Files:**

- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `CrossSectionConfigTask` contains all three-seed search inputs and a config/seed-unique checkpoint prefix.
- `_cross_section_config_result(task) -> dict[str, Any]` is module-level and writes no bundle, selection evidence, prediction, or report.
- `_ordered_search_results(configs, completed) -> list[dict[str, Any]]` restores grid order before best-config selection.

- [ ] Write a failing test that reverse-order scheduler results become increasing-index search results.
- [ ] Run that test and confirm the ordering helper is absent.
- [ ] Extract the existing three-seed config body into the worker and call the scheduler only for M1/M3 when workers exceed one. Preserve the serial loop for one worker.
- [ ] Run the complete candidate-driver suite and commit with `feat(training): parallelize cross-sectional config search`.

### Task 4: Verify the supported surface

**Files:**

- Modify: this plan to check completed steps and record exact commands/results.

- [ ] Run `python -m pytest tests/unit/training/test_reproducibility_seeds.py tests/unit/training/test_parallel_search.py tests/unit/training/test_search_resume.py tests/unit/scripts/test_run_real_candidate_training.py -q`.
- [ ] Run `python -m pytest tests/unit/evaluation -q`.
- [ ] Run `ruff check src tests scripts/run_real_candidate_training.py`.
- [ ] Run `git diff --check`, record exact results, and commit with `docs(training): record parallel search verification`.
