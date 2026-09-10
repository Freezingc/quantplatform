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

**Still open after Task 5:** the FINAL three-seed fit was still serial, and on a fold whose grid
already saturates the workers it was the only serial part left -- three fits out of a 24-config M2
fold's ~52 hours, i.e. half the fold at M2's measured ~8.6 h per fit. Taken by Task 6 below.

### Task 6: Parallelise the final three-seed fit (2026-09-10 increment)

**Why:** config-level parallelism accelerates the search only. The three final fits that produce the
frozen bundle ran serially afterwards in every family, so a fold whose 24-config grid already filled
the workers still paid three full fit-times at the end. They are independent by construction -- same
config, three seeds, `REQUIRED_SEEDS` is a fail-closed contract -- and the parent does not need the
models, only their checkpoint paths and their validation scoring.

**Interfaces:**

- `cross_section_search.CrossSectionSeedTask` / `run_cross_section_seed_task`, `temporal_search.TemporalSeedTask` / `run_temporal_seed_task`, both returning `{seed, checkpoint_path, predictions, validation_rank_ic}`.
- `_fit_one_seed` / `_fit_one_temporal_seed`: one seeded fit, defined ONCE and used by both the config-level worker (which loops the three seeds itself) and the seed-level worker (which the parent loops across processes). Returning the checkpoint path is what lets the parent freeze a bundle from a fit it never ran.

**Steps:**

- [x] Write failing tests: a spawned seed worker hands back a checkpoint that exists on disk under the same `<prefix>-seed-<seed>-epoch-<n>` name the serial path produces, and three spawned seed fits reassemble through `build_seed_ensemble` into a real ensemble with three distinct checkpoints -- for the static families and for M2.
- [x] Refactor both config-level workers onto the shared single-seed fit; their existing spawned-worker tests stay green unchanged, which is the evidence the refactor preserved behaviour.
- [x] Route the final fits of `_run_full_cross_section` (M1/M3) and `run_full_m2` through `run_parallel_search` when `--parallel-workers > 1`, indexed by seed; the serial path stays the default and keeps the identical checkpoint prefixes, so a bundle's provenance cannot depend on the worker count.
- [x] Mutation-check: renaming the parallel branch's checkpoint prefix kills exactly the test that pins serial/parallel prefix equality.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q` **2004 passed / 45 skipped**.

**Taken by Task 7 below:** seed-level granularity for the SEARCH.

### Task 7: Seed-level search granularity for narrow grids (2026-09-10 increment)

**Why:** with `--max-configs 1` -- the shape of every zero-tuning comparison run, and what M0R's
+0.009149 was measured at -- there is one config per fold, so `capped_worker_count` returned 1 and
the search stayed serial even with `--parallel-workers 48`. Task 6 parallelised the final fits, which
took such a run from six serial fits per fold to four; splitting the search's unit of work to
(config, seed) takes it to two parallel rounds, i.e. **3x**, and it is what makes an end-to-end
validation of the whole parallel path cost ~12 h instead of ~30 h.

**Rule:** per fold, adopt what the search records cover, then use config-level tasks when the
remaining configs can fill the workers (`len(unadopted) >= workers`) and seed-level tasks otherwise.
Config-level keeps the smaller return payload (scalars only); seed-level returns each seed's
validation predictions so the parent can build the ensemble, which is affordable precisely because
it is chosen only when there are few configs. Both write the same records and the same checkpoint
names, so a workspace does not record which granularity produced it.

**Interfaces:**

- `_ensemble_from_seed_fits(fits, config_payload, validation_labels) -> dict` -- rebuilds one config's
  search result from seed fits that ran in other processes, in the serial loop's exact shape, and
  refuses a seed set that is not exactly `REQUIRED_SEEDS`.
- `_constant_budget(budget)` -- a named factory for the torch families' config-independent budget,
  replacing a lambda whose type mypy could not infer at a binding site.

**Steps:**

- [x] Write failing tests for `_ensemble_from_seed_fits`: the result shape and its four keys, `worst_seed`
      is the lowest-RankIC seed, an incomplete seed set is refused, an unknown seed is refused.
- [x] Implement the seed-level branch in both torch search loops, indexed `config_index * len(REQUIRED_SEEDS) + slot`
      and decoded by integer division, so grid order is restored by the existing `_ordered_search_results`.
- [x] Mutation-check: weakening the seed-set guard to a subset test turns the refusal into a downstream
      `KeyError`, and the incomplete-seed test fails.
- [x] Clean the driver's mypy output while in it: `MYPYPATH=src mypy scripts/run_real_candidate_training.py`
      now reports **zero errors in the driver** (was 13 -- six from today's parallel wiring, six pre-existing
      `ModelWalkForwardResult` name reuse, plus the third-party `dolphindb` stub note which is not ours).
      The gate is `mypy src`, which never covered `scripts/`, so none of these were being caught.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q`
      **2008 passed / 45 skipped**.

**Still open:** no full real-data run has used any parallel path. The library level has real spawned
workers (static and temporal, config-level and seed-level, including three-seed reassembly through
`build_seed_ensemble`) and the driver level has source-wiring assertions, but the end-to-end proof is
a run. Task 7 is what made that proof affordable: a 1-config M1 run with `--parallel-workers 3
--torch-threads 1` is now ~12 h rather than ~30 h, and it would simultaneously validate the parallel
path and produce an M1 candidate under a pinned, declared thread lineage. Starting it is a decision
about numeric lineage (adjudication 2 in the compute brief), not an engineering one, so it has not
been started unilaterally.
