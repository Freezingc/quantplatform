# Candidate Search Parallelism Design

## Goal

Accelerate the expensive validation-config search for the CPU cross-sectional candidate families M1 and M3 without changing their frozen data contract, model definitions, validation metric, canonical configuration selection, final three-seed fit, or G1-G6 OOS evaluation.

## Scope

This first delivery covers only M1 (`RESIDUAL_MLP`) and M3 (`SET_ATTENTION`). M2 has a temporal-batch and device-selection contract, while M0/M0R use LightGBM-specific fitting; they remain serial and are not silently routed through the new scheduler.

A legacy invocation remains serial and retains host-observed Torch thread counts. Parallel execution is explicit opt-in and belongs to a new workspace/run lineage.

## CLI and numerical contract

`scripts/run_real_candidate_training.py` gains `--parallel-workers N` (default `1`) and `--torch-threads N`. A worker count above one requires a positive explicit thread count, which configures both Torch intra-op and inter-op threads before either parent or worker constructs a model.

The existing Torch runtime descriptor records the selected counts, so `SearchResume` refuses a record created under another runtime. No old artifact is rewritten or adopted under the new execution setting.

## Execution design

For each fold, the parent prepares the same preprocessed train/validation batches as the serial implementation. Each unadopted configuration becomes one spawned-process task. The task trains all required seeds, writes only its config/seed-namespaced checkpoints, and returns its small validation result.

The parent is the only durable writer of search records: it calls `SearchResume.persist` as results arrive, rebuilds `search_results` by increasing config index, and retains the current `max` selection rule. It alone does final three-seed fitting, bundle/evidence/prediction writes, and F1-F4 OOS aggregation.

## Safety requirements

- Spawned processes only; no thread pool and no forked Torch runtime.
- Cap worker count at the number of unadopted configs in the fold.
- Propagate any worker exception and do not select from a partial grid.
- M0, M0R, and M2 behavior stays unchanged.
- Results are canonical by config index, never completion order.

## Verification

Tests cover CLI validation, worker-count capping, reverse completion ordering, worker error propagation, explicit thread initialization, and M1/M3-only scheduling. Existing candidate-driver, training, and evaluation suites remain green in the Python 3.11 `quantyd` environment.
