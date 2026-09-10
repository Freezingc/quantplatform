# Candidate Search Parallelism Design

## Goal

Accelerate the expensive validation-config search for the CPU cross-sectional candidate families M1 and M3 without changing their frozen data contract, model definitions, validation metric, canonical configuration selection, final three-seed fit, or G1-G6 OOS evaluation.

## Scope

This first delivery covers only M1 (`RESIDUAL_MLP`) and M3 (`SET_ATTENTION`). M2 has a temporal-batch and device-selection contract, while M0/M0R use LightGBM-specific fitting; they remain serial and are not silently routed through the new scheduler.

**Increment (2026-09-10, after Tasks 1-4):** M2 (`TEMPORAL_GRU` / `TEMPORAL_TCN`) is now routed through the same scheduler under the same explicit opt-in, in `training/temporal_search.py`. The scheduler needed no change -- it is index-addressed and family-agnostic -- but the payload did: a temporal batch holds a 20-day window per row and the validation panel is train+validation concatenated, so carrying batches per task would be gigabytes each. An M2 task therefore carries identifiers only (workspace, fold, published dataset id, factor order, fitted preprocessing state) and each worker rebuilds the fold's panels and batches once, caching them for every configuration it is handed. That reconstruction is a tested equivalence, not an assumption: a worker that built a slightly different panel would report a validation RankIC for a different model than the parent finally fits.

M0/M0R remain serial and are still refused by `--parallel-workers`: their fits are minutes rather than hours and they already pin `num_threads` per fit, so there is nothing to win and a real risk of perturbing a contract that is bit-reproduced.

A legacy invocation remains serial and retains host-observed Torch thread counts. Parallel execution is explicit opt-in and belongs to a new workspace/run lineage.

## CLI and numerical contract

`scripts/run_real_candidate_training.py` gains `--parallel-workers N` (default `1`) and `--torch-threads N`. A worker count above one requires a positive explicit thread count, which configures both Torch intra-op and inter-op threads before either parent or worker constructs a model.

The existing Torch runtime descriptor records the selected counts, so `SearchResume` refuses a record created under another runtime. No old artifact is rewritten or adopted under the new execution setting.

## Execution design

For each fold, the parent prepares the same preprocessed train/validation batches as the serial implementation. Each unadopted configuration becomes one spawned-process task. The task trains all required seeds, writes only its config/seed-namespaced checkpoints, and returns its small validation result.

The parent is the only durable writer of search records: it calls `SearchResume.persist` as results arrive, rebuilds `search_results` by increasing config index, and retains the current `max` selection rule. It alone does bundle/evidence/prediction writes and F1-F4 OOS aggregation.

**Increment (2026-09-10):** the final three-seed fit is parallel too, in both the static and the temporal families. Config-level parallelism accelerated the search only, so a fold whose grid already filled the workers still paid three full fit-times at the end -- half of a 24-config M2 fold. The three fits are independent by construction and the parent needs only their checkpoint paths and validation scoring, so a seed-level task returns exactly those and the parent freezes the bundle from weights it did not train itself. One seeded fit now has a single definition per family, shared by the config-level worker (which loops the seeds) and the seed-level worker (which the parent loops across processes); both paths write the identical checkpoint names, so a bundle's provenance cannot depend on the worker count.

## Safety requirements

- Spawned processes only; no thread pool and no forked Torch runtime.
- Cap worker count at the number of unadopted configs in the fold.
- Propagate any worker exception and do not select from a partial grid.
- M0 and M0R behavior stays unchanged; M2 stays serial unless `--parallel-workers` is given.
- Results are canonical by config index, never completion order.
- The window contract (20-day history, 15-day minimum) has one definition, imported by the driver's serial panel builder, the spawned worker, and M2's search-record budget -- so a resumed run cannot adopt a record produced under a different window.

## Verification

Tests cover CLI validation, worker-count capping, reverse completion ordering, worker error propagation, explicit thread initialization, and M1/M3-only scheduling. Existing candidate-driver, training, and evaluation suites remain green in the Python 3.11 `quantyd` environment.
