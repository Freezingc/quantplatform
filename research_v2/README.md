# Quant Research V2

Independent, point-in-time A-share research platform. It reads the existing V1
source database only and writes all V2 artifacts to separately governed stores.

Implementation follows the approved V2 design and phased plans in the workspace
`docs/superpowers` directory.

## Training continuation checkpoint (2026-08-04)

The static PIT-to-training-batch boundary is now implemented in
`src/research_v2/training/static_batches.py`. It creates one keyed,
date-isolated `TrainingBatch` per signal date, pads cross-sections to a common
width, carries an explicit factor-missing mask, excludes invalid/non-finite
labels, and rejects signal dates in the locked year or later. It does not fit
preprocessing state; callers must pass values transformed by the train-fitted
`FoldPreprocessor` so validation data cannot affect preprocessing.

Formal Residual MLP, Set Attention, GRU, and TCN training has deliberately not
started. The approved selection rule in D3.5 requires at least 15 admitted
factors before a complex model can be trained. The FINAL frozen set currently
has six members (`GP_008`, `GP_011`, `GP_018`, `GP_019`, `GP_022`, `GP_064`), so
it remains suitable for B0/B1 and tabular diagnostics but is not eligible for
formal deep-model artifacts. The existing model constructors accepting 1–60
features are shape contracts, not an authorization to bypass this gate.

Verification in the requested `quantyd` Python 3.11.13 environment is green:
271 tests passed, Ruff is clean, and mypy is clean across 122 source files.
The verification command uses the explicit interpreter
`/data/zhangyudong/.conda/envs/quantyd/bin/python`; this avoids shell-wrapper
resolution differences observed with `conda run` in the current session.
