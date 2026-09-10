"""Recompute a run's candidate scores from its own bundles and compare to the signed artifacts.

  verify_prediction_reproduction.py \
    --workspace /data/zhangyudong/qrv2_research_artifacts/m0r_v2 \
    --report    /data/zhangyudong/qrv2_research_artifacts/m0r_v2/report.json

The frozen evidence chain already proves a prediction is ATTRIBUTED to a bundle: the scores are
signed, the prediction's selection id / model id / fold / dataset are matched against the frozen
selection, and that selection's bundle digest is matched against a re-hashed bundle. Nothing in it
proves the scores are what the bundle COMPUTES. This does, by recomputation.

Requires QRV2_MODEL_REGISTRY_ROOT plus the selection and prediction trust-root keys. The registry is
consulted to REPORT the run's lifecycle status, not to authorise the audit: bundles are loaded
through the evaluation path, so a run that failed a gate is auditable too and an unregistered one is
reported as UNREGISTERED rather than refused. Read-only against the workspace: it writes nothing.
Exits non-zero on any difference, so it is usable as a gate.

With ``--fold F1`` (repeatable) it audits single folds instead of the whole run, taking each
fold's bundle and TEST dataset from its signed model selection rather than from ``report.json``.
That is the mode for a run still in flight: a divergence in an already-finished fold is then
found hours before the last fold exists, the registry is not consulted, and ``--report`` is not
needed.

Covers every family the chain can produce: the LightGBM carriers (M0, M0R), the static torch
families (M1, M3), and the temporal ones (M2, whose card reads TEMPORAL_GRU or TEMPORAL_TCN after
the winning config). A temporal fold's TEST scores come from a 20-day window panel, so the audit
rebuilds that panel from the same three published splits through the one library function the
driver itself calls.

Exact equality needs no thread pinning, and that is measured: a forward pass is bitwise identical at
1, 4, 16 and 52 intra-op threads. Training is not thread-stable, which is why the search records
bind the count and this audit does not have to.

Expect minutes, not seconds: each fold's dataset split is re-hashed in full before it is scored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_v2.inference.reproduction import (  # noqa: E402
    ReproductionError,
    verify_candidate_predictions_reproduce,
    verify_fold_predictions_reproduce,
)
from research_v2.registry.promotion_evidence import PromotionEvidenceError  # noqa: E402
from research_v2.registry.registry_store import (  # noqa: E402
    ModelRegistryStore,
    RegistryStoreError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument(
        "--fold",
        action="append",
        default=[],
        help=(
            "audit only this fold, taking its bundle and TEST dataset from its signed model "
            "selection instead of from the run report (repeatable); --report then optional"
        ),
    )
    return parser.parse_args(argv)


def _audit_folds(workspace: Path, folds: list[str]) -> int:
    """Audit completed folds one at a time, reporting each and failing the exit code on any."""
    summaries: list[dict[str, object]] = []
    failed = False
    for fold_id in folds:
        try:
            summaries.append(verify_fold_predictions_reproduce(workspace, fold_id))
        except (ReproductionError, ValueError) as error:
            print(f"{fold_id} reproduction FAILED: {error}", file=sys.stderr)
            failed = True
    print(
        json.dumps(
            {"folds": summaries, "reproduced": not failed},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.fold:
        return _audit_folds(args.workspace, list(args.fold))
    if args.report is None:
        print("--report is required unless --fold is given", file=sys.stderr)
        return 2
    try:
        registry = ModelRegistryStore.from_environment()
    except RegistryStoreError as error:
        print(f"cannot open the model registry: {error}", file=sys.stderr)
        return 2
    try:
        summary = verify_candidate_predictions_reproduce(args.workspace, args.report, registry)
    except (ReproductionError, PromotionEvidenceError, ValueError) as error:
        print(f"reproduction FAILED: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
