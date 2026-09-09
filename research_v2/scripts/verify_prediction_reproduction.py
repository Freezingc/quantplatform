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

Covers the LightGBM families (M0, M0R) and the static torch families (M1, M3). M2 TEMPORAL_GRU is
refused by name, because reproducing it needs the historical panel and its 20-day window rather than
the frozen TEST frame alone.

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
)
from research_v2.registry.promotion_evidence import PromotionEvidenceError  # noqa: E402
from research_v2.registry.registry_store import (  # noqa: E402
    ModelRegistryStore,
    RegistryStoreError,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
