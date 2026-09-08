"""Emit the lifecycle registry's promotion-evidence document for a completed candidate run.

  emit_promotion_evidence.py \
    --workspace /data/zhangyudong/qrv2_research_artifacts/m0r_v2 \
    --report    /data/zhangyudong/qrv2_research_artifacts/m0r_v2/report.json \
    --out       /data/zhangyudong/qrv2_research_artifacts/m0r_v2/promotion_evidence.json

Requires both QRV2_SELECTION_EVIDENCE_HMAC_KEY and QRV2_PREDICTION_EVIDENCE_HMAC_KEY, which
hold the same single trust root by design. Reads only the workspace, never the database, so it
can be pointed at a run that finished before this tool existed. Exits non-zero if the workspace
is incomplete or if the candidate does not pass all six gates -- a failing run produces no
document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from research_v2.registry.promotion_evidence import (  # noqa: E402
    PROMOTION_EVIDENCE_FILENAME,
    PromotionEvidenceError,
    emit_promotion_evidence,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"defaults to <workspace>/{PROMOTION_EVIDENCE_FILENAME}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = args.out if args.out is not None else args.workspace / PROMOTION_EVIDENCE_FILENAME
    try:
        summary = emit_promotion_evidence(args.workspace, args.report, out_path)
    except (PromotionEvidenceError, ValueError) as error:
        print(f"promotion evidence not emitted: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
