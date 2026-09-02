"""Compare 2019 historical-window robustness results against train/validation baselines.

Reads the 2019 evaluation JSON and the decision-brief baseline table, applies the
project's standard signed gates (ICIR >= 0.50, mean RankIC >= 0.02, directional day
share >= 0.55), and classifies each candidate:

  PASS      -> 2019 signed ICIR >= 0.50 on both universe variants
  WATCH     -> signed ICIR in [0.30, 0.50) on at least one variant, no sign flip
  COLLAPSE  -> signed ICIR < 0.30 or sign flip on at least one variant

Read-only: prints the comparison table and writes a JSON artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"

BASELINE_PATH = DIAG / "historical_window_2019_baseline_brief_table.json"

GATE_ICIR = 0.50
GATE_IC = 0.02
GATE_DAY_SHARE = 0.55
WATCH_ICIR = 0.30


def classify(summary: dict) -> str:
    if not summary or summary.get("days", 0) == 0:
        return "NO_DATA"
    ic = summary.get("mean_rank_ic")
    icir = summary.get("icir")
    if ic is None or icir is None:
        return "NO_DATA"
    if ic <= 0 or icir < WATCH_ICIR:
        return "COLLAPSE"
    if icir < GATE_ICIR or ic < GATE_IC or summary.get("directional_day_share", 0) < GATE_DAY_SHARE:
        return "WATCH"
    return "PASS"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", default=[DIAG / "historical_window_2019_robustness_v1" / "historical_window_2019_robustness.json"])
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument("--output", type=Path, default=DIAG / "historical_window_2019_robustness_v1" / "historical_window_2019_comparison.json")
    args = parser.parse_args(argv)

    merged: dict[str, dict] = {}
    window = None
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged.update(payload["candidates"])
        window = window or payload.get("window")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    rows = []
    for fid, res in merged.items():
        base = baseline.get(fid, {})
        full_cls = classify(res["full_market"])
        liq_cls = classify(res["adv20_liquid"])
        official = res.get("official_cn_liquid_a")
        off_cls = classify(official) if official else None
        if off_cls is not None:
            # official universe supersedes the adv20 approximation when present
            pair = (full_cls, off_cls)
        else:
            pair = (full_cls, liq_cls)
        overall = pair[0] if pair[0] == pair[1] else (
            "COLLAPSE" if "COLLAPSE" in pair else "WATCH"
        )
        rows.append({
            "factor_id": fid,
            "direction": res["direction"],
            "train_icir_2020_2024": base.get("train_icir"),
            "val_icir_2025": base.get("val_icir"),
            "annual_gate": base.get("annual"),
            "ic_full_market": res["full_market"].get("mean_rank_ic"),
            "icir_full_market": res["full_market"].get("icir"),
            "ic_adv20_liquid": res["adv20_liquid"].get("mean_rank_ic"),
            "icir_adv20_liquid": res["adv20_liquid"].get("icir"),
            "ic_official_cn_liquid_a": official.get("mean_rank_ic") if official else None,
            "icir_official_cn_liquid_a": official.get("icir") if official else None,
            "classify_full": full_cls,
            "classify_liquid": liq_cls,
            "classify_official": off_cls,
            "classify_overall": overall,
        })

    order = {"COLLAPSE": 0, "WATCH": 1, "NO_DATA": 2, "PASS": 3}
    rows.sort(key=lambda r: (order[r["classify_overall"]], -(r["icir_full_market"] or 0)))

    header = (
        f"{'factor':28s} {'trnICIR':>7s} {'valICIR':>7s} | "
        f"{'fullIC':>8s} {'fICIR':>7s} {'liqIC':>8s} {'lICIR':>7s} {'offIC':>8s} {'oICIR':>7s} | verdict"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['factor_id']:28s} "
            f"{str(r['train_icir_2020_2024']):>7s} {str(r['val_icir_2025']):>7s} | "
            f"{str(r['ic_full_market']):>8s} {str(r['icir_full_market']):>7s} "
            f"{str(r['ic_adv20_liquid']):>8s} {str(r['icir_adv20_liquid']):>7s} "
            f"{str(r['ic_official_cn_liquid_a']):>8s} {str(r['icir_official_cn_liquid_a']):>7s} | "
            f"{r['classify_overall']} (full={r['classify_full']}, liq={r['classify_liquid']}, off={r['classify_official']})"
        )

    summary = {
        "scope": "HISTORICAL_WINDOW_2019_COMPARISON_V1",
        "window": window,
        "gates": {"signed_icir": GATE_ICIR, "signed_mean_rank_ic": GATE_IC, "directional_day_share": GATE_DAY_SHARE, "watch_icir": WATCH_ICIR},
        "counts": {
            verdict: sum(1 for r in rows if r["classify_overall"] == verdict)
            for verdict in ("PASS", "WATCH", "COLLAPSE", "NO_DATA")
        },
        "rows": rows,
        "database_writes": False,
        "locked_test_read": False,
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\ncounts: {summary['counts']}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
