"""What each lever is worth, in days and in G6, with every number read from the instrument's output.

The owner has four candidate responses to the 24 unevaluable OOS days: disclose them, move the
score-health floor, complete the STAR industry history behind yeslab_357/yeslab_471, or change which
factors define the baseline's dense subset. Three of the four change signed reports, so they are
governance decisions, and a governance decision needs the price of each option measured rather than
described. This reads the per-day coverage the intersection screen already produced for each
configuration and answers, for a sweep of floor values:

  * how many days fall below the floor, and what G6 becomes (G6's own gate is >= 0.95);
  * the largest floor that keeps every day, and how much slack that leaves;
  * how dense the coverage distribution is near the floor, which is what makes the answer fragile.

The last one matters more than it looks. At the floor in force, G6 passes with 0.0252 of headroom on
the day ratio; this measures how much headroom exists on the FLOOR axis, and the two are not the same
quantity. Nothing here writes to a database or changes a threshold.

Usage: analyze_floor_sensitivity.py --config <label>=<instrument json> [...] --out-json X --out-md Y
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# G6's own threshold, from the frozen gate contract: valid score days must be at least 95% of the
# window. Read here only to say whether a floor change would break G6, never to change it.
G6_THRESHOLD = 0.95
FLOOR_IN_FORCE = 0.90
FLOOR_SWEEP = (0.85, 0.87, 0.88, 0.885, 0.89, 0.895, 0.90, 0.905, 0.91, 0.915, 0.92, 0.93, 0.95)


def sensitivity(coverages: list[float]) -> dict[str, Any]:
    ordered = sorted(coverages)
    total = len(ordered)
    lowest = ordered[0] if ordered else None
    rows = []
    for floor in FLOOR_SWEEP:
        below = sum(1 for value in ordered if value < floor)
        valid = total - below
        rows.append(
            {
                "floor": floor,
                "days_below_floor": below,
                "valid_days": valid,
                "g6": valid / total if total else None,
                "g6_passes": (valid / total >= G6_THRESHOLD) if total else None,
                "is_the_floor_in_force": floor == FLOOR_IN_FORCE,
            }
        )
    # Histogram in 0.005 bins around the floor: this is what makes any answer here fragile.
    bins: dict[str, int] = {}
    for value in ordered:
        if 0.87 <= value < 0.96:
            key = f"{(int(value / 0.005) * 0.005):.3f}"
            bins[key] = bins.get(key, 0) + 1
    return {
        "days": total,
        "coverage_min": lowest,
        "coverage_median": ordered[total // 2] if total else None,
        "largest_floor_that_keeps_every_day": lowest,
        "slack_of_that_floor_below_the_one_in_force": (
            FLOOR_IN_FORCE - lowest if lowest is not None else None
        ),
        "sweep": rows,
        "histogram_0_005_bins_0_87_to_0_96": dict(sorted(bins.items())),
        "days_within_0_005_above_the_floor": sum(
            1 for value in ordered if FLOOR_IN_FORCE <= value < FLOOR_IN_FORCE + 0.005
        ),
        "days_within_0_005_below_the_floor": sum(
            1 for value in ordered if FLOOR_IN_FORCE - 0.005 <= value < FLOOR_IN_FORCE
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="a configuration label and the intersection screen JSON that measured it",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    configs: dict[str, dict] = {}
    for item in args.config:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--config needs LABEL=PATH, got {item!r}")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        coverages = [float(entry["coverage"]) for entry in payload["per_day"]]
        result = sensitivity(coverages)
        result["source"] = {
            "path": path,
            "factor_count": payload["diagnostics"]["factor_count"],
            "factor_ids": payload["diagnostics"]["factor_ids"],
            "floor_used_by_the_instrument": payload["summary"]["floor"],
            "days_below_floor_as_measured": payload["summary"]["days_below_floor"],
        }
        configs[label] = result

    lines = [
        "# 地板敏感性与四个杠杆的价格",
        "",
        f"G6 自己的门是 **>= {G6_THRESHOLD}**；现行地板 **{FLOOR_IN_FORCE}**。",
        "所有数字由 `analyze_floor_sensitivity.py` 从仪器的逐日输出直接算出，无手抄。",
        "",
    ]
    for label, result in configs.items():
        lines += [
            f"## {label}（{result['source']['factor_count']} 个因子）",
            "",
            f"* 天数 {result['days']}，覆盖率最低 **{result['coverage_min']:.6f}**、"
            f"中位 {result['coverage_median']:.6f}",
            f"* **能保住全部天数的最大地板 = {result['largest_floor_that_keeps_every_day']:.6f}**"
            f"（比现行地板低 {result['slack_of_that_floor_below_the_one_in_force']:.6f}）",
            f"* 地板上方 0.005 内有 **{result['days_within_0_005_above_the_floor']}** 天、"
            f"下方 0.005 内有 **{result['days_within_0_005_below_the_floor']}** 天",
            "",
            "| 地板 | 跌破天数 | 可评天数 | G6 | G6 是否通过 |",
            "|---|---|---|---|---|",
        ]
        for row in result["sweep"]:
            mark = " ←现行" if row["is_the_floor_in_force"] else ""
            lines.append(
                f"| {row['floor']:.4f}{mark} | {row['days_below_floor']} | {row['valid_days']} | "
                f"{row['g6']:.6f} | {'PASS' if row['g6_passes'] else '**FAIL**'} |"
            )
        lines += [
            "",
            "0.005 分箱的天数分布（0.87–0.96）："
            + ", ".join(
                f"{key}:{count}" for key, count in result["histogram_0_005_bins_0_87_to_0_96"].items()
            ),
            "",
        ]

    args.out_json.write_text(
        json.dumps(
            {
                "g6_threshold": G6_THRESHOLD,
                "floor_in_force": FLOOR_IN_FORCE,
                "floor_sweep": list(FLOOR_SWEEP),
                "configs": configs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    for label, result in configs.items():
        at_floor = next(r for r in result["sweep"] if r["is_the_floor_in_force"])
        print(
            f"{label}: min={result['coverage_min']:.6f} "
            f"at floor {FLOOR_IN_FORCE} -> {at_floor['days_below_floor']} days below, "
            f"G6={at_floor['g6']:.6f} ({'PASS' if at_floor['g6_passes'] else 'FAIL'}); "
            f"largest floor keeping every day = "
            f"{result['largest_floor_that_keeps_every_day']:.6f}; "
            f"days within 0.005 above floor = {result['days_within_0_005_above_the_floor']}"
        )
        next_up = next((r for r in result["sweep"] if r["floor"] > FLOOR_IN_FORCE), None)
        if next_up is not None:
            print(
                f"    地板上调到 {next_up['floor']:.3f} -> {next_up['days_below_floor']} 天跌破, "
                f"G6={next_up['g6']:.6f} ({'PASS' if next_up['g6_passes'] else 'FAIL'})"
            )
    print(f"\nwrote {args.out_json}\nwrote {args.out_md}")


if __name__ == "__main__":
    main()
