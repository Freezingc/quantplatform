#!/usr/bin/env python3
"""Why the frozen best_simple baseline loses rows: dense-factor coverage, measured per year.

The chain this explains, all of it measured rather than assumed:

1. `evaluate_frozen_oos` compares a candidate with the frozen best_simple baseline on the
   INTERSECTION of their keys, and `ScoreHealthEvidence.passes` then requires that intersection to
   cover at least 0.90 of the day's valid-label rows.
2. The baseline artifact exists exactly on the rows where all nine dense baseline-eligible factors
   are finite (whole-fold set equality, verified separately), while a candidate scores every row.
3. So a day is dropped when the dense factors are missing for more than 10% of that day's
   valid-label cross section -- and 24 OOS days in the frozen-v2 runs are dropped that way
   (1 in 2022, 23 in 2023), which is the whole of G6 = 945/969.

This script answers the question that leaves open: WHICH factors, for WHICH securities, over WHAT
period. It reads only the published dataset parquet under a run workspace -- no database, no model
weights, no predictions -- and writes JSON plus a markdown table set.

The dense/sparse split is parsed out of the driver's own source with `ast` rather than retyped, so
this cannot quietly disagree with the factor set the runs actually used.

Usage:
    analyze_dense_factor_coverage.py --workspace /data/.../m0r_v2 [--years 2022,2023] \
        [--repo /data/.../Quant-Platform-Prod_CODEX] [--out-json X.json] [--out-md X.md]
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
FACTOR_SOURCE_FILE = "research_v2/scripts/walk_forward_factor_daily.py"
STAR_PREFIXES = ("688", "689")  # STAR Market, where the gap turned out to be concentrated


def parse_factor_sets(repo: Path) -> dict[str, list[str]]:
    """Read the frozen factor sets out of the driver's own source (no import, no retyping)."""
    path = repo / FACTOR_SOURCE_FILE
    if not path.exists():
        raise SystemExit(
            f"{path} not found: --repo must point at the repository root "
            "(the directory that CONTAINS research_v2), not at research_v2 itself"
        )
    tree = ast.parse(path.read_text())
    wanted = {"FROZEN_FACTOR_SET_V2", "FROZEN_FACTOR_SET_V2_DENSE"}
    found: dict[str, list[str]] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in wanted
        ):
            found[node.target.id] = list(ast.literal_eval(node.value))
    if not wanted.issubset(found):
        raise SystemExit(f"could not parse {sorted(wanted)} out of {path}")
    return found


def dense_columns(v2: list[str], dense: list[str]) -> list[str]:
    """Feature names (f00..f14) of the dense factors, in the canonical order the runs used."""
    return [f"f{v2.index(factor):02d}" for factor in dense]


def missing_combinations(incomplete: Any, columns: list[str], top: int = 5) -> dict[str, int]:
    """Which factors are missing TOGETHER -- a single factor and a shared dependency look alike
    in a per-column NaN table and completely different here."""
    if incomplete.empty:
        return {}
    import numpy as np

    mask = incomplete[columns].isna().to_numpy()
    names = np.array(columns)
    # Iterate the ROWS that have at least one gap; `mask.any(axis=1)` is the row filter, not the
    # thing to iterate. Iterating it yields booleans and numpy then indexes names with True/False,
    # which silently produces the wrong combinations instead of an error.
    combos = [",".join(sorted(names[row])) for row in mask[mask.any(axis=1)]]
    counts: dict[str, int] = {}
    for combo in combos:
        counts[combo] = counts.get(combo, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top])


def nesting(frame: Any, inner: str, outer: str) -> dict[str, int]:
    """Is one factor's missing set a subset of the other's? `inner_only == 0` means yes."""
    inner_missing = frame[inner].isna()
    outer_missing = frame[outer].isna()
    return {
        "inner": inner,
        "outer": outer,
        "inner_missing_rows": int(inner_missing.sum()),
        "outer_missing_rows": int(outer_missing.sum()),
        "both_missing_rows": int((inner_missing & outer_missing).sum()),
        "inner_only_rows": int((inner_missing & ~outer_missing).sum()),
        "outer_only_rows": int((outer_missing & ~inner_missing).sum()),
    }


def code_prefixes(codes: Any, width: int = 3) -> dict[str, int]:
    import pandas as pd

    series = pd.Series([str(code)[:width] for code in codes])
    return {str(key): int(value) for key, value in series.value_counts().sort_index().items()}


def valid_label_rows(frame: Any) -> Any:
    """The rows the evaluator's coverage denominator is built from."""
    import numpy as np
    import pandas as pd

    # Same semantics as the evaluator's own `label_valid` handling, without the object-dtype
    # fillna downcast warning: nullable boolean, missing means not valid.
    label_valid = frame["label_valid"].astype("boolean").fillna(False).astype(bool)
    finite_return = np.isfinite(pd.to_numeric(frame["raw_return_5d"], errors="coerce"))
    return frame.loc[label_valid & finite_return]


def analyse_split(paths: list[Path], columns: list[str]) -> dict[str, Any]:
    """Everything this script reports about one test-year split, from its published parquet."""
    import pandas as pd

    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["day"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m-%d")
    frame["month"] = pd.to_datetime(frame["signal_date"]).dt.strftime("%Y-%m")
    scored = valid_label_rows(frame)
    incomplete_mask = ~scored[columns].notna().all(axis=1)
    incomplete = scored.loc[incomplete_mask]
    per_day = scored.assign(incomplete=incomplete_mask).groupby("day")["incomplete"].mean()
    per_month = scored.assign(incomplete=incomplete_mask).groupby("month")["incomplete"].mean()
    per_factor = {column: float(scored[column].isna().mean()) for column in columns}
    persistence = incomplete.groupby("ts_code")["day"].nunique()
    affected = sorted(incomplete["ts_code"].unique())
    all_codes = sorted(scored["ts_code"].unique())
    # The two most-missing dense factors, and whether one's missing set nests inside the other's.
    # A per-column table cannot tell "two independent gaps" from "one shared dependency", and the
    # distinction is the whole question: nested gaps mean fixing one factor fixes the pair.
    worst_two = sorted(per_factor, key=lambda column: -per_factor[column])[:2]
    top_pair_nesting = (
        nesting(scored, worst_two[0], worst_two[1]) if len(worst_two) == 2 else None
    )
    return {
        "valid_label_rows": int(len(scored)),
        "distinct_securities": len(all_codes),
        "dense_incomplete_rows": int(len(incomplete)),
        "dense_incomplete_share": float(len(incomplete) / len(scored)) if len(scored) else None,
        "per_factor_missing_share": per_factor,
        "top_missing_combinations": missing_combinations(incomplete, columns),
        "top_pair_nesting": top_pair_nesting,
        "per_month_incomplete_share": {
            str(month): float(share) for month, share in per_month.sort_index().items()
        },
        "per_day_incomplete_share": {
            "min": float(per_day.min()),
            "median": float(per_day.median()),
            "max": float(per_day.max()),
            "days": int(len(per_day)),
            "days_above_0.10": int((per_day > 0.10).sum()),
            "worst_5_days": {
                str(day): float(share) for day, share in per_day.nlargest(5).items()
            },
            "best_3_days": {
                str(day): float(share) for day, share in per_day.nsmallest(3).items()
            },
        },
        "affected_securities": {
            "count": len(affected),
            "share_of_frame": float(len(affected) / len(all_codes)) if all_codes else None,
            "code_prefixes": code_prefixes(affected),
            "frame_code_prefixes": code_prefixes(all_codes),
            "star_market_in_frame": int(
                sum(1 for code in all_codes if code.startswith(STAR_PREFIXES))
            ),
            "star_market_affected": int(
                sum(1 for code in affected if str(code).startswith(STAR_PREFIXES))
            ),
            "securities_incomplete_on_20_or_more_days": int((persistence >= 20).sum()),
            "top_5_by_days": {
                str(code): int(days) for code, days in persistence.nlargest(5).items()
            },
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Dense-factor coverage behind the dropped OOS days",
        "",
        f"Workspace `{result['workspace']}`. Dense set (parsed from the driver's own source): "
        + ", ".join(result["dense_factors"]),
        "",
        "| Year | Valid-label rows | Dense-incomplete | Share | Securities affected | "
        "STAR in frame | STAR affected | Days above 0.10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for year, entry in sorted(result["years"].items()):
        affected = entry["affected_securities"]
        lines.append(
            f"| {year} | {entry['valid_label_rows']} | {entry['dense_incomplete_rows']} | "
            f"{entry['dense_incomplete_share']:.4f} | {affected['count']} | "
            f"{affected['star_market_in_frame']} | {affected['star_market_affected']} | "
            f"{entry['per_day_incomplete_share']['days_above_0.10']} |"
        )
    lines += ["", "## Per-month incomplete share", "", "| Year-month | Share |", "|---|---|"]
    for _year, entry in sorted(result["years"].items()):
        for month, share in entry["per_month_incomplete_share"].items():
            lines.append(f"| {month} | {share:.4f} |")
    lines += ["", "## Which factors are missing together (top combinations per year)", ""]
    for year, entry in sorted(result["years"].items()):
        lines.append(f"* **{year}**: " + "; ".join(
            f"{combo} = {count}" for combo, count in entry["top_missing_combinations"].items()
        ))
        pair = entry.get("top_pair_nesting")
        if pair:
            nested = "nested (one factor's gaps are a subset of the other's)" if (
                pair["inner_only_rows"] == 0 or pair["outer_only_rows"] == 0
            ) else "NOT nested (two independent gaps)"
            lines.append(
                f"  * two most-missing factors `{pair['inner']}` and `{pair['outer']}`: "
                f"both missing on {pair['both_missing_rows']} rows, "
                f"{pair['inner']} only on {pair['inner_only_rows']}, "
                f"{pair['outer']} only on {pair['outer_only_rows']} -> {nested}"
            )
    lines += ["", "## Per-factor missing share", "", "| Factor | " +
              " | ".join(sorted(result["years"])) + " |", "|---" * (len(result["years"]) + 1) + "|"]
    for column, factor in zip(result["dense_columns"], result["dense_factors"], strict=True):
        cells = [
            f"{result['years'][year]['per_factor_missing_share'][column]:.4f}"
            for year in sorted(result["years"])
        ]
        lines.append(f"| {factor} ({column}) | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Read this with `analyze_yearly_baseline_deltas.py`: that script shows the days these rows "
        "cost the evaluation, this one shows which factor and which securities they belong to.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--years", default=None, help="comma-separated; default every test split")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    sets = parse_factor_sets(args.repo)
    v2 = sets["FROZEN_FACTOR_SET_V2"]
    dense = sets["FROZEN_FACTOR_SET_V2_DENSE"]
    columns = dense_columns(v2, dense)
    wanted = args.years.split(",") if args.years else None

    years: dict[str, Any] = {}
    for path in sorted(args.workspace.glob("datasets/*/split=test/year=*")):
        year = path.name.split("=")[1]
        if wanted is not None and year not in wanted:
            continue
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            continue
        print(f"  {year}: reading {len(shards)} shard(s) from {path.name}", flush=True)
        years[year] = analyse_split(shards, columns)
    result = {
        "workspace": str(args.workspace),
        "dense_factors": dense,
        "dense_columns": columns,
        "sparse_factors": [factor for factor in v2 if factor not in dense],
        "years": years,
    }
    markdown = render_markdown(result)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
