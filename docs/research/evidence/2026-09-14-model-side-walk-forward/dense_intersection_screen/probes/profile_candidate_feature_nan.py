"""What the candidate models actually train on: the NaN profile of all 15 frozen columns, per test year.

The frozen chain does not drop rows with missing factors -- `run_real_candidate_training.py` keeps
NaN and `FoldPreprocessor` handles it -- so the candidates consume all fifteen columns on every
valid-label row. The coverage screen measures those columns one at a time against the universe and
says six of them sit at median 0.46-0.49 in the OOS window. What nobody has written down is what
that means for the matrix the models actually see: how many cells are missing in total, how many
rows carry at least one missing value, and whether the missingness is spread or concentrated.

This reads the published dataset parquet under a run workspace -- no database, no weights, no
predictions -- and reports, per test year:

  * per-column missing share for all fifteen feature columns, dense and sparse alike;
  * the share of rows with at least one missing value (what the preprocessor has to cope with);
  * the distribution of how many columns are missing per row;
  * the same numbers restricted to the nine dense columns, so the baseline's row set (all nine
    finite) can be read off against the candidate's.

Usage: profile_candidate_feature_nan.py --workspace /data/.../m0r_v2 [--out out.json]
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
FACTOR_SOURCE_FILE = "research_v2/scripts/walk_forward_factor_daily.py"


def parse_factor_sets(repo: Path) -> dict[str, list[str]]:
    """Read the frozen sets out of the driver's own source, so this cannot disagree with the runs."""
    path = repo / FACTOR_SOURCE_FILE
    tree = ast.parse(path.read_text(encoding="utf-8"))
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


def feature_columns(v2: list[str]) -> list[str]:
    return [f"f{index:02d}" for index in range(len(v2))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    sets = parse_factor_sets(args.repo)
    v2 = sets["FROZEN_FACTOR_SET_V2"]
    dense = sets["FROZEN_FACTOR_SET_V2_DENSE"]
    columns = feature_columns(v2)
    dense_columns = [columns[v2.index(factor)] for factor in dense]
    print(f"workspace {args.workspace}")
    print(f"{len(v2)} feature columns {columns[0]}..{columns[-1]}; {len(dense_columns)} dense")

    result: dict[str, Any] = {
        "workspace": str(args.workspace),
        "factor_ids": v2,
        "columns": columns,
        "dense_columns": dense_columns,
        "years": {},
    }

    for path in sorted(args.workspace.glob("datasets/*/split=test/year=*")):
        year = path.name.split("=")[-1]
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            continue
        frame = pd.concat([pd.read_parquet(shard) for shard in shards], ignore_index=True)
        label_valid = frame["label_valid"].astype("boolean").fillna(False).astype(bool)
        finite_return = np.isfinite(pd.to_numeric(frame["raw_return_5d"], errors="coerce"))
        scored = frame.loc[label_valid & finite_return]
        missing = scored[columns].isna()
        per_column = {column: float(missing[column].mean()) for column in columns}
        missing_count_per_row = missing.sum(axis=1)
        dense_missing = scored[dense_columns].isna()
        entry = {
            "rows": int(len(scored)),
            "cells": int(len(scored) * len(columns)),
            "missing_cells": int(missing.to_numpy().sum()),
            "missing_cell_share": float(missing.to_numpy().mean()),
            "rows_with_any_missing": int((missing_count_per_row > 0).sum()),
            "rows_with_any_missing_share": float((missing_count_per_row > 0).mean()),
            "missing_columns_per_row": {
                str(k): int(v)
                for k, v in missing_count_per_row.value_counts().sort_index().items()
            },
            "per_column_missing_share": per_column,
            "dense_rows_all_finite": int((~dense_missing.any(axis=1)).sum()),
            "dense_rows_all_finite_share": float((~dense_missing.any(axis=1)).mean()),
        }
        result["years"][year] = entry
        print(f"\n=== test year {year}: {entry['rows']} valid-label rows ===")
        print(
            f"  missing cells {entry['missing_cells']:,} / {entry['cells']:,} "
            f"= {entry['missing_cell_share']:.4f}"
        )
        print(
            f"  rows with >=1 missing: {entry['rows_with_any_missing']:,} "
            f"= {entry['rows_with_any_missing_share']:.4f}"
        )
        print(
            f"  rows with all nine dense finite (the baseline's row set): "
            f"{entry['dense_rows_all_finite']:,} = {entry['dense_rows_all_finite_share']:.4f}"
        )
        print("  missing columns per row: " + ", ".join(
            f"{k}:{v:,}" for k, v in sorted(entry["missing_columns_per_row"].items(),
                                             key=lambda item: int(item[0]))
        ))
        print("  per-column missing share:")
        for column, factor in zip(columns, v2, strict=True):
            tag = "dense " if column in dense_columns else "sparse"
            print(f"    {column} {factor:<28} {tag} {per_column[column]:.4f}")
        del frame, scored, missing

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
