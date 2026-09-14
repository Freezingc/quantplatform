"""Did a repair/recompute overlay touch yeslab_357 / yeslab_471? That would name the mechanism.

Two facts need explaining together: the two factors lost their STAR-Market rows on the SAME months
with IDENTICAL counts (0 / 0 / 1286 / 5522 for 2023-07 / 08 / 09 / 10) even though they live on
different YesLab tables with different owners, and a control factor on each of those tables kept
full STAR coverage in the same 2026-09-03 run under the same data_version. A shared cause has to be
something both factors go through that the controls do not. The known-formula recompute path is
ruled out (`_KNOWN_FORMULA_IDS` is yeslab_391 / yeslab_433), which leaves the repair overlay, whose
decisions are persisted to `factor_repair_audit` and whose rejects go to `factor_repair_quarantine`.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes, no weights. Run from research_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402

OUT = Path("/data/zhangyudong/qrv2_research_artifacts/repair_audit_357_471_probe.json")
IDS = "'yeslab_357', 'yeslab_471', 'yeslab_381', 'yeslab_464', 'yeslab_484'"


def _run(holder: dict, sql: str) -> pd.DataFrame:
    ok, frame = query(holder, sql)
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"query failed: {sql}")
    return frame


def main() -> None:
    holder = new_holder()
    result: dict = {}
    for table in ("factor_repair_audit", "factor_repair_quarantine"):
        columns = list(_run(holder, f"select top 1 * from loadTable('{TARGET_DATABASE}','{table}')").columns)
        total = int(_run(holder, f"select count(*) as n from loadTable('{TARGET_DATABASE}','{table}')")["n"].iloc[0])
        print(f"\n=== {table}: {total} rows total ===\n  columns: {columns}", flush=True)
        result[table] = {"columns": columns, "total_rows": total}
        if not total:
            continue
        frame = _run(
            holder,
            f"select * from loadTable('{TARGET_DATABASE}','{table}') where factor_id in [{IDS}]",
        )
        print(f"  rows for the five factors under test: {len(frame)}", flush=True)
        if frame.empty:
            result[table]["rows_for_factors"] = 0
            continue
        keep = [c for c in ("factor_id", "as_of_date", "logical_partition", "repair_method",
                            "status", "decision", "reason", "created_at", "run_id") if c in frame.columns]
        grouped = (
            frame.loc[:, keep]
            .astype(str)
            .groupby([c for c in keep if c not in ("created_at", "run_id", "as_of_date", "logical_partition")], dropna=False)
            .size()
            .reset_index(name="n")
        )
        result[table]["rows_for_factors"] = int(len(frame))
        result[table]["grouped"] = grouped.to_dict(orient="records")
        for row in grouped.itertuples(index=False):
            print(f"    {row._asdict()}", flush=True)
        for extra in ("as_of_date", "logical_partition", "created_at"):
            if extra in frame.columns:
                values = frame[extra].astype(str)
                print(f"    {extra}: [{values.min()} .. {values.max()}]", flush=True)
                result[table][f"{extra}_range"] = [str(values.min()), str(values.max())]

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
