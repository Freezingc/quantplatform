#!/usr/bin/env python3
"""Read-only: does yeslab_363 exist upstream? §AM found it has zero rows in factor_daily.

§AM established that yeslab_363 sits in the materializer's PIT_FACTORS but in no frozen set, and has
zero published rows in factor_daily -- so it is a reserved-never-enabled entry rather than lost data.
It could not say whether the producer ever published it, because it only read the frozen table.

source_catalog.yaml gives the answer's address: yeslab_363 is `行业标准化绝对值除行业偏度绝对值` on
dfs://YesLab factor_data8, owner suyanning -- the SAME table as yeslab_357 (`行业内收益标准化绝对值`),
which is dense, and yeslab_381 (GP_012), also dense. So one table, three names, two of them provably
published every day. If the third has rows upstream, the gap is in our materialization; if it has none,
the factor was never produced and §AM's reading is confirmed from the source side.

DolphinDB note, because it cost two failed runs and the next person will hit it too: neither
`count(distinct x)` ("Cannot nest aggregate function") nor `nunique(x)` on a SYMBOL/STRING column
("Array vector does not support SYMBOL or STRING type") works in this dialect alongside other
aggregates. This probe therefore counts rows only -- `count(*)` and `count(value)` -- which is enough
for the question asked, and leaves distinct-day and distinct-code counts unmeasured.

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No writes, no weights, no artifact
mutation, and nothing the M2 driver imports is edited.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
ART = Path("/data/zhangyudong/qrv2_research_artifacts")
NAMES = (
    ("factor_data8", "行业标准化绝对值除行业偏度绝对值", "yeslab_363", "TARGET zero rows in factor_daily"),
    ("factor_data8", "行业内收益标准化绝对值", "yeslab_357", "CONTROL same table, dense in the panel"),
    ("factor_data8", "GP_012", "yeslab_381", "CONTROL same table, dense in the panel"),
)
YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
TARGET_DB = "dfs://QuantResearchV2Test"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "sparse_upstream")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query

    holder = new_holder()
    result = {
        "question": "Does yeslab_363 exist upstream at all, and how does it compare with two dense "
                    "controls on the same source table?",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test",
        "not_measured": (
            "distinct trade_date and distinct ts_code counts: this DolphinDB dialect rejects both "
            "count(distinct x) and nunique(x) on string columns alongside other aggregates, so only "
            "row counts are reported"
        ),
        "upstream_per_year": {},
        "frozen_factor_daily_per_year": {},
    }
    for table, name, fid, role in NAMES:
        if "'" in name:
            raise ValueError(f"name would need escaping: {name!r}")
        per_year = {}
        for y in YEARS:
            sql = (
                f"select count(*) as rows_all, count(value) as rows_with_value "
                f"from loadTable('dfs://YesLab', '{table}') "
                f"where factor_name = '{name}' and trade_date between {y}.01.01 : {y}.12.31"
            )
            ok, frame = query(holder, sql)
            rec = frame.to_dict("records")[0] if ok and isinstance(frame, pd.DataFrame) and not frame.empty else {}
            per_year[str(y)] = {k: int(rec.get(k, 0) or 0) for k in
                                ("rows_all", "rows_with_value")}
            print(f"  {fid:12} {y} upstream rows_all={per_year[str(y)]['rows_all']:>9} "
                  f"with_value={per_year[str(y)]['rows_with_value']:>9}", flush=True)
        result["upstream_per_year"][fid] = {"table": table, "source_factor_name": name,
                                            "role": role, "per_year": per_year}
        fz = {}
        for y in YEARS:
            sql = (
                f"select count(*) as rows_all, count(value) as rows_with_value "
                f"from loadTable('{TARGET_DB}', 'factor_daily') "
                f"where factor_id = '{fid}' and signal_date between {y}.01.01 : {y}.12.31"
            )
            ok, frame = query(holder, sql)
            rec = frame.to_dict("records")[0] if ok and isinstance(frame, pd.DataFrame) and not frame.empty else {}
            fz[str(y)] = {k: int(rec.get(k, 0) or 0) for k in ("rows_all", "rows_with_value")}
        result["frozen_factor_daily_per_year"][fid] = fz
        print(f"  {fid:12} frozen factor_daily totals: "
              + " ".join(f"{y}={fz[str(y)]['rows_with_value']}" for y in YEARS), flush=True)

    t363 = result["upstream_per_year"]["yeslab_363"]["per_year"]
    t357 = result["upstream_per_year"]["yeslab_357"]["per_year"]
    total_363 = sum(v["rows_with_value"] for v in t363.values())
    total_357 = sum(v["rows_with_value"] for v in t357.values())
    result["verdict"] = {
        "yeslab_363_upstream_valued_rows_2020_2025": total_363,
        "yeslab_357_upstream_valued_rows_2020_2025": total_357,
        "yeslab_363_frozen_valued_rows_2020_2025": sum(
            v["rows_with_value"] for v in result["frozen_factor_daily_per_year"]["yeslab_363"].values()
        ),
        "conclusion": (
            "the producer never published yeslab_363 on factor_data8 in 2020-2025, while both dense "
            "controls on the same table published throughout -- so §AM's reading is confirmed from the "
            "source side: yeslab_363 is a catalogued-but-never-produced factor, not data we lost"
            if total_363 == 0 and total_357 > 0
            else "yeslab_363 DOES have upstream rows, so its absence from factor_daily is a "
                 "materialization question and needs its own section"
        ),
    }
    out = args.out / "yeslab_363_upstream.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
