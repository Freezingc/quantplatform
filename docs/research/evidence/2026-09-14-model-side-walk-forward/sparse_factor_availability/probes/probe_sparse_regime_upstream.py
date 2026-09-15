#!/usr/bin/env python3
"""Read-only: are the six sparse factors' day-level availability switches upstream, or ours?

§BA measured that the six candidate_only factors are missing on whole days together -- daily coverage
std 0.144-0.229, i.e. 17.7-28.2x what independent per-security missingness would give, with
yeslab_372/389/390 swinging between 0.019 and 1.000 -- and that the two co-moving clusters
{372,389,390} and {375,384,406} cut ACROSS the two source tables. The catalog says all six are
`availability_policy: SAME_DAY_CLOSE` with `max_staleness_calendar_days: 0`, so a day the producer did
not publish becomes a NaN day here with no forward fill.

This settles upstream-versus-ours with one comparison per factor per day: read the SAME day from the
SAME source table the materialization reads, and compare its row count with the row count that reached
the frozen factor_daily table.

  zero rows upstream                     -> producer-side gap; nothing to fix here, hand to the owner
  rows upstream, zero in factor_daily    -> our materialization dropped a usable day; a bug here
  rows in both, panel still NaN          -> the loss is between factor_daily and the assembled panel

Controls matter more than targets: yeslab_381 (GP_012) sits on factor_data8 with 372/375/384 and
yeslab_425 (GP_056) sits on factor_data9 with 389/390/406, and both are complete on all 969 OOS days
(min daily coverage 0.9834). If a whole table went dark on a day, those two would be empty too.

SAFETY: SELECT-only against dfs://YesLab and dfs://QuantResearchV2Test. No writes to any database, no
artifact mutation, no weight loading, and nothing the M2 driver imports at runtime is edited -- the
shared read-only connection helper is imported, not changed.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
FOLD_DATASETS = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
# source_catalog.yaml, verified by reading it -- table, source factor_name, factor_id, role
CATALOG = (
    ("factor_data8", "GP_003", "yeslab_372", "target cluster-A"),
    ("factor_data9", "GP_020", "yeslab_389", "target cluster-A"),
    ("factor_data9", "GP_021", "yeslab_390", "target cluster-A"),
    ("factor_data8", "GP_006", "yeslab_375", "target cluster-B"),
    ("factor_data8", "GP_015", "yeslab_384", "target cluster-B"),
    ("factor_data9", "GP_037", "yeslab_406", "target cluster-B"),
    ("factor_data8", "GP_012", "yeslab_381", "CONTROL same table as 372/375/384, dense"),
    ("factor_data9", "GP_056", "yeslab_425", "CONTROL same table as 389/390/406, dense"),
)
COL_OF = {"yeslab_372": "f02", "yeslab_375": "f03", "yeslab_384": "f05",
          "yeslab_389": "f06", "yeslab_390": "f07", "yeslab_406": "f08",
          "yeslab_381": "f04", "yeslab_425": "f09"}
CLUSTER_A = ["yeslab_372", "yeslab_389", "yeslab_390"]
CLUSTER_B = ["yeslab_375", "yeslab_384", "yeslab_406"]
TARGET_DB = "dfs://QuantResearchV2Test"


def panels(artifacts: Path) -> pd.DataFrame:
    cols = ["signal_date", "label_valid", "raw_return_5d"] + sorted(set(COL_OF.values()))
    frames = []
    for ds in FOLD_DATASETS.values():
        for ws in WORKSPACES:
            p = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / "split=test" / "*" / "*.parquet")))
            if p:
                frames.append(pd.concat([pd.read_parquet(x, columns=cols) for x in p], ignore_index=True))
                break
    panel = pd.concat(frames, ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"]).dt.date
    return panel.loc[
        panel["label_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(panel["raw_return_5d"], errors="coerce"))
    ]


def pick_days(daily: pd.DataFrame, per_cluster: int) -> list[tuple[str, str]]:
    out = []
    for tag, members in (("A", CLUSTER_A), ("B", CLUSTER_B)):
        mean_cov = daily[members].mean(axis=1).sort_values()
        for d in mean_cov.index[:per_cluster]:
            out.append((f"cluster-{tag}-lowest", d))
        for d in mean_cov.index[-per_cluster:]:
            out.append((f"cluster-{tag}-highest", d))
        mid = mean_cov.iloc[(len(mean_cov) // 2) - 1:(len(mean_cov) // 2) + 1]
        for d in mid.index:
            out.append((f"cluster-{tag}-median", d))
    seen, uniq = set(), []
    for tag, d in out:
        if d not in seen:
            seen.add(d)
            uniq.append((tag, d))
    return uniq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "sparse_upstream")
    ap.add_argument("--per-cluster", type=int, default=3)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query  # read-only helper, imported not modified

    panel = panels(args.artifacts)
    daily = pd.DataFrame({
        n: pd.to_numeric(panel[COL_OF[n]], errors="coerce").notna()
        .groupby(panel["signal_date"].to_numpy()).mean()
        for n in COL_OF
    }).sort_index()
    print(f"panel: {len(panel)} scoreable rows, {len(daily)} days", flush=True)

    days = pick_days(daily, args.per_cluster)
    holder = new_holder()
    result = {
        "question": (
            "Are the six sparse factors' day-level availability switches upstream (the producer did not "
            "publish that day) or ours (the day exists upstream but never reached factor_daily, or "
            "reached it but never reached the panel)?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test; no writes anywhere",
        "catalog_used": [
            {"table": t, "source_factor_name": n, "factor_id": f, "role": r} for t, n, f, r in CATALOG
        ],
        "panel_days": int(len(daily)),
        "panel_scoreable_rows": int(len(panel)),
        "days_examined": [],
    }

    for tag, day in days:
        dl = day.isoformat().replace("-", ".")
        row = {"tag": tag, "signal_date": day.isoformat(),
               "panel_daily_coverage": {n: float(daily.loc[day, n]) for n in COL_OF}}
        for table in ("factor_data8", "factor_data9"):
            names = [n for t, n, _, _ in CATALOG if t == table]
            quoted = ", ".join(f"'{n}'" for n in names)
            sql = (
                f"select factor_name, count(*) as rows_all, count(value) as rows_with_value "
                f"from loadTable('dfs://YesLab', '{table}') "
                f"where trade_date = date('{dl}') and factor_name in ({quoted}) "
                f"group by factor_name"
            )
            ok, frame = query(holder, sql)
            got = {}
            if ok and isinstance(frame, pd.DataFrame) and not frame.empty:
                for rec in frame.to_dict("records"):
                    got[str(rec["factor_name"])] = {
                        "rows_all": int(rec["rows_all"]),
                        "rows_with_value": int(rec["rows_with_value"]),
                    }
            for n in names:
                row.setdefault("upstream", {})[n] = got.get(n, {"rows_all": 0, "rows_with_value": 0})
        ids = ", ".join(f"'{f}'" for _, _, f, _ in CATALOG)
        sql = (
            f"select factor_id, data_version, count(*) as rows_all, count(value) as rows_with_value "
            f"from loadTable('{TARGET_DB}', 'factor_daily') "
            f"where signal_date = date('{dl}') and factor_id in ({ids}) "
            f"group by factor_id, data_version"
        )
        ok, frame = query(holder, sql)
        frozen = {}
        if ok and isinstance(frame, pd.DataFrame) and not frame.empty:
            for rec in frame.to_dict("records"):
                frozen.setdefault(str(rec["factor_id"]), {})[str(rec["data_version"])[:12]] = {
                    "rows_all": int(rec["rows_all"]), "rows_with_value": int(rec["rows_with_value"])
                }
        row["frozen_factor_daily_by_version"] = {
            f: frozen.get(f, {}) for _, _, f, _ in CATALOG
        }
        # verdict per factor
        verdicts = {}
        for _, name, fid, _ in CATALOG:
            up = row["upstream"][name]["rows_with_value"]
            fz = sum(v["rows_with_value"] for v in row["frozen_factor_daily_by_version"][fid].values())
            cov = row["panel_daily_coverage"][fid]
            if up == 0 and fz == 0:
                v = "ABSENT_UPSTREAM_AND_FROZEN"
            elif up == 0 and fz > 0:
                v = "FROZEN_HAS_ROWS_UPSTREAM_DOES_NOT"
            elif up > 0 and fz == 0:
                v = "UPSTREAM_HAS_ROWS_FROZEN_DOES_NOT"
            elif cov < 0.20 and fz > 0:
                v = "ROWS_REACHED_FROZEN_BUT_PANEL_IS_STILL_EMPTY"
            else:
                v = "CONSISTENT"
            verdicts[fid] = {"upstream_rows_with_value": up, "frozen_rows_with_value": fz,
                             "panel_coverage": cov, "verdict": v}
        row["verdicts"] = verdicts
        result["days_examined"].append(row)
        print(f"  {day} [{tag}] " + " ".join(
            f"{fid.replace('yeslab_','')}={verdicts[fid]['verdict'].split('_')[0]}("
            f"{verdicts[fid]['upstream_rows_with_value']}/{verdicts[fid]['frozen_rows_with_value']}/"
            f"{verdicts[fid]['panel_coverage']:.2f})"
            for _, _, fid, _ in CATALOG), flush=True)

    tally = {}
    for row in result["days_examined"]:
        for fid, v in row["verdicts"].items():
            tally.setdefault(fid, {}).setdefault(v["verdict"], 0)
            tally[fid][v["verdict"]] += 1
    result["verdict_tally"] = tally
    result["reading"] = (
        "ABSENT_UPSTREAM_AND_FROZEN on the low-coverage days, with the two dense controls (381 on "
        "factor_data8, 425 on factor_data9) present on the very same days, is a producer-side "
        "per-factor publication gap: the table was not dark, those six names were not published. "
        "Any UPSTREAM_HAS_ROWS_FROZEN_DOES_NOT or ROWS_REACHED_FROZEN_BUT_PANEL_IS_STILL_EMPTY would "
        "instead be a defect on our side and would need its own section."
    )
    out = args.out / "sparse_regime_upstream.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(tally, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
