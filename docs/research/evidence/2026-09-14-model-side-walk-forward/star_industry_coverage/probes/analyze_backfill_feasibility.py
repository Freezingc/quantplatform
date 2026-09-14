"""Option 1's only unverified precondition: can the vendor supply STAR industry membership before 2023-09-25?

Two membership APIs exist at the vendor. The cache in use was built from ``index_member(index_code=...)``.
This measures the other one, ``index_member_all``, against it -- per day, on the 24 OOS days that the
0.90 intersection floor drops -- and separately measures how much the two disagree where they overlap,
because a backfill that silently restates labels already baked into published factors is a bigger
lineage cost than a backfill that only adds rows.

Reads: vendor JSON already fetched under --raw, the existing local cache, and the instrument's per-day
output. Writes: --out only. No database, no repo file, no weights.
"""
from __future__ import annotations
import argparse, glob, hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd

STAR = ("688", "689")
CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
PERDAY = Path("/tmp/dense_intersection_2022_2025_v2.json")


def _interval(members: pd.DataFrame) -> pd.DataFrame:
    m = members.copy()
    m["in_p"] = pd.to_datetime(m["in_date"].astype(str), format="%Y%m%d", errors="coerce")
    out = m["out_date"]
    m["out_p"] = pd.to_datetime(out.astype(str).str.replace(r"\.0$", "", regex=True),
                                format="%Y%m%d", errors="coerce")
    m.loc[out.isna() | (out.astype(str).isin(["None", "nan", "", "NaT"])), "out_p"] = pd.NaT
    return m.dropna(subset=["in_p"])


def active_by_day(m: pd.DataFrame, days: pd.DatetimeIndex, code_col: str) -> pd.Series:
    """active(d) = #(in_date <= d) - #(out_date <= d); right-exclusive, the vendor's own rule."""
    ins = np.sort(m["in_p"].to_numpy())
    outs = np.sort(m["out_p"].dropna().to_numpy())
    dv = days.to_numpy()
    return pd.Series(np.searchsorted(ins, dv, "right") - np.searchsorted(outs, dv, "right"), index=days)


def active_map(m: pd.DataFrame, day: pd.Timestamp, code_col: str) -> dict[str, str]:
    """security -> l1 index_code active on `day` (right-exclusive)."""
    sel = m[(m["in_p"] <= day) & (m["out_p"].isna() | (m["out_p"] > day))]
    out: dict[str, list[str]] = {}
    for _, r in sel.iterrows():
        out.setdefault(str(r[code_col]), []).append(str(r["l1_code"] if "l1_code" in m.columns else r["index_code"]))
    return {k: (v[0] if len(v) == 1 else "|".join(sorted(set(v)))) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # ---- source A: the cache in use (built from index_member) ----
    files = sorted(glob.glob(str(CACHE / "pit_industry_members_*_SI.json")))
    a_frames = [pd.read_json(Path(f), orient="table") for f in files]
    A = pd.concat(a_frames, ignore_index=True)
    A = A.rename(columns={"con_code": "code", "index_code": "l1_code"})
    A = _interval(A)
    a_digest = hashlib.sha256()
    for f in files:
        a_digest.update(hashlib.sha256(Path(f).read_bytes()).digest())

    # ---- source B: index_member_all, both is_new halves ----
    B_raw = json.loads((args.raw / "member_all_combined.json").read_text())
    B = pd.DataFrame(B_raw).rename(columns={"ts_code": "code"})
    B = _interval(B)

    perday = json.loads(PERDAY.read_text())
    below = [r for r in perday["per_day"] if r["below_floor"]]
    days = pd.DatetimeIndex([pd.Timestamp(r["day"]) for r in below])
    floor = float(perday["floor"])

    starA = A[A["code"].str.startswith(STAR)]
    starB = B[B["code"].str.startswith(STAR)]

    # full-window daily series for the shape of the hole
    sweep = pd.bdate_range("2019-06-01", "2025-12-31")
    serA = active_by_day(starA, sweep, "code")
    serB = active_by_day(starB, sweep, "code")

    rows = []
    for r, day in zip(below, days):
        V, N = int(r["valid_label_rows"]), int(r["intersection_rows"])
        need = math.ceil(floor * V) - N
        mA = active_map(starA, day, "code")
        mB = active_map(starB, day, "code")
        gained = sorted(set(mB) - set(mA))
        lost = sorted(set(mA) - set(mB))
        disagree = sorted(k for k in set(mA) & set(mB) if mA[k] != mB[k])
        rows.append({
            "day": r["day"], "valid_label_rows": V, "intersection_rows": N,
            "coverage": r["coverage"], "rows_needed_to_reach_floor": need,
            "star_active_cache": len(mA), "star_active_member_all": len(mB),
            "star_gained": len(gained), "star_lost": len(lost), "star_label_disagreed": len(disagree),
            "gained_covers_need": bool(len(gained) >= need),
            "headroom_gained_minus_need": len(gained) - need,
        })

    # ---- restatement risk: agreement where BOTH sources have a STAR label, OOS window ----
    oos = pd.bdate_range("2023-09-25", "2025-12-31")
    agree_days, agree_tot, agree_same, multi_l1 = [], 0, 0, 0
    for day in oos[::10]:  # every 10th business day: enough to characterise, cheap to compute
        mA, mB = active_map(starA, day, "code"), active_map(starB, day, "code")
        both = set(mA) & set(mB)
        same = sum(1 for k in both if mA[k] == mB[k])
        multi_l1 += sum(1 for v in mB.values() if "|" in v)
        agree_tot += len(both); agree_same += same
        agree_days.append({"day": str(day.date()), "both": len(both), "same": same,
                           "agreement": (same / len(both)) if both else None})

    # ---- coverage of codes the cache never had ----
    codesA, codesB = set(starA["code"]), set(starB["code"])

    out = {
        "question": "Can the vendor supply STAR-Market industry membership before 2023-09-25?",
        "answer": None,  # filled below
        "two_apis": {
            "cache_in_use": {
                "api": "index_member(index_code=...)",
                "files": len(files), "combined_sha256": a_digest.hexdigest(), "rows": int(len(A)),
                "star_rows": int(len(starA)), "star_codes": len(codesA),
                "star_min_in_date": str(starA["in_p"].min().date()),
                "all_min_in_date": str(A["in_p"].min().date()),
            },
            "alternative": {
                "api": "index_member_all(l1_code=..., is_new in {Y,N})",
                "rows": int(len(B)), "star_rows": int(len(starB)), "star_codes": len(codesB),
                "star_min_in_date": str(starB["in_p"].min().date()),
                "all_min_in_date": str(B["in_p"].min().date()),
            },
        },
        "star_codes_only_in_member_all": sorted(codesB - codesA),
        "star_codes_only_in_cache": sorted(codesA - codesB),
        "problem_days": rows,
        "problem_days_all_covered": all(r["gained_covers_need"] for r in rows),
        "problem_days_covered_count": sum(1 for r in rows if r["gained_covers_need"]),
        "min_headroom_gained_minus_need": min(r["headroom_gained_minus_need"] for r in rows),
        "daily_star_active_shape": {
            "note": "active STAR memberships per business day, both sources, 2019-06..2025-12",
            "first_day_cache_nonzero": str(serA[serA > 0].index.min().date()) if (serA > 0).any() else None,
            "first_day_member_all_nonzero": str(serB[serB > 0].index.min().date()) if (serB > 0).any() else None,
            "sampled": [{"day": str(d.date()), "cache": int(serA[d]), "member_all": int(serB[d])}
                        for d in sweep[::60]],
        },
        "restatement_risk_on_the_overlap": {
            "window": "2023-09-25..2025-12-31, every 10th business day",
            "pairs_compared": agree_tot, "labels_identical": agree_same,
            "agreement_rate": (agree_same / agree_tot) if agree_tot else None,
            "member_all_rows_with_multiple_active_l1": multi_l1,
            "per_day": agree_days,
        },
        "caveat": (
            "An industry label existing is a NECESSARY condition for yeslab_357/471 to be non-null, not a "
            "sufficient one: those factors are produced by the upstream pipeline, and this measures the "
            "label supply, not a replay of the pipeline. rows_needed_to_reach_floor is the arithmetic gap "
            "at the 0.90 floor; star_gained is an upper bound on how many rows a backfill could add."
        ),
        "database_writes": False, "weights_written": False,
    }
    ok = out["problem_days_all_covered"] and len(codesB - codesA) >= 0 and out["two_apis"]["alternative"]["star_min_in_date"] < "2023-09-25"
    out["answer"] = (
        "YES -- index_member_all carries STAR membership from "
        f"{out['two_apis']['alternative']['star_min_in_date']}, versus "
        f"{out['two_apis']['cache_in_use']['star_min_in_date']} in the cache built from index_member. "
        + ("On all 24 problem days the gained labels outnumber the rows needed to clear the 0.90 floor."
           if out["problem_days_all_covered"] else
           f"On {out['problem_days_covered_count']}/24 problem days the gained labels cover the gap.")
    ) if ok else "INCONCLUSIVE -- see per-day table"

    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("answer", "problem_days_all_covered",
                                         "problem_days_covered_count", "min_headroom_gained_minus_need")},
                     ensure_ascii=False, indent=1))
    print("two_apis:", json.dumps(out["two_apis"], ensure_ascii=False))
    print("only_in_member_all:", len(out["star_codes_only_in_member_all"]),
          "only_in_cache:", len(out["star_codes_only_in_cache"]))
    print("restatement:", json.dumps({k: v for k, v in out["restatement_risk_on_the_overlap"].items() if k != "per_day"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
