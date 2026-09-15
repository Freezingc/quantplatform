#!/usr/bin/env python3
"""Read-only local analysis of sparse_regime_upstream.json: what shape does each upstream gap have?

The probe's own verdict labels only asked "did rows reach factor_daily and the panel". The decisive
question is one level up: on a cluster's low day, did the producer emit NO rows, or emit rows with NULL
values, and how does that compare with the two dense controls sitting on the very same tables on the
very same day? This recomputes that from the JSON the probe already wrote -- no database, no re-query.
"""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
TARGETS_A = ["yeslab_372", "yeslab_389", "yeslab_390"]
TARGETS_B = ["yeslab_375", "yeslab_384", "yeslab_406"]
CONTROL_OF = {"factor_data8": "yeslab_381", "factor_data9": "yeslab_425"}
NAME_OF = {"yeslab_372": "GP_003", "yeslab_375": "GP_006", "yeslab_384": "GP_015",
           "yeslab_389": "GP_020", "yeslab_390": "GP_021", "yeslab_406": "GP_037",
           "yeslab_381": "GP_012", "yeslab_425": "GP_056"}
TABLE_OF = {"yeslab_372": "factor_data8", "yeslab_375": "factor_data8", "yeslab_384": "factor_data8",
            "yeslab_381": "factor_data8", "yeslab_389": "factor_data9", "yeslab_390": "factor_data9",
            "yeslab_406": "factor_data9", "yeslab_425": "factor_data9"}

d = json.loads((HERE / "sparse_regime_upstream.json").read_text(encoding="utf-8"))
days = d["days_examined"]
out = {"source": "sparse_regime_upstream.json", "days": len(days), "per_day": [], "summary": {}}

for row in days:
    up = row["upstream"]
    fz = row["frozen_factor_daily_by_version"]
    fzv = {f: sum(v["rows_with_value"] for v in vs.values()) for f, vs in fz.items()}
    fza = {f: sum(v["rows_all"] for v in vs.values()) for f, vs in fz.items()}
    rec = {"signal_date": row["signal_date"], "tag": row["tag"], "factors": {}}
    for f in list(TARGETS_A) + list(TARGETS_B) + list(CONTROL_OF.values()):
        name = NAME_OF[f]
        ua, uv = up[name]["rows_all"], up[name]["rows_with_value"]
        ctrl = CONTROL_OF[TABLE_OF[f]]
        cu = up[NAME_OF[ctrl]]["rows_with_value"]
        rec["factors"][f] = {
            "source_name": name, "table": TABLE_OF[f],
            "upstream_rows_all": ua, "upstream_rows_with_value": uv,
            "upstream_null_value_rows": ua - uv,
            "upstream_null_share": ((ua - uv) / ua) if ua else None,
            "frozen_rows_all": fza[f], "frozen_rows_with_value": fzv[f],
            "frozen_over_upstream_values": (fzv[f] / uv) if uv else None,
            "panel_daily_coverage": row["panel_daily_coverage"][f],
            "control_on_the_same_table_and_day": {
                "factor_id": ctrl, "source_name": NAME_OF[ctrl],
                "upstream_rows_with_value": cu,
                "frozen_rows_with_value": fzv[ctrl],
                "panel_daily_coverage": row["panel_daily_coverage"][ctrl],
            },
            "target_over_control_upstream": (uv / cu) if cu else None,
        }
        uv_ = uv
        if uv_ == 0 and ua == 0:
            shape = "NO_ROWS_UPSTREAM"
        elif ua > 0 and uv_ == 0:
            shape = "ROWS_UPSTREAM_ALL_VALUES_NULL"
        elif cu and uv_ / cu < 0.25:
            shape = "UPSTREAM_COLLAPSED_vs_control" if ua < cu else "UPSTREAM_MOSTLY_NULL_vs_control"
        else:
            shape = "PRESENT"
        rec["factors"][f]["upstream_shape"] = shape
    out["per_day"].append(rec)

# summarise: on each cluster's low days, what shape and what magnitude?
for tag, members in (("A", TARGETS_A), ("B", TARGETS_B)):
    low = [r for r in out["per_day"] if r["tag"] == f"cluster-{tag}-lowest"]
    high = [r for r in out["per_day"] if r["tag"] == f"cluster-{tag}-highest"]
    def agg(rows, key):
        vals = [r["factors"][m][key] for r in rows for m in members
                if r["factors"][m][key] is not None]
        return {"min": min(vals), "median": float(median(vals)), "max": max(vals)} if vals else None
    shapes = {}
    for r in low:
        for m in members:
            s = r["factors"][m]["upstream_shape"]
            shapes[s] = shapes.get(s, 0) + 1
    out["summary"][f"cluster_{tag}"] = {
        "members": members,
        "low_days": [r["signal_date"] for r in low],
        "high_days": [r["signal_date"] for r in high],
        "low_day_upstream_shapes": shapes,
        "low_day_upstream_rows_with_value": agg(low, "upstream_rows_with_value"),
        "high_day_upstream_rows_with_value": agg(high, "upstream_rows_with_value"),
        "low_day_upstream_null_share": agg(low, "upstream_null_share"),
        "high_day_upstream_null_share": agg(high, "upstream_null_share"),
        "low_day_target_over_control_upstream": agg(low, "target_over_control_upstream"),
        "high_day_target_over_control_upstream": agg(high, "target_over_control_upstream"),
        "low_day_frozen_over_upstream_values": agg(low, "frozen_over_upstream_values"),
        "high_day_frozen_over_upstream_values": agg(high, "frozen_over_upstream_values"),
        "low_day_panel_coverage": agg(low, "panel_daily_coverage"),
        "high_day_panel_coverage": agg(high, "panel_daily_coverage"),
        "control_panel_coverage_on_the_low_days": agg(low, "panel_daily_coverage") and {
            "min": min(r["factors"][m]["control_on_the_same_table_and_day"]["panel_daily_coverage"]
                       for r in low for m in members),
            "median": float(median([r["factors"][m]["control_on_the_same_table_and_day"]["panel_daily_coverage"]
                                    for r in low for m in members])),
        },
        "control_upstream_rows_on_the_low_days": {
            "min": min(r["factors"][m]["control_on_the_same_table_and_day"]["upstream_rows_with_value"]
                       for r in low for m in members),
            "median": float(median([r["factors"][m]["control_on_the_same_table_and_day"]["upstream_rows_with_value"]
                                    for r in low for m in members])),
        },
    }

ratios = [r["factors"][f]["frozen_over_upstream_values"] for r in out["per_day"]
          for f in r["factors"] if r["factors"][f]["frozen_over_upstream_values"] is not None]
out["summary"]["frozen_over_upstream_value_ratio"] = {
    "what": "the share of upstream valued rows that survive into factor_daily, i.e. the universe filter",
    "n": len(ratios), "min": min(ratios), "median": float(median(ratios)), "max": max(ratios),
    "reading": (
        "if this is roughly constant across targets, controls, and low/high days then the frozen table "
        "is a stable universe-in filter on whatever the producer published, and every coverage swing in "
        "§BA is inherited from upstream rather than created here"
    ),
}
(HERE / "regime_shapes.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {HERE / 'regime_shapes.json'}")
for tag in ("A", "B"):
    s = out["summary"][f"cluster_{tag}"]
    print(f"\n=== cluster {tag} {s['members']} ===")
    print(" low days :", s["low_days"], "shapes", s["low_day_upstream_shapes"])
    print(" high days:", s["high_days"])
    for k in ("upstream_rows_with_value", "upstream_null_share", "target_over_control_upstream",
              "frozen_over_upstream_values"):
        print(f"  low  {k:34} {s['low_day_' + k]}")
        print(f"  high {k:34} {s['high_day_' + k]}")
    print(f"  low  panel_daily_coverage             {s['low_day_panel_coverage']}")
    print(f"  high panel_daily_coverage             {s['high_day_panel_coverage']}")
    print("  control upstream rows on low days:", s["control_upstream_rows_on_the_low_days"])
    print("  control panel coverage on low days:", s["control_panel_coverage_on_the_low_days"])
print("\n", json.dumps(out["summary"]["frozen_over_upstream_value_ratio"], ensure_ascii=False, indent=1))
