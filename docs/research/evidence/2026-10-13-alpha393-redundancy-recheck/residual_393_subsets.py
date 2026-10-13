"""Follow-up: 393's incremental information under three conditioning sets (read-only).

The all-15 residual was measured on the 25.8% of days/rows where all fifteen are finite (the six
sparse factors are NaN on down days), so it is subsample-selected. This re-measures the residual
on (a) the dense nine, (b) the three most-correlated incumbents, (c) all fifteen, with the
coverage of each conditioning set reported alongside.
"""
from __future__ import annotations
import json, pathlib
import numpy as np, pandas as pd, pyarrow.dataset as pad, dolphindb as ddb

OUT = pathlib.Path(__file__).resolve().parent
PANEL = "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/diagnostics/gp_candidate_similarity_2025_values.parquet"
DB = "dfs://QuantResearchV2Test"
V2 = json.loads(pathlib.Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/diagnostics/factor_set_frozen_v2.json").read_text())["factors"]
DENSE9 = ["yeslab_96","yeslab_206","yeslab_357","yeslab_381","yeslab_425","yeslab_443_reg30_max20_v1","yeslab_464","yeslab_471","yeslab_484"]
TOP3 = ["yeslab_425","yeslab_484","yeslab_471"]
KEYS = ["signal_date","ts_code"]; MIN_CS = 30

def load_panel(fid):
    d = pad.dataset(PANEL, format="parquet")
    df = d.to_table(filter=pad.field("factor_id")==fid, columns=["signal_date","ts_code","value"]).to_pandas()
    df["signal_date"] = pd.to_datetime(df["signal_date"]); return df

def residual_ic(sub, cond_cols):
    days, raw, res, cov = [], [], [], []
    have = sub.dropna(subset=["a393","label"])
    n_all = len(have)
    for d, g in have.groupby("signal_date"):
        X = g[cond_cols].to_numpy(dtype=float)
        fin = np.isfinite(X).all(axis=1)
        gg = g.loc[fin]
        if len(gg) < MIN_CS: continue
        Xg = gg[cond_cols].to_numpy(dtype=float)
        ry = pd.Series(gg["a393"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
        rr = pd.Series(gg["label"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
        rx = np.apply_along_axis(lambda c: pd.Series(c).rank(pct=True).to_numpy(), 0, Xg)
        A = np.column_stack([np.ones(len(rx)), rx])
        beta, *_ = np.linalg.lstsq(A, ry, rcond=None)
        resid = ry - A @ beta
        if np.std(resid) < 1e-12 or np.std(rr) < 1e-12: continue
        days.append(str(d.date())); raw.append(float(np.corrcoef(ry, rr)[0,1])); res.append(float(np.corrcoef(resid, rr)[0,1]))
        cov.append(len(gg)/len(g))
    return {"days": len(days), "mean_raw_rank_ic": float(np.mean(raw)), "mean_residual_rank_ic": float(np.mean(res)),
            "retention": float(np.mean(res)/np.mean(raw)), "mean_within_day_coverage": float(np.mean(cov)),
            "row_coverage_of_all_labelled": float(len(days)/243), "daily": {"signal_date": days, "raw_ic": raw, "residual_ic": res}}

s = ddb.session(); s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
a393 = load_panel("yeslab_393").rename(columns={"value":"a393"})
lab = s.run(f"select signal_date, ts_code, raw_return from loadTable('{DB}','label_daily') where label_id='forward_open_5d' and label_version=1 and label_valid=true and signal_date between 2025.01.01 : 2025.12.31").rename(columns={"raw_return":"label"})
lab["signal_date"] = pd.to_datetime(lab["signal_date"])
wide = a393.merge(lab, on=KEYS, how="inner")
for fid in V2:
    df = s.run(f"select signal_date, ts_code, value from loadTable('{DB}','factor_daily') where factor_id='{fid}' and signal_date between 2025.01.01 : 2025.12.31")
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    wide = wide.merge(df.rename(columns={"value": fid}), on=KEYS, how="left")
s.close()
print("wide rows:", f"{len(wide):,}")
out = {"scope":"ALPHA393_RESIDUAL_SUBSETS_V1","window":"2025","conditioning_sets":{}}
for name, cols in (("dense9", DENSE9), ("top3_most_correlated", TOP3), ("all15", V2)):
    r = residual_ic(wide, cols)
    out["conditioning_sets"][name] = {k: v for k, v in r.items() if k != "daily"}
    out["conditioning_sets"][name]["n_factors"] = len(cols)
    print(f"{name:22s} days={r['days']:3d} raw={r['mean_raw_rank_ic']:+.5f} resid={r['mean_residual_rank_ic']:+.5f} retention={r['retention']:.3f} within-day cov={r['mean_within_day_coverage']:.1%}")
    (OUT/f"residual_393_{name}_daily.json").write_text(json.dumps(r["daily"], indent=1))
(OUT/"residual_393_subsets.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
print("wrote residual_393_subsets.json")
