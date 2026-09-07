"""Watch-pair combination review for the frozen factor set.

Pairs under review (required by the frozen contract warnings and the admission
decision sheet section B before training starts):
  - yeslab_381 <-> yeslab_405   (validation-window corr 0.865/0.888)
  - yeslab_398 <-> yeslab_381   (validation-window corr 0.837/0.824)

Read-only against DolphinDB. Pulls factor_daily values (2020-2025, the
materialized frozen-set table), forward_open_5d:v1 labels (label_valid only),
and the two approved regime indicator CSVs. For each pair computes, per year
and per regime bucket:
  - daily cross-sectional Spearman rho (distribution stats + high-corr shares)
  - signed RankIC and ICIR of each factor
  - residualized RankIC (factor rank regressed on partner rank, per day) and
    the retention ratio residual/raw -- the incremental information each factor
    carries given its partner.

Verdict rules (pre-registered, aligned with the admission framework):
  R1 near-duplicate: |median rho| >= 0.90 in any year -> FAIL (mutual-exclusion level)
  R2 incremental info: retention < 0.30 in either direction (pooled 2020-2025)
     -> FAIL; 0.30-0.50 or any single year < 0.30 -> REVIEW; >= 0.50 -> PASS
  R3 regime robustness: any regime-ON bucket with pooled retention < 0.30 or a
     sign flip of mean residual IC vs mean raw IC -> REVIEW
Outputs JSON + markdown evidence under
research_v2/artifacts/diagnostics/watch_pair_review/.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import dolphindb as ddb

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
OUT_DIR = DIAG / "watch_pair_review"
DB = "dfs://QuantResearchV2Test"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
FACTORS = ["yeslab_381", "yeslab_405", "yeslab_398"]
PAIRS = [("yeslab_381", "yeslab_405"), ("yeslab_398", "yeslab_381")]
MIN_NAMES_PER_DAY = 30
REGIME_FILES = {
    "largecap_dominance": (DIAG / "largecap_dominance_regime_v1" / "largecap_dominance_regime_daily.csv", "regime_on"),
    "liquidity_shock": (DIAG / "liquidity_shock_regime_v1" / "liquidity_shock_regime_daily.csv", "shock_on"),
}


def connect():
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    return s


def fetch_factors(s) -> pd.DataFrame:
    frames = []
    for f in FACTORS:
        parts = []
        for y in YEARS:
            q = (f"select signal_date, ts_code, value from loadTable('{DB}','factor_daily') "
                 f"where factor_id='{f}' and signal_date between {y}.01.01 : {y}.12.31")
            parts.append(s.run(q))
        df = pd.concat(parts, ignore_index=True)
        df = df.rename(columns={"value": f})
        frames.append(df)
    wide = frames[0]
    for df in frames[1:]:
        wide = wide.merge(df, on=["signal_date", "ts_code"], how="outer")
    return wide


def fetch_labels(s) -> pd.DataFrame:
    parts = []
    for y in YEARS:
        q = (f"select signal_date, ts_code, raw_return from loadTable('{DB}','label_daily') "
             f"where label_id='forward_open_5d' and label_version=1 and label_valid=true "
             f"and signal_date between {y}.01.01 : {y}.12.31")
        parts.append(s.run(q))
    return pd.concat(parts, ignore_index=True).rename(columns={"raw_return": "label"})


def daily_moments(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-day means of cols and their pairwise products (for corr/cov via moments)."""
    work = df.dropna(subset=cols)
    work = work[work.groupby("signal_date")[cols[0]].transform("size") >= MIN_NAMES_PER_DAY]
    prod_cols = []
    for i, a in enumerate(cols):
        for b in cols[i:]:
            name = "__x__".join(sorted([a, b]))
            work[name] = work[a] * work[b]
            prod_cols.append(name)
    g = work.groupby("signal_date").agg(
        **{c: (c, "mean") for c in cols + prod_cols},
        n=("signal_date", "size"),
    )
    return g


def _pkey(x: str, y: str) -> str:
    return "__x__".join(sorted([x, y]))


def corr_from_moments(g: pd.DataFrame, x: str, y: str) -> pd.Series:
    cov = g[_pkey(x, y)] - g[x] * g[y]
    vx = g[_pkey(x, x)] - g[x] ** 2
    vy = g[_pkey(y, y)] - g[y] ** 2
    return cov / np.sqrt(vx.clip(lower=1e-18) * vy.clip(lower=1e-18))


def beta_from_moments(g: pd.DataFrame, y: str, x: str) -> pd.Series:
    """OLS slope of y on x (per day) from moments."""
    cov = g[_pkey(x, y)] - g[x] * g[y]
    vx = g[_pkey(x, x)] - g[x] ** 2
    return cov / vx.clip(lower=1e-18)


def resid_ic_from_moments(g: pd.DataFrame, a: str, b: str, y: str) -> pd.Series:
    """corr(a - beta*b, y) with beta from per-day OLS of a on b."""
    beta = beta_from_moments(g, a, b)
    cov_ay = g[_pkey(a, y)] - g[a] * g[y]
    cov_by = g[_pkey(b, y)] - g[b] * g[y]
    cov_ab = g[_pkey(a, b)] - g[a] * g[b]
    va = g[_pkey(a, a)] - g[a] ** 2
    vb = g[_pkey(b, b)] - g[b] ** 2
    vy = g[_pkey(y, y)] - g[y] ** 2
    cov_resid_y = cov_ay - beta * cov_by
    var_resid = (va - 2 * beta * cov_ab + beta ** 2 * vb).clip(lower=1e-18)
    return cov_resid_y / np.sqrt(var_resid * vy.clip(lower=1e-18))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = connect()
    # provenance asserts
    prov = s.run(f"select universe_id, factor_version, count(*) as n "
                 f"from loadTable('{DB}','factor_daily') where factor_id in {json.dumps(FACTORS)} "
                 f"group by universe_id, factor_version")
    print(prov)
    assert set(prov["universe_id"]) == {"cn_liquid_a"}, f"unexpected universes: {set(prov['universe_id'])}"
    factors = fetch_factors(s)
    labels = fetch_labels(s)
    s.close()
    print(f"factors: {len(factors):,} rows; labels: {len(labels):,} rows")

    # rank-transform per day (Spearman = Pearson on ranks); use pct ranks of raw values
    df = factors.merge(labels, on=["signal_date", "ts_code"], how="inner")
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    for c in FACTORS + ["label"]:
        df["r_" + c] = df.groupby("signal_date")[c].rank(pct=True)
    rcols = ["r_" + c for c in FACTORS + ["label"]]

    daily = {}
    for y in YEARS:
        sub = df[df["year"] == y]
        g = daily_moments(sub, rcols)
        out = pd.DataFrame(index=g.index)
        out["n"] = g["n"]
        out["rho_381_405"] = corr_from_moments(g, "r_yeslab_381", "r_yeslab_405")
        out["rho_398_381"] = corr_from_moments(g, "r_yeslab_398", "r_yeslab_381")
        for f in FACTORS:
            out["ic_" + f] = corr_from_moments(g, "r_" + f, "r_label")
        out["resic_381_given_405"] = resid_ic_from_moments(g, "r_yeslab_381", "r_yeslab_405", "r_label")
        out["resic_405_given_381"] = resid_ic_from_moments(g, "r_yeslab_405", "r_yeslab_381", "r_label")
        out["resic_398_given_381"] = resid_ic_from_moments(g, "r_yeslab_398", "r_yeslab_381", "r_label")
        out["resic_381_given_398"] = resid_ic_from_moments(g, "r_yeslab_381", "r_yeslab_398", "r_label")
        out["year"] = y
        daily[y] = out
    alld = pd.concat(daily.values())
    alld.index = pd.to_datetime(alld.index)

    # regime flags
    for name, (path, col) in REGIME_FILES.items():
        r = pd.read_csv(path, parse_dates=["trade_date"]).set_index("trade_date")[col]
        alld[name] = r.reindex(alld.index).fillna(0).astype(int)

    def agg_window(w: pd.DataFrame) -> dict:
        if len(w) == 0:
            return {"days": 0}
        res = {"days": int(len(w))}
        for rho_col, fa, fb in [("rho_381_405", "381", "405"), ("rho_398_381", "398", "381")]:
            if rho_col in w:
                r = w[rho_col].dropna().abs()
                res[f"{fa}_{fb}"] = {
                    "mean_rho": float(w[rho_col].mean()),
                    "median_abs_rho": float(r.median()),
                    "p10_abs_rho": float(r.quantile(0.10)),
                    "p90_abs_rho": float(r.quantile(0.90)),
                    "share_ge_085": float((r >= 0.85).mean()),
                    "share_ge_090": float((r >= 0.90).mean()),
                }
        for f in ["381", "405", "398"]:
            ic = w["ic_yeslab_" + f].dropna()
            res[f"ic_{f}"] = {"mean": float(ic.mean()), "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else None}
        for key in ["resic_381_given_405", "resic_405_given_381", "resic_398_given_381", "resic_381_given_398"]:
            ric = w[key].dropna()
            base = key.split("_given_")[0].replace("resic_", "")
            partner = key.split("_given_")[1]
            raw = w["ic_yeslab_" + base].dropna()
            res[key] = {
                "mean": float(ric.mean()),
                "icir": float(ric.mean() / ric.std()) if ric.std() > 0 else None,
                "retention_vs_raw": float(ric.mean() / raw.mean()) if abs(raw.mean()) > 1e-9 else None,
            }
        return res

    results = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "source": {"factor_daily": DB, "label": "forward_open_5d:v1 (label_valid=true)",
                          "universe": "cn_liquid_a (materialization-aligned)", "years": YEARS,
                          "provenance": json.loads(prov.to_json(orient="records"))},
               "per_year": {}, "pooled": {}, "regime": {}, "pairs": {}}
    for y in YEARS:
        results["per_year"][str(y)] = agg_window(alld[alld["year"] == y])
    results["pooled"] = agg_window(alld)
    for name in REGIME_FILES:
        results["regime"][name] = {"on": agg_window(alld[alld[name] == 1]),
                                   "off": agg_window(alld[alld[name] == 0])}

    # verdicts
    for fa, fb, rho_col, k1, k2 in [
        ("381", "405", "rho_381_405", "resic_381_given_405", "resic_405_given_381"),
        ("398", "381", "rho_398_381", "resic_398_given_381", "resic_381_given_398"),
    ]:
        v = {"rule_R1_near_duplicate": "PASS", "rule_R2_incremental": "PASS", "rule_R3_regime": "PASS", "notes": []}
        worst_year_rho = max(results["per_year"][str(y)][f"{fa}_{fb}"]["median_abs_rho"] for y in YEARS)
        v["max_yearly_median_abs_rho"] = worst_year_rho
        if worst_year_rho >= 0.90:
            v["rule_R1_near_duplicate"] = "FAIL"
            v["notes"].append(f"median |rho| reached {worst_year_rho:.3f} >= 0.90 in at least one year")
        ret_pooled = [results["pooled"][k]["retention_vs_raw"] for k in (k1, k2)]
        v["pooled_retention"] = {k1: ret_pooled[0], k2: ret_pooled[1]}
        worst_ret = min(r for r in ret_pooled if r is not None)
        worst_year_ret = min(results["per_year"][str(y)][k]["retention_vs_raw"] for y in YEARS for k in (k1, k2)
                             if results["per_year"][str(y)][k]["retention_vs_raw"] is not None)
        v["worst_year_retention"] = worst_year_ret
        if worst_ret < 0.30:
            v["rule_R2_incremental"] = "FAIL"
        elif worst_ret < 0.50 or worst_year_ret < 0.30:
            v["rule_R2_incremental"] = "REVIEW"
            v["notes"].append(f"pooled retention {worst_ret:.3f} or worst-year {worst_year_ret:.3f} below 0.50/0.30 band")
        for name in REGIME_FILES:
            on = results["regime"][name]["on"]
            for k in (k1, k2):
                base = k.split("_given_")[0].replace("resic_", "")
                ret = on[k]["retention_vs_raw"]
                raw_mean = on[f"ic_{base}"]["mean"]
                res_mean = on[k]["mean"]
                if ret is not None and ret < 0.30:
                    v["rule_R3_regime"] = "REVIEW"
                    v["notes"].append(f"{name}-ON retention for {k} = {ret:.3f} < 0.30")
                if raw_mean * res_mean < 0 and abs(res_mean) > 1e-4:
                    v["rule_R3_regime"] = "REVIEW"
                    v["notes"].append(f"{name}-ON sign flip for {k} (raw {raw_mean:+.4f} vs resid {res_mean:+.4f})")
        order = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
        v["overall"] = max([v["rule_R1_near_duplicate"], v["rule_R2_incremental"], v["rule_R3_regime"]],
                           key=lambda x: order[x])
        results["pairs"][f"{fa}<->{fb}"] = v

    (OUT_DIR / "watch_pair_review_v1.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    # markdown summary
    lines = ["# 关注对组合评审（冻结集训练前置）", "",
             f"> 生成：{results['generated_at']} · 数据：factor_daily 2020–2025 × forward_open_5d:v1（label_valid）",
             f"> 判定规则：R1 近冗余 |median rho|≥0.90；R2 增量信息保留率 pooled≥0.50 且单年≥0.30；R3 体制 ON 桶保留率≥0.30 且无符号翻转", ""]
    for pair, v in results["pairs"].items():
        lines += [f"## {pair} — 总判定：**{v['overall']}**", "",
                  f"- 各年 median|rho| 最大值：{v['max_yearly_median_abs_rho']:.3f}（R1 {v['rule_R1_near_duplicate']}）",
                  f"- pooled 保留率：{v['pooled_retention']}（R2 {v['rule_R2_incremental']}，最差单年 {v['worst_year_retention']:.3f}）",
                  f"- R3 {v['rule_R3_regime']}", ""]
        if v["notes"]:
            lines += ["注："] + [f"- {n}" for n in v["notes"]] + [""]
    lines += ["## 逐年明细", "", "| 年 | 对 | mean rho | median|rho| | ≥0.85占比 | IC_A | IC_B | 残差IC A|B | 保留率 A|B | 残差IC B|A | 保留率 B|A |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for y in YEARS:
        py = results["per_year"][str(y)]
        for (fa, fb, rho_key, k1, k2) in [("381", "405", "381_405", "resic_381_given_405", "resic_405_given_381"),
                                          ("398", "381", "398_381", "resic_398_given_381", "resic_381_given_381".replace("381_given_381", "381_given_398"))]:
            r = py[rho_key]
            def fmt(x, spec):
                return format(x, spec) if x is not None else "NA"
            lines.append(f"| {y} | {fa}↔{fb} | {r['mean_rho']:+.3f} | {r['median_abs_rho']:.3f} | {r['share_ge_085']:.1%} | "
                         f"{py['ic_' + fa]['mean']:+.4f} | {py['ic_' + fb]['mean']:+.4f} | "
                         f"{py[k1]['mean']:+.4f} | {fmt(py[k1]['retention_vs_raw'], '.2f')} | {py[k2]['mean']:+.4f} | {fmt(py[k2]['retention_vs_raw'], '.2f')} |")
    lines += ["", "## 体制分桶（pooled 2020–2025）", "", "| 体制 | 桶 | 天数 | rho381_405 | rho398_381 | 保留率381|405 | 保留率405|381 | 保留率398|381 | 保留率381|398 |", "|---|---|---|---|---|---|---|---|---|"]
    for name in REGIME_FILES:
        for bucket in ("on", "off"):
            w = results["regime"][name][bucket]
            if w["days"] == 0:
                continue
            def fmtr(x):
                return format(x, '.2f') if x is not None else "NA"
            lines.append(f"| {name} | {bucket.upper()} | {w['days']} | {w['381_405']['mean_rho']:+.3f} | {w['398_381']['mean_rho']:+.3f} | "
                         f"{fmtr(w['resic_381_given_405']['retention_vs_raw'])} | {fmtr(w['resic_405_given_381']['retention_vs_raw'])} | "
                         f"{fmtr(w['resic_398_given_381']['retention_vs_raw'])} | {fmtr(w['resic_381_given_398']['retention_vs_raw'])} |")
    (OUT_DIR / "watch_pair_review_v1.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR/'watch_pair_review_v1.json'}")
    print(f"wrote {OUT_DIR/'watch_pair_review_v1.md'}")
    for pair, v in results["pairs"].items():
        print(pair, "->", v["overall"], "| max median|rho| =", round(v["max_yearly_median_abs_rho"], 3),
              "| pooled retention =", {k: (round(x, 3) if x is not None else None) for k, x in v["pooled_retention"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
