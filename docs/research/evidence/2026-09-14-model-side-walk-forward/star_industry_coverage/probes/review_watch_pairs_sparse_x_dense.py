"""Watch-pair combination review for the frozen factor set (generic pairs).

Default pairs (frozen contract warnings + admission decision sheet section B,
all among ADMITTED factors):
  - yeslab_381 <-> yeslab_405                (validation-window corr 0.865/0.888)
  - yeslab_398 <-> yeslab_381                (validation-window corr 0.837/0.824)
  - yeslab_381 <-> yeslab_443_reg30_max20_v1 (validation-window corr 0.811/0.837)

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
  R2 incremental info: retention < 0.30 in either direction (pooled) -> FAIL;
     0.30-0.50 pooled or any single year < 0.30 -> REVIEW; >= 0.50 -> PASS
  R3 regime robustness: any regime-ON bucket with pooled retention < 0.30 or a
     sign flip of mean residual IC vs mean raw IC -> REVIEW

Usage: python review_watch_pairs.py [out_suffix]   (default out_suffix=v1)
Outputs JSON + markdown evidence under
research_v2/artifacts/diagnostics/watch_pair_review/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import dolphindb as ddb

# parents[2] is correct for the original location (research_v2/scripts/) but not for this copy, so the
# repository root is pinned explicitly; DIAG is only ever read from, never written to.
REPO_ROOT = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
# COPY of research_v2/scripts/review_watch_pairs.py, run from the research artifacts area so that nothing is
# written under research_v2/artifacts/ (a standing constraint on this line of work). Only OUT_DIR and PAIRS
# differ from the original; the statistics, the pre-registered verdict rules R1/R2/R3 and the DolphinDB
# queries are untouched. Read-only against the database.
OUT_DIR = Path("/data/zhangyudong/qrv2_research_artifacts/watch_pair_review_sparse_x_dense")
DB = "dfs://QuantResearchV2Test"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
# All 54 sparse-by-dense pairs. The six candidate_only factors of frozen v2 split into two clusters by daily row-count correlation
# (problems_found.md §AN.4): A = {372, 389, 390} at 0.9992-1.0000 and B = {375, 384, 406} at
# 0.9991-0.9995, with only 0.266-0.297 across clusters. Row-count identity is necessary but not sufficient
# for value redundancy, so every within-cluster pair is run through the same instrument that removed
# yeslab_398 and yeslab_405 for retention 20.3% and 7.6% against yeslab_381, plus one cross-cluster pair as
# a control that should NOT look redundant.
PAIRS = [
    ("yeslab_372", "yeslab_206"),
    ("yeslab_372", "yeslab_357"),
    ("yeslab_372", "yeslab_381"),
    ("yeslab_372", "yeslab_425"),
    ("yeslab_372", "yeslab_443_reg30_max20_v1"),
    ("yeslab_372", "yeslab_464"),
    ("yeslab_372", "yeslab_471"),
    ("yeslab_372", "yeslab_484"),
    ("yeslab_372", "yeslab_96"),
    ("yeslab_375", "yeslab_206"),
    ("yeslab_375", "yeslab_357"),
    ("yeslab_375", "yeslab_381"),
    ("yeslab_375", "yeslab_425"),
    ("yeslab_375", "yeslab_443_reg30_max20_v1"),
    ("yeslab_375", "yeslab_464"),
    ("yeslab_375", "yeslab_471"),
    ("yeslab_375", "yeslab_484"),
    ("yeslab_375", "yeslab_96"),
    ("yeslab_384", "yeslab_206"),
    ("yeslab_384", "yeslab_357"),
    ("yeslab_384", "yeslab_381"),
    ("yeslab_384", "yeslab_425"),
    ("yeslab_384", "yeslab_443_reg30_max20_v1"),
    ("yeslab_384", "yeslab_464"),
    ("yeslab_384", "yeslab_471"),
    ("yeslab_384", "yeslab_484"),
    ("yeslab_384", "yeslab_96"),
    ("yeslab_389", "yeslab_206"),
    ("yeslab_389", "yeslab_357"),
    ("yeslab_389", "yeslab_381"),
    ("yeslab_389", "yeslab_425"),
    ("yeslab_389", "yeslab_443_reg30_max20_v1"),
    ("yeslab_389", "yeslab_464"),
    ("yeslab_389", "yeslab_471"),
    ("yeslab_389", "yeslab_484"),
    ("yeslab_389", "yeslab_96"),
    ("yeslab_390", "yeslab_206"),
    ("yeslab_390", "yeslab_357"),
    ("yeslab_390", "yeslab_381"),
    ("yeslab_390", "yeslab_425"),
    ("yeslab_390", "yeslab_443_reg30_max20_v1"),
    ("yeslab_390", "yeslab_464"),
    ("yeslab_390", "yeslab_471"),
    ("yeslab_390", "yeslab_484"),
    ("yeslab_390", "yeslab_96"),
    ("yeslab_406", "yeslab_206"),
    ("yeslab_406", "yeslab_357"),
    ("yeslab_406", "yeslab_381"),
    ("yeslab_406", "yeslab_425"),
    ("yeslab_406", "yeslab_443_reg30_max20_v1"),
    ("yeslab_406", "yeslab_464"),
    ("yeslab_406", "yeslab_471"),
    ("yeslab_406", "yeslab_484"),
    ("yeslab_406", "yeslab_96"),
]
FACTORS = sorted({f for p in PAIRS for f in p})
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
        df = pd.concat(parts, ignore_index=True).rename(columns={"value": f})
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


def _pkey(x: str, y: str) -> str:
    return "__x__".join(sorted([x, y]))


def daily_moments(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    work = df.dropna(subset=cols)
    work = work[work.groupby("signal_date")[cols[0]].transform("size") >= MIN_NAMES_PER_DAY]
    prod_cols = []
    for i, a in enumerate(cols):
        for b in cols[i:]:
            name = _pkey(a, b)
            work[name] = work[a] * work[b]
            prod_cols.append(name)
    g = work.groupby("signal_date").agg(
        **{c: (c, "mean") for c in cols + prod_cols},
        n=("signal_date", "size"),
    )
    return g


def corr_from_moments(g, x, y):
    cov = g[_pkey(x, y)] - g[x] * g[y]
    vx = g[_pkey(x, x)] - g[x] ** 2
    vy = g[_pkey(y, y)] - g[y] ** 2
    return cov / np.sqrt(vx.clip(lower=1e-18) * vy.clip(lower=1e-18))


def beta_from_moments(g, y, x):
    cov = g[_pkey(x, y)] - g[x] * g[y]
    vx = g[_pkey(x, x)] - g[x] ** 2
    return cov / vx.clip(lower=1e-18)


def resid_ic_from_moments(g, a, b, y):
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


def rho_key(a, b): return f"rho__{a}__{b}"
def res_key(a, b): return f"resic__{a}__given__{b}"
def ic_key(f): return f"ic__{f}"


def fmt(x, spec=".3f"):
    return format(x, spec) if x is not None else "NA"


def main() -> int:
    suffix = sys.argv[1] if len(sys.argv) > 1 else "v1"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = connect()
    prov = s.run(f"select universe_id, factor_version, count(*) as n "
                 f"from loadTable('{DB}','factor_daily') where factor_id in {json.dumps(FACTORS)} "
                 f"group by universe_id, factor_version")
    print(prov)
    assert set(prov["universe_id"]) == {"cn_liquid_a"}, f"unexpected universes: {set(prov['universe_id'])}"
    factors = fetch_factors(s)
    labels = fetch_labels(s)
    s.close()
    print(f"factors: {len(factors):,} rows; labels: {len(labels):,} rows")

    df = factors.merge(labels, on=["signal_date", "ts_code"], how="inner")
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year
    for c in FACTORS + ["label"]:
        df["r_" + c] = df.groupby("signal_date")[c].rank(pct=True)
    rlabel = "r_label"

    daily = {}
    for y in YEARS:
        sub = df[df["year"] == y]
        frames = []
        # per-pair moments on the pairwise intersection (not the global one)
        for a, b in PAIRS:
            cols = ["r_" + a, "r_" + b, rlabel]
            g = daily_moments(sub, cols)
            o = pd.DataFrame(index=g.index)
            o[rho_key(a, b)] = corr_from_moments(g, "r_" + a, "r_" + b)
            o[res_key(a, b)] = resid_ic_from_moments(g, "r_" + a, "r_" + b, rlabel)
            o[res_key(b, a)] = resid_ic_from_moments(g, "r_" + b, "r_" + a, rlabel)
            frames.append(o)
        # per-factor IC on the factor x label intersection
        for f in FACTORS:
            cols = ["r_" + f, rlabel]
            g = daily_moments(sub, cols)
            o = pd.DataFrame(index=g.index)
            o[ic_key(f)] = corr_from_moments(g, "r_" + f, rlabel)
            frames.append(o)
        out = pd.concat(frames, axis=1)
        out["year"] = y
        daily[y] = out
    alld = pd.concat(daily.values())
    alld.index = pd.to_datetime(alld.index)
    for name, (path, col) in REGIME_FILES.items():
        r = pd.read_csv(path, parse_dates=["trade_date"]).set_index("trade_date")[col]
        alld[name] = r.reindex(alld.index).fillna(0).astype(int)

    def agg_window(w: pd.DataFrame) -> dict:
        if len(w) == 0:
            return {"days": 0}
        res = {"days": int(len(w))}
        for a, b in PAIRS:
            rho = w[rho_key(a, b)].dropna()
            ar = rho.abs()
            res[rho_key(a, b)] = {
                "mean_rho": float(rho.mean()), "median_abs_rho": float(ar.median()),
                "p10_abs_rho": float(ar.quantile(0.10)), "p90_abs_rho": float(ar.quantile(0.90)),
                "share_ge_085": float((ar >= 0.85).mean()), "share_ge_090": float((ar >= 0.90).mean()),
            }
            for x, y_ in ((a, b), (b, a)):
                ric = w[res_key(x, y_)].dropna()
                raw = w[ic_key(x)].dropna()
                res[res_key(x, y_)] = {
                    "mean": float(ric.mean()),
                    "icir": float(ric.mean() / ric.std()) if ric.std() > 0 else None,
                    "retention_vs_raw": float(ric.mean() / raw.mean()) if abs(raw.mean()) > 1e-9 else None,
                }
        for f in FACTORS:
            ic = w[ic_key(f)].dropna()
            res[ic_key(f)] = {"mean": float(ic.mean()),
                              "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else None}
        return res

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {"factor_daily": DB, "label": "forward_open_5d:v1 (label_valid=true)",
                   "universe": "cn_liquid_a (materialization-aligned)", "years": YEARS,
                   "pairs": [list(p) for p in PAIRS],
                   "provenance": json.loads(prov.to_json(orient="records"))},
        "per_year": {str(y): agg_window(alld[alld["year"] == y]) for y in YEARS},
        "pooled": agg_window(alld),
        "regime": {name: {"on": agg_window(alld[alld[name] == 1]), "off": agg_window(alld[alld[name] == 0])}
                   for name in REGIME_FILES},
        "pairs": {},
    }

    for a, b in PAIRS:
        v = {"rule_R1_near_duplicate": "PASS", "rule_R2_incremental": "PASS", "rule_R3_regime": "PASS", "notes": []}
        worst_year_rho = max(results["per_year"][str(y)][rho_key(a, b)]["median_abs_rho"] for y in YEARS)
        v["max_yearly_median_abs_rho"] = worst_year_rho
        if worst_year_rho >= 0.90:
            v["rule_R1_near_duplicate"] = "FAIL"
            v["notes"].append(f"median |rho| reached {worst_year_rho:.3f} >= 0.90 in at least one year")
        rets = [results["pooled"][res_key(x, y_)]["retention_vs_raw"] for x, y_ in ((a, b), (b, a))]
        v["pooled_retention"] = {res_key(a, b): rets[0], res_key(b, a): rets[1]}
        valid = [r for r in rets if r is not None]
        worst_ret = min(valid) if valid else None
        year_rets = [results["per_year"][str(y)][res_key(x, y_)]["retention_vs_raw"]
                     for y in YEARS for x, y_ in ((a, b), (b, a))]
        worst_year_ret = min(r for r in year_rets if r is not None)
        v["worst_year_retention"] = worst_year_ret
        if worst_ret is not None and worst_ret < 0.30:
            v["rule_R2_incremental"] = "FAIL"
        elif (worst_ret is not None and worst_ret < 0.50) or worst_year_ret < 0.30:
            v["rule_R2_incremental"] = "REVIEW"
            v["notes"].append(f"pooled retention {fmt(worst_ret)} / worst-year {fmt(worst_year_ret)} inside REVIEW band")
        for name in REGIME_FILES:
            on = results["regime"][name]["on"]
            if on["days"] == 0:
                continue
            for x, y_ in ((a, b), (b, a)):
                k = res_key(x, y_)
                ret = on[k]["retention_vs_raw"]
                raw_mean = on[ic_key(x)]["mean"]
                res_mean = on[k]["mean"]
                if ret is not None and ret < 0.30:
                    v["rule_R3_regime"] = "REVIEW"
                    v["notes"].append(f"{name}-ON retention for {k} = {fmt(ret)} < 0.30")
                if raw_mean * res_mean < 0 and abs(res_mean) > 1e-4:
                    v["rule_R3_regime"] = "REVIEW"
                    v["notes"].append(f"{name}-ON sign flip for {k} (raw {raw_mean:+.4f} vs resid {res_mean:+.4f})")
        order = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
        v["overall"] = max([v["rule_R1_near_duplicate"], v["rule_R2_incremental"], v["rule_R3_regime"]],
                           key=lambda x: order[x])
        results["pairs"][f"{a}<->{b}"] = v

    (OUT_DIR / f"watch_pair_review_{suffix}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    lines = ["# 关注对组合评审（冻结集训练前置）", "",
             f"> 生成：{results['generated_at']} · 数据：factor_daily 2020–2025 × forward_open_5d:v1（label_valid）",
             "> 判定规则：R1 近冗余 |median rho|≥0.90；R2 增量保留率 pooled≥0.50 且单年≥0.30；R3 体制 ON 桶保留率≥0.30 且无符号翻转", ""]
    for pair, v in results["pairs"].items():
        lines += [f"## {pair} — 总判定：**{v['overall']}**", "",
                  f"- 各年 median|rho| 最大值：{v['max_yearly_median_abs_rho']:.3f}（R1 {v['rule_R1_near_duplicate']}）",
                  f"- pooled 保留率：{ {k.split('__')[1] + '|' + k.split('__')[3]: round(x, 3) if x is not None else None for k, x in v['pooled_retention'].items()} }（R2 {v['rule_R2_incremental']}，最差单年 {fmt(v['worst_year_retention'])}）",
                  f"- R3 {v['rule_R3_regime']}", ""]
        if v["notes"]:
            lines += ["注："] + [f"- {n}" for n in v["notes"]] + [""]
    lines += ["## 逐年明细", "", "| 年 | 对 | mean rho | median|rho| | ≥0.85占比 | IC_A | IC_B | 残差IC A|B | 保留率 A|B | 残差IC B|A | 保留率 B|A |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for y in YEARS:
        py = results["per_year"][str(y)]
        for a, b in PAIRS:
            r = py[rho_key(a, b)]
            k1, k2 = res_key(a, b), res_key(b, a)
            lines.append(f"| {y} | {a.replace('yeslab_', '')}↔{b.replace('yeslab_', '')} | {r['mean_rho']:+.3f} | {r['median_abs_rho']:.3f} | {r['share_ge_085']:.1%} | "
                         f"{py[ic_key(a)]['mean']:+.4f} | {py[ic_key(b)]['mean']:+.4f} | "
                         f"{py[k1]['mean']:+.4f} | {fmt(py[k1]['retention_vs_raw'], '.2f')} | {py[k2]['mean']:+.4f} | {fmt(py[k2]['retention_vs_raw'], '.2f')} |")
    lines += ["", "## 体制分桶（pooled 2020–2025）", ""]
    for name in REGIME_FILES:
        lines += [f"### {name}", "", "| 桶 | 天数 | " + " | ".join(f"rho {a.replace('yeslab_','')}↔{b.replace('yeslab_','')}" for a, b in PAIRS) + " | " + " | ".join(f"保留率 {x.replace('yeslab_','')}|{y_.replace('yeslab_','')}" for a, b in PAIRS for x, y_ in ((a, b), (b, a))) + " |",
                  "|---|---|" + "---|" * (len(PAIRS) * 3)]
        for bucket in ("on", "off"):
            w = results["regime"][name][bucket]
            if w["days"] == 0:
                continue
            cells = [f"{w[rho_key(a, b)]['mean_rho']:+.3f}" for a, b in PAIRS]
            rets = [fmt(w[res_key(x, y_)]['retention_vs_raw'], ".2f") for a, b in PAIRS for x, y_ in ((a, b), (b, a))]
            lines.append(f"| {bucket.upper()} | {w['days']} | " + " | ".join(cells) + " | " + " | ".join(rets) + " |")
        lines.append("")
    (OUT_DIR / f"watch_pair_review_{suffix}.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / f'watch_pair_review_{suffix}.json'}")
    print(f"wrote {OUT_DIR / f'watch_pair_review_{suffix}.md'}")
    for pair, v in results["pairs"].items():
        print(pair, "->", v["overall"], "| max median|rho| =", round(v["max_yearly_median_abs_rho"], 3),
              "| pooled retention =", {k: (round(x, 3) if x is not None else None) for k, x in v["pooled_retention"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
