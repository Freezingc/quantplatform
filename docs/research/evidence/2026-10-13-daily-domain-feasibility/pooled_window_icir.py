"""Re-pool the per-year screen statistics onto the benchmark window (2022-2024).

screen_daily_domains.py records, for every candidate/lag, the per-year mean RankIC, per-year
ICIR (= mean/std with ddof=1) and the valid-day count.  Those three numbers determine each
year's variance exactly (std_i = mean_i / icir_i), so the pooled mean and pooled ddof=1
variance over any subset of years follow from the law of total variance:

    N            = sum n_i
    grand_mean   = sum n_i * mean_i / N
    pooled_var   = [ sum (n_i - 1) * var_i + sum n_i * (mean_i - grand_mean)^2 ] / (N - 1)

This makes the candidates directly comparable with benchmark_frozen_v2_icir_2022_2024.csv,
which is computed from factor_rank_ic_daily over 2022-2024 (that table has no 2025 rows).
No new fitting, no new data reads: pure post-processing of analysis/ic_screen.json.
"""
from __future__ import annotations

import json
import pathlib

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
BENCH_YEARS = ("2022", "2023", "2024")


def pooled(yearly: dict, years=BENCH_YEARS) -> dict | None:
    n = m = ss = 0.0
    parts = []
    for y in years:
        e = yearly.get(y)
        if not e or not e.get("icir") or e.get("days", 0) < 2:
            return None
        ni, mi = float(e["days"]), float(e["mean_rank_ic"])
        vi = (mi / float(e["icir"])) ** 2
        parts.append((ni, mi, vi))
    if not parts:
        return None
    N = sum(p[0] for p in parts)
    grand = sum(p[0] * p[1] for p in parts) / N
    var = (sum((p[0] - 1) * p[2] for p in parts) + sum(p[0] * (p[1] - grand) ** 2 for p in parts)) / (N - 1)
    if var <= 0:
        return None
    return {"days": int(N), "mean_rank_ic": grand, "icir": grand / var**0.5}


def main() -> int:
    res = json.loads((OUT / "analysis" / "ic_screen.json").read_text())
    rows = []
    for source, entries in res["sources"].items():
        for e in entries:
            if e.get("icir") is None:
                continue
            p = pooled(e.get("yearly") or {})
            rows.append({"source": source, "candidate": e["candidate"], "lag": e["lag"],
                         "icir_2022_2025": round(e["icir"], 4),
                         "mean_ic_2022_2025": round(e["mean_rank_ic"], 5),
                         "icir_2022_2024": round(p["icir"], 4) if p else None,
                         "mean_ic_2022_2024": round(p["mean_rank_ic"], 5) if p else None,
                         "days_2022_2024": p["days"] if p else None,
                         "yearly_icir": {y: round(v["icir"], 3) for y, v in (e.get("yearly") or {}).items() if v.get("icir")},
                         "coverage": round(e.get("coverage_of_universe_member_days", float("nan")), 4)})
    df = pd.DataFrame(rows).sort_values("icir_2022_2024", key=lambda s: s.abs(), ascending=False)
    df.to_csv(OUT / "analysis" / "ic_screen_pooled_windows.csv", index=False)
    bench = pd.read_csv(OUT / "analysis" / "benchmark_frozen_v2_icir_2022_2024.csv")
    lo, hi = bench["abs_icir"].min(), bench["abs_icir"].max()
    print(f"frozen v2 benchmark |ICIR| 2022-2024: min={lo:.3f} median={bench['abs_icir'].median():.3f} max={hi:.3f}")
    print(f"admission bar (signed ICIR, non-annualised): 0.50\n")
    cols = ["source", "candidate", "lag", "icir_2022_2024", "mean_ic_2022_2024", "icir_2022_2025", "coverage"]
    print(df[cols].head(24).to_string(index=False))
    best = df["icir_2022_2024"].abs().max() if len(df) else float("nan")
    print(f"\nbest candidate |ICIR| 2022-2024 = {best:.3f} vs weakest frozen factor {lo:.3f} "
          f"-> {'CLEARS' if best >= 0.50 else 'BELOW'} the 0.50 admission bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
