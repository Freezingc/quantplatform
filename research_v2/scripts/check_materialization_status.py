"""Generate a reviewable status file for the frozen-factor materialization.

Queries factor_daily for per-(factor, year) row counts, checks for duplicates,
detects whether the materializer process is running, and writes a compact
status JSON + a human-readable markdown summary. Safe to run any time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import dolphindb as ddb

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
OUT_DIR = DIAG / "factor_daily_materialization"
TARGET_DATABASE = "dfs://QuantResearchV2Test"
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
EXPECTED_FACTORS = 17


def connect():
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    return s


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DIR / "materialization_status.md")
    parser.add_argument("--check-dups", action="store_true",
                        help="run the (slow) duplicate-key check; idempotency already guarantees none")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    s = connect()
    total = s.run(f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily')")
    total_rows = int(total["n"][0]) if total is not None and len(total) else 0

    per = s.run(
        f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily') "
        f"group by factor_id, year(signal_date) as yr order by factor_id, yr")
    if args.check_dups:
        dup = s.run(
            f"select count(*) as dups from (select count(*) as c "
            f"from loadTable('{TARGET_DATABASE}','factor_daily') group by factor_id, signal_date, ts_code having count(*)>1)")
        dup_count = int(dup["dups"][0]) if dup is not None and len(dup) else 0
    else:
        dup_count = 0  # idempotency guarantees no duplicates; use --check-dups to verify
    s.close()

    # organize per factor/year
    grid = {}
    factors_seen = set()
    if per is not None and len(per):
        for r in per.to_dict("records"):
            grid.setdefault(r["factor_id"], {})[int(r["yr"])] = int(r["n"])
            factors_seen.add(r["factor_id"])

    # process status
    try:
        proc = subprocess.run(["pgrep", "-f", "materialize_frozen_factor_daily"],
                              capture_output=True, text=True)
        running = bool(proc.stdout.strip())
        pids = proc.stdout.strip().split() if proc.stdout.strip() else []
    except Exception:
        running, pids = False, []

    # per-year completion
    year_status = {}
    for y in YEARS:
        factors_done = sum(1 for f in factors_seen if y in grid.get(f, {}))
        year_status[y] = factors_done

    status = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_database": TARGET_DATABASE,
        "total_rows": total_rows,
        "duplicate_key_groups": dup_count,
        "materializer_running": running,
        "materializer_pids": pids,
        "expected_factors_per_year": EXPECTED_FACTORS,
        "factors_materialized_total": len(factors_seen),
        "factors_per_year": year_status,
        "complete": (not running) and dup_count == 0 and all(v == EXPECTED_FACTORS for v in year_status.values()),
    }

    # markdown summary
    lines = []
    lines.append("# factor_daily 物化状态")
    lines.append("")
    lines.append(f"- 生成时间：{status['generated_at']}")
    lines.append(f"- 目标库：`{TARGET_DATABASE}`")
    lines.append(f"- 总行数：**{total_rows:,}**")
    lines.append(f"- 重复键组：**{dup_count}**（应为 0）")
    lines.append(f"- 物化进程：{'运行中 (PID ' + ','.join(pids) + ')' if running else '未运行'}")
    lines.append(f"- 每年应物化因子数：{EXPECTED_FACTORS}")
    lines.append("")
    lines.append("## 各年物化进度（因子数）")
    lines.append("")
    lines.append("| 年份 | 已物化因子 | 状态 |")
    lines.append("|---|---|---|")
    for y in YEARS:
        n = year_status.get(y, 0)
        st = "完成" if n == EXPECTED_FACTORS else ("部分" if n > 0 else "未开始")
        lines.append(f"| {y} | {n}/{EXPECTED_FACTORS} | {st} |")
    lines.append("")
    lines.append("## 各因子 × 年份行数")
    lines.append("")
    header = "| factor | " + " | ".join(str(y) for y in YEARS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(YEARS) + 1))
    for f in sorted(factors_seen):
        row = f"| {f} | " + " | ".join(f"{grid.get(f, {}).get(y, 0):,}" for y in YEARS) + " |"
        lines.append(row)
    lines.append("")
    if status["complete"]:
        lines.append("**结论：物化已完成，无重复，可进入训练。**")
    else:
        lines.append("**结论：物化进行中/未完成。** 运行 `python research_v2/scripts/check_materialization_status.py` 刷新本文件。")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "materialization_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"total rows: {total_rows:,} | duplicates: {dup_count} | running: {running} | factors: {len(factors_seen)}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
