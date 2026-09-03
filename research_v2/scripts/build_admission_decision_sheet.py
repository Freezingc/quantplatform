"""Build the consolidated one-stop admission decision sheet.

Merges the explicit admission checklist (23 READY candidates), the
four-dimension quality grading, and the special decision points from the brief
into a single decision sheet a human can approve in one pass. Read-only
generator; re-runnable from the source artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"
BRIEF = REPO_ROOT / "docs" / "handoffs" / "2026-09-01-admission-decision-brief.md"
OUT = REPO_ROOT / "docs" / "handoffs" / "2026-09-03-admission-decision-sheet.md"

GRADE_ORDER = {"S": 0, "A": 1, "B": 2, "C": 3, "D": 4}

MONITOR_LABEL = {
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW": "无附加监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_APPROVAL": "隔离范围批准",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_AND_PAIR_MONITORING": "隔离范围 + 关注对监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_AND_FORMULA_RECONCILIATION_MONITORING": "隔离范围 + 公式核对监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_AND_HIGH_VOL_MONITORING": "隔离范围 + 高波动监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_MONTHLY_AND_REGIME_MONITORING": "月度 + 市场状态监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_REGIME_MONITORING": "体制监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_PIT_AND_ANNUAL_MONITORING": "PIT 行业 + 年度监控",
    "READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_2020_ANNUAL_MONITORING": "2020 年度门槛监控",
}

# Hard mutual-exclusion pairs for recommendation building (both admitted => conflict).
EXCLUSIVE_PAIRS = [
    ("yeslab_114", "yeslab_484"),
    ("yeslab_357", "yeslab_363"),
    ("yeslab_379", "yeslab_375"),
    ("yeslab_370", "yeslab_372"),
    ("yeslab_379", "yeslab_384"),
]

# Mutual-exclusion / joint-admission constraints accumulated across 4a-4t.
CONSTRAINTS = [
    "114 ↔ 484：禁止同时准入（验证期冗余 0.9079）",
    "357 ↔ 363：禁止同时准入（行业族互斥）",
    "379 ↔ 375：禁止联合准入（近冗余 0.886/0.894）",
    "370 ↔ 372：禁止联合准入（跨体制冗余 ≥0.93，4t；保留 372）",
    "379 ↔ 384：禁止联合准入（跨体制冗余 ≥0.92，4t；保留 379）",
    "381 ↔ 405：关注对监控（0.865/0.888），同时准入需单独组合评审",
    "381 ↔ 443：关注对监控（0.811/0.837）",
    "398 ↔ 381：关注对监控（0.837/0.824）",
    "114 ↔ 471：跨体制关注对监控（4t）",
]


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)

    checklist = load_json(DIAG / "explicit_admission_review_checklist_v1.json")["checklists"]
    grading = load_json(DIAG / "factor_quality_grading_v1" / "factor_quality_grading_v1.json")
    grade_by_fid = {e["factor_id"]: e for e in grading["entries"]}

    # The 206 candidate under review is the median-repair variant; the grading
    # keys it as yeslab_206_median_abs_v2 while the checklist uses source id yeslab_206.
    grade_alias = {"yeslab_206": "yeslab_206_median_abs_v2"}
    ready = [e for e in checklist if e["status"].startswith("READY")]
    rows = []
    for e in ready:
        fid = e["source_factor_id"]
        g = grade_by_fid.get(grade_alias.get(fid, fid), {})
        grade = g.get("grade", "?")
        tot = g.get("total")
        train_icir = g.get("train_icir")
        val_icir = g.get("validation_icir")
        regime_min = g.get("regime_min_official_icir")
        edge = g.get("max_nine_edge") or g.get("max_intra_formal_edge_train")
        monitor = MONITOR_LABEL.get(e["status"], e["status"])
        rows.append({
            "fid": fid, "name": e.get("factor_name", ""),
            "grade": grade, "tot": tot, "train": train_icir, "val": val_icir,
            "regime": regime_min, "edge": edge, "monitor": monitor,
            "status": e["status"],
        })
    rows.sort(key=lambda r: (GRADE_ORDER.get(r["grade"], 9), -(r["tot"] or 0)))

    # Greedy conflict-free recommendation: iterate strongest-first; a factor is
    # recommended only if it does not clash (exclusive pair) with one already kept.
    kept: list[str] = []
    rec_by_fid: dict[str, str] = {}
    for r in rows:
        fid = r["fid"]
        if r["grade"] in ("S", "A", "B"):
            clash = next((k for k in kept if any({fid, k} == set(pr) for pr in EXCLUSIVE_PAIRS)), None)
            if clash:
                rec_by_fid[fid] = f"与{clash}互斥，二选一"
            else:
                kept.append(fid)
                rec_by_fid[fid] = "准入" if r["grade"] in ("S", "A") else "准入(带监控)"
        elif r["grade"] == "C":
            rec_by_fid[fid] = "缓议"
        else:
            rec_by_fid[fid] = "缓议/不入"

    fmt = lambda v, nd=3: ("—" if v is None else f"{v:.{nd}f}")
    lines = []
    lines.append("# 准入决策清单（一次性决策表）")
    lines.append("")
    lines.append(f"> 生成日期：{date.today().isoformat()}（由 `scripts/build_admission_decision_sheet.py` 从清单+分级产物自动生成）")
    lines.append("> 配套：决策简报 `2026-09-01-admission-decision-brief.md`（4a-4aa 全证据）。本表只做决策汇总，不替代简报证据。")
    lines.append("")
    lines.append("## A. 23 个 READY 候选（按质量分级排序）")
    lines.append("")
    lines.append("「建议」列是依据分级与体制证据给出的默认处置；你只需在「决定」列填 准入 / 不入 / 缓议。")
    lines.append("")
    lines.append("| # | 因子 | 名称 | 级 | 分 | 训练ICIR | 验证ICIR | 体制最低 | 最大边 | 监控要求 | 建议 | 决定 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        rec = rec_by_fid[r["fid"]]
        lines.append(
            f"| {i} | {r['fid']} | {r['name']} | {r['grade']} | {r['tot']} | "
            f"{fmt(r['train'])} | {fmt(r['val'])} | {fmt(r['regime'])} | {fmt(r['edge'])} | "
            f"{r['monitor']} | {rec} | ☐ |"
        )
    lines.append("")
    lines.append("**分级口径**：四维（训练预测力/样本外/体制稳健/独特性）各 0-3 分；S≥11，A 9-10，B 7-8，C 5-6，D≤4；体制崩塌或训练未过门槛封顶 D。体制最低=2017/2018/2019 官方口径 ICIR 最小值。")
    lines.append("")
    lines.append("## B. 互斥 / 联合准入约束（必须遵守）")
    lines.append("")
    for c in CONSTRAINTS:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## C. 特别决策点")
    lines.append("")
    lines.append("| # | 决策点 | 选项 | 我的建议 | 你的决定 |")
    lines.append("|---|---|---|---|---|")
    lines.append("| D1 | 443 处置（2017 崩塌，已找到修复变体） | A=修复变体替换 / B=维持MONITORED | **A**（修复变体全评估已过） | ☐ |")
    lines.append("| D2 | 370×372、379×384 互斥 + 114×471 监控 | 采纳/不采纳 | **采纳** | ☐ |")
    lines.append("| D3 | 437/431 处置（正式因子，2017 崩塌） | a=监控+体制加权 / b=授权修复探索 / c=维持 | **a** | ☐ |")
    n_s = sum(1 for r in rows if r["grade"] == "S")
    n_a = sum(1 for r in rows if r["grade"] == "A")
    lines.append(f"| D4 | 23 候选整体准入 | 按 A 表逐行决定 | **S级{n_s}个 + A级{n_a}个先入** | ☐ |")
    lines.append("")
    lines.append("（365 已闭环：不恢复，无需决策——见简报 4aa。）")
    lines.append("")
    lines.append("## D. 建议的首批冻结因子集（若按建议执行）")
    lines.append("")
    s_factors = [r["fid"] for r in rows if r["grade"] == "S"]
    a_factors = [r["fid"] for r in rows if r["grade"] == "A"]
    lines.append(f"- **S 级（{len(s_factors)}）**：" + "、".join(s_factors))
    lines.append(f"- **A 级（{len(a_factors)}）**：" + "、".join(a_factors))
    lines.append(f"- B 级（{sum(1 for r in rows if r['grade']=='B')}）与 C 级（{sum(1 for r in rows if r['grade']=='C')}）：带监控准入或缓议，见 A 表。")
    lines.append("")
    lines.append("**注意**：首批集合仍需按 B 节约束去重（如 379/384 二选一、381/405 关注对）。最终冻结集以你在 A 表的逐行决定为准。")
    lines.append("")
    lines.append("## E. 决策后流程（已预置脚本）")
    lines.append("")
    lines.append("1. 在本表填好决定 → 2. 运行 `scripts/apply_admission_decisions.py`（按本表更新清单状态、写 `factor_quality_latest`、生成冻结因子集契约并入库）→ 3. 交接模型训练（见 `2026-09-03-model-training-handoff.md`）。")
    lines.append("")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"rows: {len(rows)} | S:{sum(1 for r in rows if r['grade']=='S')} A:{sum(1 for r in rows if r['grade']=='A')} B:{sum(1 for r in rows if r['grade']=='B')} C:{sum(1 for r in rows if r['grade']=='C')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
