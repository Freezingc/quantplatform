# 冻结因子集与训练链因子集口径决策备忘

> 生成时间：2026-09-05 12:25 CST（Asia/Shanghai）
> 分支 `codex/publish-quant-platform` · 状态：**待人工裁决**（本备忘只记录事实与选项，不擅自变更任何运行中的链）

## 1. 事实核查（全部可复核）

1. **冻结集 = 17 因子**：`factor_set_frozen_v1.json`（frozen_date=2026-09-03，
   入库 `dfs://QuantResearchV2Test.factor_set_frozen`，17 行）：
   206, 357, 372, 375, 381, 384, 389, 390, 398, 405, 406, 425,
   **443_reg30_max20_v1**（修复变体）, 464, 471, 484, 96。
2. **物化已完成**：`factor_daily` 中 2020–2025 各年 17/17（68,363,458 行，0 重复键；
   `research_v2/artifacts/diagnostics/factor_daily_materialization/materialization_status.md`）。
3. **当前训练链用的是遗留 14 因子集**：`scripts/walk_forward_common.py` 的
   `WALK_FORWARD_FACTOR_IDS`（376,377,380,381,387,388,391,396,400,405,425,432,433,442），
   与冻结集仅 3 个交集（381/405/425）；因子值路径为 `factor_observation_clean`（PIT 观测）。
4. **09-03 模型训练交接的明确指令**（`2026-09-03-model-training-handoff.md`）：
   「冻结因子集落地后，把冻结集合作为训练的因子输入清单，替换当前占位」，
   且训练因子值来源为 `factor_daily`（隔离规范版本，勿混历史 YesLab 值）。
5. **候选训练计划的预留条款**（`2026-09-04-real-candidate-model-training-plan.md`
   Global Constraints）：「若人工准入决策扩集，以冻结清单为准并在报告记录」。
   准入决策（09-03 决策清单）已生效 → 条款已触发。
6. **正在运行的链**：基线重跑 #3（复现 criterion-303，遗留 14 因子）→ 自动化
   M1→M2→M3（同 14 因子）。其目的是证据链机制验证与复现性签发（见
   `2026-09-05-candidate-training-chain-ops.md`），**不是**冻结集上的正式训练。

## 2. 口径结论（按既有文书推导）

正式模型训练阶段应使用：**冻结 17 因子 × `factor_daily` 值 × 官方 `universe_daily`
`cn_liquid_a` 宇宙**；在此之前需要一次冻结集上的基线 walk-forward（B0–B3）
作为新的信任根（现 criterion-303 信任根属于遗留 14 因子集，G4 增量口径随之失效于正式判定）。

## 3. 缺口（冻结集训练的技术前置）

1. `walk_forward_common` 的数据装配硬编码 `factor_observation_clean`；
   需要 `factor_daily`（版本绑定、可复现）的装配路径。
2. `WALK_FORWARD_FACTOR_IDS` → 冻结 17 清单（含修复变体名 `yeslab_443_reg30_max20_v1`
   到实际物化列名的映射核验）。
3. 体制感知加权（交接 §4，D3=a：大盘股主导 ON 时对 437/431/443 变体降权，
   先经 OOS 验证再固化）——属模型阶段工作，不在基线重建前置内。
4. 新基线的判定文档与证据归档（复用现有判定器/核验器模式）。

## 4. 选项（供裁决）

- **A（推荐）**：当前 14 因子链继续跑完（机制验证 + 复现证据，不打断），
  并行启动冻结集基线重建的接线工作（上面缺口 1/2），冻结集基线完成后
  在冻结集上重跑正式候选序列；现 M1–M3 结果标注为「遗留集管线验证，非正式业绩证据」。
- **B**：立即终止 14 因子链，全部资源转冻结集重建（损失已运行的重跑工时，
  且证据链机制未在完整时长上验证过）。
- **C**：维持现状，14 因子结果作为正式结果（与 09-03 交接指令冲突，不建议）。

## 5. 复核命令

```bash
python3 -c "import json; print(json.load(open('research_v2/artifacts/diagnostics/factor_set_frozen_v1.json'))['count'])"   # → 17
grep -c yeslab research_v2/artifacts/diagnostics/factor_set_frozen_v1.json
grep -n 'factor_observation_clean' research_v2/scripts/walk_forward_common.py   # 现装配路径
head -8 research_v2/artifacts/diagnostics/factor_daily_materialization/materialization_status.md  # 17/17 物化
```
