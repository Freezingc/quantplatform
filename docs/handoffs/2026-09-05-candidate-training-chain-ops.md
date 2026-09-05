# 候选训练链运维说明（基线重跑 #3 → M1 → M2 → M3）

> 生成时间：2026-09-05 11:50 CST（Asia/Shanghai）
> 仓库：`/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX` · 分支 `codex/publish-quant-platform`
> 用途：记录基线重跑完成后自动衔接 M1→M2→M3 顺序训练的无人值守链路、标记文件语义与人工恢复步骤。

## 0. 一句话状态

基线重跑 #3（含扩展选拔字段的证据链再生成）完成后，链路自动执行：一致性核验 →
候选消费 schema 预检 → M1 全量 →（若通过）M2 全量 →（若通过）M3 全量。
任一闸门失败即写 HOLD 标记并停止，绝不在未验证证据上启动下游家族。

## 1. 链路与组件

```
run_real_walk_forward.py (pid 406300, 03:05 启动, 预计 ~18:40)
  └─ watcher /tmp/baseline_artifacts_watcher.sh (pid 407725)
       落盘 → 写 /tmp/baseline_artifacts_done_at.txt → 运行一致性核验器
  └─ orchestrator /tmp/baseline_to_m1_orchestrator.sh (pid 801189)
       等 done → 读核验结果（10 分钟未出则自行补跑核验器）→ schema 预检（含深检）
       → 双过: 启动 M1 (/tmp/run_m1_full.sh) + M1 watcher；否则 HOLD
  └─ orchestrator /tmp/family_chain_orchestrator.sh (pid 810759)
       等 m1_report → 渲染判定文档+对比文档 → all_passed? 启动 M2 → 同前 → M3 → chain_complete
```

## 2. 标记文件地图（/tmp）

| 文件 | 含义 |
|---|---|
| `baseline_artifacts_done_at.txt` | 重跑 #3 报告落盘（watcher 写） |
| `baseline_artifacts_failed_at.txt` | 重跑 #3 进程死亡且无报告（watcher 写） |
| `baseline_rerun_verification.txt` | 一致性核验结果（首行应为 CONSISTENT） |
| `baseline_schema_gate.txt` | schema 预检结果（首行应为 SCHEMA PASS） |
| `m1_launch_hold.txt` | 一致性核验或 schema 预检失败，M1 未启动 |
| `m1_full_pid.txt` / `m1_full.log` | M1 进程号 / 训练日志（每折每配置有进度行） |
| `m1_full_done_at.txt` / `m1_full_failed_at.txt` | M1 完成 / 失败 |
| `m1_gate_summary.txt` | M1 门控摘要（watcher 写） |
| `chain_hold.txt` | 某家族失败或进程死亡，链停止 |
| `chain_complete.txt` | M1/M2/M3 全部通过，可进 Task 5 汇总 |

对比文档 `docs/reports/<date>-real-candidate-models-comparison.md` 由链式编排器在每个家族阶段结束时自动渲染（含 HOLD 时的未运行状态记录）。

M2/M3 同构（前缀 `m2_` / `m3_`）。

## 3. 人工恢复步骤

编排器均为 `setsid` 脱离会话的 bash 循环；机器重启后不会自动恢复，按以下顺序手动续接：

1. 判断当前位置：按第 2 节标记文件定位（先看 `chain_complete.txt`、`chain_hold.txt`、
   `m1_launch_hold.txt`，再看各 `_done_at` / `_failed_at`）。
2. 若重跑 #3 报告已落盘且核验文件缺失：手动运行
   `scripts/verify_baseline_rerun_consistency.py --report <报告> --reference
   docs/reports/2026-09-04-real-walk-forward-criterion-303.json` 与
   `scripts/verify_baseline_report_candidate_schema.py --report <报告>`，双过后手动
   `bash /tmp/run_m1_full.sh`（脚本仍在则直接复用）。
3. 若某家族报告已落盘：直接渲染判定文档
   `scripts/summarize_real_candidate_walk_forward.py --report <家族报告>
   --baseline-report <baseline_oos_report.json> --out docs/reports/<date>-<家族>-real-candidate-walk-forward.md`，
   依 `oos.all_passed` 决定是否 `bash /tmp/run_<下一家族>_full.sh`。
4. 顺序守卫由驱动 `--prior-report` fail-closed 校验（要求前置家族 full 模式且
   all_passed），手动启动也必须带该参数（`/tmp/run_m2_full.sh`、`run_m3_full.sh` 已含）。

## 4. 复核命令

```bash
# 链上进程是否存活
for pid in 406300 407725 801189 810759; do ps -o pid,etime,cmd -p $pid | tail -1; done
# 训练心跳（CPU 时间应持续增长）
awk '{print $14+$15}' /proc/406300/stat
# M1 进度（落盘后）
tail -5 /tmp/m1_full.log
```

## 5. 不变约束（复查红线）

1. 任一闸门失败不得跳过；不得用 2026 锁定段数据补救任何失败。
2. 信任根密钥只经环境变量 `QRV2_SELECTION_EVIDENCE_HMAC_KEY` 注入，文件权限 600，不入库。
3. M2/M3 必须经 `--prior-report` 证据链启动，禁止绕过顺序守卫。
4. 判定文档渲染器输出为证据的呈现形式；报告 JSON 才是判定源。

## 6. 关联文书（2026-09-05 补充）

- `docs/handoffs/2026-09-05-frozen-factor-set-training-decision.md`：冻结 17 因子集与
  本链所用遗留 14 因子集的口径分岔备忘（选项 A/B/C 待裁决）。**本链产物定性为
  遗留集管线验证，不作为正式业绩证据**；正式证据需在冻结集上重建基线后产生。
- `docs/superpowers/plans/2026-09-05-frozen-set-baseline-reconstruction-plan.md`：
  冻结集基线重建计划草案（含 `factor_daily` 登记式发布约束 Task 1.6 与 2026-09-05
  只读探测实测证据）。
- 裁决为 A 时：本链不中断，重建计划并行起步；两线在 Task 5（正式候选序列）合流。

## 7. 并行作业与新标记文件（2026-09-05 21:00 起）

本节作业与第 1 节的遗留链**并行**运行，互不写入对方路径。三者合计约 50–60 核 / 104 核。

```
backfill_2014_2019.sh (顺序 6 年 --apply, 每年写后核验, 任一年失败即停)
  └─ 完成 → frozen_chain_orchestrator.sh
       └─ 登记 2014-01..2019-12 指针 (dry-run 必须 72/72 REGISTER、0 拒绝) → --execute
            → 校验 factor_daily 达 144 分区且三张遗留表仍各 151 分区
            → 装配 F01/train (2014–2018) 走冻结路径, 断言 5 年齐全 + 17 因子列
            → 全部通过 → run_frozen_baseline.sh (Task 3.1 新信任根) + watcher
                 └─ 报告落盘 → 渲染判定文档 + 候选消费 schema 预检
legacy_flagcheck (Task 2.4, 独立): run_real_walk_forward.py 不带新旗标全量重跑
  └─ watcher → verify_baseline_rerun_consistency.py 对 criterion-303 比对
```

| 文件（/tmp） | 含义 |
|---|---|
| `backfill_2014_2019.log` / `_progress.txt` | 补齐日志 / 每年核验通过记录（`YYYY OK <时间>`） |
| `backfill_2014_2019_done_at.txt` / `_failed_at.txt` | 六年全部完成 / 某年失败停链 |
| `frozen_unblock.log` / `_done_at.txt` / `_hold.txt` | 登记+分区校验+F01 装配证明的进度 / 完成 / HOLD |
| `frozen_registration_2014_2019_{dryrun,execute}.log` | 新 72 指针的 dry-run 与执行输出（副本存 artifacts） |
| `frozen_partition_check.txt` | 144 分区 + 遗留表 151 分区不变的断言输出 |
| `frozen_chain_orchestrator.log` / `frozen_chain_hold.txt` | 冻结链编排进度 / HOLD |
| `frozen_baseline.log` / `_pid.txt` / `_done_at.txt` / `_failed_at.txt` | Task 3.1 运行日志与状态 |
| `frozen_baseline_verdict_render.txt` / `_schema_gate.txt` | Task 3.2 判定渲染 / Task 3.3 schema 门结果 |
| `legacy_flagcheck.log` / `_pid.txt` / `_verification.txt` / `_failed_at.txt` | Task 2.4 复现重跑与核验结果（首行应为 CONSISTENT） |
| `m1_fit_cadence.log` | 只读探针：M1 检查点出现时间与驱动日志增量（用于测单拟合时长） |

**产物路径（git 树外，`/data/zhangyudong/qrv2_research_artifacts/`）**：
冻结集一律在 `frozen/` 子树下——`frozen/baseline_oos_report.json` 与
`frozen/baseline_oos/`。**这不是命名偏好而是硬约束**：`save_baseline_oos_artifacts`
写的是 `args.report.parent / "baseline_oos"`（目录名硬编码），若把冻结集报告与遗留报告
放在同一父目录，冻结集工件会覆盖遗留链 M2/M3 消费的 `baseline_oos/F0*.parquet`。

**人工恢复**：编排器均为 `setsid` bash 循环，机器重启不会自恢复。按标记文件定位后：
补齐未完成 → 重跑 `bash /tmp/backfill_2014_2019.sh`（物化器按
(factor_id, 窗口, data_version) 幂等，已完成年份会被 skip，核验器照常复核）；
登记未完成 → 先 dry-run 后 `--execute`（幂等，重跑为 SKIP_CURRENT）；
Task 3.1 未完成 → `bash /tmp/run_frozen_baseline.sh` 再
`setsid bash /tmp/frozen_baseline_watcher.sh &`；
Task 2.4 未完成 → 重跑 `run_real_walk_forward.py`（不带 `--factor-source`）到
`/tmp/legacy_flagcheck/report.json`，再手动运行一致性核验器。
复核命令与口径详见 `research_v2/runbooks/real-walk-forward-reproduction.md`
（已增补「Frozen-set baseline (the new trust root)」章节）。
