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
