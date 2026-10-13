# M3 启动：顺序偏离机制、血缘选择与发射记录（2026-10-13）

本记录只有文字与数字，不含权重、检查点或预测明细。

## 1. 阻塞点（复述，已由代码证实）

`run_real_candidate_training.py` 中 `FAMILY_ORDER_PREREQUISITES["M3"] = ("M1", "M2")`，
而 `prior_families_from_reports` 要求每份 `--prior-report` 指向的报告满足 family 相符、`mode=full`
且 `oos.all_passed` 为真，否则 `SystemExit`。`m2_v1/report.json` 的 `all_passed` 是 `false`
（G4 0.0049591733 < 0.005，G6 0.9339525284 < 0.95），所以 M3 在现行契约下无法启动。
`2026-10-08-candidate-family-summary.md` 第 6 节把这明确定性为**计划层决定**，并且当时没有擅自放宽守卫。

本次由所有者授权偏离，并**新增机制**而不是削弱默认行为。

## 2. 机制：默认仍然 fail-closed

分支 `codex/m3-order-deviation`（自 `33d5d19` 新开），提交 `5cc1489`。

偏离必须同时满足三条，缺一条即拒绝启动：

1. `--family-order-deviation REASON`，理由长度 ≥ `MINIMUM_DEVIATION_REASON_LENGTH` = 24 字符，
   空串或口令式短语被拒；
2. `--family-order-deviation-owner NAME`，记录授权人；
3. `--prior-report-failed FAMILY=PATH`，**每一个缺失的前置都必须附上真实失败报告**。
   该报告被校验为 family 相符、`mode=full`、且 `all_passed` 不为真——
   一份实际通过的报告若被当作“失败”提交会被拒（它属于 `--prior-report`），
   这样调用方无法掩盖到底缺的是哪个前置。

另外两条防滥用约束：**不需要的偏离也被拒**（前置都齐时传这三个参数会报错），
所以审计块永远不会声称一次运行并不依赖的授权。

`main()` 在偏离下运行时会打印 WARNING，并把 `family_order_deviation` 块写入报告，内含
理由、授权人、前置清单、哪些前置有通过证据、哪些被承认为失败，以及每份失败报告的
`report_sha256`、`all_passed`、`failed_gates`、`observed_gates`。
日后任何人读 M3 报告，都能直接看到它的前置并未全部通过、以及为什么被允许。

测试：新增 12 条契约测试，驱动自身契约套件 119 passed，全量单测 **2127 passed, 1 skipped**。

## 3. 血缘选择（更正一个会浪费 40 小时的错误假设）

原先准备在本研究工作树 `codex/research-conditional-factor-review`（`0193b7c`）里跑 M3。这是错的：

- `m1_full_v1` 与 `m2_v1` 产自血缘 `33d5d19`，而 `33d5d19` = 主树 `2502014` **+ 恰好 1 个提交**
  （`git rev-list --left-right --count 2502014...33d5d19` = `0  1`）。
- 研究工作树 `0193b7c` 自分叉点 `fd59aab` 起有 166 个自有提交，同时**缺主树 130 个提交**。
  可验证的差异之一：该树驱动脚本**没有** `--worker-memory-gb`（主树与 `33d5d19` 各有 3 处命中），
  而 `m2_v1.log` 第一行正是 `worker memory budget: 24 x 17.1 GiB ...`。
  即产出官方 M1/M2 数字的驱动比研究工作树更新。

在较旧的驱动上跑 M3，其数字与 M1/M2 不可比，`--resume` 的采纳校验也可能失配。
因此 M3 从**新工作树** `/data/zhangyudong/qrv2_worktrees/m3-order-deviation`
（分支 `codex/m3-order-deviation`，自 `33d5d19`）运行。主树保持干净停在 `2502014`；
文档仍记在本研究分支。

**为什么 M3 不需要 `33d5d19` 引入的时序缺口放行**：M3 是 `SET_ATTENTION`，走
`_run_full_cross_section` + `build_static_training_batches`，候选侧对每个有效标签行都打分。
M1 同理，实测 F1 候选行 803,572 = 可打分标签行 803,572，缺口为 0。
那个放行是为 M2 的 17,306 行时序缺口准备的，横截面家族用不到。

## 4. 发射记录

- 时间：2026-10-13 17:36（本地）；驱动 PID `1866097`；日志 `/data/zhangyudong/qrv2_research_artifacts/m3_v1.log`
- 工作区 `/data/zhangyudong/qrv2_research_artifacts/m3_v1`，报告 `m3_v1/report.json`
- 参数：`--family M3 --factor-source frozen_v2 --parallel-workers 48 --torch-threads 1
  --worker-memory-gb 2.5 --resume`，基线工件 `baseline_v2/baseline_oos` + `baseline_v2/report.json`
  （与 `m0r_v2` / `m1_full_v1` **同一套**），`--prior-report M1=m1_full_v1/report.json`，
  `--prior-report-failed M2=m2_v1/report.json`
- 内存预算：`48 x 2.5 GiB + parent 0.6 GiB = 120.6 GiB of 458.6 GiB available`
- **输入同一性检查（关键）**：日志显示 `factor source: frozen_v2 (15 factors,
  tables=universe_daily,factor_daily,label_daily)`，且 F1 缓存数据集行数
  **2,296,573 / 696,906 / 804,575**，与 `m1_full_v1.log` 的三行逐位相同。
  这说明 M3 消费的是与 M1/M2 完全相同的冻结数据集——预注册的可比性前提成立。

规模与成本：M3 网格 16 个配置（`hidden_dim × num_heads × inducing_points × num_blocks`，
dropout=0.1、lr=0.0003 固定），每配置 3 个种子（20260801/02/03）→ 每折 48 个任务，
48 worker 正好一波。实测单价 430.6 ms/date（`family_device_benchmark.txt`，CPU；
同表记录 M3 在 CUDA 上**慢 2.09 倍**，所以用 CPU），1,206 个日期约 8.7 min/epoch。
四折先前估算约 31–40 小时墙钟。

## 5. 预注册交叉核对：M3 的 G6 应当通过

`m3_preregistration.md`（登记于 2026-09-14T19:48:14Z，即评价跑之前）预测：

- **G6 = 945/969 = 0.9752321981424149**
- 候选侧逐日覆盖率恒为 **1.0000**；`days_dropped_by_the_baseline_side` = **24/24**
- 逐折可评天数 F1 241 / F2 219 / F3 242 / F4 243
- 24 个不可评日期的完整清单（起于 2022-07-29，其余在 2023-04-28 → 2023-07-19）

本次从**已冻结的 m0r_v2 与 m1_full_v1 工件**独立复算，得到两者基线侧不健康天数均为
F1 1 天 + F2 23 天 = **24 天**，G6 均为 **0.975232**，候选自身不健康天数均为 **0**
（即候选侧覆盖率恒为 1）。与预注册逐位一致，因此预注册仍然有效。

推论：M3 作为同类横截面候选，**G6 预期通过**（0.9752 ≥ 0.95），除非输入在 m0r_v2/m1_full_v1
与 M3 之间发生了变化——而第 4 节的行数同一性检查已排除这一点。
按预注册的口径，一致只说明消费了同一套冻结输入，**不说明候选好坏**。

因此决定 M3 成败的实际是 **G4（≥0.005）**：M0R 实测 0.009149、M1 实测 0.007827、
M2 实测 0.0049591733（差 0.0000408267 未过）。G1/G2/G3/G5 在 M0R、M1、M2 上全部通过，
且余量很大（G1 门槛 0.03 对实测 0.090–0.094；G5 门槛 0.50 对实测 0.279–0.284）。

## 6. 关联记录

M2 侧的覆盖率归因与门槛定价见同目录
[`2026-10-13-m2-coverage-counterfactual-analysis.md`](2026-10-13-m2-coverage-counterfactual-analysis.md)。
两份记录分开写、分开对应各自的实验，不共用结论。
