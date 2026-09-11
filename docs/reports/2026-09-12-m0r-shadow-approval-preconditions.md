# M0R 影子审批前置清单（计划 §5.3 交付物）

**这份文档做什么**：执行 `2026-09-04-real-candidate-model-training-plan.md` 的未勾选项
**§5.3**——「若有 `WALK_FORWARD_PASSED` 候选：记录其进入影子审批的前置清单
（F6.1 部署审批、60 有效日影子运行、监控与回滚演练），移交人工审批」。
该条件**现在已满足**，且这一项不依赖 M1/M2/M3 跑完。

**这份文档不做什么**：**不启动 SHADOW、不做任何状态转换、不写 registry、不写数据库。**
`WALK_FORWARD_PASSED → SHADOW` 按 F 计划 F6.1 是**人工**动作（"人工核验……随后执行"），
本文只把核验所需的材料备齐。撰写时（2026-09-12 05:25）registry 里 M0R 的状态**仍是
`WALK_FORWARD_PASSED`**，本文没有改动它。

---

## 1. 当前权威状态（逐项从签名工件读出，不是转述）

| 项 | 值 | 来源 |
|---|---|---|
| model_id / lineage_id | `sha256:32315c89fb9a65b8ceab193036fc3a0a870d24f2e6d99c6ff74049dccc75788b` | `model_registry/models/32315c89…/record.json` |
| **status** | **`WALK_FORWARD_PASSED`** | 同上（签名记录，signature `0b30530292e79f9b…`） |
| 状态事件链 | `None → DRAFT`（2026-09-08T19:51:27Z）→ `DRAFT → WALK_FORWARD_PASSED`（19:51:31Z） | 同目录 `events.jsonl` |
| 转换理由 / 执行者 | `all G1-G6 walk-forward gates passed` / `codex-m0r-track` | 同上 |
| 转换所绑证据哈希 | `64b413507bf4…` | 同上 |
| 折级 bundle | `m0r-f1` `m0r-f2` `m0r-f3` `m0r-f4` 四个都在 | `m0r_v2/models/` |
| 晋级证据工件 | `promotion_evidence.json`，sha256 `92b000c0d246556baf6dd077…`，129,509 字节 | `m0r_v2/`；该哈希与 2026-09-11 交接 §13 只读复核时打印的一致 |
| 判定文档 | `docs/reports/2026-09-12-m0r-real-candidate-walk-forward.md`（本次新归档）与 `m0r_v2/verdict.md`，两份与临时渲染**三份 md5 全同** `2bfa79833db1bcb9b5bb8036ceac48d1` | 渲染器确定性已验 |

**六道门（`m0r_v2/report.json`，`oos.all_passed = true`）**：

| 门 | 定义 | 实测 | 阈值 |
|---|---|---|---|
| G1 | OOS mean RankIC ≥ 0.030 | 0.09437596139252168 | 0.030 |
| G2 | HAC t ≥ 2.0 | 12.905448712183105 | 2.0 |
| G3 | 正年占比 ≥ 0.70（2022–2025） | 1.0 | 0.70 |
| **G4** | Δ RankIC vs 折匹配 best_simple ≥ 0.005 | **0.009149095118820283** | 0.005 |
| G5 | 最大单年正贡献占比 ≤ 0.50 | 0.27908240329707823 | 0.50 |
| G6 | 有效得分日占比 ≥ 0.95 | 0.9752321981424149 | 0.95 |

`valid_day_count = 945`、`baseline_mean_rank_ic = 0.0852268662737014`、`icir = 0.7681728838370345`。

**与 M1 的可比性已核实**（详见 2026-09-12 交接 §6 验收 B）：`m0r_v2` 与 `m1_parallel_v1`
同为 `mode=full`、4 折、`valid_day_count = 945`、`baseline_mean_rank_ic` **16 位全同**，
所以 M0R 的 +0.009149 与 M1 单配置载体的 +0.005845 可直接相减（约 1.57 倍）。

## 2. F6.1 要求人工核验的五项，及其当前证据状态

F 计划 F6.1（**未勾选**）：「人工核验历史 G1–G6、artifact、推理 manifest contract、
V2Test 演练和回滚准备，随后执行 `WALK_FORWARD_PASSED → SHADOW`。」

| F6.1 核验项 | 证据在哪 | 状态 |
|---|---|---|
| 历史 G1–G6 | 本文 §1 的表；渲染判定文档；`report.json` | **已备齐**，待人工核验 |
| artifact | `m0r_v2/models/m0r-f1..f4` 四个 bundle；registry 签名记录；`promotion_evidence.json`（sha256 `92b000c0…`）；`load_walk_forward_evidence` 只读复核已通过、registry 自己独立重算的 G4 与报告逐位一致（2026-09-11 交接 §13） | **已备齐**，待人工核验 |
| 推理 manifest contract | F 计划 F1–F5 的代码与测试已完成（F 计划第 552 行注记） | 待人工核验具体条目 |
| V2Test 演练 | F 计划第 15 行：「F 的管线可先用 V2Test fixture 验证」；F6.1 明记 **Test 环境 smoke run 不计入 60 日** | 待人工核验 |
| 回滚准备 | **F10.5「影子切换/回滚桌面演练」已勾选完成** | **已完成** |

## 3. 批准后即生效的约束（这些不是建议，是 F 计划里写死的规则）

1. **60 日的计数口径**：只有「状态为 `SHADOW` **且** shadow pointer 指向该模型」的成功日才计入；
   Test 环境 smoke run **不计**（F6.1）。有效影子日的定义在 F6.2（已勾选）。
2. **2026 期间不得看收益**：2026 年未结束且最后 T+6 标签未成熟前，
   **不计算 2026 RankIC/ICIR/PnL，不以影子结果调参**（F 计划第 17 行）。
   影子阶段只监控数据与分数。
3. **SHADOW ≠ 交易授权**：生产切换另需 locked test 通过与独立审批（第 18 行）。
4. **PSI 的双向影响**：单纯 PSI ERROR 不自动阻断当日 SHADOW 预测，但
   **连续 3 个有效日 ERROR 触发人工 review，并禁止新的模型晋级与 active pointer 切换**（第 286 行）。
   这条对本文的决策有直接含义：一旦 M0R 进入 SHADOW 并出现连续 ERROR，
   **后续 M1/M2/M3 的晋级也会被这条挡住**，不只是 M0R 自己的问题。
5. **计满天数不会自动升状态**：达到 60 日且无未解决 ERROR 后生成 `shadow_evidence.json`、
   **保持 `SHADOW`**；进入 `LOCKED_TEST_APPROVED` 还必须等 F9 的年份成熟、预注册与**双人审批**，
   系统不因计满天数自动 transition（第 329 行）。
6. **申请人与批准人必须不同**；审批先验证 `shadow_evidence.json` 与当前状态 `SHADOW`，
   再执行转换，并签发绑定 preregistration hash / model id / label watermark / 截止时间的一次性能力
   （用后即标记 consumed；过期、hash 改变、第二次调用都失败并写 audit）（第 460 行）。

## 4. 需要所有者裁断的一个解释问题（我不代为决定）

F 计划第 15 行写的**正式运营前置**是：「至少一个**复杂候选模型**处于 `WALK_FORWARD_PASSED`」。

M0R 是 **LightGBM 的 rankreg 载体**（`model_family = LGBM_RANKREG`），不是 torch 架构。
它确实通过了全部六道门、确实显著超过折匹配的 `best_simple` 基线（+0.009149，余量 +83%），
而且它的网格被**钉死为单配置**，所以这个数字**不含验证集选择偏差**——
从"证据强度"看它比 M1 那个 +0.005845 更干净（M1 的载体同样只搜 1 格，但 M1 全网格正在跑）。

但"复杂候选"这个词在设计语境里指的是相对 `best_simple` 的**建模复杂度升级**，
而 M0/M0R 被引入的理由本身是「LambdaRank 载体顶不到 G4 线」（驱动 docstring）。
**所以有两种读法**，且结论不同：

* **读法 A**：M0R 属于候选家族、已过门，前置满足 → 可以现在启动 F6.1 审批与 60 日影子计时。
* **读法 B**：「复杂候选」意在 torch 家族（M1/M2/M3），M0R 是 LightGBM 载体、不算 →
  前置**尚未**满足，影子计时应等 M1（约 25 小时后出全网格结果）或 M2/M3。

这个区别会移动 60 日计时的起点，而 60 有效日又卡在 2026 标签成熟与 F9 之前，
**所以它是有时间代价的决定，不是文书细节**。本文把它原样交给所有者，不预设答案。

## 5. 另一个时间点上的相关事实

M1 全 27 格运行正在进行（工作区 `m1_full_v1`，2026-09-12 02:05 发射，41 路），
预计 **09-13 06:45 前后**出报告。若所有者倾向读法 B、或希望在做影子决定前先看全网格结果，
**等约 25 小时即可**；若倾向读法 A，则 M0R 的材料**现在已经齐备**，不必等。
两者不冲突：M0R 与 M1 是两个不同 model_id，晋级其中一个不影响另一个的证据。

## 6. 本文没有做的事（明确列出，便于审计）

* 未执行任何状态转换；registry `record.json` 与 `events.jsonl` **未被写入**。
* 未启动 SHADOW，未设置 shadow pointer，未触碰 active model pointer。
* 未写数据库，未触碰 `materialization_status.{json,md}`、`/tmp/frozen_chain_hold.txt`、2026 locked 段。
* 未勾选 F 计划的 F6.1（那是人工审批项），也未勾选本计划 §5.1/§5.2
  （它们要求 M1–M3 全部结果，尚不具备）。
* §5.3 本身：本文即其交付物；是否据此**判定 §5.3 完成**留待所有者确认，
  因为它对 M1/M2/M3 尚未适用（那三个还没有 `WALK_FORWARD_PASSED` 的全网格候选）。
