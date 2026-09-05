# 裁决 4 的落地半径与一处需更正的拦停理由

generated_at: 2026-09-06T05:25+08:00
状态：**只读核查所得，未改任何接线**。`/tmp/frozen_chain_hold.txt` 仍在生效，Task 3.1 仍未启动。

本文回答三个问题：拦停理由中哪一条需要更正、选项 (iii) 的完整改动半径有多大、
以及一个此前未被记录的耦合如何让「保留 17」的代价比已知的更大。

## 1. 拦停理由的一条需更正：G1–G6 全部按构造折匹配

`/tmp/frozen_chain_hold.txt` 写：

> G4 (delta RankIC vs the fold-matched baseline) would then compare different
> cross-sections and is not fold-matched evidence; a baseline weakened by selection
> also makes G4 easier to pass

**前半句不成立。** `model_evaluator.py:135-163` 的 `_match_prediction_coverage` 对候选与
基线取 `(signal_date, ts_code)` 的**内连接**（`validate="one_to_one"`），而
`evaluate_frozen_oos:60-64` 把 `candidate_predictions` **重新赋值**为匹配后的版本，
该版本随后喂给 `compute_daily_rank_ic` → `summary` → **G1/G2/G3/G5/G6**，
匹配后的基线版本喂给 `delta` → **G4**。故**全部六道门都计算在同一键交集上**，
G4 并不比其它门更折匹配或更不折匹配。该函数还在重复业务键上 fail-closed
（`duplicated(keys) → raise`），与 `combine_seed_percentile_ranks` 是同一套契约。

**后半句成立且是本调查的核心**：基线被 `dropna` 限制在方向选择性更强的子样本上，
作为比较对象被削弱，于是 G4 更易通过、信任根不保守。这条理由已由标定后的覆盖率门
定量（train 中位 310–673 对已接受基线 1402–2794）。

**结论：拦停应当维持，但依据是「比较对象被削弱」，不是「横截面不匹配」。**

## 2. 新发现的耦合：基线覆盖率**决定**候选的评价口径

因为候选被限制到与基线的交集，**基线覆盖多宽，候选就在多宽的横截面上被评价**。
这不是推断，是实测：冻结 F01 test 装配帧 **666,421 行**，`dropna` 后 **169,107 行**
= **25.38%**（`assembly_coverage` 实测，与计划为选项 (ii) 估算的 25.4% 吻合）。

即**现状（保留 17）下，每个候选的 G1–G6 都被隐式限制在「全 17 齐全」子群上**——
那正是选项 (ii) 的口径，但它是被匹配逻辑**强加**的，而不是被**选择**的。
这一点此前没有出现在任何文书里，而它改变了三个选项的相对代价：

| 选项 | 基线 train 中位 | 候选评价横截面（F01 test 中位） | 说明 |
|---|---|---|---|
| 保留 17（现状） | 310 | **599** | 候选评价被压到 25.4% 的配对上 |
| (i) ≥11/17 | 1624 | **2699** | 评价口径放宽 ~4.5× |
| (iii) C6.2 PASS 11 | 1446 | **2652** | 评价口径放宽 ~4.4× |
| 遗留 14（已接受） | 1402 | 2657 | 参照 |

**故「保留 17」的代价不止于训练侧**：它同时把**候选**的整个 G1–G6 评价压到约四分之一的
横截面上，而 G6 的「每日 ≥30 有效性」与 G3 的「逐年正贡献」都会因此更容易受少数股票支配。
选项 (i) 与 (iii) 都把评价横截面恢复到与已接受基线同量级（2652/2699 对 2657）。

## 3. 选项 (iii) 的完整改动半径（含一个设计岔口）

### 3.1 岔口：缩训练集，还是也缩物化表？

- **(iii-a) 只缩训练所用的因子集，`factor_daily` 仍保留全部 17。** 推荐。
  不需要重物化；`verify_factor_daily_backfill.py` 的 `FROZEN_FACTOR_COUNT = 17`
  （该脚本按年断言 `factor_daily` 恰含 17 个不同 `factor_id`）**保持正确**；
  6 个缺陷因子的数据仍在库，C6.2 筛查可继续度量它们，将来若要重新准入仍有据可查。
- **(iii-b) 连物化表一起缩到 11。** 不推荐：需要重物化或删数据，
  且会让 backfill 核验器（硬编码 17）失效，并销毁 6 个因子的证据。

### 3.2 契约应当**新发 V2**，而不是就地修改 V1

`artifacts/diagnostics/factor_set_frozen_v1.json` 不是普通配置，它带
`frozen_date: 2026-09-03`、`data_version: 78203c90…` 摘要、以及人工裁决记录
`special_decisions: {D1: A, D2: 采纳, D3: a, D4: 按表}` 与两条 MONITOR 警告。
就地改写等于**重写一份已签署、已定日期的审计记录**。
项目其余部分（`published_version` 指针、按月 `data_version`）一律是**追加新版本而非改旧版本**，
故此处应同样新发 `factor_set_frozen_v2.json`（11 个因子、新的 `frozen_date`、自己的
`data_version`，并记录它由 V1 经 C6.2 筛查导出）。

### 3.3 已经做好的部分

**`scoped_cache_key`（`walk_forward_factor_daily.py:127-152`）已存在且正是为此而建**：
`fold_cache_key` 只哈希 `(fold_id, role, versions)`，因子集是模块常量、不在其载荷内，
故若把 `FROZEN_FACTOR_SET_V1` 缩小，键不变、会**静默交出 17 因子的旧缓存帧（列都不对）**。
该函数把因子集作为一条伪表项折进键里，并有测试
`test_cache_key_is_scoped_to_the_factor_set`，其 docstring 原文即
「shrinking FROZEN_FACTOR_SET_V1 (adjudication 4 option iii …)」。
**即 (iii) 最危险的失效模式已被关闭并有测试覆盖。**

### 3.4 需要改动的文件（(iii-a) + V2 契约）

| 文件 | 改动 | 性质 |
|---|---|---|
| `artifacts/diagnostics/factor_set_frozen_v2.json` | 新建（11 因子、新 frozen_date/data_version、记录导出自 V1） | 契约 |
| `scripts/walk_forward_factor_daily.py` | 新增 `FROZEN_FACTOR_SET_V2` 与 `FACTOR_SOURCES` 中的 `frozen_v2`；`factor_source_plan` 返回 V2 | 接线 |
| `scripts/verify_admission_consistency.py` | **无需改**——已有 `--contract` 旗标（:213） | — |
| `scripts/preflight_frozen_baseline.py` | 需加 `--contract` 透传：:149 现以**无参**调用该核验器，故走 V1 默认 | 接线 |
| `scripts/screen_factor_daily_coverage.py` | **必须保持筛查全部 17**（见 3.5） | 治理 |
| `configs/models/{residual_mlp,set_attention,temporal}.yaml` | `raw_factor_count_min: 15` → ≤11（`max: 60` 无需改） | 声明 |
| `tests/unit/scripts/test_walk_forward_factor_daily.py` | :205 `== 17`、:284 `len(set(names)) == 17`、:304 `frozen_count == 17`，以及 `minimum > 14` 断言 | 测试 |
| `scripts/verify_frozen_fold_coverage.py` + 其测试 | `C62_PASS_11` 与 `--factor-set` 预设的语义需随 V2 调整 | 工具 |
| 计划/简报/runbook | Task 2.1、2.2 与复现命令 | 文书 |

`raw_factor_count_min` **无任何代码消费点**（模型侧硬约束只有 `raw_factor_count >= 1`
与 `<= 60`），故降到 11 **不会引起运行期失败**；但「声明 min 15、实际训练 11 个因子」
是一份不真实的声明，所以 yaml 必须改。

### 3.5 一个必须避免的治理倒退

`screen_factor_daily_coverage.py:198-213` **遍历 `FROZEN_FACTOR_SET_V1`**。
若把冻结集缩到 11，这道 C6.2 筛查就会**停止筛查那 6 个缺陷因子**——
而它正是**发现**它们的那道筛查，也是 `verify_admission_consistency.py`
（前置门之一）读取证据的来源。故 (iii) 落地时**必须让该筛查继续覆盖全部 17**
（或至少那 6 个），否则准入记录变得不可审计，将来的重新准入也无从度量。

### 3.6 唯一真正需要「重新裁决」的条款

守卫测试 `test_frozen_set_satisfies_every_model_yaml_factor_bound` 的第二条断言
`minimum > 14`，其失败文案原文为「the documented legacy-14 tension would need
re-adjudication」。该断言是重建计划 Task 2.2 为保留「遗留 14 才违反声明下限」这一
可解释对照而写的，张力条款当时据此**正式结案**。选 (iii) 就要重开它：
一旦冻结集本身是 11，「min > 14」所保留的那个对照就失去指涉对象，
应改写为按因子源分别断言边界，而非钉住一个跨时代的常数。
**这是 (iii) 相对 (i) 的唯一治理成本，(i) 完全不触发**（保留 17，yaml 与守卫测试均不动）。

## 4. 选项 (i) 的改动半径（对照）

因子集与契约**不变**，`factor_daily`、backfill 核验器、准入一致性门、C6.2 筛查、
三份 yaml、守卫测试**全部不动**。需要改的是缺失值语义：

- `scripts/run_real_walk_forward.py:177-179` 的一律 `dropna(subset=feature_order)`
  改为「保留 ≥11/17 个有限因子的行」（即 B1 `minimum_factor_coverage=0.60` 的口径）；
- 但 **B2/B3 会随之失败**：`linear.py:240` 与 `lgbm_ranker.py:294` 的 `_finite_matrix`
  对任一 NaN 直接 `raise`。故 (i) 必须二选一：给 B2/B3 加插补（**会为公式未定义的值造数**），
  或让 B2/B3 退出基线集（**改变 B0–B3 的选拔口径**，进而改变 G4 的比较对象）。
- 候选链无需改：`run_real_candidate_training.py` 本就不 `dropna`
  （冻结抽取器保留 NaN、由 `FoldPreprocessor` 处理）。

## 5. 三个窗口的一致性（本轮新测，补齐 2025 缺口）

C6.2 筛查此前只覆盖 2014–2019（1464 天）与 2020–2024（1212 天），
合计 2676 天，而 2014–2025 共 2919 天——**2025 全年（243 天）从未被筛过**。
本轮补测（`coverage_screen_factor_daily_2025.{txt,json}`）：

- 17 个因子**全部 243/243 天**、`day_ratio 1.0000`、`dups 0`；
- **PASS 11/17，ELIGIBILITY_FAIL 恰为同六个**（372/375/384/389/390/406）。

故 (iii) 的因子集分类在**三个窗口全部一致**，不是只在早年成立。
且合并三窗口可得：**冻结 17 在 2014–2025 全期零个缺失 (因子,日期) 单元**——
与遗留 14 恰有 1 个（`yeslab_387` = GP_018 = `turnover * shift_1(amount)`，2017-10-09）
形成对照。**这支持在选 (i) 或 (iii) 后把覆盖率门的「容忍 1 天」收回为 0。**

### 5.1 那一个遗留单元的根因：三个假说已否证，剩余空间在本仓库之外

已确证的事实：`yeslab_387` = **GP_018**（`prepare_factor_repair_batch.py:33`），
存于 `factor_data9`，`lookback_days=1`，计算器
`数据库/factor_calculators/GP_Factor_018_Calculator.py` 的表达式为
**`mul(turnover, shift_1(amount))`**，`required_factors = ["amount", "turnover_rate"]`。

**两个假说已被否证**：

- **不是「长假后回看窗口不足」**：`lookback_days=1`，而前一交易日 2017-09-29 的
  `yeslab_387` 有 3,168 行（正常）；若该假说成立，每个长假后首日都会复发，
  而 2014–2025 全期扫描显示**只有这一天**。
- **不是通用的 `turnover_rate` 源缺口**：`数据库/factor_calculators/` 下
  **100 个 GP 计算器中有 64 个把 `turnover_rate` 列入 `required_factors`**
  （实测 `grep -l turnover_rate GP_Factor_*.py | wc -l` = 64）。若该日 `turnover_rate`
  整体缺失，会同时打击大量因子；而实测**同日其余 13 个遗留因子全部正常**
  （2,645–3,177 行），且 2014–2025 全期 40,866 个 (因子,日期) 单元中**只有这一个缺失**。

**第三个假说也已否证（2026-09-06 06:15）：不是 10 日历日预热窗的边界效应。**
`数据库/factor_calculators/base.py:126-152` 的 `preprocess_time_range` 先取
`pstart_time = start_time − 10 × lookback_days` **日历日**，再在交易日历上**向前数
`lookback_days` 个交易日**。对 GP_018（`lookback_days=1`）与 `start_time=2017-10-09`：
回看 10 天 = **2017-09-29**，而它正是所需的那唯一一个前置交易日——**零余量、恰好压在边界上**。
这解释了为何 2017 年特殊（2016 首个交易日 Oct 10 → 边界 Sep 30；2018 Oct 8 → Sep 28；
2019 Oct 8 → Sep 28 而前置交易日为 Sep 30，均有余量；**2017 是 2014–2025 中唯一
让单交易日预热恰好落在 10 日历日边界上的一年**）。
**但边界是「恰好够」而非「差一天」**：`数据库/get_data.py:685-703` 的 `get_trade_calendar`
用 `cal_date between date(start_time) and date(end_time)`，**双端包含**，故 2017-09-29
在日历内、会被计入那 1 个交易日，`amount.shift(1)` 本应拿到前置行。**故该假说不成立。**

**另一个确证的判别事实**：`yeslab_387`(GP_018) 是遗留 14 中**唯一**使用 `shift(1)` 的因子
（`shift(1)` 的全部使用者为 GP_013/018/025/054/080，其中 GP_054 → `yeslab_423` 不在遗留 14 内，
GP_013/025/080 未映射到任何准入因子）；而 GP_011(`yeslab_380`) 的表达式是
`mul(turnover, amount)`、**与 GP_018 的 `mul(turnover, shift_1(amount))` 只差一个 `shift_1`**，
且 GP_011 在该日正常产出 3,177 行。故缺失在**特征上**与 `shift_1` 绑定，
但窗口算术核对下来并不缺前置行。

**结论（诚实版）**：三个假说——通用 `turnover_rate` 源缺口、长假回看不足、
10 日历日预热边界——**均已否证**。剩余解释空间是「该 (因子,日期) 的一次性生产缺失」：
GP_018 在 V1 每日任务配置表中的 `lookback_days` 取值、一次失败或部分完成的运行、
或一次未落盘的写入。这些都在 **V1 生产系统**（`数据库/daily_task_new/`）内，
属本仓库 V2 范围之外、且由他人维护（核查时该目录的 `update.py` 正由另一用户的 cron 运行）。
继续深挖需要读另一个团队的生产日志，**且对任何裁决都不必要**，故止于此并如实记录未确诊。

**对决策而言已足够的两点**：
1. **该值可修**——它并非「本质上无定义」：所需的 `turnover_rate`(2017-10-09) 与
   `amount`(2017-09-29) 都在库，标准预热窗经核对也确实覆盖前置交易日。
   故用既有机制（`factors/repair.py::apply_repair_overlay`，配合
   `audit_formula_replay_consistency.py` 中 `yeslab_387` 的 `ReplayDefinition`）重跑该日即可。
   **修复后，覆盖率门的「容忍 1 天」就可以、也应当收回为 0。**
2. **它不阻塞任何裁决**——占遗留 train 的 1/1,214 = 0.08%；且修复必须等 Task 2.4 落地之后
   （改 `factor_observation_clean` 会推进版本指针、使遗留折缓存失效，破坏逐位复现的前提）。
   注意重放时 `lookback_days=1` 的预热窗对 2017-10-09 是**零余量**的，
   故重放应显式给足至少 2 个交易日预热，不要依赖那个恰好够的边界。

**已有的修复机制**（同类先例，非同一根因）：`audit_turnover_gap_repair.py` 处理
2020-09-18 的 turnover 源缺口，其 `DEFAULT_FACTOR_IDS` 亦含 `yeslab_387`，
修复路径为 `research_v2/factors/repair.py::apply_repair_overlay`，
且该脚本自述为**只读**（用 Tushare 仅补齐缺失证据、在内存中重构受影响的 GP 表达式、
在任何 repair overlay 被考虑之前先检查有限源重叠）。

## 6. 关联文书

- `docs/handoffs/2026-09-05-compute-and-coverage-decision-brief.md`（裁决 2/3/4 与标定结论）
- `docs/superpowers/plans/2026-09-05-frozen-set-baseline-reconstruction-plan.md`（Task 2.2、3.1b）
- `research_v2/runbooks/real-walk-forward-reproduction.md`（前置步骤与三道门）
- 实测证据：`coverage_gate_c62pass11.txt`、`coverage_gate_option_i_ge11of17.txt`、
  `coverage_screen_factor_daily_2025.{txt,json}`、`legacy_train_coverage_calibration.{txt,json}`、
  `train_coverage_calibration_and_2017_hole.txt`
