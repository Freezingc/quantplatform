# C 计划逐款核验：待裁决清单与未接线治理路径

> 生成时间：2026-09-06 10:40 CST（Asia/Shanghai）
> 分支 `codex/publish-quant-platform` · 状态：**七项待人工裁决 / 决定，其中一项阻塞一条性能重写**
> 来源：C 计划勾选审计续七～续十二（`docs/superpowers/plans/2026-08-01-quant-research-v2-c-factor-governance-plan.md`）
> 本文书**只汇总已实测的事实与选项**，未代做任何一项决定，未变更任何运行中的链。

## 0. 为什么单独出这份清单

逐款核验的结论散在计划的六节审计里（续七～续十二），每节都写明了"记录不代改"的理由。
但**累积起来它们不是七条脚注，而是一份待办**：其中若干项互相依赖，
顺序错了要改两遍。本文书把它们按**依赖关系**排出，并标出哪几项其实是同一个问题。

**一个贯穿性的事实先摆在前面**：C 计划里**四处**治理实现是"写了但没接上生产路径"的。
这不是七条独立缺陷的背景，而是决定其中几条**该怎么修、以及现在修是否安全**的关键——
未接线的代码可以改契约而不影响任何已物化数据，已接线的则不行。

## 1. 未接线的治理路径（四处，已逐一 grep 核实）

| 实现 | 计划条目 | 生产调用者 | 生产中真实使用的替代路径 |
|---|---|---|---|
| `FactorRegistry` / `FactorStatus` | C1.2–C1.4 | **0** | 无（没有任何东西注册因子定义或推进其状态） |
| `create_factor_set` / `FactorSet` | C8.1–C8.2 | **0**（24 处命中全在 `tests/`） | `artifacts/diagnostics/factor_set_frozen_v1.json`，由 `scripts/apply_admission_decisions.py` 直接写成 JSON dict；成员资格来自**人工裁决 D1–D4** + `data_version`，其身份**不是** `create_factor_set` 的内容寻址哈希 |
| `select_cluster_representatives` | C7.3–C7.4 | **0** | 无 |
| `expand_fundamental_events` | C4.2–C4.4 | **0** | 只有 `materialize_technical_availability` 被接线（`materializer.py:101`）；**基本面可用性整条路径未接线** |
| C6.2 结构质量快照模块 | C6.1–C6.2 | **0** | `scripts/screen_factor_daily_coverage.py` 自带一套阈值与逐日计数，不调用该模块（续四已记） |

**含义**：这五处的缺陷**当下都没有在污染已发布数据**（好消息），
但也意味着 **C 计划中相当一部分治理是"可被审阅但尚未承重"的**（坏消息）。
接线顺序本身需要决定——见 §2 的决定 A。

## 2. 待裁决 / 待决定

### 决定 A（**最上游，其余多项依赖它**）：以哪条路径为准

`factor_set_frozen_v1.json`（人工裁决 D1–D4 + `data_version`）与
`create_factor_set`（内容寻址 `factor_set_id` + registry definition hashes）是**两套互不相通的冻结集表示**。
C1.4 要求的「只有 `ACTIVE` 因子可加入 factor set」**在两套里都不成立**：
`create_factor_set` 收裸元组、从不接触 registry（其 docstring 却写着
"from **active-registry** member identities"——**docstring 声称了一个代码不执行的约束**），
而 `apply_admission_decisions.py` 里 grep `ACTIVE|registry|FactorStatus` **零命中**。

- **(A1) 以 JSON 工件为准**：则 C1.4 的 ACTIVE 门与 C8.2 的内容寻址**属于未采用的设计**，
  应在计划里明确降级为"设计参考"，不再作为待实现项挂着。
- **(A2) 以 registry + `create_factor_set` 为准**：则需要一条从 `source_catalog.yaml`
  到 `FactorDefinition` 到 registry 状态推进的**装配路径**，并把冻结集的生成改为经它产出；
  这会**改变已结案契约**（见裁决 4 的同类问题：应新发 V2 而非就地改 `factor_set_frozen_v1.json`）。
- **(A3) 两套并存、各司其职**：需要写明各自的权威范围，否则第三次发现同一模式时仍然无从判断。

**为什么它最上游**：C1.4 的 ACTIVE 门、C8.2 的内容寻址、以及"要不要给 `create_factor_set`
加 registry 参数"都取决于此。**在 A 未定前实现任何一项都是在猜。**

### 决定 B（**新增，本清单里唯一会算错数的一项**）：跨报告期的基本面版本优先级

**实测缺陷**：`expand_fundamental_events` 的版本排序键是
`["available_date", "report_period", "source_revision_id"]` 全降序，**`available_date` 是第一键**。
构造 FY2024（`report_period` 2024-12-31，公告 2025-03-28，available 03-31）与
**对更旧 Q3 2024 的迟到重述**（公告 2025-04-10，available 04-11），在 signal date 2025-04-15：

```
signal_date report_period  raw_value  available_date  source_revision_id  stale_days
 2025-04-15    2024-09-30        1.5      2025-04-11    q3-2024-restated           4
```

**发布的是被重述的 Q3（1.5），FY2024（2.0）被丢弃**——报告期**向后退了一格**
（同一 fixture 在 2025-04-01 发布的是 FY2024，测试对此有显式断言）。
按通常 PIT 基本面语义这是错的：某日的值应来自**当时可得的最新报告期**，修订只在**同一期内**决胜。

**规格两可**：C4.3 把区间定义为到 `next_version_available_date - 1 trading day`
（**单一线性版本序列**，即现状）；C4.4 只讲**同一 report period 内部**的修订。**都没说跨期谁压谁。**

- **(B-i) 最新报告期优先，修订仅在同期内决胜**：排序键改为
  `report_period DESC, available_date DESC, source_revision_id DESC`。
  代价：C4.3 的区间公式须重述为**按报告期分组**，因为不同期的有效区间会**重叠**、不再线性。
- **(B-ii) 最新可用版本优先（现状）**：与 C4.3 字面一致。代价：如上，报告版本会因旧期重述而倒退。
- **(B-iii) 并列输出、由下游按因子语义决定**：代价：展开不再是每 (signal_date, ts_code) 一行，
  波及 C5 的统一 observation 输出契约。

**附带更正一条测试名的过度承诺**：既有测试
`test_fundamental_event_prefers_latest_report_period_before_revision` 的名字断言"报告期优先于修订"，
但其 fixture 给两个版本**同一 `ann_date`**，故 `available_date` 相同、排序落到第二键才正确。
**名字读起来像一条 precedence 保证，实际只是一条 tie-break 保证。**（测试本身是好的。）

### 决定 C（**被 B 阻塞**）：基本面展开的性能重写

**实测成本**：`expand_fundamental_events` 是**逐 universe 行的 Python 循环**，
每行对**整个 events 帧**做一次布尔掩码 → `O(universe_rows × events)`。
输出**不是**笛卡尔积（只为实际 universe 成员产出行），**但时间是**。

```
 universe  events   seconds    us/universe-row  rows out
      500     100      0.83             1664.4       190
     1000     200      1.29             1286.0       190
     2000     400      2.74             1371.1       380
     4000     400      6.71             1677.9      1350
     8000     800     13.99             1748.4      2339
```

外推（~1676 µs/行）：**单因子单折**（5000×250 = 1,250,000 行）≈ **0.6 h**；
**487 因子单折** ≈ **12 因子·天**；**6 折** ≈ **71 天**。
每行成本在 universe 规模 16 倍区间内持平到上升，**不是预热假象**。
证据：`/data/zhangyudong/qrv2_research_artifacts/fundamental_expansion_scaling.txt`。

**正解**（as-of/区间连接：按证券把版本按 `available_date` 排序，用 `merge_asof`/`searchsorted`，
`O((N+M) log M)`）**必须逐位复现现在的 tie-break 顺序**——而那正是决定 B 争议的东西。
**故：先定 B，再重写 C。先做 C 等于把有争议的规则固化进更快的路径，日后要改两处。**

### 决定 D：C7.4 代表因子排序键——以规格还是以实现为准

规格：`eligible → FDR pass → direction stability → median abs IC → coverage → lower conflict rate → factor_id ASC`
实现：`(fdr_q_value ASC, -signed_mean_rank_ic, -positive_year_count, -coverage, conflict_rate ASC, factor_id ASC, factor_version ASC)`

- **(a) 中间两项顺序相反（会改变选出的代表因子）**：规格 direction stability 先、IC 后；
  实现 IC 先、`positive_year_count` 后。**当一个因子 IC 更高而方向稳定性更差时，代表不同**，即模型输入不同。
- **(b) `mean` vs `median`**：候选帧里只有 `signed_mean_rank_ic`，**没有 median IC 字段**，
  故按规格实现必须**在上游新增一个统计量**。（"signed vs abs"已证在本管线上无差别：
  进入代表排序的候选必有 `signed_mean_rank_ic > 0`。）
- 「不得读取 validation/test 指标」**已满足**。

**注意 `select_cluster_representatives` 未接线**（§1），故 (a) 当下不改变任何已训练模型的输入；
但接线前必须定，否则第一次接线就是第一次受影响。

### 决定 E：C7.1 的 scope 绑定与 C7.2 的覆盖率分母（**两项须一并定**）

- **C7.1**：`compute_factor_similarity(observations, *, start_date, end_date, minimum_common_members,
  minimum_valid_days)` —— **没有 `universe_id`，没有 `data_version`，模块内无任何 query builder / SQL**。
  它接收调用方已装配好的内存长表，故 scope **只由日期界定**：
  **同一 universe 换个数据版本、或换个 universe 而日期相同，算出的相似度无从区分，也不会报错。**
  与 C7.3 已交付的 `cluster_id` 内容寻址形成对照——**簇有稳定身份，而产生它的相似度矩阵
  没有记录自己是在哪个 universe / 哪个数据版本上算的**，故跨折或跨版本比较簇时无法判断可比性。
  日期闭区间与 scope 隔离**已满足且有测试**。
- **C7.2**：四款中三款成立（同日共同非空 ≥30、有效日 `abs(rho)` 中位数、有效日 <60 标 INSUFFICIENT
  且不连边）；**缺"覆盖"字段**。`valid_days` 不能代替：两个 pair 可同为 `valid_days == 60`，
  一个每天在 4000 个标的上算、另一个每天刚好卡在 30 的下限，**当前输出无法区分这两种证据强度**，
  而下游正是拿这个中位数去连边、去选代表。
- **为何须一并定**：覆盖至少三种口径——(i) 有效日共同成员中位数 ÷ 该日 universe 全量；
  (ii) ÷ 两因子各自覆盖的并集；(iii) 有效天数 ÷ scope 内总交易日数（已被 `valid_days` 隐含）。
  **(i) 最贴近"证据强度"，但它需要 universe 口径输入——正是 C7.1 缺的那个绑定。**
  先挑一个分母写进去、事后再改，会改变 `INSUFFICIENT` 之外的**所有输出数值**，
  进而可能改变连边与代表选择。

### 决定 F：quarantine 证据列的 JSON 格式（**已实测为非法 JSON**）

发布层聚合 quarantine 时用 `json.dumps`，过滤条件是 `pd.notna`——**`pd.notna(inf)` 为 True**，
故 `distinct_values_json` 可以是 **`"[Infinity]"`**，**RFC 8259 不允许**，
严格解析器（数据库 JSON 函数、JS `JSON.parse`、serde_json）**直接拒绝**。
可达路径已实测：C2.3 刻意把 ±inf 留在 `raw_value`（转为 invalid flag 而非 NaN）→
某键观测全为非有限值 → 以 `NO_FINITE_VALUE` 进 quarantine → 聚合出 `"[Infinity]"`。
**代价落在最痛处**：quarantine 就是"不可发布观测"的审计轨迹，
而**恰好是描述源里最坏数据的那条记录无法被标准工具读取**。
同一次聚合还**丢掉了 reader 的 `invalid_reason`**，故发布行**无法区分
"源说了无穷"与"源什么都没说"**——而这个区分正是 C2.3 特意建立的。
**clean 批次全程不受影响**，纯属证据格式问题。

三个候选：字符串哨兵（`"Infinity"`）／ `null` 加标志位 ／ 把 `invalid_reason` 带下去。
**三者都会改变已持久化列的内容从而改变 `record_hash`**，且 migrations 已定义该列——
故需要决定，不宜代改。

### 决定 G：C3.1 的两处去重偏差（**须等 Task 2.4 落地后才能动**）

与 §1 不同，`deduplicate_factor_observations` **在活的物化路径上**（`materializer.py:98`），
故收紧判定会**重新分类真实行、改变 `factor_observation_clean`**——
红线：会推进版本指针、使遗留折缓存失效、破坏逐位复现。**Task 2.4 落地前不得动。**

- **(a) `IDENTICAL_COLLAPSE` 只比 `raw_value`，不比"所有非审计字段"**。
  两行业务键相同、值相同、但 **`source_available_date` 不同**时会被折叠，
  而**谁存活由最小的 `source_row_id` 决定**——那是**溯源字段、与语义无关**。
  即：**可用日（决定该值何时可被下游使用的 PIT 字段）是被溯源排序"顺手挑中"的。**
- **(b) `SINGLE_NONNULL` 不要求"恰有一个有限非空值"**：`[1.0, 1.0, NaN]` 也被标为 SINGLE_NONNULL。
  **发布值不受影响**（两值相等；不等即冲突），故是**标签不精确**而非数字错误——
  但该标签是复核者理解"这行为什么活下来"的依据，且会被持久化。
- **(c)「源如声明量化精度则先按该精度规范化再比较」——对本源为空条款，已核实**：
  `FactorDefinition`（17 字段）、`SourceBatch`（8 字段）均无精度字段，
  `source_catalog.yaml` 中 `precision|quantiz` **命中 0 次**，`unit` 全为 `UNKNOWN`。
  若日后接入声明精度的源需重新核验——届时"先规范化再比较"会与
  「不用宽松 isclose 吞掉真实冲突」直接张力，须明确**规范化只用于折叠判定、不用于发布值**。

## 3. 缺输入而非缺决定的一项

**C3.4**（用已知问题构造回归 fixture）要求"从 Phase 0 审计中抽取匿名化的 GP_017、GP_077 重复键样例"。
**那份审计在本仓库不存在**：全仓 grep `GP_017|GP_077` 命中 13 个文件，逐个核过性质——
候选筛查 IC/覆盖率统计、准入清单、训练质量报告，**无一是"重复业务键"审计**。
唯一以 GP_077 为主角的诊断 `gp_turnover_scalar_degeneracy_diagnostic_v1.json` 讲的是**标量退化**
（`turnover × 每日横截面标量` → 与 `turnover_rate` 完全冗余，`yeslab_436/446/468` 三者 RankIC 逐日完全相同），
**那是 C7 聚类的问题域，不是 C3 去重的**。
**顺带确立映射：GP_017 = `yeslab_386`，GP_077 = `yeslab_446`**
（出自 `gp_candidate_screen_2020_2024/summary.csv`）；注意 `yeslab_386` 与待查的
`yeslab_387`@2017-10-09 是相邻一对。

两条出路：**(i)** 在 V1 树（`数据库/daily_task_new/`，他人维护）里找到 Phase 0 审计原件；
**(ii)** 对 `yeslab_386`/`yeslab_446` 跑一次**有界的只读重复键普查**据实构造 fixture。
**(ii) 刻意未做**：要对 `factor_data*` 发聚合查询，而 Task 2.4 正在跑第 13 小时、
是**裁决 3 与裁决 4 的共同前置**；在关键路径收尾阶段给源库加负载换一个可以晚做的 fixture，不划算。
**等 2.4 落地后即可跑。** **不代造 fixture**：C3.4 要的是真实审计抽取的样例，
凭空编一个会让回归测试**看起来有真实出处而没有**，那比缺这条测试更糟。

## 4. 建议的处理顺序

1. **等 Task 2.4 落地**（`/tmp/legacy_flagcheck/report.json` → 自动核验 → 期望 `CONSISTENT`），
   勾计划 2.4，并据其结果处理裁决 3。
2. **决定 A**（以哪条冻结集路径为准）——它决定 C1.4/C8.2 是"待实现"还是"未采用的设计"。
3. **决定 B → 决定 C**（先定跨期语义，再重写 as-of 连接）——B 是本清单里唯一会算错数的一项。
4. **决定 E**（C7.1 绑定 + C7.2 分母，一并）——在 C7 接线前定，代价最低。
5. **决定 D**（C7.4 排序键）——同样在 `select_cluster_representatives` 接线前定。
6. **决定 F**（quarantine JSON 格式）——独立，随时可定。
7. **决定 G**（去重偏差）——**必须在 Task 2.4 落地之后**，且会改变已物化数据，需配合版本指针策略。
8. **C3.4**——2.4 落地后跑只读普查即可解锁，无需裁决。

## 5. 本轮已交付（均已提交推送，无需裁决）

- **C7.5** 四款案例齐全并变异验证；其中一条非有限值测试被查出**在正确与被破坏实现下结论相同**，
  已重写为两条结论相反的测试（29 < 30 下限 → 必须整天跳过；恰好 30 → 必须保留）。
- **C1.5** catalog 对源 schema 的 dry-run：只发 `schema(...).colDefs`（实跑记录恰好 11 条查询、无一含 `select`），
  不构造任何写会话（用 AST 而非文本钉住），表名白名单在**发出任何查询前**校验。
  实跑：**487 mapped / 0 blocked / 0 retired / 无未映射源列**；11 张表 schema 完全一致。
- **C2.1** 的唯一缺口（额外未注册列**被报告**）随 C1.5 关闭；并补上第三条边——
  catalog 登记 ↔ 读路径硬编码投影 `_SOURCE_COLUMNS` 的一致性（此前无人校验，
  一个登记了别名列的条目会被两道现有检查同时放行）。
- **C1.1** 已勾并**实测可复现**：从活的 `factor_metadata` 重跑生成器，
  **snapshot hash 与 487 条条目逐字节相同**、双向差集为空；离线复演报告逐字节相同。
  并补上生成器缺失的通配符拒绝（此前会把 `GP_*` 照字面收进 catalog）；
  **加防护后 hash 与条目完全不变**。
- **C3.2** 已勾：`DedupStatistics` 四个计数中 **`invalid_groups` 与 `single_nonnull_groups`
  此前在整个套件中从未被断言过**，而 `conflict_groups` 是持久化快照的存储列。
- **C4.1** 已勾：补齐"不 ffill"，且**覆盖两个读法**（不凭空造行 / 已有行的缺失值不得继承前一日）——
  只测行数的版本在 `groupby.ffill()` 变异下**全绿**。
- **C1.4 状态机**：此前只有一条 happy path 测试，补 20 条（只能向前、被拒迁移留在原状态、
  RETIRED 终态、任一活状态可直接退役、新版本与旧版本并存且各有生命周期）。

**累计变异验证 25 个，最终全部被预期测试捕获**，但过程中查出**四次测试自身的缺陷**——
均已修正测试而非放过变异：
**两次**是"测试在正确实现与被破坏实现下结论相同"（续八：非有限值测试把恰好 30 个有限对
断言为 PASS，而把 inf 计入也是 40 ≥ 30、结论一样；续十二：只断言行数的不-ffill 测试
在 `groupby.ffill()` 下全绿，因为填值不改变行数）；
**一次**是变异没落在被测路径上（续九：把 `if reasons: break` 插在类型检查前，
而该测试的两个缺陷都是"列不存在"、走更早的 `continue`，根本到不了那一行——
**变异没生效不等于性质没被测到**）；
**一次**是测试辅助函数自己走错了状态（续十：`_registry_at(DRAFT)` 实际走到了 ACTIVE，
于是"非法迁移被拒"的测试以错误的理由通过；**实现是对的、测试是错的**）。
统一判据：**一条测试是否有效，看"被破坏的实现下它是否给出相反结论"，
且必须对每一类可能的破坏分别验证**——覆盖了"造行"不等于覆盖了"改值"，
覆盖了"类型不对"不等于覆盖了"缺列"。

**全量：1431 passed / 45 skipped**，`ruff check src tests` 清、`mypy src` 157 文件清。

## 6. 关联文书

- 计划与逐款审计：`docs/superpowers/plans/2026-08-01-quant-research-v2-c-factor-governance-plan.md`（续七～续十二）
- 算力与覆盖决策（裁决 1–4）：`docs/handoffs/2026-09-05-compute-and-coverage-decision-brief.md`
- 裁决 4 影响面：`docs/handoffs/2026-09-06-adjudication-4-implementation-blast-radius.md`
- 证据目录：`/data/zhangyudong/qrv2_research_artifacts/`
  （`catalog_schema_dry_run{.txt,_report.json,_schemas.json}`、
  `catalog_reproducibility{_report.json,.txt}`、`factor_metadata_frozen_snapshot.parquet`、
  `fundamental_expansion_scaling.txt`）
