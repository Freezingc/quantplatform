# 模型侧走查证据包（2026-09-14）

配套叙述文档：`../../2026-09-14-model-side-walk-forward-results-v0.1.md`。
这个目录放的是**机读证据 + 被引用的原件副本**，全部是文本；
**没有权重、没有检查点、没有 parquet 面板、没有数据库导出**。

**如果要做的决定是"那 24 个不可评日怎么办"，直接读
[`../../2026-09-15-the-24-days-decision-sheet-v0.1.md`](../../2026-09-15-the-24-days-decision-sheet-v0.1.md)**：
四个选项的价格表（跌破天数 / G6 / 余量 / 对照强度 / 前置条件 / 是否开新谱系）、
三个已被实测排除的"伪选项"、以及一个此前没人量过的悬崖——
**地板上调 0.005 就会让 G6 从 0.9752 掉到 0.9319 而不及格**。
它下面的每一个数字都由 `dense_intersection_screen/probes/analyze_floor_sensitivity.py`
从仪器输出生成，机读件在 `dense_intersection_screen/measurements/floor_sensitivity.json`。

## 文件

| 文件 | 生成方式 | 内容 |
|---|---|---|
| `candidate_standings.json` | `extract_from_reports.py` | 9 个运行工作区（含基线与复现副本）的完整门控 G1–G6、OOS 指标、逐年 RankIC、逐折明细、`report.json` 与 `promotion_evidence.json` 的 sha256；基线的逐折 best_simple 选择与落选原因 |
| `per_fold_winners.json` | 同上 | 逐折胜者配置、集成与逐 seed 验证 RankIC、每折 top3、冻结选择点 id |
| `yearly_rank_ic.json` | 同上 | 逐年 RankIC + 门控状态，便于直接出表 |
| `factor_sets.json` | 同上（用 `ast` 从驱动源码解析，不 import） | 三个冻结因子集的完整 id 列表、规范特征名 `f00…f14`、基线可见性规则的原文理由 |
| `datasets.json` | 同上 | 每折已发布数据集的行数、列序、分片年份、manifest 与分片哈希、数据版本摘要 |
| `in_progress_search_records.json` | 同上 | **M2 中途快照**：逐格搜索记录（配置、集成与逐 seed 验证 RankIC、记录 sha256 与写入时刻）。**验证集口径，不是 OOS** |
| `provenance.json` | 同上 | 所有被读文件的 sha256、主树 git HEAD 与分支、提取时刻、刻意排除的内容清单 |
| `problems_found.md` | 手写，逐条给出出处 | 问题记录：评估阶段六个阻塞点、成本模型三次低估（含本轮我自己的一次 ETA 误算）、stale 门禁默认值、四处永不触发的守卫、两个驱动缺陷、GPU/内存/主机与治理未决项 |
| `comparison_interim.md` | `research_v2/scripts/summarize_candidate_comparison.py`（主分支 `39ee96a`） | **工具渲染**的跨候选对比：可比性分组与排除理由、G1–G6 矩阵、两两增量矩阵（带"共享基线恒等式"校验）、领先候选的逐年分布与逐折最差 seed。同输入两次渲染 md5 全同 `c79e5545ebad6d36c3e935935176f3b7`。**M2/M3 出报告后重跑即得计划 Task 5.1 的那份文档** |
| `yearly_baseline_deltas/` | `research_v2/scripts/analyze_yearly_baseline_deltas.py`（主分支 `78216af`，sha256 前缀 `c36aadec52042ac4`） | 三个已完成候选各一份 json + md：**基线在候选 945 日集上的逐年值与逐年增量**（此前任何工件里都没有）、四道逐位验收的实测值、**24 个不可评日的逐日清单**、`coverage_attribution`（**逐侧**覆盖率：候选四年恒为 **1.0000**、基线 0.89–0.99、**24 天全部由基线一侧决定**）、以及关掉 0.90 地板后的敏感性重算（结论更强，且与基线自己的签名报告**逐位相同**） |
| `dense_factor_coverage/` | `research_v2/scripts/analyze_dense_factor_coverage.py`（主分支 `f96c4b2`） | 那 24 天的**再上一层原因**：稠密因子不完整的逐年/逐月/逐日份额、缺失**组合**与**嵌套**判定、受影响证券的代码前缀分布与科创板占比。结论：`yeslab_357` 与 `yeslab_471` 成对且嵌套地对**科创板 688/689** 证券缺失、**止于 2023 年 9 月**，而"不完整份额 > 10% 的日子"与被丢掉的 24 天**逐日相同** |
| `star_industry_coverage/` | 7 个**只读**探针（`probes/`，select-only 查 `dfs://YesLab` 与 `dfs://QuantResearchV2Test`，最后一个只读本地缓存）+ `make_provenance.py` | 那 24 天的**根因，且推翻了 `dense_factor_coverage/` 留下的成因方向**：不是因子生产端，而是**我们重放时的 PIT 行业输入**。`materialize_frozen_factor_daily.py` 只给 `PIT_FACTORS = {yeslab_357, yeslab_363, yeslab_471}` 传行业提供方，而 SW2021 成分缓存里**科创板最早的 `in_date` 是 2023-09-25**（559 只同日入册；非科创板到 1990-12-10）。含四段链路同日对齐（上游有值 → `factor_daily` 科创板 0 行 → pin==manifest → 面板报缺）、两处算术闭合（1,286÷4 交易日=321.5/日；8 月 RankIC 时序证明当时截面完整）、三条被排除的假设、`689009.SH` 四年全缺的解释、论文措辞与三个处置选项。**行业缓存 32 份原件已随包复制**——数据商重述过这段历史，重新取数不能复现 |
| `coverage_gates_vs_the_intersection_floor/` | 1 个**只读**探针（复用 `screen_factor_daily_coverage.py` 的纯函数，阈值单点来源；把 factor_id 过滤推进 SQL，扫描量小 30 倍）+ `make_provenance.py` | 那 24 天**为什么没有任何一道门拦住**：把冻结 **v2 的 15 个因子**（此前只为 v1 的 17 个跑过）在**实际消费的 `factor_daily`** 上、在 **OOS 窗口 2022–2025**（此前只跑过训练窗口）筛查，并把**两台在用仪器的阈值都应用到同一组计数**上。结果 **9 个稠密因子全部 PASS**，含 `yeslab_357`（median 0.9992 / p05 **0.9131**）与 `yeslab_471`（0.9936 / **0.9091**）；而面板实测 2023 年九因子交集覆盖率中位数 **0.9093**，与之差 0.0002——因为 357 嵌套在 471 里，**交集就等于 471 自己**。门看单因子分位数（p05 门 0.30/0.70），地板管九因子交集逐日值（0.90），**被度量的量与被约束的量不是同一个**。另测出：**6 个稀疏因子在 OOS 窗口两台仪器都 FAIL**（median 0.46–0.49，候选消费全部 15 个）；**`universe_daily` 的多版本分母陷阱**（20 天有多个 `is_member=1` 版本，最坏 5.007×，冻结链因 `filter_current` 安全、治理脚本的 `minimum_coverage_ratio` 全被污染成 ≈0.199，但判定未翻转） |
| `reproduction_audits/` | `research_v2/scripts/verify_prediction_reproduction.py`（主分支既有工具，本轮首次对这两个运行使用） | **预测复现审计**的 JSON 摘要：`m0r_v2` 四折 3,557,762 行与 `m1_full_v1` 四折 3,553,163 行，**全部 `identical=true`、`max_absolute_difference=0.0`**；含机器读出的 registry 状态（`WALK_FORWARD_PASSED` / `UNREGISTERED`）与实际打分线程数（52 / 1）。这是"分数确由该 bundle 算出"的唯一证据——签名链只证明归属 |
| `dense_intersection_screen/` | 新只读仪器 `screen_dense_intersection_coverage.py`（分支 `codex/dense-intersection-screen` 提交 `c162073`，**未合并**；副本在 `probes/`）+ `make_provenance.py` | `coverage_gates…` 缺的那台仪器：**只读已发布的表就能预测将被丢掉的日集**，不需要任何训练运行。分子=当日 9 个稠密因子全有行**且当日有有效标签**的证券数（都取已发布 `data_version`），分母=当日 `label_valid=1` 行数（2023-07-05 = 3,627，与面板逐行相同），地板**运行时从 `gates.py` 解析**并有漂移测试盯着。**验证：2022–2025 预测 24 天，与 `m0r_v2` 实测逐日完全相同（`identical=True`，零误报零漏报），且逐年覆盖率到浮点末位相同**（2023 min 0.8902674386545354 两边 16 位全同）。含一处差点让仪器失效的设计错误：第一版分子未限制在有效标签证券上，**只预测出 21/24、漏 3 天且全在不安全方向**（2023-04-28 未限制 0.9070 对限制后 0.89997），修法是 `ej()` 脚本模式连接（DDB 不给子查询/`loadTable` 起别名）。**这是测量工具不是策略**：无人调用、不改门值，`--fail-on-prediction` 默认关闭 |
| `extract_from_reports.py` | — | 只读提取器，stdlib only |
| `vendor_sources.py` | — | 原件复制器，同时重算 `vendored/MANIFEST.json` |
| `vendored/` | `vendor_sources.py` | 14 份仓库文档 + 7 份签名 `report.json` 的**逐字节副本**，`MANIFEST.json` 记源路径、sha256、取值时的 git HEAD。新增的一份是 `docs/reports/2026-09-14-yearly-deltas-and-the-24-unevaluable-days.md`——逐年增量与那 24 天归属分析的完整方法、§4.1 逐侧覆盖率表、§4.2 的设计观察与三种处置、§6 敏感性、§8 边界 |
| `vendored/docs/reports/2026-09-14-*.md` | 同上 | 逐年增量与 24 天归属（含 §4.1 逐侧覆盖率表、§4.2 设计观察与三种处置、§6 敏感性）、稠密因子覆盖缺口的完整叙述（含给因子侧的移交问题与论文方法学段落该怎么写）、预测复现审计（两个领先候选四折逐位相同）、以及 M1 的影子审批前置清单（§3.1 是服务侧线程钉住这条硬前置） |

### `dense_intersection_screen/` 的追加测量（§9–§13，2026-09-15 02:20）

同一台仪器又量了四件事，其中第三件**直接关系到"15 个因子最终用哪些"这个待定决定**：

1. **交叉验证到三个运行**：24 天预测与 `m0r_v2`、`m1_full_v1`、`m1_parallel_v1` 三份实测日集
   **逐日全同**。这**不是三次独立验证**（三者共用同一套基线工件与标签），
   它证明的是"这个日集是**基线工件 + 标签**的性质、与具体候选无关"——
   也就是这台仪器能在候选训练之前算出它的原因。
2. **训练/验证窗口 2014–2021：1,045 / 1,948 天在地板之下**（2014–2016 三年全部，
   2015 最差 median 0.7631 / min 0.4453，最差十天全在 2015-07～10）。
   不使任何已发布结果失效（地板只在 TEST 日判定），但**冻结基线在 2014–2017 的行集远比 OOS 窗口薄**。
   **归因已测到因子**：2015 下半年 `yeslab_206` **146/146 天都是最弱的**（median 0.7162），
   其次 `yeslab_443_reg30_max20_v1`（0.7665），而 `yeslab_357` 在这里是 **0.9973**——
   **稠密 9 的薄弱是两个时代两个原因**（2014–2017 是两个修复变体，2022–2023 是缺行业标签），
   科创板 2019-07 才开板，两段互不重叠。
3. **全 15 因子口径：968 / 969 天跌破地板、中位覆盖率仅 0.2194**。
   若基线工件定义在"15 个全齐备"的行上，G6 会变成 ≈0.001。
   所以稠密/稀疏的分工是**评价可行性的前提**：**稀疏因子可作候选输入**（冻结链不 `dropna`），
   **但进了稠密子集就会让评价失效**。换因子名单前该跑的就是这台仪器（60 秒），不是九天后在评价阶段发现。
4. **候选真正吃进去的矩阵**：缺失单元 **20.2%–21.9%**、至少缺 1 列的行 **73.6%–77.0%**，
   缺失**按 3 列 / 6 列成组到达**（不是均匀散布）。
   **对账**：2022 年"九列全有限"= **746,521** 行 = §A 阻塞点 3 记的 747,461 − 940，逐行吻合。

### `star_industry_coverage/` 的追加测量（2026-09-15 04:35）：选项 1 的前置**已确认成立**，并撤回两处措辞

新增 5 个只读探针（`probes/` 现 12 个）、6 份机读件（`measurements/option1_*.json`、
`cache_halves_not_independent.json`）、1 份叙述（`option1_feasibility.md`），
以及数据商另一支 API 应答的字节副本 `vendored_member_all/`（7,912 行，含 `MANIFEST.json` 与 `combined_sha256`）。
结论已并入决策单 **v0.2**（`../../2026-09-15-the-24-days-decision-sheet-v0.2.md`，取代 v0.1）。

1. **决策单 v0.1 §5 唯一标着"尚未确认"的前置，答案是"能"。**
   数据商有两支行业成员 API：在用的缓存由 `index_member(index_code=…)` 建成，
   **这支 API 把每只科创板股票的 `in_date` 都截断成 2023-09-25**；
   `index_member_all` 给真实入册日，**最早 2019-06-26**，且是在用缓存的**严格超集**
   （多 9 只、少 0 只，含此前"缓存里完全不存在"的 `689009.SH`，其真实入册日 = 上市当日 2020-10-29）。
   → **21 个月的板块级空洞不是数据商没有数据，是我们问错了接口。**
2. **机制被精确证明**（先逐日复现仪器：分子 24/24、分母 24/24 全同，复现不过即拒绝分解）：
   那 24 天的科创板有效标签行 181–330 行，**已集齐九因子的是 0 行**；
   科创板行里 `yeslab_357` 与 `yeslab_471` 命中数 **24 天全为 0**，其余七个因子命中率 **0.9936–1.0031**。
   到地板还差 **1–36 行**、可修 **181–330 行**、**最小余量 164 行**
   （对照被否掉的"只移 206"：余量 0–8 行）。
3. **选项 1 的价格**（上下两界，在最差日重合）：跌破 **24 → 0**、**G6 = 1.000000**、
   最差日 2022-07-29 覆盖率 **0.949031**（高于地板 +0.049031），
   但**按 v0.1 §4.1 自己提的 0.95 线不达标，差 4 行**；选项 2 的 0.971370 达标。
   **【本条已被下面第 5 条推翻，0.949031 只是"仅算科创板行"的下界，正确值是 0.976751–0.979136，过线】**
4. **重述风险 ≈ 0**：两支 API 都有科创板标签的重叠窗口（2023-09-25…2025-12-31，每第 10 个交易日）
   **34,571 对里 34,570 对 L1 相同，一致率 0.999971**（唯一一处 2024-05-20 `688292.SH`）。
   → 换 API 重建缓存对已发布值几乎无改动，是**加法不是重述**，选项 1 的谱系代价比 v0.1 设想的小。

**两处措辞撤回**（详见 `star_industry_coverage/root_cause.md` 内的三条更正框与 `option1_feasibility.md` §5）：

* 上表 `star_industry_coverage/` 一行末尾的"**数据商重述过这段历史，重新取数不能复现**"**不成立**。
  `688002.SH` 在 `index_member_all` 下是两条连贯记录：`801740 国防军工 20190626→20260820` 与
  `801080 电子 20260821→今`。缓存是 **2026-08-14** 建的，那天 `801740` **确实还是现行行业**；
  改行业发生在建缓存**之后一周**。随包复制缓存**仍然必要**，但理由是"在用的 API 会截断科创板 `in_date`"，
  不是"数据商不可靠"。
* 上文与 `dense_intersection_screen/finding.md` §10.3 里"**SW2014 与 SW2021 两支缓存互相印证**"
  **不构成两次独立观测**：两半重叠的 **27 个行业文件全部字节级相同**（一致率 1.0000），
  且 `pit_industry_sw2021_members_*` **一个都不存在**（SW2021 走遗留文件名回退）。
  根因：`audit_tushare_pit_industry.py:472` 的 `index_member` **不接受 `src` 参数**，
  随 `src` 变的只有 `index_classify` 的分类表（28 vs 31 个 L1）。洞本身是真的，已用两支**不同** API 独立复核。

**未做**：没有重放上游管线（357/471 由上游按行业标签生成，本轮量的是标签供给与行数算术）；
没有改任何代码、缓存、门值、冻结集、签名工件或数据库；`research_v2/artifacts/tushare_cache/` 一个字节未动。

#### 同日 05:15 追加：三条实测，其中一条**推翻了上面第 3 条**

再加 6 个只读探针（`probes/` 现 18 个）、9 份机读件、`option1_feasibility.md` §8–§12，
决策单就地更正（仍是 v0.2，顶部加了更正框）。

5. **上面第 3 条的"0.949031、差 4 行不到 0.95"是错的，撤回。**
   错因是口径漏项：只把科创板行算作补齐收益。按证券级别把 24 天的 **8,782** 个缺口行四分类
   （先逐日复现仪器，分子分母 24/24 全同）：可修科创板 **7,544** + 可修非科创板 **136** +
   **已有标签、缺的是其余七因子之一的 1,101（补标签救不了）** + 两支 API 都无标签的 **1**；
   可修的 7,680 行里 **7,639 行（99.47%）正好缺那两个行业因子**。
   最差日 2022-07-29：0.895082 → 只算科创板 0.949031 → **科创板+非科创板 0.979136** →
   **严格界 0.976751**，**24/24 天都过 0.95**，即**高于**选项 2 的 0.971370。
   顺带发现一个此前没记录的缺口：**缓存每天还少 460–614 个非科创板证券的标签**，贯穿整个窗口。
6. **选项 1 必须做成并集，不能做成替换。**
   两支 API 把年度成分调整的空洞放在**不同的日子**（缓存 2022-07-29 到期 143 条；
   `index_member_all` 2022-07-28 到期 243 条，失标证券**只重合 5 个**；2023 年 07-05 对 **07-03**，
   2024 年 07-30 对 **07-29**）。969 天定价（科创板收益在 **422** 天实测，聚合口径分母与仪器 **420/420** 全同）：
   并集跌破 **0** 天、G6 **1.000000**、距地板 0.01 以内 **0** 天；
   替换跌破也是 **0** 天、G6 也是 **1.000000**，**但把 2022-07-28 压到 0.903697、只剩 12.4 行余量**，
   与已被否掉的"只移 `yeslab_206`"（0–8 行）同一类。
   并集还需一条**显式优先级规则**（两支在 39 天上有 **114 对**取值不同），**未实现未验证**；
   并集会**掩盖真实到期**（固有代价，未量化）。
   *这一条的第一版把替换算成"19 天跌破、124 天在 0.01 以内"，因为对未实测的天一律假设科创板收益为 0；
   实测进来后变成 0 天与 1 天。第一版保留在 `option1_replace_vs_union_first_pass.json`。*
7. **另外两处失效形状一并消失，所以选项 1 同时是训练年的修复。**
   2014 左端：缓存 2014-01-02 只有 **20/31** 个行业、2014-02-21 一次性入册 **912** 只；
   `index_member_all` 自 2014-01-02 即 **31/31**、当天入册 **1** 只。
   2021-12-10 cutover 空洞：缓存当天 **535** 条到期、在册 4,338 → **3,806**；
   `index_member_all` 当天到期 **1** 条、在册 4,803 → **4,812**，**无空洞**。
   但**不要**由此推断训练年那 1,045/1,948 个跌破日都会被救——§9–§13 已归因出 2014–2017 的薄弱
   主要来自 `yeslab_206` 与 `yeslab_443_reg30_max20_v1`，与行业标签无关。
8. **上面第 4 条之外，§10.3 那九个"批量到期日"里有 5 个也是接口造成的。**
   阈值同为 ±50：缓存 **21** 个大幅变动日、`index_member_all` **17** 个，**只 4 个重合**；
   九个日期里 **5 个消失**（2015-06-30、2016-06-30、2017-06-30、2019-07-05、**2021-12-10**），
   **4 个存续**（2018-07-13、2022-07-29、2024-07-30、2026-07-01）。
   "每年 6 月底/7 月初有一次集中调整"仍成立，但**论文里不要引用那九个日期**，日期取决于用哪支接口。

## 为什么要复制原件

本分支 `codex/research-conditional-factor-review` 比 `codex/publish-quant-platform`
落后约 95 个提交，叙述文档引用的那些报告与交接文档**在本分支上并不存在**。
复制过来是为了让这一包在另一台机器上自包含；引用前请用 `vendored/MANIFEST.json`
里的 sha256 与主分支上的原件比对，确认拿到的是同一份。

## 重新生成

两个脚本都是**只读**的：不写工件树、不碰数据库、不动主工作树
（主树上有一个多日的 M2 运行在跑，`research_v2/scripts/run_real_candidate_training.py`
在它跑完前不能被编辑，因为下一折的 worker 会从磁盘重新导入它）。

```bash
cd docs/research/evidence/2026-09-14-model-side-walk-forward
python3 extract_from_reports.py   # 默认 --artifacts /data/zhangyudong/qrv2_research_artifacts
python3 vendor_sources.py         # 默认 --repo /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX
```

重跑逐年增量分析（每个工作区约 20 分钟、单进程约 8 GiB 峰值、**只读**；
`m0r_v2` 与 `m1_full_v1` 带 `--sensitivity-no-score-health`，已被取代的载体不带）：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2
A=/data/zhangyudong/qrv2_research_artifacts
for W in m0r_v2 m1_full_v1; do
  python scripts/analyze_yearly_baseline_deltas.py --workspace $A/$W --report $A/$W/report.json \
    --key $A/secrets/selection_evidence_hmac.key --sensitivity-no-score-health \
    --out-json $A/yearly_baseline_deltas/$W.json --out-md $A/yearly_baseline_deltas/$W.md
done
```

重渲染 `comparison_interim.md`（在主工作树里跑，M2/M3 报告落地后把 `--report` 补上即可）：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2
A=/data/zhangyudong/qrv2_research_artifacts
python scripts/summarize_candidate_comparison.py \
  --report M0R=$A/m0r_v2/report.json --report M1-full=$A/m1_full_v1/report.json \
  --report M1-carrier=$A/m1_parallel_v1/report.json --report M0-refined=$A/m0_v3/report.json \
  --report M0-base=$A/m0_v2/report.json --report M0R-replicate=$A/m0r_v3/report.json \
  --baseline-report $A/baseline_v2/report.json --out comparison_interim.md
```

`in_progress_search_records.json` 会随 M2 的进度变化；其余文件只在有新运行归档后才变。
M2 出报告后请重跑一次提取器，并把叙述文档升到 v0.2、用最终 OOS 结果整节替换第 3 节。

## 提取时刻

2026-09-14（UTC 时间戳见 `provenance.json` 的 `extracted_at_utc`），
主树 HEAD `c490353`、分支 `codex/publish-quant-platform`、工作树干净。

`star_industry_coverage/` 是同日 23:55 追加的一批（根因排查），
**自带独立的 `provenance.json`**（由 `make_provenance.py` 生成，记当时的主树 HEAD、分支与是否干净、
7 个探针与 11 份测量输出的 sha256、10 份被引用源文件的 sha256、行业缓存的合并 sha256）。
引用该目录时以它自己的 `provenance.json` 为准。
