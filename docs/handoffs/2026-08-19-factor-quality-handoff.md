# 因子质量改进与服务器迁移交接文档

> 状态日期：2026-08-31（Asia/Shanghai）
>
> 当前工作区：`/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX`
>
> 当前分支：`codex/publish-quant-platform`
>
> 代码基线提交：以当前分支 HEAD 为准（迁移时执行 `git rev-parse HEAD`）
>
> 持续目标：先完成严格、可复现的因子质量筛选，再考虑模型训练

## 1. 交接结论

当前工作已经完成了因子库的广覆盖诊断、正式九因子的质量记账、若干候选因子的修复实验，以及精简版质量表的代码实现。模型训练尚未开始，2026 锁定测试没有被本轮因子质量工作读取。

当前已完成 `yeslab_484_tushare_shsz_v1` 的严格、独立 Tushare SH/SZ 复核：它通过所有固定质量门槛，状态为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW`。这只是候选的人工准入评审资格，未改变正式九因子、未启动模型训练，也未读取 2026 锁定测试。

六个隔离候选的统一两两相似度审计也已完成：训练期和 2025 验证期各 15 条边，所有边分别覆盖 1212/243 个交易日。`yeslab_114_tushare_shsz_v2` 与 `yeslab_484_tushare_shsz_v1` 的验证期绝对 Spearman 中位数为 `0.90787`，标记为 `REDUNDANT`；`yeslab_381_tushare_shsz_v1` 与 `yeslab_405_tushare_shsz_v1` 为 `0.86451/0.88821`，标记为 `ELEVATED_PAIR_REDUNDANCY_MONITOR`。该审计仅是组合比较证据，不改变任何候选状态或正式九因子。

新表 `factor_quality_latest` 已按最新结论发布并回读校验：491 行、491 个唯一因子 ID、9 个正式可用因子。发布的是当前质量结论，不会自动把 Alpha484 加入正式因子集；下一关键路径是对候选逐个进行人工准入评审。

迁移到新服务器时，不能只克隆 GitHub 仓库。Tushare 缓存和一部分诊断产物被 Git 忽略，必须单独复制；迁移前使用 `git rev-list --left-right --count origin/codex/publish-quant-platform...HEAD` 核对实际推送差异。

## 2. 不可变研究口径

后续工作必须保持以下约束，除非用户再次明确批准变更：

| 项目 | 当前口径 |
|---|---|
| 行情主源 | Tushare；YesLab 仅作为历史源和交叉核对来源 |
| 股票范围 | 上交所、深交所；暂不考虑北交所 `.BJ` |
| 主标签 | `forward_open_5d:v1`，原始 5 个交易日复权开盘价远期收益 |
| 训练区间 | 2020-01-02 至 2024-12-31 |
| 验证区间 | 2025-01-02 至 2025-12-31 |
| 测试区间 | 2026，锁定；普通质检和训练选择不得读取 |
| 模型评估规划 | 5 年训练 + 1 年验证 + 1 年测试，按年滚动 |
| 方向 | 只从训练期冻结；负方向不是失败，报告同时使用 signed 指标 |
| 预测门槛 | signed mean RankIC `> 0.02`；signed ICIR `>= 0.50`；方向日占比 `>= 0.55` |
| 年度稳定性 | 训练五年中至少 4 年通过年度门槛 |
| 相似度 | 对正式因子的日截面 Spearman 绝对值中位数 `< 0.90` |
| 覆盖率 | 日覆盖中位数 `>= 0.80`；日覆盖 P05 `>= 0.70`；月度最低覆盖 `>= 0.70` |
| 数据库原则 | 不修改已有 YesLab/V2 表结构；允许新建独立表 |
| 准入原则 | 候选通过质量门槛后只进入人工准入复核，不自动加入正式因子集 |
| 模型原则 | 暂不训练；复杂模型要求至少 15 个正式准入因子，当前只有 9 个 |

此外，不得为了让候选通过而事后改变窗口、方向、标签、股票范围或阈值。所有修复候选必须使用独立 `factor_id`，不能覆盖历史因子值。

## 3. 当前可信状态和证据优先级

状态判断应按以下顺序取证：

1. DolphinDB 当前表和迁移历史；
2. 当前 Git 工作树与提交历史；
3. `research_v2/artifacts/diagnostics/` 中的审计 JSON/CSV；
4. 已批准的 spec/plan；
5. README 和聊天摘要。

README 中仍保留过“正式六因子”的旧检查点，不能作为当前正式集合的权威来源。当前正式集合以 `tushare_shsz_factor_quality_scoreboard_v4.json` 和 `tushare_shsz_factor_audit_coverage_v1.json` 为准。

关键证据文件：

- `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json`
- `research_v2/artifacts/diagnostics/tushare_shsz_factor_quality_scoreboard_v4.json`
- `research_v2/artifacts/diagnostics/formal_monitored_factor_regime_audit_v1.json`
- `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/admission_review.json`
- `research_v2/artifacts/diagnostics/tushare_alpha95_relative_amount_volatility_v1/quality_summary.json`
- `research_v2/artifacts/diagnostics/tushare_shsz_alpha484_admission_audit_2020_2025_v1.json`
- `research_v2/artifacts/diagnostics/tushare_shsz_alpha484_rematerialized_quality_2020_2024_v1.json`
- `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1.json`
- `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1/train_pairs.csv`
- `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1/validation_pairs.csv`
- `research_v2/artifacts/checkpoints/alpha484_canonical_v1/integrity.json`
- `research_v2/artifacts/diagnostics/tushare_alpha443_canonical_candidate_review_v1.json`
- `docs/superpowers/specs/2026-08-14-factor-quality-latest-alpha484-canonical-design.md`
- `docs/superpowers/plans/2026-08-14-factor-quality-latest.md`
- `docs/superpowers/plans/2026-08-14-alpha484-canonical-admission.md`

## 4. 已完成工作

### 4.1 因子库诊断覆盖

因子目录来自 `research_v2/configs/factors/source_catalog.yaml`，共 487 个因子。

当前质量台账记录：

| 指标 | 数量 | 含义 |
|---|---:|---|
| 目录因子 | 487 | 当前 YesLab 源目录中的因子身份数 |
| 至少有诊断证据 | 487 | 每个目录因子至少有一份初筛或专项诊断证据 |
| 无任何诊断产物 | 0 | 已完成广覆盖扫描 |
| 主重物化子集的完整预测质量 | 16 | 仅表示该子集完成主要 RankIC/ICIR 检验，不等同于准入 |
| 深度代表性候选 | 28 | 经过更多年度、公式或相似度复核的候选 |
| 正式准入 | 9 | 当前唯一可供正式准入流程读取的集合 |

必须区分三个概念：

- “已诊断”不等于完成所有严格门槛；
- “完整预测质量”不等于公式、PIT、覆盖率和相似度全部通过；
- 只有正式九因子被标记为当前可训练候选输入，且复杂模型仍因因子数不足而禁止训练。

### 4.2 正式九因子

当前正式九因子如下。表中 RankIC/ICIR 是原始方向值；除 `yeslab_442` 外，训练冻结方向为负，进入模型前使用冻结方向进行符号统一。

| 因子 | 训练 RankIC | 训练 ICIR | 2025 RankIC | 2025 ICIR | 状态 |
|---|---:|---:|---:|---:|---|
| `yeslab_155` | -0.04280 | -0.64639 | -0.04891 | -0.66108 | 通过 |
| `yeslab_376` | -0.08219 | -0.70571 | -0.09448 | -0.78317 | 通过 |
| `yeslab_380` | -0.08170 | -0.59312 | -0.08427 | -0.54570 | 通过，需监控 |
| `yeslab_396` | -0.07483 | -0.59475 | -0.08244 | -0.61365 | 通过 |
| `yeslab_431` | -0.06952 | -0.55564 | -0.08531 | -0.74605 | 通过，需监控 |
| `yeslab_437` | -0.06685 | -0.50908 | -0.08561 | -0.74988 | 通过，需监控 |
| `yeslab_442` | +0.07282 | +0.59758 | +0.07923 | +0.62236 | 通过 |
| `yeslab_94` | -0.03861 | -0.50853 | -0.05501 | -0.69812 | 通过，需监控 |
| `yeslab_98` | -0.04360 | -0.56559 | -0.06136 | -0.75979 | 通过 |

监控要求：

- `yeslab_380`：非正趋势和高波动环境；
- `yeslab_431`：训练期非正趋势、低波动环境；
- `yeslab_437`：训练期非正趋势、低波动环境；
- `yeslab_94`：高波动和非正趋势环境。

这些监控触发条件不是自动剔除条件；应按月复核，若后续验证窗在相同市场状态下持续低于 RankIC/ICIR 门槛，再启动人工复审。

### 4.3 已完成的修复和候选审计

#### Alpha95 相对成交额波动修复

独立候选：

- `yeslab_163_relative_amount_volatility_10_v1`
- `yeslab_163_relative_amount_volatility_20_v1`
- `yeslab_163_relative_amount_volatility_40_v1`

结论：三个候选均拒绝。10 日版本最接近，但训练 signed ICIR 为 `0.47750`，年度通过 3/5；20 日和 40 日版本年度通过均为 2/5。候选没有写数据库、没有改变正式集合，也没有启动训练。

#### Alpha484 历史研究证据

历史 `yeslab_484` 在既有 Tushare 重物化证据中表现较强：

- 训练 signed RankIC `0.07008`，signed ICIR `0.58073`，年度通过 5/5；
- 2025 signed RankIC `0.06863`，signed ICIR `0.50200`；
- 与正式九因子的最大日截面 Spearman 绝对值中位数：训练 `0.69439`、验证 `0.72541`。

但是历史值存在 BJ 横截面、复权因子版本和源覆盖差异，不能直接作为新 SH/SZ 规范版本的准入证据。因此建立了独立候选：

`yeslab_484_tushare_shsz_v1`

该候选不会覆盖 `yeslab_484`。2026-08-27 已完成独立 Tushare SH/SZ 全量复核，结论如下：

| 检查项 | 训练 2020-2024 | 验证 2025 | 门槛 / 结论 |
|---|---:|---:|---|
| 冻结方向 | `-1` | 复用训练方向 | 未在验证集重新选择 |
| signed RankIC | `0.06992` | `0.06916` | 均 `> 0.02` |
| signed ICIR | `0.58075` | `0.50850` | 均 `>= 0.50` |
| 方向日占比 | `72.36%` | `67.08%` | 均 `>= 55%` |
| 年度门槛 | `5/5` | - | 至少 `4/5`，通过 |
| 覆盖率 | 日中位数 / P05 / 月度最低均通过 | 同口径通过 | 通过 |
| 最大相似度中位数 | `0.69098` | `0.72308` | 均 `< 0.90`，最近 `yeslab_376` |

最终审计文件为 `research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/admission_review.json`。它的状态是 `READY_FOR_EXPLICIT_ADMISSION_REVIEW`，但 `model_eligible=false`、`database_writes=false`、`locked_test_read=false`。

#### Alpha405 隔离规范候选

2026-08-27 补齐了 `yeslab_405_tushare_shsz_v1` 的独立 Tushare SH/SZ 复评。训练和验证均使用同一 SH/SZ Tushare 面板，结果如下：

| 检查项 | 训练 2020-2024 | 验证 2025 | 结论 |
|---|---:|---:|---|
| 冻结方向 | `-1` | 复用训练方向 | 未在验证集重选 |
| signed RankIC | `0.06709` | `0.09017` | 均 `> 0.02` |
| signed ICIR | `0.62364` | `1.03910` | 均 `>= 0.50` |
| 方向日占比 | `73.10%` | `86.42%` | 均 `>= 55%` |
| 年度严格门槛 | `4/5` | - | 通过最低 `4/5`，2020 年仅 ICIR 不足 |
| 覆盖率 | 中位数 `0.99912`，P05 `0.99742`，月度最低 `0.99563` | 中位数 `0.99898`，P05 `0.99798`，月度最低 `0.99714` | 通过 |
| 正式九因子最大相似度中位数 | `0.68707`（`yeslab_396`） | `0.71773`（`yeslab_396`） | 均 `< 0.90` |

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha405_canonical_candidate_review_v1.json`。它的状态是 `READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_APPROVAL`：历史 YesLab GP_036 含 `552` 个 BJ 代码，不能与新 SH/SZ 候选值混用；因此该候选仍 `model_eligible=false`，没有写入因子值，也没有改变正式九因子。只有明确批准创建隔离版本后，才可进入实际准入流程。

#### Alpha381 隔离规范候选

2026-08-27 以与 Alpha405 相同的 Tushare SH/SZ 面板完成了 `yeslab_381_tushare_shsz_v1` 的独立复评：训练期 signed RankIC / ICIR 为 `0.07540 / 0.61529`，2025 验证期为 `0.09506 / 0.89045`，年度严格门槛 `4/5`，训练和验证覆盖率均通过。它对正式九因子的最大相似度中位数分别为 `0.78453` 与 `0.79331`，均低于 `0.90`，最近正式因子均为 `yeslab_431`。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha381_canonical_candidate_review_v1.json`，状态为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_AND_PAIR_MONITORING`。历史 YesLab GP_012 含 `551` 个 BJ 代码，不能和新 SH/SZ 候选值混用；此外，381 与 `yeslab_405_tushare_shsz_v1` 的训练 / 验证相似度中位数为 `0.84484 / 0.87119`，并有单日峰值超过 `0.92`。因此 381 只能作为单独的隔离候选人工审查；若未来同时考虑 381 与 405，必须先通过额外的组合冗余审批。它仍为 `model_eligible=false`，未写因子值、未改正式九因子、未启动训练。

#### Alpha114 统一口径候选

2026-08-27 用现行统一评估器完成 `yeslab_114_tushare_shsz_v2` 的连续窗口复评。训练期 signed RankIC / ICIR 为 `0.07568 / 0.54959`，2025 验证期为 `0.07624 / 0.52272`，年度严格门槛 `4/5`；正式九因子最大相似度中位数为训练 `0.80593`、验证 `0.80642`，均低于 `0.90`。

该因子必须保留月度和市场状态监控：训练月度 signed mean / 方向 / 严格月度 ICIR 通过数为 `47/60`、`46/60`、`32/60`，验证为 `9/12`、`8/12`、`7/12`；2025 关注月份为 `01/02/04/12`，且非正趋势、高波动状态的验证 ICIR 分别为 `0.25135`、`0.14745`。评审包为 `research_v2/artifacts/diagnostics/tushare_alpha114_canonical_candidate_review_v2.json`，状态为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_MONTHLY_AND_REGIME_MONITORING`，仍未发布、未写因子值、未改变正式九因子、未启动训练。

本次连续五年结果与早期按年度拆分的旧摘要有轻微差异：2020 单年按同样方式重跑可逐项复现旧值，差异来自训练期代码面板并集与逐年代码并集不同。两种结果不能混合，后续以连续窗口 v2 评审包为准。

#### Alpha425 隔离规范候选

2026-08-28 用与 405/381 相同的 Tushare SH/SZ 面板完成 `yeslab_425_tushare_shsz_v2` 的连续窗口复评。训练期 signed RankIC / ICIR 为 `0.07095 / 0.55768`，2025 验证期为 `0.09559 / 0.84938`，年度严格门槛 `4/5`；覆盖率通过，正式九因子最大相似度中位数为训练 `0.69210`、验证 `0.72325`，均低于 `0.90`，最近因子为 `yeslab_376`。

425 与 `yeslab_405_tushare_shsz_v1` 的训练 / 验证相似度中位数为 `0.59737 / 0.61095`，候选对不冗余；但两者仍不能自动组合准入。历史 YesLab GP_056 含 `552` 个 BJ 代码，必须先明确批准新的隔离 SH/SZ 版本；若日后准入，425 还需保留训练高波动状态监控（signed ICIR `0.49010`）。评审包为 `research_v2/artifacts/diagnostics/tushare_alpha425_canonical_candidate_review_v2.json`，状态为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_AND_HIGH_VOL_MONITORING`，并继续保持 `model_eligible=false`、不写因子值、不改正式九因子、不启动训练。

#### Alpha206 中位数修复候选

2026-08-28 重跑确认 `yeslab_206_median_abs_v2` 的 2025 评估结果可逐日复现：signed RankIC / ICIR 为 `0.05046 / 0.73323`，共 `243` 个交易日。此前 `yeslab_206` 的 `0.39748` 验证 ICIR 属于历史的 60 日**均值**偏离公式；修复候选使用 60 日**中位数**偏离，两者不能交叉引用。新候选训练期 signed RankIC / ICIR 为 `0.04085 / 0.56053`、年度严格门槛 `4/5`，相对正式九因子最大相似度中位数为训练 `0.27229`、验证 `0.27627`。

评审包为 `research_v2/artifacts/diagnostics/alpha206_median_abs_candidate_review_v2.json`。它只可作为隔离的 Tushare 规范修复候选参与人工准入审查，仍要求监控训练低波动（signed ICIR `0.47133`）和 2025 非正趋势（`0.41302`）状态；不得自动发布、写入因子值、变更正式九因子、加入模型输入或启动训练。

#### Alpha443 隔离规范候选

2026-08-31 使用当前 `evaluate_tushare_scope_validation.py` 对 `yeslab_443_tushare_shsz_v1` 做了独立的 Tushare SH/SZ 全训练期与验证期重放，并重新计算与正式九因子的全日相似度。训练期 signed RankIC / ICIR 为 `0.06668 / 0.58257`，2025 验证期为 `0.07875 / 0.71210`，冻结方向为 `-1`，严格年度 ICIR 门槛通过 `4/5`。训练与验证覆盖率中位数分别为 `0.99142` 和 `0.99184`，P05 分别为 `0.98177` 和 `0.98957`，均通过覆盖门槛。

与正式九因子的最大绝对 Spearman 中位数为训练 `0.74651`（最近 `yeslab_431`）、验证 `0.74141`（最近 `yeslab_431`），两期 9/9 边均完整覆盖 `1212/243` 个交易日，低于 `0.90` 冗余阈值。历史 YesLab `GP_074` 源池包含 `552` 个 BJ 代码，历史值不能与新的 SH/SZ Tushare 值混用；因此该候选状态为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW_WITH_SCOPE_APPROVAL`，仍未发布、未写数据库、未改变正式九因子、未加入模型输入。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha443_canonical_candidate_review_v1.json`；训练/验证原始 RankIC 与相似度文件位于同名 `tushare_shsz_alpha443_formal9_*` 目录。该候选只进入人工范围审批，不因质量通过而自动准入。

#### Alpha432 冗余复核结论

2026-08-31 使用同一评估器完成 `yeslab_432_tushare_shsz_v1` 的 Tushare SH/SZ 全训练期与验证期重放。训练期 signed RankIC / ICIR 为 `0.07150 / 0.59249`，2025 验证期为 `0.08582 / 0.79153`，冻结方向为 `-1`，严格年度 ICIR 门槛通过 `4/5`；训练和验证覆盖率中位数分别为 `0.99912` 和 `0.99898`，均通过覆盖门槛。

但它与正式 `yeslab_431` 的全日绝对 Spearman 中位数为训练 `0.95381`、验证 `0.96155`，均超过 `0.90` 冗余阈值；9/9 条正式九因子边均完整覆盖 `1212/243` 个交易日。因此 Alpha432 的预测质量虽通过，结构准入必须拒绝为正式因子的重复输入，不能以“质量通过”绕过冗余门槛。历史 YesLab `GP_063` 源池包含 `552` 个 BJ 代码，历史值也不能和隔离的 SH/SZ Tushare 值混用。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha432_canonical_candidate_review_v1.json`；训练/验证原始 RankIC 与相似度文件位于同名 `tushare_shsz_alpha432_formal9_*` 目录。当前结论为 `REJECT_REDUNDANT_WITH_FORMAL_YESLAB_431`，仍保持 `model_eligible=false`，不发布、不写数据库、不改变正式九因子、不加入模型输入、不启动训练。

#### Alpha377 早停冗余拒绝

2026-08-31 基于已完成的 16 因子 Tushare SH/SZ 面板复核了 `yeslab_377_tushare_shsz_v1`。训练期 signed RankIC / ICIR 为 `0.08153 / 0.54575`，2025 验证期为 `0.08347 / 0.51002`，覆盖率两期均为 `1.00000`；严格年度门槛为 `3/5`。

在正式因子相似度方面，Alpha377 与正式 `yeslab_380` 的绝对 Spearman 中位数已达到训练 `0.97005`、验证 `0.97555`，且两条边分别完整覆盖 `1212/243` 个交易日。根据早停规则，单条完整边超过 `0.90` 已足以拒绝新增输入，因此没有将其余 8 条正式边误报为“已完成审计”。历史 YesLab `GP_008` 源池含 `551` 个 BJ 代码，未来如需研究也必须隔离 SH/SZ 版本。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha377_canonical_redundancy_rejection_review_v1.json`，状态为 `REJECT_REDUNDANT_WITH_FORMAL_YESLAB_380`。该因子不发布、不写数据库、不改变正式九因子、不加入模型输入、不启动训练。

#### Alpha387/388 早停冗余拒绝

2026-08-31 复用已完成的 16 因子 Tushare SH/SZ 面板，对 `yeslab_387_tushare_shsz_v1`（GP_018）和 `yeslab_388_tushare_shsz_v1`（GP_019）完成质量、覆盖和首条决定性相似度边复核。两者训练/验证 signed RankIC / ICIR 分别为 `0.08121/0.57704`、`0.08398/0.53292`（Alpha387）和 `0.08198/0.57039`、`0.08303/0.50032`（Alpha388），年度严格门槛均为 `4/5`；覆盖率中位数均超过 `0.997`。

两者与正式 `yeslab_380` 的绝对 Spearman 中位数均在训练/验证期超过 `0.97`，每条决定性边均完整覆盖 `1212/243` 个交易日。因此按早停规则拒绝新增输入，且明确没有声称其余 8 条正式边已经审计。历史 GP_018/GP_019 源池含 BJ 证券，不能与未来隔离的 SH/SZ Tushare 版本混用。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha387_alpha388_redundancy_rejection_review_v1.json`。两个因子均不发布、不写数据库、不改变正式九因子、不加入模型输入、不启动训练。

#### Alpha391 完整九边边界冗余复核

2026-08-31 对 `yeslab_391_tushare_shsz_v1`（GP_022）完成了训练期和 2025 验证期与正式九因子的完整 9 条相似度边复核。训练/验证 signed RankIC / ICIR 为 `0.07970/0.57624` 与 `0.08280/0.54981`，严格年度门槛为 `4/5`，公式重放四窗口通过。

与正式 `yeslab_380` 的完整边中位绝对 Spearman 在训练期为 `0.90411`，超过不变的 `0.90` 冗余门槛；验证期为 `0.88781`，低于门槛且作为边界监控记录。9 条边分别覆盖训练/验证 `1212/243` 个交易日。按既有单阶段冗余先例，训练期完整边越线足以拒绝新增输入；本次没有因验证期回落到门槛以下而放宽拒绝，也没有把验证期结果解释为通过。

评审包为 `research_v2/artifacts/diagnostics/tushare_alpha391_canonical_candidate_review_v1.json`，状态为 `REJECT_REDUNDANT_WITH_FORMAL_YESLAB_380_TRAIN_BOUNDARY_VALIDATION_MONITOR`。该因子不发布、不写数据库、不改变正式九因子、不加入模型输入、不启动训练。

#### 六候选统一两两相似度审计

2026-08-28 已完成 `yeslab_114_tushare_shsz_v2`、`yeslab_381_tushare_shsz_v1`、`yeslab_405_tushare_shsz_v1`、`yeslab_425_tushare_shsz_v2`、`yeslab_484_tushare_shsz_v1`、`yeslab_206_median_abs_v2` 的统一比较。公式计算使用完整 SH/SZ Tushare 市场面板，输出再限制到共同的 V2 `is_member` 且 `label_valid` 业务键；标签查询显式锁定 `forward_open_5d`/版本 `1`。Alpha484 使用带代码、manifest、内容哈希和日期集合指纹的已完成 Tushare 重放检查点，完整指纹记录在 `research_v2/artifacts/checkpoints/alpha484_canonical_v1/integrity.json`；加载器同时绑定代码内的规范阶段覆盖/日期摘要和文件哈希，替换 `integrity.json` 也不能让稀疏检查点通过；Alpha206 使用 60 日滚动中位数修复。

结果摘要如下：

| 比较对 | 训练绝对 Spearman 中位数 | 验证绝对 Spearman 中位数 | 状态 |
|---|---:|---:|---|
| `114` - `484` | `0.85086` | `0.90787` | `REDUNDANT` |
| `381` - `405` | `0.86451` | `0.88821` | `ELEVATED_PAIR_REDUNDANCY_MONITOR` |
| 其余 13 对 | `0.21092–0.59732` | `0.20460–0.64956` | `PAIR_UNIQUE` |

每条边训练/验证有效日数均为 `1212/243`，阈值为监控 `0.80`、冗余 `0.90`。完整逐边结果在 `research_v2/artifacts/diagnostics/tushare_canonical_candidate_pair_similarity_v1.json` 及其两个 CSV 中。该结果是 comparison-only 证据：六个候选仍全部 `model_eligible=false`，不授权组合准入、不写数据库、不读取 2026 锁定测试。

六个当前候选的唯一统一索引为 `research_v2/artifacts/diagnostics/tushare_shsz_candidate_admission_review_matrix_v2.json`。该矩阵不再提供跨公式、跨范围的优先级排名；它只列出每个隔离版本的当前证据、必需监控和组合限制。所有候选仍需逐个明确审批，任何两个候选同时进入正式集合前都必须单独评估组合冗余。

六候选矩阵不覆盖后来单独完成的单因子复核。当前单因子状态由 `research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json` 中的 `current_single_candidate_registry` 索引：包括 `yeslab_443`、`yeslab_432`、`yeslab_377`、`yeslab_387`、`yeslab_388`、`yeslab_391`、`yeslab_433`、`yeslab_395`、`yeslab_400`、`yeslab_363`、`yeslab_357`、`yeslab_96` 和 `yeslab_208`。该索引仅汇总证据与状态，不构成自动准入；任何候选仍需单独人工审批，组合准入仍需单独组合冗余评估。

人工准入前的“隔离版本契约”由 `research_v2/artifacts/diagnostics/canonical_candidate_version_contract_index_v1.json` 统一索引，覆盖 10 个待审或监控候选：六候选矩阵中的 405/425/381/114/484/206，以及 443/363/357/96。该契约只说明每个候选的规范版本来源、市场/标签窗口、监控要求和证据文件，不批准准入、不写数据库、不改变正式九因子。

组合准入证据缺口已由 `research_v2/artifacts/diagnostics/candidate_combination_evidence_gap_index_v1.json` 记录并完成补充：六候选矩阵原有 15 条训练/验证边；2026-08-31 又补充了 363/357/96/443 的完整扩展组合边，包括对六候选的 24 条边和扩展候选彼此之间的 6 条边（`supplemental_candidate_pair_similarity_train_v1.json` / `supplemental_candidate_pair_similarity_validation_v1.json`）。训练/验证最大中位绝对 Spearman 分别为 `0.84518/0.87878`，均来自 `357-363`：高于 `0.80` 监控阈值但低于 `0.90` 冗余阈值。357/363 不得在未单独组合审查前同时准入；其余扩展组合边低于监控阈值。组合准入仍需单独人工审批，且保持 `0.90` 冗余和 `0.80` 监控阈值不变。

#### 其他值得保留的候选状态

| 候选 | 当前结论 | 后续动作 |
|---|---|---|
| `yeslab_114_tushare_shsz_v2` | 预测和相似度通过，但月度/市场状态敏感 | 需明确批准隔离版本，并保留月度与 regime 监控 |
| `yeslab_206_median_abs_v2` | 中位数修复公式已逐日复现，需低波动/非正趋势监控 | 仅隔离版本人工审查，不自动加入正式集合 |
| `yeslab_405_tushare_shsz_v1` | 预测、覆盖和正式九因子相似度通过，等待范围审批 | 必须明确批准隔离 SH/SZ 版本；不能使用历史 YesLab 值 |
| `yeslab_425_tushare_shsz_v2` | 预测、覆盖、正式九因子和 405 候选对相似度通过 | 需明确批准隔离 SH/SZ 版本，并保留训练高波动监控 |
| `yeslab_381_tushare_shsz_v1` | 预测、覆盖和正式九因子相似度通过；与 405 组合冗余偏高 | 先明确批准隔离 SH/SZ 版本；若与 405 同时考虑，必须另行组合审批 |
| `yeslab_443_tushare_shsz_v1` | 预测、覆盖和正式九因子全日相似度通过；历史源含 BJ | 必须明确批准隔离 SH/SZ 版本，不能混用历史 YesLab 值 |
| `yeslab_432_tushare_shsz_v1` | 预测和覆盖通过，但与正式 `yeslab_431` 冗余（训练/验证 `0.95381/0.96155`） | 拒绝作为新增输入；保持正式九因子不变，不发布、不训练 |
| `yeslab_377_tushare_shsz_v1` | 预测质量和覆盖通过，但与正式 `yeslab_380` 冗余（训练/验证 `0.97005/0.97555`） | 早停拒绝新增输入；其余正式边未声称完成审计 |
| `yeslab_387_tushare_shsz_v1` | 预测质量和覆盖通过，但与正式 `yeslab_380` 冗余（训练/验证 `0.97775/0.97503`） | 早停拒绝新增输入；其余正式边未声称完成审计 |
| `yeslab_388_tushare_shsz_v1` | 预测质量和覆盖通过，但与正式 `yeslab_380` 冗余（训练/验证 `0.97999/0.97741`） | 早停拒绝新增输入；其余正式边未声称完成审计 |
| `yeslab_391_tushare_shsz_v1` | 完整九边复核通过；训练期对正式 `yeslab_380` 为 `0.90411`，验证期为 `0.88781` | 训练期冗余拒绝新增输入；验证期仅作为边界监控 |
| `yeslab_433_tushare_shsz_v1` | 2025 验证通过，但严格年度门槛仅 `3/5`；已知对正式 `yeslab_380` 未越线 | 按年度稳定性拒绝新增输入；相似度不是本次拒绝依据 |
| `yeslab_395_tushare_shsz_v1` | 严格年度门槛 `4/5`，但历史公式重放被源横截面范围阻断 | 按公式重放阻断拒绝；需源公式对账和范围决策 |
| `yeslab_400_tushare_shsz_v1` | 2025 验证通过，但严格年度门槛仅 `3/5`；已知对正式 `yeslab_380` 未越线 | 按年度稳定性拒绝新增输入；相似度不是本次拒绝依据 |
| `yeslab_163`（Alpha95 原始） | 完整复核年度门槛 `3/5`；正式九因子相似度未越线；三个相对金额波动变体全部拒绝 | 保留 `DEFER_ANNUAL_STABILITY_3_OF_5`，不进入模型输入；不因变体拒绝改变原始因子的保留状态 |
| `yeslab_363_tushare_shsz_v1` | 完整九边相似度通过（训练/验证最大 `0.32094/0.34519`）；年度门槛 `4/5`；使用版本化 PIT 行业 | 保留待人工准入；需保留行业 PIT 与年度监控 |
| `yeslab_96`（修正日间反转） | 完整回放年度门槛 `4/5`，相似度最大 `0.55556/0.61206`；与 `yeslab_95` 偏高但未达冗余 | 待人工准入；保留 2020 年度监控与配对监控；旧 2020 单年拒绝仅作为历史证据 |
| `yeslab_385` | 与 `yeslab_405` 相似度 0.90134，冗余失败 | 拒绝 |
| `yeslab_208` | 2025 验证通过、相似度未越线，但严格年度 ICIR 仅 `1/5`；成交量修复变体全部不改善年度门槛 | 保留监控研究候选，不准入；不因验证通过放宽年度稳定性 |
| `yeslab_165` | 历史年度稳定但公式重放被源市场覆盖阻断 | 保留监控研究候选，不准入；需源市场覆盖修复 |
| `yeslab_167` | 历史年度稳定但公式重放被市场源版本阻断 | 保留监控研究候选，不准入；需市场源版本修复 |
| `yeslab_168` | 历史年度稳定但公式重放被源市场覆盖阻断 | 保留监控研究候选，不准入；需源市场覆盖修复 |
| `yeslab_401` | 2025 验证通过，但 2020 严格预测门槛失败；公式精度仍有局部不匹配 | 拒绝新增准入；公式精度问题仅作对账记录 |
| `yeslab_471` | 版本化 PIT 行业映射已接入，但 2025-01 重放仍有 `90,469` 行不匹配 | 历史源公式阻断；继续隔离，不进入训练/发布 |
| `yeslab_357` / `yeslab_362` | 版本化 PIT 行业映射已接入，但 2025-01 重放分别有 `88,196/88,047` 行不匹配；362 另有年度拒绝 | 历史源公式阻断；继续隔离，不进入训练/发布 |
| `yeslab_357_tushare_shsz_v1` | 完整九边相似度通过（训练/验证最大 `0.40066/0.42540`）；年度门槛 `5/5`；使用版本化 PIT 行业；历史源重放仍阻断 | 隔离候选待人工准入；不得混用历史值，并保留源公式对账监控 |

### 4.4 精简质量表代码

用户要求只保留当前质量结论，不需要在新表中保留大量工程版本字段。已经完成：

- 新迁移：`research_v2/migrations/012_factor_quality_latest.dos`
- 行构建与校验：`research_v2/src/research_v2/quality/latest.py`
- 刷新与读取 CLI：`research_v2/scripts/refresh_factor_quality_latest.py`
- 单元测试：
  - `research_v2/tests/unit/db/test_migrations.py`
  - `research_v2/tests/unit/quality/test_latest.py`
  - `research_v2/tests/unit/scripts/test_refresh_factor_quality_latest.py`

新表设计：

- 一因子一行，只保存当前结论；
- 仅用 `effective_date` 表示当前结论日期；
- 不包含 `run_id`、`data_version`、`created_at`、`source_watermark_id`、`record_hash`；
- 预期 491 行：487 个目录因子 + 3 个 Alpha95 拒绝候选 + 1 个 Alpha484 独立候选；
- 正式可用因子必须恰好为 9 个；
- 默认 dry-run，只有显式 `--publish` 才写数据库；
- 读取器只返回 `model_eligible=true`、`market_scope=SHSZ`、`label_id=forward_open_5d:v1` 的行，不自动创建正式因子集。

相关提交：

- `af79815`：新增精简表迁移；
- `460eedb`：质量行构建与校验；
- `944f7dc`：dry-run 优先的刷新和读取 CLI；
- `55cf367`：记录实现进度。

### 4.5 测试状态

2026-08-19 在 `quantyd` 环境实际执行：

```text
pytest: 703 passed, 42 warnings, 15.66s
```

全仓库 Ruff 当前不是绿色，`ruff check research_v2/src research_v2/scripts research_v2/tests` 报 186 个问题，主要分布在历史审计脚本和测试中的行长、导入排序及少量 lint 规则。这是当前基线债务，不能在交接时写成“全仓库 Ruff 通过”。

质量表和 Alpha484 相关文件应单独执行 focused Ruff；迁移后恢复工作前也应重新跑完整 pytest。

## 5. 未完成工作

### 5.1 Alpha484 独立规范复核历史现场

失败命令：

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
PYTHONPATH=research_v2/scripts:research_v2/src \
python -u research_v2/scripts/evaluate_alpha484_canonical.py \
  --output research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1
```

已观察到的事实：

- 运行约 3 小时 48 分后进程退出；
- 前约 2 小时 40 分主要执行 CSAD 滚动相关矩阵；
- 训练计算使用约一个 CPU 核，内存约 3.8 GiB；
- 随后按月读取五年、九个正式因子明细以计算相似度；
- DolphinDB 连接累计向 Python 传输约 891 MB；
- 进程退出前内存约 2.7 GiB，系统仍有约 506 GiB 可用内存；
- 系统日志没有 OOM/Killed 证据；
- 原进程 stdout/stderr 没有落日志，无法还原准确异常栈；
- 此次历史失败不能解释为通过或拒绝，也不能把历史 Alpha484 结果冒充新候选结果。

已确认的性能根因：

1. `HerdingCSADCore.py` 对约 5,000 只股票、120 日窗口、约 1,333 个训练日期逐日重算完整相关矩阵，复杂度接近 `O(T*N^2)`，当前实现基本串行；
2. `_load_formal_values` 按 60 个自然月下载九因子所有明细，并在 Python 中一次性拼接，传输和内存生命周期过长；
3. 输出目录和结果文件只在训练、验证、相似度全部完成后创建，任何后段失败都会丢失前段数小时计算；
4. 运行未使用独立日志文件，异常退出后无法恢复 traceback。

### 5.2 可恢复审计已实现并完成重跑

2026-08-27 已完成可恢复执行改造，代码位于：

- `research_v2/scripts/evaluate_alpha484_canonical.py`：支持 `--phase train|validation|all`、`--checkpoint-dir`、`--resume` 和 `--log-file`；
- `research_v2/scripts/evaluate_tushare_scope_validation.py`：相似度函数支持保存每日 Spearman 序列，恢复汇总与一次性计算保持同一统计口径；
- `research_v2/tests/unit/scripts/test_evaluate_alpha484_canonical.py`：覆盖原子检查点、指纹校验、阶段运行、失败日志和恢复路径；
- `research_v2/tests/unit/scripts/test_evaluate_tushare_scope_validation.py`：覆盖每日 Spearman 保留行为。

当前执行语义：

- 训练阶段先保存候选值 `candidate_values.pkl`，再按月写入 `similarity/YYYY-MM.json`；
- 检查点采用临时文件 + `os.replace`，不会留下半写文件；
- `manifest.json` 状态为 `running` 或 `complete`，包含候选身份、公式来源、时间窗、标签、市场范围、代码提交和缓存文件指纹；
- 缓存只允许新增文件，已有文件被删除或修改会拒绝恢复；
- 验证阶段必须读取训练阶段的冻结方向检查点，不能单独猜方向；
- 异常写入 JSONL 日志并返回退出码 1；
- 相似度检查点保存每日序列，跨月恢复时重新计算全周期中位数和最大值，不用月度中位数近似。

实际执行先完成训练阶段、再使用训练阶段冻结方向完成 2025 验证阶段，最后由 `--phase all --resume` 复用候选值和月度相似度检查点生成最终审计。运行中发现跨月相似度汇总遗漏 `valid_day_count` 字段，已用回归测试覆盖并修复；最终 `admission_review.json` 已生成且所有固定门槛通过。

该过程严格使用 Tushare SH/SZ、`forward_open_5d:v1`、训练 2020-2024、验证 2025，未读取 2026。正式九因子相似度因旧 V2 latest 表缺少其中 3 个值序列，改由相同完整 Tushare SH/SZ 市场面板和原始计算器公式回放；这保证九个比较基准的横截面口径一致。

### 5.3 数据库迁移和质量表发布已完成

2026-08-27 已完成迁移、发布和独立回读：

| 项目 | 当前状态 |
|---|---|
| DolphinDB 当前地址 | `10.176.55.211:8848` |
| DolphinDB 目标库 | `dfs://QuantResearchV2Test` 可连接 |
| `schema_migration` | 已应用 001–012，012 checksum 与本地一致 |
| 迁移 012 | 已应用，仅创建新 `factor_quality_latest` 表 |
| 迁移 013 | 未应用；它属于无关的 OOS 列变更，明确隔离于本次质量表发布 |
| `factor_quality_latest` | 已发布 491 行，491 个唯一 ID，9 个 eligible |
| `factor_quality_snapshot` | 存在，2,612 行，仅作旧审计归档 |
| compact latest 表 | `factor_observation_clean_latest`、`label_daily_latest`、`universe_daily_latest`、`latest_manifest` 均存在 |

先执行了本地 491 行 / 9 eligible dry-run，再将只包含 001–012 的迁移集合交给 `MigrationRunner`。001–011 被同 checksum 跳过，只有 012 执行。首次发布已写入正确数据，但 DolphinDB 拒绝 `count(distinct factor_id)` 的嵌套聚合语法；发布器已改为回读 `factor_id/model_eligible` 后在 Python 侧验证，回归测试覆盖该方言行为。最终发布状态为 `published`。

发布证据：`research_v2/artifacts/diagnostics/factor_quality_latest_refresh_v1.json`。其中记录决策分布 `5 APPROVED + 4 APPROVED_MONITOR + 1 PENDING_EXPLICIT_ADMISSION + 3 REJECT + 478 UNSCREENED`，并确认 Alpha95 三个候选与 Alpha484 独立候选均 `model_eligible=false`。

### 5.4 模型训练未开始

Residual MLP、Set Attention、Temporal GRU/TCN 以及 LightGBM/XGBoost 等模型均未进入正式训练。当前仅有模型组件、数据边界和训练计划，不能把构造器或测试产物理解为训练结果。

现行 D3.5 规则要求复杂模型至少 15 个正式准入因子；当前正式集合只有 9 个。B0/B1 等简单诊断可运行，但本阶段仍应优先完成因子准入和质量表，而不是为了达到数量门槛放宽质量标准。

## 6. Git 与文件迁移状态

### 6.1 GitHub

远端：`https://github.com/Freezingc/quantplatform.git`

远端与本地提交数会随推送变化，迁移前执行以下命令核对：

```bash
git fetch origin
git rev-list --left-right --count origin/codex/publish-quant-platform...HEAD
git status --short
```

本轮新增的质量表发布审计被 artifacts 忽略，必须以 `git add -f` 显式纳入提交。

### 6.2 Git 不包含的数据

| 路径 | 大小/数量 | Git 状态 |
|---|---:|---|
| `research_v2/artifacts/` | 约 3.3 GiB | 整体默认忽略，仅选择性强制跟踪证据 |
| `research_v2/artifacts/tushare_cache/` | 约 2.9 GiB，6,150 文件 | 0 个文件被 Git 跟踪 |
| `research_v2/artifacts/diagnostics/` | 约 308 MiB，2,258 文件 | 约 1,004 个被 Git 跟踪，其余仅本机存在 |
| `数据库/factor_calculators/` | 约 3.5 MiB，515 个跟踪文件 | 已进入 Git |
| `.env` | 本机存在 | 被 Git 忽略，必须安全迁移 |

只克隆 GitHub 会丢失 Tushare 缓存和约 1,254 个本地诊断文件。

## 7. 新服务器迁移步骤

以下步骤假设 DolphinDB 仍运行在现有内部服务上，新服务器只迁移代码、环境、缓存和研究证据。如果 DolphinDB 本身也要迁移，必须另行执行数据库备份/恢复和表级校验，不属于本文档已完成范围。

### 7.1 先固化 Git 历史

推荐方式：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX
git status --short
git add docs/handoffs/2026-08-19-factor-quality-handoff.md
git commit -m "docs: hand off factor quality work"
git push origin codex/publish-quant-platform
```

如 GitHub 推送受限，使用 bundle：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX
git bundle create /tmp/quantplatform-20260819.bundle codex/publish-quant-platform
git bundle verify /tmp/quantplatform-20260819.bundle
```

然后把 bundle 复制到新服务器并克隆：

```bash
git clone /path/to/quantplatform-20260819.bundle Quant-Platform-Prod_CODEX
cd Quant-Platform-Prod_CODEX
git switch codex/publish-quant-platform
git remote set-url origin https://github.com/Freezingc/quantplatform.git
```

### 7.2 复制 Git 忽略的数据

从旧服务器执行：

```bash
rsync -a --partial --info=progress2 \
  /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/ \
  <new-server>:/path/to/Quant-Platform-Prod_CODEX/research_v2/artifacts/
```

复制完成后两端核对：

```bash
du -sh research_v2/artifacts research_v2/artifacts/tushare_cache research_v2/artifacts/diagnostics
find research_v2/artifacts/tushare_cache -maxdepth 1 -type f | wc -l
find research_v2/artifacts/diagnostics -type f | wc -l
```

期望大致为：总计 3.3 GiB、缓存 2.9 GiB/6,150 文件、诊断 308 MiB/2,258 文件。文件数和大小应一致；若需要强校验，可使用 `rsync --checksum --dry-run`。

### 7.3 安全迁移环境变量

不要把密码或 Token 写入本文档、Git 提交或聊天。新服务器通过安全渠道创建 `.env`，至少包含：

```dotenv
QRV2_DDB_HOST=10.176.55.211
QRV2_DDB_PORT=8848
QRV2_DDB_USERNAME=<authorized-user>
QRV2_DDB_PASSWORD=<secret>
QRV2_TUSHARE_TOKEN=<secret>
```

也支持 `TUSHARE_TOKEN` 作为兼容变量。完成后：

```bash
chmod 600 .env
```

注意：部分现有连接辅助函数仍带开发环境默认账号配置。这是安全债务；新服务器必须显式设置环境变量，后续应移除代码内默认凭据并轮换已暴露凭据。

### 7.4 重建 Python 环境

当前环境：

```text
Python 3.11.13
dolphindb 3.0.3.0
numpy 2.3.3
pandas 2.3.3
pyarrow 21.0.0
scikit-learn 1.7.2
scipy 1.16.2
statsmodels 0.14.6
torch 2.6.0
tushare 1.4.29
lightgbm 4.6.0
pytest 8.4.2
ruff 0.16.1
```

重建命令：

```bash
conda create -n quantyd python=3.11.13 -y
conda activate quantyd
python -m pip install -e 'research_v2[dev]'
```

若需要完全复刻，迁移前可在旧服务器额外导出：

```bash
conda env export -n quantyd --no-builds > /tmp/quantyd-20260819.yml
```

### 7.5 网络与数据库烟雾测试

```bash
set -a
source .env
set +a
nc -vz "$QRV2_DDB_HOST" "$QRV2_DDB_PORT"
```

Python 连接测试：

```bash
PYTHONPATH=research_v2/scripts:research_v2/src python - <<'PY'
from refresh_factor_quality_latest import _connect_v2

client, session = _connect_v2()
try:
    print(client.run("1 + 1"))
    print(session.target_database)
finally:
    client.close()
PY
```

期望返回 `2` 和 `dfs://QuantResearchV2Test`。

### 7.6 基线验证

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate quantyd
PYTHONPATH=research_v2/scripts:research_v2/src python -m pytest -q
```

期望当前基线为 703 个测试通过。警告数量可能随依赖小版本变化；测试失败必须先诊断，不应直接开始长任务。

不要以全仓库 Ruff 绿色作为当前迁移门槛，因为当前基线有 186 个 lint 问题。应先对将要修改的文件运行 focused Ruff，并把全仓库清理作为独立任务。

## 8. 新服务器上的执行计划

### P0：恢复并验证现场

1. 固化并取得包含本地 228 个提交的分支；
2. 复制全部 artifacts，尤其是 Tushare cache；
3. 安全恢复 `.env`；
4. 建立 `quantyd`；
5. 验证 Git 提交、文件数、DolphinDB 连接和 703 个测试。

### P1：可恢复 Alpha484 审计（已完成）

实现、训练阶段、验证阶段、最终汇总和门槛复核均已完成。结论为 `READY_FOR_EXPLICIT_ADMISSION_REVIEW`，不得因为通过候选审计而自动修改正式九因子。

### P2：Alpha484 严格复核（已完成）

输出目录仍建议使用：

`research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/`

历史重跑命令和检查点保留用于复现：

```bash
source /data/share/miniconda3/etc/profile.d/conda.sh
conda activate quantyd
set -a
source .env
set +a
PYTHONPATH=research_v2/scripts:research_v2/src \
python -u research_v2/scripts/evaluate_alpha484_canonical.py \
  --phase train \
  --checkpoint-dir research_v2/artifacts/checkpoints/alpha484_canonical_v1 \
  --log-file research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/train.log \
  --output research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1
```

训练阶段、验证阶段和最终汇总均已完成。最终审计确认候选通过全部固定门槛，但保持 `model_eligible=false`，等待单独的人工准入决策。

最终必须存在并相互一致：

- `daily_rank_ic.csv`
- `coverage_summary.json`
- `similarity_summary.json`
- `quality_summary.json`
- `admission_review.json`
- 运行日志和检查点 manifest

最终状态只允许：

- `READY_FOR_EXPLICIT_ADMISSION_REVIEW`：所有门槛通过，但仍不自动准入；
- `REJECT_CANONICAL_GATE`：至少一个固定门槛失败；
- 明确的失败/不完整状态：任务异常，不得生成伪最终结论。

### P3：发布 `factor_quality_latest`（已完成）

已完成 012 应用、dry-run、显式发布和回读：

```text
row_count = 491
unique_factor_count = 491
eligible_count = 9
alpha95 = 3 REJECT / model_eligible=false
alpha484 = PENDING_EXPLICIT_ADMISSION / model_eligible=false
migration_013_applied = false
```

如需只读检查当前正式可用因子，运行：

```bash
PYTHONPATH=research_v2/scripts:research_v2/src \
python research_v2/scripts/refresh_factor_quality_latest.py \
  --model-eligible
```

### P4：候选人工准入评审（当前执行点）

质量表完成后，按独立决策审查 `yeslab_405_tushare_shsz_v1`、`yeslab_425_tushare_shsz_v2`、`yeslab_381_tushare_shsz_v1`、`yeslab_484_tushare_shsz_v1`、`yeslab_114_tushare_shsz_v2`、`yeslab_206_median_abs_v2`、`yeslab_443_tushare_shsz_v1` 等候选。六候选统一两两相似度审计已完成并登记为 comparison-only 证据：`114-484` 标记为 `REDUNDANT`，`381-405` 标记为 `ELEVATED_PAIR_REDUNDANCY_MONITOR`，其余边为 `PAIR_UNIQUE`；Alpha443 已另行完成与正式九因子的 9 条训练/验证边复核；Alpha432 已完成同口径复核但因与正式 `yeslab_431` 冗余而拒绝新增准入；Alpha377、Alpha387、Alpha388 均以至少一条完整决定性边早停判定冗余，未声称其余边已审计；Alpha391 已完成完整 9 条训练/验证边复核，训练期对正式 `yeslab_380` 越线，验证期作为边界监控；Alpha433 按严格年度门槛 `3/5` 拒绝；Alpha395 虽然年度门槛 `4/5`，但公式重放阻断仍拒绝新增准入；Alpha400 按严格年度门槛 `3/5` 拒绝；Alpha363 已完成完整九边相似度和年度复核，保留为待人工准入的隔离候选；Alpha357 也已完成完整九边相似度和年度复核，历史源公式阻断作为监控保留。下一步仍是逐个候选的明确人工准入和组合决策；任何候选都不能因为需要凑满 15 个因子而批量放宽门槛。

### P5：模型工作仍保持暂停

只有正式准入因子达到现行 D3.5 门槛并冻结 factor set 后，才讨论 LightGBM/XGBoost、Residual MLP、Set Attention、GRU/TCN 的正式训练和年度滚动样本外评估。2026 锁定测试仍需要单独预注册和一次性授权。

## 9. 明确禁止事项

- 不读取或利用 2026 标签进行因子修复、选择或调参；
- 不把旧 `yeslab_484` 历史指标直接写给新候选；
- 不覆盖 YesLab 原表或现有 V2 表结构；
- 不在 Alpha484 结果缺失时发布伪造/占位质量结论；
- 不自动把“待人工准入”候选加入正式因子集；
- 不把负 direction 解读为负质量；
- 不仅凭 Git clone 判断迁移完整；
- 不重新运行当前单体 Alpha484 脚本，除非先完成检查点和增量相似度改造；
- 不在代码、文档、提交或日志中打印数据库密码和 Tushare Token；
- 不宣称全仓库 Ruff 通过，直到 186 个当前 lint 问题被单独清理。

## 10. 新对话开场模板

迁移后可在新服务器的新对话中直接发送：

```text
请先完整阅读 docs/handoffs/2026-08-19-factor-quality-handoff.md，
并以当前工作区、Git、数据库和审计产物为权威证据继续工作。

持续目标：持续推进因子质量改进，暂不训练模型。
确认 Alpha484 可恢复审计已完成，审查
`research_v2/artifacts/diagnostics/tushare_alpha484_canonical_v1/admission_review.json`。
保持 Tushare SH/SZ、forward_open_5d:v1、2020-2024 训练、
2025 验证、2026 锁定禁读、正式九因子不自动变更。

接下来按文档执行 migration 012 dry-run、491 行质量表 dry-run、
显式发布和回读校验。全过程记录证据，不泄露凭据。
```

## 11. 当前接手点

接手者的第一项工作是阅读 `factor_quality_latest_refresh_v1.json`、`tushare_shsz_candidate_admission_review_matrix_v2.json`、`tushare_canonical_candidate_pair_similarity_v1.json`、`candidate_combination_evidence_gap_index_v1.json`、`canonical_candidate_version_contract_index_v1.json`、`explicit_admission_review_checklist_v1.json` 和 `extension_candidate_monthly_stability_validation_v1.json`。六候选统一两两审计已经完成；扩展候选 363/357/96 的完整组合边也已完成，训练/验证各 21 条边。当前需按独立、明确的人工决策审查六候选、443、363、357、96 及对应版本契约。`114-484` 不得在组合中同时使用；`381-405` 与 `357-363` 需保留冗余/监控审查；任何单因子通过或 pair `PAIR_UNIQUE` 都不构成自动准入。质量表发布后仍不得自动改变正式九因子、启动模型训练或读取 2026 数据。
