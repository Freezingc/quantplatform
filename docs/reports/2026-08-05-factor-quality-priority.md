# 因子质量补救优先级清单

日期：2026-08-05  
数据范围：`SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1`（2020-01-02–2024-12-31）  
锁定测试：未读取  
数据库写入：仅向新建的 `factor_repair_candidate` 表追加候选隔离批次，并向既有
`factor_quality_snapshot` 追加独立的 `FULL_DAY26_V3` 审计快照；未修改任何业务数据表、
`factor_observation_clean` 或原始 YesLab 因子表

## 结论

当前 source-qualified 集合有 47 个因子。47 个因子的值冲突键均为 0、完整日期范围均通过，但它们仍全部处于 `SOURCE_DQ_PASS_PENDING_SIMILARITY`，不能直接作为模型输入。

主要问题不是“47 个因子都失效”，而是两个可分离的数据质量问题：

- 38 个因子的有限值比例低于 0.80，需要先确认缺失是否来自可重算的行情/财务依赖；
- 27 个因子的原始行重复比例为 0.50，需要做确定性去重并重新计算覆盖率；重复行不能计入有效覆盖；
- 所有因子的 PIT 数字目前仍是去重前代理统计，尚未完成 compact universe 交集检查。

## 分层结果

| 优先级 | 因子数 | 判定条件 | 处理动作 |
| --- | ---: | --- | --- |
| P0-公式门禁 | 1 | 已知公式重叠值不一致 | 保持阻断，核对源值版本/输入，不放宽 tolerance |
| P1-去重+低覆盖 | 20 | duplicate_ratio=0.50 且 finite_ratio<0.80 | 先 exact dedup，再做依赖补值和 compact-universe 覆盖 |
| P2-低覆盖 | 18 | finite_ratio<0.80 且无重复 | 查缺失依赖、按月重算；不能用横截面填充 |
| P3-仅去重 | 7 | finite_ratio≥0.80 且 duplicate_ratio=0.50 | exact dedup 后重做 PIT/相似度 |
| P4-当前结构最干净 | 2 | finite_ratio≥0.80 且无重复 | 直接进入 train-only similarity/公式哈希复核 |

P0 仅指已在代表性公式 smoke 中发现的 `yeslab_433/GP_064`。这不是预测失败结论，而是源表存值与原始计算器重算值不一致的公式一致性阻断。

## 因子分组

### P0：公式重叠阻断

`yeslab_433 (GP_064)`

2020-12 精确使用 `stock_factor_daily` 和原始预热边界后仍有 6 个 mismatch；最大被拒绝行绝对误差 `66.086938`，最大相对误差 `0.012042`。必须先审计 `factor_data9` 的源值版本或生成批次。

`yeslab_391 (GP_022)` 已经没有公式重叠 mismatch，但该月仍有非有限源值隔离，因此不等于可以发布。

### P1：重复与有限覆盖同时存在（20）

```text
yeslab_395, yeslab_391, yeslab_388, yeslab_117, yeslab_118,
yeslab_96, yeslab_114, yeslab_115, yeslab_95, yeslab_98,
yeslab_385, yeslab_376, yeslab_393, yeslab_382, yeslab_387,
yeslab_377, yeslab_380, yeslab_381, yeslab_383, yeslab_396
```

处理顺序固定为：

1. 按 `(trade_date, ts_code, factor_name)` 只删除完全相同的 revision 行；
2. 对去重后的 finite coverage 重新计算月度/横截面覆盖；
3. 只有存在可审计公式和同口径行情时，才允许补非有限值；
4. 将未能补齐的行保留在 quarantine，不把它们转成 0 或横截面均值。

### P2：低覆盖但未发现重复（18）

```text
yeslab_471, yeslab_433, yeslab_458, yeslab_473, yeslab_456,
yeslab_400, yeslab_437, yeslab_401, yeslab_442, yeslab_443,
yeslab_439, yeslab_431, yeslab_94, yeslab_484, yeslab_425,
yeslab_41, yeslab_405, yeslab_432
```

这组优先检查依赖行情的缺口、上市日/停牌边界和滚动窗口预热，不应把低 finite ratio 直接解释成预测能力差。

### P3：先去重（7）

```text
yeslab_163, yeslab_362, yeslab_363, yeslab_365,
yeslab_366, yeslab_357, yeslab_359
```

这些因子的 finite ratio 已达到 0.80 以上，但重复行会抬高 PIT 代理覆盖，必须去重后再进入相似度和 compact-universe 门禁。

### P4：结构上最干净（2）

```text
yeslab_155, yeslab_79
```

这只表示 source DQ 结构较干净，不代表已经通过预测稳定性、train-only similarity 或最终准入。

## 下一轮执行顺序

1. 完成 P1/P3 的确定性去重物化，并保存去重前后行数、finite ratio、月度 coverage 对照；
2. 对 P2 及 P1 的低覆盖因子逐个核对依赖字段和滚动预热，沿用 `NO_FINITE_CANDIDATE` 隔离；
3. 对 `yeslab_433` 单独完成源值版本核对，未解释前禁止任何覆盖；
4. 对去重后仍通过的因子运行全 TRAIN 日 cross-sectional similarity、FDR/年度稳定性和 compact-universe PIT；
5. 所有层级通过后才生成新的候选 factor set；本清单不改变当前六因子 frozen contract，也不启动模型训练。

## 代表性去重 smoke

在不写库的情况下对一个 P1 和一个 P3 因子执行 2020-12 dry-run：

| 因子 | 分组 | clean 行 | 隔离行 | 状态 | 隔离原因 |
| --- | --- | ---: | ---: | --- | --- |
| `yeslab_395/GP_026` | P1 | 89,495 | 47,447 | `PASS_WITH_UNRESOLVED_QUARANTINE` | `NO_EVIDENCE_BACKFILL` |
| `yeslab_163/Alpha95` | P3 | 97,085 | 39,420 | `PASS_WITH_UNRESOLVED_QUARANTINE` | `NO_EVIDENCE_BACKFILL` |

P1 smoke 的审计统计为 `136,942` 个 exact duplicate group/row、`94,894` 个源非有限行；P3 smoke 为 `136,505` 个 exact duplicate group/row、`78,840` 个源非有限行。经过确定性去重后，12 月有限覆盖分别只有 `89,495/136,942=0.653525` 和 `97,085/136,505=0.711219`，隔离行分别为 `47,447` 和 `39,420`，没有伪造填充值。这也说明全范围 finite ratio 不能替代逐月门禁。

这两个 smoke 说明“无冲突”并不等于“可直接发布”：原始重复行和非有限行仍被隔离，未得到候选证据的行没有被填成常数。对应 manifest：

- [`P1 yeslab_395`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p1_yeslab395_v2_manifest.json)
- [`P3 yeslab_163`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p3_yeslab163_v2_manifest.json)

另外重跑了此前受 V2 行情起始日影响的 `2020-01 / yeslab_391 (GP_022)`：切换到
YesLab 精确 `stock_factor_daily` 后，结果仍为 `clean=56,920`、
`quarantine=38,344`、`NO_FINITE_CANDIDATE`，且 mismatch 为 0。这说明这 38,344
行不是简单的 V2 行情表缺失，而是对应证券在原始滚动窗口内确实没有足够的有限候选；
不能为了提高覆盖率而用常数或横截面填充。详见
[`2020-01 GP_022 exact-input smoke`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_jan2020_source_factor_market_v1_manifest.json)。

### P1/P3 全部 27 个因子的 compact-universe 复核

对 P1/P3 全部 27 个因子执行了 2020-12 的只读审计，分母为
`cn_liquid_a/v1` 与 `forward_open_5d/v1` 的有效交集，共 64,306 个成熟观测。
27 个因子全部达到当前 compact coverage 门槛：

| 指标 | 27 个因子最小值 |
| --- | ---: |
| 整体 finite coverage | 0.976892 |
| 最低日覆盖 | 0.974212 |
| 日覆盖 P05 | 0.974498 |
| 月覆盖 | 0.976887 |

因此，P1/P3 的全市场低 finite ratio 不能直接作为模型输入拒绝理由；它们需要继续通过全 TRAIN 月度稳定性、PIT 版本、相似度和 FDR 门禁。该结论也不能外推到 P2 或 P0，后两类仍需单独审计。

完整只读结果见 [`P1/P3 compact coverage JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_dec2020_p1_p3_all27_v1.json)。

### P1/P3 全 TRAIN（2020–2024）compact-universe 复核

随后将同一口径扩展到完整 TRAIN 区间（2020-01-01–2024-12-31）。本次查询按
`store_factor` 分组并在 DolphinDB 服务端执行有限值 `(trade_date, ts_code)` 去重；成熟分母仍是
`cn_liquid_a/v1 ∩ forward_open_5d/v1`，只使用会话级临时键，不写入任何数据库表。

27 个因子的结果为：20 个通过结构覆盖门槛、7 个因最低日覆盖失败。通过这里只表示
`COMPACT_DQ_PASS_PENDING_SIMILARITY`，尚未通过相似度、FDR、年度稳定性或准入发布门禁。

| 结果 | 数量 | 因子 |
| --- | ---: | --- |
| `COMPACT_DQ_PASS_PENDING_SIMILARITY` | 20 | `yeslab_114, 115, 117, 118, 163, 357, 359, 362, 363, 365, 366, 376, 381, 383, 391, 393, 396, 95, 96, 98` |
| `COMPACT_DQ_FAIL` | 7 | `yeslab_377, 380, 382, 385, 387, 388, 395` |

全 TRAIN 的总体范围如下：

| 指标 | 27 个因子最小值 | 说明 |
| --- | ---: | --- |
| 整体 finite coverage | 0.983577 | GP_022；仍明显高于全市场分母下的代理值 |
| 最低日覆盖 | 0.481387 | GP_026；7 个失败因子的共同门禁触发项约为 0.48–0.49 |
| 日覆盖 P05 | 0.968101 | GP_022；失败因子并非全期间普遍缺失 |
| 最低月覆盖 | 0.788443 | GP_019；其余失败因子月度最低值仍在 0.88 以上 |

这 7 个因子的失败主要表现为少数交易日的断崖式缺失，而不是整体覆盖率不足；不能用
横截面均值或常数填充掩盖。下一步应定位这些最低覆盖交易日，核对对应公式的滚动预热、依赖字段
及源表生成批次；在没有可复算证据前，将这些日期的记录保留在 quarantine。其余 20 个因子可以
进入下一道 train-only similarity/FDR/年度稳定性检查，但不能据此直接扩大 frozen factor set。

完整结果见 [`P1/P3 全 TRAIN compact coverage JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_p1_p3_all27_v1.json)。

### 2020-09-18 换手率断档的补救审计

对上述 7 个失败因子继续做了日期定位和外部证据核对。异常并非因子公式本身首先产生：
`stock_factor_daily` 在 `2020-09-18` 的成熟宇宙中只有 `1,354/2,740` 条
`turnover_rate` 有限值，缺失行的 `amount`、`vol` 仍存在，但股本字段也同时缺失，
因此不能从 YesLab 现有字段安全反推换手率。5/10 日滚动公式将同一缺口传导到
`2020-09-22`，解释了最低日覆盖约 0.49 的现象。

Tushare `daily_basic(20200918)` 返回 4,070 条有限 `turnover_rate`，对成熟宇宙的
1,386 条缺失记录全部有可用替代值。基于这些值只在内存中重算，并对原始有限重叠值
执行严格核对：

| 因子 | 重叠 mismatch | compact 可补行 | 结论 |
| --- | ---: | ---: | --- |
| `yeslab_377 / GP_008` | 0 | 1,386 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_380 / GP_011` | 0 | 1,386 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_382 / GP_013` | 0 | 1,385 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_385 / GP_016` | 0 | 1,377（含滚动传导共 6,853） | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_387 / GP_018` | 0 | 1,385 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_388 / GP_019` | 0 | 1,377（含滚动传导共 8,204） | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_395 / GP_026` | 2,016 | 1,348 | `TURNOVER_REPAIR_BLOCKED_OVERLAP` |

六个候选的有限重叠值完全一致，并已通过纯内存 overlay dry-run：状态均为
`REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE`。它们只填充有有限候选的缺失行，
其余 `NO_FINITE_CANDIDATE` 行继续隔离；这不代表可以覆盖 YesLab 原表，也不代表已经发布。
`GP_026` 的重叠值仍有较大不一致（最大绝对差约 `42,249.21`），overlay 为
`REPAIR_BLOCKED`，不能因为外部换手率完整就强行补值。

补救审计脚本和结果：

- [`turnover gap audit script`](../../research_v2/scripts/audit_turnover_gap_repair.py)
- [`turnover gap audit JSON`](../../research_v2/artifacts/diagnostics/turnover_gap_repair_audit_2020-09-18_v1.json)
- [`six-factor candidate batch manifest`](../../research_v2/artifacts/diagnostics/turnover_gap_repair_candidates_2020-09-18_v1.json)

首批候选批次先以本地、可复核的 CSV 形式生成，包含 6 个因子的有限候选补行（分别为
1,386、1,386、1,385、6,853、1,385、8,204 行），不包含 `GP_026`。每个批次都有
SHA-256；写入新 V2 隔离表时必须用该哈希和同一来源水印做幂等校验。

随后已将 migration 009/010 应用到 `dfs://QuantResearchV2Test`，并把通过 preflight 的
20,599 行写入新表 `factor_repair_candidate`：

- `data_version=2e74fb7328665c5ef3344f43ce13a22d891b1b890458cd88310d754f395bc731`；
- 6 个因子均有有限候选值，`source_value` 有限值为 0，业务键无重复；
- 日期范围为 2020-09-18–2020-09-25，`GP_026` 未写入；
- `factor_observation_clean` 行数保持 `170,333,348`，没有推进 clean 对象指针；
- `published_version` 只新增了 `factor_repair_candidate/2020-09` 的隔离指针。

发布前检查报告和发布结果分别见 [`candidate preflight`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_preflight_2020-09-18_v1.json)
和 [`candidate publish report`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_publish_2020-09-18_v1.json)。

首批候选随后已物化到新的 V2 隔离对象；后续延长窗口和全 TRAIN 复核见下一节。
`GP_026` 始终留在阻断清单，暂不进入候选表。任何正式发布仍需单独批准。

### GP_019 延长窗口补救与全 TRAIN 复核

对 `yeslab_388 / GP_019` 继续追踪到 2020-10-15 后，发现源表在
2020-09-18–2020-10-09 对大量证券各写入两条空值 revision（每个键的两条记录均为
非有限值），因此只修复 9 月批次仍会在 10 月 9 日的 10 日滚动窗口留下缺口。这是
源表 revision/生成批次问题，不是候选公式与源值不一致。

本次处理采取按月隔离的方式：

- 重新用同一 Tushare `daily_basic(20200918)` 水印和原始预热区间重算 GP_019；
- 对 2020-10-01 起可补的 1,343 行生成单独的 `2020-10` 候选批次；有限源值重叠
  `37,633` 行，mismatch `0`，候选值全为有限数，未覆盖任何有限源值；
- preflight 的五个阻断检查全部 PASS，批次已发布到新表，
  `data_version=ad03741eca14ea75458938ce121df3b1643afdeb87c4b4360f2ca559797b86a7`，
  `factor_repair_candidate/2020-10` 指针已推进；原始因子表与 clean 表仍未改变。

数据库核验显示候选表物理行数为 `46,601`（包括首批 9 月版本、9 月延长版本和 10 月批次，
表本身保持 append-only）；当前版本指针暴露 9 月最新 `24,659` 行和 10 月 `1,343` 行，
候选值有限行 `46,601`，`source_value` 有限行为 `0`。`factor_observation_clean` 仍为
`170,333,348` 行，schema migration 记录为 10 条。

局部 2020-09-01–10-31 复核显示 GP_019 的最低日覆盖率为 `0.981063`、最低月覆盖率
为 `0.991532`。随后对 2020-01-01–2024-12-31 的六个换手率候选因子执行完整
compact-universe 复核（成熟分母为 `cn_liquid_a/v1 ∩ forward_open_5d/v1`，共
`3,996,740` 个观测）：

| 因子 | 整体覆盖 | 最低日覆盖 | 最低月覆盖 | 结果 |
| --- | ---: | ---: | ---: | --- |
| GP_008 / `yeslab_377` | 0.999998 | 0.998548 | 0.999887 | PASS_PENDING_SIMILARITY |
| GP_011 / `yeslab_380` | 0.999998 | 0.998548 | 0.999887 | PASS_PENDING_SIMILARITY |
| GP_013 / `yeslab_382` | 0.999758 | 0.997811 | 0.999263 | PASS_PENDING_SIMILARITY |
| GP_016 / `yeslab_385` | 0.998109 | 0.987259 | 0.995044 | PASS_PENDING_SIMILARITY |
| GP_018 / `yeslab_387` | 0.999758 | 0.997811 | 0.999263 | PASS_PENDING_SIMILARITY |
| GP_019 / `yeslab_388` | 0.997580 | 0.981063 | 0.991532 | PASS_PENDING_SIMILARITY |

将这六行结果与未受 overlay 影响的其余 21 个因子合并后，27 个 P1/P3 因子中有
26 个达到 compact 双重门槛，唯一仍失败的是 `yeslab_395 / GP_026`。GP_026 继续保持
`TURNOVER_REPAIR_BLOCKED_OVERLAP`，不进入候选表，也不进入模型输入。完整合并结果见
[`turnover repaired full TRAIN compact JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_turnover_repaired6_v3.json)，
六因子独立复核见
[`six-factor focus JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_turnover_repaired6_focus_v3.json)。

在全日相似度和年度稳定性复核完成前，状态仍是“结构质量通过、相似度/FDR/年度稳定性
待审”，不是最终准入；GP_026 保持隔离，期间不启动任何模型训练。复核结果见下一节。

### 26 因子 TRAIN 全日相似度与年度稳定性结果

已完成 26 个结构通过因子的 2020–2024 全日横截面相似度计算。相似度计算使用每月
`published_version` 指向的最新 `factor_observation_clean` 版本；clean 表缺少的因子从
YesLab 原始表按月读取并做确定性有限值去重；6 个换手率候选继续按“只填充源值缺失键”
叠加。每个因子对均有 `1,212` 个有效交易日，未读取 2026 locked-test。

结果为 19 个冗余、7 个唯一：

- 冗余簇代表候选包括 `yeslab_114/115`（0.9474）、`yeslab_117/118`（0.9448）、
  `yeslab_357/359`（0.9675）、`yeslab_365/366`（0.9793）、
  `yeslab_377/382`（0.9823）、`yeslab_387/388`（0.9824）等；
- 唯一候选为 `yeslab_163, yeslab_363, yeslab_385, yeslab_396, yeslab_95,
  yeslab_96, yeslab_98`，它们与最近邻的绝对 Spearman 均低于 0.90。

现有五年年度预筛要求每个年度 ICIR 也达到 0.50，因此只有
`yeslab_357, yeslab_359, yeslab_376, yeslab_393, yeslab_98` 通过该保守门槛；与全日
唯一集合交集后，当前最保守的全门禁交集只有 `yeslab_98`。另一方面，主计划规定的
“五年中至少三年同向、年度均值/正向日占比达标”稳定性规则下，26 个因子均未在年度
均值或方向占比上失败；这 7 个唯一因子可作为下一轮候选，但不应在未确认年度 ICIR
规则前直接发布为 frozen factor set。

当前结论：

1. `GP_026 / yeslab_395` 仍因公式 overlap mismatch 阻断；
2. 其余 26 个通过结构覆盖、日级 HAC/FDR 和全日相似度证据；
3. 若要求每个因子本身都必须是全日唯一，严格“年度 ICIR + 全日唯一”交集只有
   `yeslab_98`；若按冗余簇选代表，严格年度 ICIR 通过的代表为
   `yeslab_357, yeslab_376, yeslab_98`；
4. 尚未训练任何模型，也未推进 frozen factor set。下一步应先审阅年度 ICIR 的规则
   是否作为每年硬门槛，再决定对 7 个唯一候选做第二类补救或保守发布。

完整门禁汇总见
[`TRAIN26 gate report`](../../research_v2/artifacts/diagnostics/factor_quality_train26_gate_report_v4.json)，
全日原始结果见
[`TRAIN26 full-day similarity`](../../research_v2/artifacts/diagnostics/train_similarity_full_day_2020_2024_repaired26_v3.csv)，
clean 版本指针审计见
[`V2 clean pointer audit`](../../research_v2/artifacts/diagnostics/v2_clean_pointer_version_audit_train26_v1.json)。

### 修复后 RankIC 再检验

按照数据修复设计，未直接沿用修复前的 RankIC。针对 2020-09-18–2020-10-09 的换手率
候选影响窗口，重新计算 6 个修复因子的每日横截面 RankIC，并把这 60 条受影响记录
替换到本地 2020–2024 RankIC 面板，再重新计算 26 因子的 HAC/FDR/ICIR。结果显示：

- 6 个因子各自仍有 `1,212` 个有效 TRAIN 日；
- 60 条 RankIC 记录发生替换，26 因子的每日 predictive gate 全部保持通过；
- 指标变化很小，没有因修复而制造新的显著性或方向性：例如 GP_019 的 mean RankIC
  从 `-0.082160` 变为 `-0.081981`，ICIR 从 `-0.573622` 变为 `-0.570411`，仍满足
  当前绝对值阈值；
- 该再检验只写本地 CSV/JSON，不写入已有 `factor_rank_ic_daily` 表，也不改变模型或
  frozen factor 指针。

对应证据见
[`repaired RankIC report`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1.json)、
[`repaired RankIC impact rows`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1_impact.csv)、
[`repaired TRAIN metrics`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1_metrics.csv)
和更新后的
[`TRAIN26 gate report V3`](../../research_v2/artifacts/diagnostics/factor_quality_train26_gate_report_v5.json)。

### 冗余簇代表候选（尚未冻结）

已保存完整的 325 个因子对矩阵。按全日相似度阈值 `0.90` 的全部边组成连通分量，再按
`FDR q → signed mean RankIC → |ICIR| → factor_id` 排序选代表，可以得到 14 个代表候选：

```text
yeslab_115, yeslab_118, yeslab_163, yeslab_357, yeslab_363,
yeslab_365, yeslab_376, yeslab_380, yeslab_383, yeslab_385,
yeslab_396, yeslab_95, yeslab_96, yeslab_98
```

该代表集略低于主计划的 15–40 个特征目标，但目标是 soft target，不应为了凑数放回冗余
因子。完整 pair matrix 的每对有效日数均为 1,212，连通分量和阈值边均已落盘；因此这仍
是“候选代表集”，不能直接冻结，冻结前还需确认年度 ICIR 硬门槛采用“每年硬失败”还是
主计划中的“至少三年同向”规则。候选代表提案见
[`TRAIN26 full-pair cluster proposal`](../../research_v2/artifacts/diagnostics/train26_cluster_representative_proposal_v2.json)，
完整 pair matrix 见
[`TRAIN26 pair matrix`](../../research_v2/artifacts/diagnostics/train_similarity_pairs_2020_2024_repaired26_v1.csv)。
年度规则的严格/主计划口径对照见
[`TRAIN26 annual rule comparison`](../../research_v2/artifacts/diagnostics/train26_annual_rule_comparison_v1.json)。

### 14 个代表因子的 2025 验证期复核

为避免把旧的 21 因子验证目录误当成当前代表集证据，新增了一个只读评估器，对上述
14 个代表逐月从 YesLab 源表重新做严格 PIT 清洗，并使用 V2 最新的成熟股票池与原始
5 日复权收益标签计算验证期横截面 RankIC。训练期方向从
`SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2` 的已审计门禁报告固定读取，验证期不重新
估计方向，也不读取 2026 locked-test；为保证 5 日标签成熟度，去除了 2025-12-24–12-31
的末端日期。

核验结果如下：

| 指标 | 结果 |
| --- | ---: |
| 代表因子 | 14 |
| 日度 RankIC 行数 | 3,318 |
| 每因子有效交易日 | 237（2025-01-02–2025-12-23） |
| 训练方向 | 14 个均为 -1，全部来自 TRAIN 报告 |
| 当前双重门槛预筛 | 14/14 通过 |
| 验证覆盖率 | 0.994156–1.000000 |
| 2026 日期读取 | 0 |
| 数据库写入 | 0 |

代表中验证期表现最强的几个因子为：`yeslab_357`（mean RankIC `-0.061495`，ICIR
`-1.1740`，方向一致率 `89.87%`）、`yeslab_363`（`-0.057695`，`-0.9145`，
`82.70%`）和 `yeslab_365`（`-0.054612`，`-0.9108`，`84.81%`）。其余代表也满足
当前验证门槛，**但这只表示验证期 predictive 预筛通过**，尚未完成验证期相似度复核，
也没有改变年度 ICIR 规则尚未决策、候选集尚未冻结的状态。

评估器会把“没有有效 RankIC 的因子”显式记录为 `VALIDATION_2025_NO_RANKIC`，而不是
静默丢弃；本次 14 个代表没有触发该状态。脚本和结果如下：

- [`2025 representatives validation evaluator`](../../research_v2/scripts/evaluate_factor_representatives_validation.py)
- [`2025 representatives validation quality`](../../research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1/quality_summary.csv)
- [`2025 representatives validation daily RankIC`](../../research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1/daily_rank_ic.parquet)
- [`2025 representatives validation coverage`](../../research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1/coverage.csv)
- [`2025 representatives validation metadata`](../../research_v2/artifacts/diagnostics/train26_representatives_validation_2025_v1/metadata.json)

下一步应把这 14 个代表与 2025 验证期的日级相似度矩阵合并，再按已经选定的年度门槛
生成“可训练候选集”；在此之前不发布新 factor set，也不启动模型训练。

### 2025 验证期日级相似度复核

在验证期 RankIC 证据之上，又对 14 个代表的 91 个因子对执行了完整日级绝对 Spearman
矩阵。每个 pair 均有 `237` 个有效交易日；因子值读取优先使用每月
`published_version` 指向的最新 clean 版本，clean 缺少的因子才从 YesLab 源表做有限值
去重 fallback。该命令仍是只读，不写 `factor_quality_snapshot` 或任何业务表。

结果显示，训练期代表关系在验证期发生了结构变化：

- `yeslab_357` 与 `yeslab_365` 的验证期相似度为 `0.946569`，超过 `0.90`，因此这两个
  在训练期不同簇中的代表在验证期应视为冗余，候选代表应优先保留训练期 signed RankIC
  更强的 `yeslab_357`；
- 其余 12 个代表的最近邻相似度均低于 `0.90`，其中 `yeslab_115/118` 为 `0.833597`、
  `yeslab_95/96` 为 `0.843193`，暂不合并；
- 因此按验证期阈值做“只合并确认冗余”的保守提案，14 个代表暂时压缩为 13 个候选，
  但这仍是 OOS 结构证据，不是已冻结的 factor set。

验证期相似度 artifact：

- [`2025 validation similarity summary`](../../research_v2/artifacts/diagnostics/validation_similarity_full_day_2025_representatives_v1.csv)
- [`2025 validation pair matrix`](../../research_v2/artifacts/diagnostics/validation_similarity_pairs_2025_representatives_v1.csv)
- [`2025 validation similarity metadata`](../../research_v2/artifacts/diagnostics/validation_similarity_full_day_2025_representatives_v1.json)

当前仍保留两个决策分支：若采用主计划的年度方向稳定性规则，OOS 相似度提案对应 13 个
候选；若继续采用每年 ICIR `>=0.50` 的保守硬门槛，训练期年度稳定代表只有
`yeslab_357, yeslab_376, yeslab_98`，验证期相似度不会把这三者再合并。年度规则明确前，
不发布上述 13 个候选，也不启动模型训练。

### 年度门槛审阅建议

现有数据支持把“每个年度 ICIR `>=0.50`”降为诊断项，而不是再叠加为硬门槛：14 个代表
在训练期总体绝对 ICIR、HAC/FDR、年度方向稳定性和 2025 固定方向验证期双重门槛均有
证据；但逐年 ICIR 硬门槛会把 `yeslab_163`（最低年度绝对 ICIR `0.2779`）、
`yeslab_95`（`0.4409`）等仍在验证期稳定通过的因子直接排除。这一额外硬门槛也不是
主计划中“至少三年同向、年度均值/正向日占比达标”的原始规则。

因此建议采用以下准入顺序（仍需在发布前确认）：

1. TRAIN 总体门槛：绝对 mean RankIC、绝对 ICIR、HAC/FDR、覆盖率和修复后 RankIC 全部通过；
2. 年度稳定性硬门槛：五年中至少三年同向，且年度 signed mean RankIC/正向日占比达标；
3. 2025 验证期：固定 TRAIN 方向，双重 RankIC/ICIR/FDR 门槛通过；
4. OOS 相似度：验证期 pair 的绝对 Spearman `>=0.90` 才合并，当前仅合并
   `yeslab_357/yeslab_365`，形成 13 个“待训练候选”；
5. 逐年 ICIR、相似度边界和覆盖最低月作为监控/人工复核项，不以它们单独制造第二层
   过拟合筛选。

若坚持严格年度 ICIR 硬门槛，候选会收缩到
`yeslab_357, yeslab_376, yeslab_98`；这是一条明确的保守备选，不是当前默认提案。
无论采用哪条分支，`yeslab_395 / GP_026` 的公式 overlap mismatch 仍然阻断，不能进入
候选集。

### 剩余 21 个 source-DQ 通过因子的结构复核与输入修复

此前 47 个 source-DQ 通过因子中，有 21 个已经通过 TRAIN predictive 门槛，但尚未完成
compact-universe 交集检查。本轮对这 21 个因子执行了 2020–2024 的只读成熟分母审计：

- 未修复前 `16/21` 通过 compact coverage；
- `yeslab_395/GP_026`、`yeslab_400/GP_031`、`yeslab_439/GP_070`、
  `yeslab_456/GP_087`、`yeslab_471/AttentionSpillover` 在 2020-09-18 出现断崖式最低日覆盖；
- GP_031、GP_070、GP_087 与此前 6 个 GP 一样依赖 `turnover_rate`，但原有修复函数没有
  覆盖这 3 个公式，现已按源文件计算器语义补齐候选重算。

扩展 Tushare turnover 审计结果：

| 因子 | overlap mismatch | gap 可补行 | compact 修复后最低日覆盖 | 状态 |
| --- | ---: | ---: | ---: | --- |
| `yeslab_400 / GP_031` | 0 | 1,348 | 0.963599 | 候选通过 |
| `yeslab_439 / GP_070` | 0 | 1,364 | 0.981578 | 候选通过 |
| `yeslab_456 / GP_087` | 0 | 1,348 | 0.963599 | 候选通过 |
| `yeslab_395 / GP_026` | 2,016 | 1,348 | — | 继续阻断 |

3 个新候选经过 preflight 的有限值、无覆盖有限源值、业务键唯一、未读锁定测试等 5 个
阻断检查后，已追加到新建 `factor_repair_candidate` 隔离表；没有修改 YesLab 或
`factor_observation_clean`。数据库核验为：

- `factor_repair_candidate` 物理行数由 `46,601` 增至 `50,661`，新增 `4,060` 行；
- `factor_observation_clean` 仍为 `170,333,348` 行；
- 2020–2024 影响窗口的 9 个换手率修复因子重新计算 RankIC 后，全部保持 TRAIN
  predictive prefilter，通过方向均为 `-1`；新 3 个因子的 mean RankIC/ICIR 分别为
  `GP_031: -0.077127/-0.546584`、`GP_070: -0.075355/-0.509775`、
  `GP_087: -0.076192/-0.532066`。

`AttentionSpillover` 的断档来自 `total_mv`，不是换手率。Tushare 在 2020-09-18 提供
4,070 条有限 `total_mv`，但按其计算器语义重算后与已有因子值的 35,024 个有限重叠键中
有 11,214 个 mismatch（最大绝对差 `485.66`），因此该修复被阻断，没有生成或发布候选。

相关证据：

- [`remaining 21 compact coverage`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_remaining21_v1.json)
- [`extended turnover repair audit`](../../research_v2/artifacts/diagnostics/turnover_gap_repair_extended_2020-09-18_v1.json)
- [`extended turnover candidate preflight`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_preflight_extended_new3_2020-09-18_v1.json)
- [`extended turnover candidate publish`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_publish_extended_new3_2020-09-18_v1.json)
- [`extended repair compact overlay`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_turnover_extended3_v1.json)
- [`extended repaired RankIC`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train29_extended_v1.json)
- [`AttentionSpillover total_mv audit`](../../research_v2/artifacts/diagnostics/attention_spillover_gap_repair_audit_2020-09-18_v1.json)

这 19 个结构通过因子还没有完成全 TRAIN 相似度和 2025 固定方向验证，因此暂不加入
当前 13 个 OOS 候选；下一步继续做扩展因子的 train-only 相似度与验证期 RankIC/ICIR。

### 合并 45 因子相似度与扩展代表验证

随后将原有 26 个结构因子与这批修复后通过结构门禁的 19 个因子合并，执行了完整的
45 因子 TRAIN 矩阵：共 `990` 对、每对 `1,212` 个有效交易日、无 NaN。以绝对
Spearman `0.90` 组成连通分量后，45 个因子得到 28 个 TRAIN 代表候选。新增的显著关系
包括：

- `yeslab_400/yeslab_456`：`0.974844`；
- `yeslab_431/yeslab_432/yeslab_458/yeslab_79`：连通簇，最高 `0.999996`；
- `yeslab_433` 与已有 `yeslab_391`：`0.910099`；
- 原有换手率簇扩展为 `yeslab_377/380/382/387/388/391/433`。

因此跨批次合并后，旧代表中的 `115、118、365、383` 被更强的同簇代表
`114、117、362、381` 替代；不能把之前 14 个代表直接与扩展因子拼接。完整 45 因子
提案（仅提案，不冻结）见：

- [`TRAIN45 full-day summary`](../../research_v2/artifacts/diagnostics/train_similarity_full_day_2020_2024_combined45_v2.csv)
- [`TRAIN45 full pair matrix`](../../research_v2/artifacts/diagnostics/train_similarity_pairs_2020_2024_combined45_v2.csv)
- [`TRAIN45 cluster representatives`](../../research_v2/artifacts/diagnostics/train45_cluster_representative_proposal_v1.json)

对新增的 18 个 TRAIN 代表补做 2025 固定方向验证后，结果为 `14/18` 通过、4 个失败：
`yeslab_117`、`yeslab_41`、`yeslab_439`、`yeslab_473`，失败原因均为验证期 ICIR
低于 `0.50`，不是覆盖或方向翻转。每个因子均有 237 个有效日，日期范围为
2025-01-02–2025-12-23，无 2026 读取。

将原有 10 个仍属于合并簇代表且已验证通过的因子加入后，得到 24 个验证期 predictive
通过的代表。对这 24 个执行验证期 276 对相似度矩阵，只有一条冗余边：
`yeslab_114/yeslab_484 = 0.905240`。按 TRAIN 排名保留 `yeslab_114` 后，当前 OOS
保守候选为 23 个。

这 23 个仍是待审候选，不是 frozen factor set。若采用严格逐年 ICIR 硬门槛，23 个候选中
只有 `yeslab_155、yeslab_357、yeslab_376、yeslab_442、yeslab_98` 通过；这进一步说明
逐年 ICIR 应与主计划规则分开，作为监控或保守分支，而不能在没有确认的情况下隐式叠加。

完整扩展 OOS 对照见：

- [`TRAIN45 new representatives validation`](../../research_v2/artifacts/diagnostics/train45_new_representatives_validation_2025_v1/quality_summary.csv)
- [`TRAIN45 final24 validation similarity`](../../research_v2/artifacts/diagnostics/validation_similarity_full_day_2025_final24_v1.csv)
- [`TRAIN45 final24 validation pair matrix`](../../research_v2/artifacts/diagnostics/validation_similarity_pairs_2025_final24_v1.csv)
- [`TRAIN45 OOS representative quality`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v1.csv)
- [`TRAIN45 OOS admission proposal`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v1.json)

当前质量结论是：`yeslab_395/GP_026` 和 `yeslab_471/AttentionSpillover` 仍阻断；扩展
因子中有 4 个通过 TRAIN 但未通过 2025 predictive 的因子；其余 23 个才具备进入下一步
年度规则审阅的证据。任何候选发布和模型训练仍保持暂停。

### 阻断因子的公式复现一致性重放

为区分“输入字段缺失”与“源因子本身无法由当前计算器复现”，本轮新增了只读的公式重放
门禁。它直接调用 `数据库/factor_calculators` 中的当前计算器，在相同日期窗口重算，再与
YesLab 已存有限值逐键比较；有限重叠 mismatch 会阻断后续输入修复。

2020-09-14–2020-09-25 的重放结果如下：

| 因子 | 有限重叠 | mismatch | mismatch 日期数 | 最大绝对差 | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| `yeslab_395 / GP_026` | 35,431 | 2,433 | 10 | 83,498.21 | `FORMULA_REPLAY_BLOCKED` |
| `yeslab_471 / AttentionSpillover` | 35,024 | 9,682 | 10 | 485.66 | `FORMULA_REPLAY_BLOCKED` |

两者 mismatch 都覆盖整个审计窗口，而不是只发生在 2020-09-18。因此即使 Tushare 能
补齐 `turnover_rate` 或 `total_mv`，也不能把当前计算器输出直接写成这两个因子的历史
替代值。当前正确处置是继续隔离，并追溯生成源版本/历史输入快照；不降低 overlap 门槛，
也不把这两个因子并入 23 个 OOS 候选。

重放脚本和证据：

- [`公式复现一致性审计脚本`](../../research_v2/scripts/audit_formula_replay_consistency.py)
- [`阻断因子重放结果`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_blocked_factors_2020-09-14_2020-09-25_v1.json)
- [`源输入横截面漂移审计`](../../research_v2/artifacts/diagnostics/source_input_universe_drift_2020-09-14_2020-09-25_v1.json)

该门禁只读 YesLab，`database_writes=false`，且未读取 2026 locked-test。

进一步核查显示，这不是简单的单日缺失：

- 2020-09-14–25，`factor_data9/GP_026` 每日有 `11,908` 条原始行、`5,954` 个唯一业务键，
  即每个键有一条完全重复记录；去重后仍保留 `3,694–3,724` 个有限值。
- 同一窗口，`factor_data10/AttentionSpillover` 每日有 `5,976` 个唯一键，有限值为
  `1,991–3,699` 个。
- 当前 `stock_factor_daily` 每日只有 `4,042–4,080` 个唯一键；2020-09-18 的有限
  `total_mv/turnover_rate` 更只有 `2,200` 个。
- 已知旧 `dfs://dayFactorDB/dayFactorTB` 在该 2020 窗口没有记录，无法提供生成时的
  历史输入横截面。

这解释了为什么部分简单换手率因子可以精确重放，而 GP_026 和 AttentionSpillover
不能：后两者包含横截面排序/行业市值分组或依赖历史横截面集合，源因子生成时的输入
集合已经无法从现有表中恢复。没有历史输入快照前，任何“按当前行情重算并覆盖”的做法
都会把源版本漂移伪装成修复，继续禁止发布。

### 23 个 OOS 候选的公式复现复核

将同一公式重放门禁扩展到当前 23 个 OOS 候选后，结果为 `19 PASS / 4 BLOCKED`，没有
计算器接口错误。被阻断的 4 个因子为：

- `yeslab_357`：37,473/37,833 个有限重叠 mismatch；
- `yeslab_362`：37,433/37,793 个有限重叠 mismatch；
- `yeslab_363`：37,433/37,793 个有限重叠 mismatch；
- `yeslab_443 / GP_074`：23 个有限重叠 mismatch，最大绝对差 `0.324917`。

前 3 个行业因子在补通 `DataFetcher.get_stock_industry_info()` 后仍大面积 mismatch，
说明将当前 `industry` 别名为 `l1_name/l1_code` 不能恢复历史行业分类；GP_074 则是少量
跨全窗口 mismatch，仍不能通过“精确 overlap”修复门槛。

因此，公式复现门禁后的 OOS 候选从 23 个收缩为 19 个：

`yeslab_114, yeslab_155, yeslab_163, yeslab_376, yeslab_380, yeslab_381,
yeslab_385, yeslab_396, yeslab_400, yeslab_401, yeslab_405, yeslab_425,
yeslab_431, yeslab_437, yeslab_442, yeslab_94, yeslab_95, yeslab_96, yeslab_98`。

严格逐年 ICIR 分支相应从 5 个变为 4 个：`yeslab_155、yeslab_376、yeslab_442、yeslab_98`。
这仍是候选提案，不代表冻结或开始模型训练。

复核产物：

- [`OOS23 公式复现结果`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_oos23_2020-09-14_2020-09-25_v2.json)
- [`OOS 质量提案 v2`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v2.json)
- [`OOS 质量明细 v2`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v2.csv)

### 19 个候选的多时间窗复现门禁

为避免把单个缺失日窗口当作完整公式证据，又对 v2 的 19 个候选执行了四个只读窗口：

- 2020-01-02–2020-01-31；
- 2022-01-04–2022-01-31；
- 2024-01-02–2024-01-31；
- 2025-01-02–2025-01-31。

其中 2020 和 2022 窗口的 19 个因子全部通过；2024 和 2025 窗口均发现
`yeslab_401 / GP_032` mismatch（分别 2 个和 6 个有限键），其余 18 个因子四个窗口
全部通过。因此当前多窗口公式门禁后的 OOS 候选进一步收缩为 18 个，`yeslab_401` 进入
阻断清单。严格年度 ICIR 分支仍为 4 个。

这 18 个是更严格的待审提案，不是 frozen set；在此之前不启动模型训练。多窗口证据和
提案见：

- [`OOS19 多窗口复现汇总`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_oos19_multiwindow_v1.json)
- [`OOS 质量提案 v3`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v3.json)
- [`OOS 质量明细 v3`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v3.csv)

对 `GP_032` 和 `GP_074` 的少量 mismatch 进一步抽查发现，YesLab `amount` 通常保留两位
小数，而 Tushare `daily` 保留三位小数，例如 `2024-01-15/688607.SH` 为
`16631.67` 对 `16631.672`。这说明局部差异很可能来自行情精度修订，而不是表达式本身
改变；但尚未用高精度 amount 完成全窗口重放，因此不能据此放宽复现门槛或解除阻断。

证据见 [`GP_032/GP_074 精度修订抽查`](../../research_v2/artifacts/diagnostics/precision_revision_mismatch_audit_gp032_gp074_v1.json)。

随后在内存中对相应 warm-up 输入逐日替换 Tushare 三位小数 `amount` 并重放，结果仍不通过：
GP_032 在 2024/2025 窗口分别剩余 2/4 个 mismatch，GP_074 的 mismatch 反而从 23 个增至
336 个。因此 Tushare 精度只能解释差异来源之一，不能作为历史源值的等价替代；三个因子
继续阻断，不发布 amount overlay 候选。

完整结果见 [`高精度 amount overlay 重放`](../../research_v2/artifacts/diagnostics/precision_amount_overlay_replay_gp032_gp074_v1.json)。

### 结构覆盖失败但预测通过因子的 warm-up 复核

对 TRAIN predictive-pass 但结构覆盖门禁失败的 23 个因子做了只读 raw-source 全范围聚合。
其中 7 个因子的 2020–2024 原始有限值比例达到 `0.70`，但标准 compact 门禁仍会因为公式
预热期的首个交易日覆盖率为零而失败。根据各公式的实际嵌套滚动窗口，增加了一个仅用于
复核的 `warm-up=60` 交易日排除版本；该版本通过的 4 个因子为：

`yeslab_398/GP_029`、`yeslab_438/GP_069`、`yeslab_451/GP_082`、`yeslab_464/GP_095`。

另外 3 个原始有限值比例足够但 warm-up 后仍有长期断档：`yeslab_418/GP_049` 的最小日
覆盖率约 `0.488`，`yeslab_430/GP_061` 和 `yeslab_444/GP_075` 存在零覆盖月份，继续
阻断。剩余 16 个因子的原始有限值比例低于 `0.70`。这一步没有修改全局结构门禁，也没有
把 warm-up 规则直接用于发布，只把 4 个因子列入复核队列。

证据：

- [`23 个结构失败因子 raw 聚合`](../../research_v2/artifacts/diagnostics/structural_fail_predictive_pass_raw_agg_2020_2024_v1.json)
- [`7 个 raw 通过因子的标准 compact`](../../research_v2/artifacts/diagnostics/structural_fail_predictive_pass_compact_7_2020_2024_v1.json)
- [`7 个因子的 warm-up compact`](../../research_v2/artifacts/diagnostics/structural_fail_predictive_pass_compact_7_warmup60_2020_2024_v1.json)

### Tushare canonical market replay 与 4 因子复核结论

按当前约定，公式复现的行情输入改用 `Tushare.daily`，YesLab 仅作为已存因子观测的对照
目标。4 个公式的实际计算预热窗口按嵌套滚动路径扩展为 180 个交易日，并在 64 只按代码
排序的受控股票样本上执行 2020-09、2022-01、2024-01、2025-01 四个窗口的只读重放。
该样本审计用于识别“计算器是否能重现源值”，不是对全市场发布的替代。

结果稳定分成两组：

- `yeslab_438/GP_069` 和 `yeslab_451/GP_082`：4 个窗口均为
  `FORMULA_REPLAY_PASS`，有限重叠约 `94%–100%`；
- `yeslab_398/GP_029` 和 `yeslab_464/GP_095`：4 个窗口均为
  `FORMULA_REPLAY_BLOCKED`，每个窗口的有限重叠几乎全部 mismatch，最大相对差异可达
  数十至数百倍。

因此，GP_029/GP_095 的问题不能再归因于 YesLab `amount` 小数精度；即使直接使用
Tushare 行情，当前源文件计算器仍无法重现历史因子值，继续隔离。GP_069/GP_082 只具备
“可复现候选”证据，仍需完成全市场分块重放后才能进入正式 OOS 候选池。

2025 固定 TRAIN 方向（4 个方向均为 `-1`）的只读验证已完成：每个因子 237 个有效日，
日期为 2025-01-02–2025-12-23，4 个因子均通过双重 predictive 门槛：

| 因子 | mean RankIC | ICIR | signed mean RankIC | 方向日占比 |
| --- | ---: | ---: | ---: | ---: |
| `GP_029` | -0.088695 | -0.843199 | 0.088695 | 0.801688 |
| `GP_069` | -0.085053 | -0.679709 | 0.085053 | 0.742616 |
| `GP_082` | -0.085063 | -0.679593 | 0.085063 | 0.742616 |
| `GP_095` | -0.066156 | -0.649646 | 0.066156 | 0.751055 |

验证结果不能解除公式复现阻断；当前正式 18 个候选保持不变，GP_069/GP_082 进入“待全市场
复现的 recovered review queue”，GP_029/GP_095 继续 blocked。没有写入 YesLab/V2 新表，
也没有启动模型训练。

相关产物：

- [`Tushare canonical 复现脚本`](../../research_v2/scripts/audit_formula_replay_tushare.py)
- [`2020-09 Tushare 复现`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered4_tushare_sample64_2020-09-14_2020-09-25_v2.json)
- [`2022-01 Tushare 复现`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered4_tushare_sample64_2022-01-04_2022-01-31_v1.json)
- [`2024-01 Tushare 复现`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered4_tushare_sample64_2024-01-02_2024-01-31_v1.json)
- [`2025-01 Tushare 复现`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered4_tushare_sample64_2025-01-02_2025-01-31_v1.json)
- [`4 因子 Tushare 多窗口汇总`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered4_tushare_sample64_multiwindow_v1.json)
- [`4 因子 TRAIN 方向来源`](../../research_v2/artifacts/diagnostics/train_structural_recovered4_direction_source_v1.json)
- [`4 因子 2025 验证汇总`](../../research_v2/artifacts/diagnostics/structural_recovered4_validation_2025_v1/quality_summary.csv)

### GP_069/GP_082 全市场复现与相似度复核

随后将 Tushare 复现从 64 只样本提升为按股票批次的全市场审计，并将代码集合限制为实际
训练输入 `universe_daily_latest.is_member=true`。4 个窗口的 V2 universe 代码均实现完整
Tushare 覆盖，且两个因子在所有窗口都为 `FORMULA_REPLAY_PASS`、`mismatch_rows=0`：

| 窗口 | universe 代码数 | Tushare 行数 | GP_069/GP_082 |
| --- | ---: | ---: | --- |
| 2020-09-14–25 | 2,746 | 581,716 | PASS / PASS |
| 2022-01-04–31 | 3,337 | 733,375 | PASS / PASS |
| 2024-01-02–31 | 4,084 | 873,646 | PASS / PASS |
| 2025-01-02–31 | 4,136 | 897,229 | PASS / PASS |

不限制 V2 universe 时出现的 7 个缺失代码全部是 `.BJ`，且不属于实际训练集合；这次显式
限制 universe 后 `market_missing_code_count=0`，没有把源表外代码静默排除。

全日 TRAIN 相似度矩阵进一步显示，20 个因子（原 18 个候选加 GP_069/GP_082）形成一个
高相似度连通簇：

- `GP_069/GP_082`：两者绝对 Spearman 中位数 `0.999999`；
- `yeslab_431/GP_069`：`0.907429`；
- `yeslab_431/GP_082`：`0.907413`。

按 `0.90` 相似度门槛，三者属于同一簇。`yeslab_431` 已在当前正式 18 因子候选中，且
2025 固定方向 signed mean RankIC 为 `0.088039`，高于 GP_069 的 `0.085053` 和 GP_082
的 `0.085063`，因此保留 `yeslab_431` 作为簇代表，不扩充候选池。当前正式候选仍为 18
个；这两个 recovered 因子已经完成质量闭环，但因冗余门禁不新增输入维度。

证据：

- [`全市场 Tushare 多窗口汇总`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_recovered2_tushare_full_v2universe_multiwindow_v1.json)
- [`全市场复现脚本`](../../research_v2/scripts/audit_formula_replay_tushare.py)
- [`TRAIN 全日相似度汇总`](../../research_v2/artifacts/diagnostics/train_similarity_full_day_oos18_recovered2_2020_2024_v1.csv)
- [`TRAIN 全日相似度 pair 矩阵`](../../research_v2/artifacts/diagnostics/train_similarity_pairs_oos18_recovered2_2020_2024_v1.csv)
- [`recovered 因子相似度审计结论`](../../research_v2/artifacts/diagnostics/train_similarity_recovered2_full_day_review_v1.json)

### 低覆盖因子的输入缺失与 turnover overlay 复核

对剩余 16 个结构失败、但 TRAIN predictive-pass 的因子，按计算脚本的
`required_factors` 做了输入字段审计。2020–2024 的 `stock_factor_daily` 结果为：

- `amount/vol/open/high` 的整体有限值比例为 `1.0`；
- `pct_chg` 的整体有限值比例为 `0.9992458`；
- `turnover_rate` 的整体有限值比例为 `0.9911484`，唯一低于 70% 的交易日是
  `2020-09-18`，有限值为 `2,200/4,069`。

这说明 13 个 GP turnover 因子和 `日换手率波动偏离` 确实存在一个可定位的行情输入缺口，
但缺口并不自动意味着因子可以修复。使用 Tushare `daily_basic.turnover_rate` 对该输入做
只读 overlay 后，15 个因子在 2020-09-14–25 的受控重放中只有 `GP_002、GP_020、GP_045`
没有有限源值 mismatch；将这 3 个提升到 V2 实际训练 universe 的全市场分块审计后，仍然
只有约 `26.1% / 32.8% / 66.5%` 的有限覆盖，低于当前 `70%` 结构门槛。因此这次 overlay
没有生成任何可发布修复候选。

其余 12 个因子在 overlay 后仍有全窗口 mismatch 或没有有限重叠；其中不依赖 turnover 的
`GP_021/GP_035` 也被当前计算器复现门禁阻断，说明它们不是单纯的 turnover 缺口问题。

证据：

- [`stock_factor_daily 输入覆盖`](../../research_v2/artifacts/diagnostics/stock_factor_daily_input_coverage_2020_2024_v1.json)
- [`15 个低覆盖因子 Tushare overlay`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_lowcoverage15_tushare_basic_sample64_2020-09-14_2020-09-25_v2.json)
- [`GP_002/GP_020/GP_045 全市场 overlay`](../../research_v2/artifacts/diagnostics/formula_replay_consistency_turnover3_tushare_full_v2universe_2020-09-14_2020-09-25_v1.json)
- [`低覆盖因子 overlay 结论`](../../research_v2/artifacts/diagnostics/low_coverage_turnover_overlay_review_v1.json)

本轮结论是：不降低有限覆盖门槛，不把 Tushare overlay 直接写入旧表；16 个因子继续隔离，
后续优先转向重新定义公式或重新生成源因子，而不是继续做行情字段填补。

### GP_002/GP_020 signed-power 公式变体探索

进一步检查计算脚本后发现：

- `GP_002 = pow(ts_regression_slope_60(vol), turnover_rate)`；
- `GP_020 = pow(pct_chg, ts_sum_10(turnover_rate)) + pct_chg`。

两者都会在负底数与非整数指数的组合下产生 NaN，这解释了为什么底层行情字段基本完整，
因子有限覆盖却只有约 39%。在不覆盖旧因子值的前提下，做了 signed-power 变体探索：
`sign(x) * abs(x)^p`。2020-09 全市场 smoke 中，GP_002/GP_020 的候选有限覆盖分别
提升到 `80.87%/82.54%`，但与已有有限源值仍有 `110/642` 个 mismatch。因此它们不是
可直接填缺的修复，而是需要重新生成全量因子的新公式版本。

2025-01-02–2025-01-23 的 16 个交易日 predictive smoke（方向固定为 `-1`）结果为：

| 变体 | signed mean RankIC | 原始 ICIR | 方向日占比 |
| --- | ---: | ---: | ---: |
| GP_002 signed-power | `0.062903` | `-0.526213` | `68.75%` |
| GP_020 signed-power | `0.064346` | `-0.383737` | `75.00%` |

这只能作为变体筛查证据，不能替代完整 5 年训练、1 年验证和 1 年测试。当前不发布新版本、
不改变 18 个正式候选，下一步需要在隔离的新因子版本中完成完整 RankIC/ICIR、相似度和
样本外门禁。

证据：

- [`signed-power 覆盖与源值 smoke`](../../research_v2/artifacts/diagnostics/signed_power_repair_smoke_2020-09-14_2020-09-25_v1.json)
- [`signed-power 2025 predictive smoke`](../../research_v2/artifacts/diagnostics/signed_power_repair_validation2025_v1/quality_summary.json)

### Signed-power 全周期 walk-forward（Tushare 口径）

按 `5 年训练（2020-01-01–2024-12-31）+ 1 年验证（2025-01-01–2025-12-31）`，
原始 5 日复权收益 `raw_return`，并对每个阶段末尾 purged 6 个交易日，完成了
GP_002/GP_020 signed-power 变体的完整日度 RankIC/ICIR 评估。`vol/pct_chg` 来自
YesLab 中由 Tushare 导入的 `stock_factor_daily`，`turnover_rate` 使用直接 Tushare
`daily_basic` 覆盖；没有读取 2026、没有写数据库。

| 变体 | 训练 signed mean RankIC | 训练 ICIR | 训练方向日占比 | 2025 signed mean RankIC | 2025 ICIR | 2025 方向日占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yeslab_371` / GP_002 | `0.037042` | `-0.323795` | `63.35%` | `0.052701` | `-0.454263` | `70.46%` |
| `yeslab_389` / GP_020 | `0.014305` | `-0.099958` | `54.31%` | `0.025377` | `-0.183533` | `56.12%` |

有限覆盖也不是主要问题：训练期 GP_002/GP_020 的最低日覆盖分别为 `96.36%/98.93%`，
验证期最低日覆盖为 `98.22%/99.32%`。但按当前双重门槛（`abs(mean RankIC) >= 0.02`、
`abs(ICIR) >= 0.50`、有效日数至少 60、方向日占比至少 55%），两个变体都因 ICIR
不足 `0.50` 未通过；GP_020 训练期的 signed mean RankIC 也不足 `0.02`。因此本轮
不发布新因子版本、不扩充正式候选池，变体仅保留为后续公式重构的研究证据。

证据：

- [`walk-forward metadata`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v1/metadata.json)
- [`训练摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v1/train_summary.csv)
- [`2025 验证摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v1/validation_summary.csv)
- [`训练日度覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v1/train_daily_rank_ic_with_coverage.csv)
- [`2025 日度覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v1/validation_daily_rank_ic_with_coverage.csv)

### GP_026 全横截面复现复核与 GP_021 signed-power 变体

为排除 GP_026 因子包含 `cs_rank` 而被 256 只分批复现器误阻断的可能性，新增了允许隔离
因子显式进入 Tushare 重放的审计入口，并用完整 V2 训练宇宙单批（`2,796` 只证券）重算
2020-09-14–25。结果仍有 `19,438/26,464` 个有限重叠 mismatch，最大相对差异约
`119.5`，因此 GP_026 的阻断不是分批横截面造成，继续禁止行情 overlay 或版本发布。

GP_021 的原式为 `pow(pct_chg, cs_rank(vol)) + ts_delta_ratio_60(amount)`。由于负的
`pct_chg` 与 0–1 分数指数会产生非法定义域，新增 signed-power 变体：
`sign(pct_chg) * abs(pct_chg)^cs_rank(vol) + ts_delta_ratio_60(amount)`，并以 Tushare
导入行情和原始 5 日收益完成完整 walk-forward：

| 变体 | 训练 signed mean RankIC | 训练 ICIR | 训练方向日占比 | 2025 signed mean RankIC | 2025 ICIR | 2025 方向日占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yeslab_390` / GP_021 signed-power | `0.044670` | `-0.359040` | `63.85%` | `0.063484` | `-0.542591` | `72.15%` |

GP_021 的训练/验证有效日数为 `1,206/237`，训练期最低有限覆盖 `99.36%`，验证期最低
有限覆盖 `99.63%`。验证期达到双重门槛，但训练期绝对 ICIR 仍低于 `0.50`，因此不发布
新因子版本、不加入正式候选池；它保留为“公式修复后预测性改善、稳定性不足”的后续研究
候选。GP_026 仍为 `FORMULA_REPLAY_BLOCKED`，GP_021 变体和前述 GP_002/GP_020 变体
均不改变当前正式候选或模型训练状态。

证据：

- [`GP_026 全横截面 Tushare 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp026_tushare_full_unbatched_2020-09_v1.json)
- [`GP_021/GP_035 原式全横截面 Tushare 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp021_gp035_tushare_full_unbatched_2020-09_v1.json)
- [`signed-power v2 metadata`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v2/metadata.json)
- [`signed-power v2 训练摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v2/train_summary.csv)
- [`signed-power v2 验证摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v2/validation_summary.csv)
- [`signed-power v2 训练覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v2/train_daily_rank_ic_with_coverage.csv)
- [`signed-power v2 验证覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v2/validation_daily_rank_ic_with_coverage.csv)

### GP_001/GP_003 signed-power 扩展筛查

继续检查剩余低覆盖 GP 因子后，GP_001/GP_003 也确认存在负底数分数幂：

- GP_001：`pow(pct_chg, ts_sum_10(turnover))`；
- GP_003：`pow(pct_chg, ts_delta_ratio_20(turnover))`。

新增 signed-power 变体并完成同一 Tushare walk-forward 后，结果如下：

| 变体 | 训练 signed mean RankIC | 训练 ICIR | 训练方向日占比 | 2025 signed mean RankIC | 2025 ICIR | 2025 方向日占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yeslab_370` / GP_001 | `0.036254` | `-0.278` | `61.77%` | `0.033627` | `-0.260` | `62.45%` |
| `yeslab_372` / GP_003 | `0.046356` | `-0.359` | `64.76%` | `0.042334` | `-0.303` | `64.56%` |

两者有限覆盖均明显改善（训练最低覆盖分别 `98.93%/94.68%`，验证最低覆盖分别
`99.32%/95.30%`），但 ICIR 在训练和验证均未达到 `0.50`。这说明定义域修复改善了
可用样本，却没有形成稳定预测增益；不发布新版本、不扩充正式输入，后续不再对这两式做
无约束调参。

同时，GP_021/GP_035 使用源因子实际证券集合的单批 Tushare 重放仍分别有
`15,929/17,288`、`14,101/32,409` mismatch，横截面 universe 漂移也不能解释全部差异。

证据：

- [`signed-power v3 metadata`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v3/metadata.json)
- [`signed-power v3 训练摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v3/train_summary.csv)
- [`signed-power v3 验证摘要`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v3/validation_summary.csv)
- [`signed-power v3 训练覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v3/train_daily_rank_ic_with_coverage.csv)
- [`signed-power v3 验证覆盖`](../../research_v2/artifacts/diagnostics/signed_power_walk_forward_tushare_v3/validation_daily_rank_ic_with_coverage.csv)
- [`GP_021/GP_035 源 universe 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp021_gp035_tushare_source_universe_2020-09_v1.json)

### 正式 OOS 候选的 Tushare canonical replay

将当前 18 个 OOS 候选直接切换到 Tushare `daily + daily_basic`，使用完整 V2 宇宙和单批
横截面重放。2025-01 窗口只有 4 个因子逐键无 mismatch：
`yeslab_163/Alpha95`、`yeslab_380/GP_011`、`yeslab_431/GP_062`、`yeslab_437/GP_068`；
其余 14 个在该窗口即被 `FINITE_OVERLAP_MISMATCH` 阻断，不能把 YesLab 派生值直接当作
Tushare 口径的可复现输入。

对这 4 个通过者再执行 2020-01、2022-01、2024-01 三个窗口，四者全部保持
`FORMULA_REPLAY_PASS`，且每个窗口 `market_missing_code_count=0`。因此当前形成了一个
更严格的 Tushare canonical 核心复核集合，但这不是对现有 18 因子表的删除或发布操作；
现有候选仍保留在提案中，任何新准入审查必须先通过这道行情一致性门槛。

| Tushare canonical 状态 | 因子数 | 因子 |
| --- | ---: | --- |
| 四窗口全部 PASS | 4 | `yeslab_163, yeslab_380, yeslab_431, yeslab_437` |
| 2025-01 即 BLOCKED | 14 | `yeslab_114, 155, 376, 381, 385, 396, 400, 405, 425, 442, 94, 95, 96, 98` |

这 4 个核心因子仍需结合年度稳定性、相似度和最终准入规则审阅，尚未冻结，也没有启动
模型训练。数据库写入为 `false`，没有读取 2026。

证据：

- [`Tushare canonical 汇总`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_canonical_summary_v1.json)
- [`2025-01 18 因子重放`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_2025-01_v1.json)
- [`2020-01 4 因子重放`](../../research_v2/artifacts/diagnostics/formula_replay_oos4_tushare_full_2020-01_v1.json)
- [`2022-01 4 因子重放`](../../research_v2/artifacts/diagnostics/formula_replay_oos4_tushare_full_2022-01_v1.json)
- [`2024-01 4 因子重放`](../../research_v2/artifacts/diagnostics/formula_replay_oos4_tushare_full_2024-01_v1.json)

### Tushare canonical 核心因子的 predictive/年度复核

对四个四窗口重放通过者，使用现有 V2 `factor_rank_ic_daily` 与 `raw_return` 结果做训练和
2025 验证汇总。四个因子的全 TRAIN ICIR 绝对值为 `0.509–0.593`，2025 验证 ICIR 绝对值
为 `0.572–0.780`，方向全部稳定为 `-1`；因此它们是当前最可信的 Tushare 口径候选。

| 因子 | TRAIN signed mean RankIC | TRAIN ICIR | 2025 signed mean RankIC | 2025 ICIR | 年度 ICIR 达标年数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `yeslab_163` / Alpha95 | `0.06655` | `-0.544` | `0.07274` | `-0.589` | `3/5` |
| `yeslab_380` / GP_011 | `0.08167` | `-0.593` | `0.08812` | `-0.572` | `4/5` |
| `yeslab_431` / GP_062 | `0.06952` | `-0.556` | `0.08804` | `-0.770` | `3/5` |
| `yeslab_437` / GP_068 | `0.06685` | `-0.509` | `0.08868` | `-0.780` | `2/5` |

如果把“每个年度 ICIR 必须达到 0.50”作为硬门槛，四个因子都不能直接冻结；如果采用主
计划的“至少三年同向、年度均值和方向占比达标”稳定性规则，四个因子可继续进入相似度、
PIT、FDR 和准入决策审阅。当前不修改原 18 因子提案、不写旧表、不启动模型训练。

证据：[`Tushare canonical 核心质量汇总`](../../research_v2/artifacts/diagnostics/tushare_canonical_core_quality_v1.json)

四个核心因子的 TRAIN 和 2025 验证期六个两两相似度均低于 `0.90`，不存在需要再删减的
内部冗余边。由此形成两种透明的准入分支：按主计划“至少 3 年稳定”可暂列
`yeslab_163/380/431`，`yeslab_437` 进入监控；按严格“五年每年 ICIR≥0.50”则四个都只能
监控。该分支尚未代替既有规则，提案仅供批准后执行。

证据：[`Tushare canonical 准入提案`](../../research_v2/artifacts/diagnostics/tushare_canonical_admission_proposal_v1.json)

### 跳空因子 `pre_close` 语义补救审计

对 `yeslab_155 / 跳空去均值绝对值` 检查了一个明确的计算器语义问题：当前计算器以
`close.shift(1)` 代替 Tushare `pre_close`。新增候选公式
`abs(open / pre_close - cross_section_mean(open / pre_close))` 后，2025-01 在 V2 宇宙中
仍有 `75,792/75,792` 个有限重叠 mismatch；再换用 YesLab 源因子实际证券集合重放，仍有
`92,273/92,274` mismatch，并且 Tushare 对源集合缺少 `248` 个证券。

结论是：该因子还存在源生成横截面/历史输入语义漂移，不能把 `pre_close` 变体当作安全
补值或新版本候选；旧因子继续隔离，后续需追溯生成时的完整证券集合与行情快照。

证据：

- [`pre_close 变体审计`](../../research_v2/artifacts/diagnostics/formula_repair_preclose_variant_2025-01_v1.json)
- [`源 universe 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp155_tushare_source_universe_2025-01_v1.json)

对含横截面 z-score 的 `GP_007/GP_012/GP_016` 也做了源因子证券集合重放，2025-01 的
有限重叠 mismatch 分别为 `91,941/91,945`、`92,150/92,150`、`92,150/92,150`；
即使放宽到源集合仍无法重现，继续归类为历史输入/生成语义阻断，不做行情填补。

证据：[`GP_007/012/016 源 universe Tushare 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp007_gp012_gp016_tushare_source_universe_2025-01_v1.json)

最后用四窗口 replay artifact 和核心质量摘要实际调用了新的 canonical 选择 API，返回四个
因子，且每个因子的 `canonical_replay_window_count=4`、选择原因均为
`DUAL_GATE_STABLE_DIRECTION_TUSHARE_CANONICAL`。这只是选择执行证据，不代表已发布或冻结；
年度政策分支仍需单独决定。

证据：[`canonical 选择执行结果`](../../research_v2/artifacts/diagnostics/tushare_canonical_selection_execution_v1.json)

### YesLab 与直接 Tushare 的 replay 分歧分类

对 2025-01 的 18 个候选做了来源交叉诊断：使用 YesLab `stock_factor_daily` 和源 universe
时，18 个当前计算器全部可以重现存储因子；切换到直接 Tushare `daily/daily_basic` 后，
只有 4 个仍然通过。由此，剩余 14 个不能简单归类为“计算器写错”，更准确的分类是
`MARKET_SNAPSHOT_OR_UNIVERSE_DIVERGENCE`：历史因子生成时使用的行情快照、证券集合或字段
修订，与当前直接 Tushare 口径不一致。

处置原则已固定：不把直接 Tushare 重算值覆盖到 YesLab；新准入必须先通过 canonical replay；
剩余 14 个只有在恢复历史输入快照后才允许重新评估。

证据：[`YesLab/Tushare replay 分歧汇总`](../../research_v2/artifacts/diagnostics/tushare_vs_yeslab_replay_divergence_oos18_v1.json)

### 分批横截面与原始行情字段复核

进一步检查了回放实现本身：此前 `full-market` 默认按 256 只股票分批，含
`cs_mean/cs_zscore/cs_rank` 的因子可能被错误地按批次计算。将 GP_007/012/016 改为 V2
证券集合的单批完整横截面，并自动加载其声明的 `turnover_rate`（来自 `daily_basic`）后，
三者仍分别有 `75,619/75,619`、`75,720/75,720`、`75,720/75,720` 个有限重叠 mismatch，
因此本轮阻断不是 256 分批造成的假阻断。

同时对 2024-11-18 至 2025-01-31 的原始字段做了只读对账。重叠证券的 `open/high/low/close/
pre_close/pct_chg/vol/turnover_rate` 在浮点容差内一致；`amount` 最大差异约 `0.005`，属于
小数精度差异。当前真正影响横截面重放的是 Tushare 缓存相对 YesLab 少了 `248` 个证券，且
全部是 `.BJ`；YesLab 对这些证券仍有完整的 50 日原始行情。由此，剩余候选继续标记为
历史证券集合/生成快照语义阻断，不能用当前值直接补写或发布。

证据：

- [`GP_007/012/016 单批 canonical 重放`](../../research_v2/artifacts/diagnostics/formula_replay_gp007_gp012_gp016_tushare_single_batch_v2_2025-01_fixed.json)
- [`横截面单批规则 smoke`](../../research_v2/artifacts/diagnostics/formula_replay_gp007_tushare_auto_full_cross_section_2025-01_fixed.json)
- [`YesLab/Tushare 原始行情字段对账`](../../research_v2/artifacts/diagnostics/tushare_yeslab_market_field_parity_2024-11-18_2025-01-31_v1.json)

用同一规则补跑其余 14 个候选后，仍为 `14/14 BLOCKED`。但 mismatch 密度可用于安排
补救优先级：

| 因子 | mismatch / 有限重叠 | 最大绝对差 | 当前判断 |
| --- | ---: | ---: | --- |
| `yeslab_94` | `1,722 / 91,606` | `5.4668` | 低 mismatch，优先追 BJ 快照后复核 |
| `yeslab_98` | `2,057 / 91,606` | `0.04545` | 低 mismatch，优先追 BJ 快照后复核 |
| `yeslab_95` | `29,978 / 91,606` | `1.62303` | 中等 mismatch，暂阻断 |
| `yeslab_96` | `31,162 / 91,606` | `5.4668` | 中等 mismatch，暂阻断 |
| `yeslab_400` | `49,166 / 90,139` | `2.99353` | 低覆盖且阻断 |
| `yeslab_425` | `63,645 / 91,945` | `0.4` | 中高 mismatch，暂阻断 |
| 其余 8 个 | `≥90,000` | 依公式而异 | 历史横截面/输入语义优先排查 |

所有行同时带有 `TUSHARE_MARKET_MISSING_CODES`，缺失证券数为 `248` 且全部为 `.BJ`；上述
低 mismatch 不是准入通过，只表示恢复完整 BJ 快照后更值得优先复核。

证据：[`14 个候选单批 canonical 重放`](../../research_v2/artifacts/diagnostics/formula_replay_oos14_tushare_full_single_batch_2025-01_fixed.json)

针对低 mismatch 的 `yeslab_94` 和 `yeslab_98`，做了一个不参与准入的诊断性补齐实验：仅将
Tushare 缺失的 248 个 BJ 证券用 YesLab 原始行情补入横截面，其他证券仍使用 Tushare。两者
都恢复为 `0` mismatch，最大绝对误差分别为 `1.8e-15` 和 `1.1e-17`，证明当前差异由 BJ
横截面缺口解释，而不是计算器公式差异。正式状态仍保持 BLOCKED；待 Tushare 能提供完整 BJ
日行情后，94/98 是第一优先级复核对象。

证据：[`94/98 BJ 补齐诊断`](../../research_v2/artifacts/diagnostics/formula_replay_bj_supplement_diagnostic_94_98_2025-01_v1.json)

随后将补齐实验扩大到全部 14 个阻断候选，并使用完整 `stock_factor_daily` 原始证券集合
`5,396` 个代码作为横截面。14 个因子全部恢复为 `0 mismatch`；这包括此前几乎全量阻断的
`yeslab_114`、`yeslab_381`、`yeslab_385`、`yeslab_400`、`yeslab_425` 和 `yeslab_442`。
因此当前证据链已经足以把这 14 个候选统一归因于 Tushare 当前交易日接口缺少 BJ 证券，而
不是逐个公式需要修补。补齐值仍仅来自 YesLab 诊断，不能用于发布或替代 Tushare canonical
输入。

为解决这个数据覆盖问题，canonical 获取器已增加严格的缺失代码检测：当按交易日接口整段
缺少证券时，会尝试用 Tushare 按 `ts_code + start_date/end_date` 补拉 `daily`，需要换手率
时同步补拉 `daily_basic`；补拉后仍缺失的代码继续触发 `TUSHARE_MARKET_MISSING_CODES`，不
会回退到 YesLab。

证据：

- [`14 因子 BJ 补齐诊断`](../../research_v2/artifacts/diagnostics/formula_replay_bj_supplement_diagnostic_oos14_2025-01_v1.json)
- [`Tushare 缺失代码补拉单测`](../../research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py)

### BJ 代码迁移后的多窗口 canonical 结果

将 BJ 迁移映射按窗口独立生成后，2024-01 和 2025-01 的 18 个候选均达到
`18/18 FORMULA_REPLAY_PASS`，且 `market_missing_code_count=0`。2020-01 和 2022-01 的
可覆盖证券上公式误差也已消失，但 Tushare 分别缺少 9 和 16 个历史证券，故严格门禁仍为
`18/18 BLOCKED`；这不是通过降低覆盖门槛可以解决的问题。

| 窗口 | BJ 映射数 | Tushare 覆盖缺口 | 通过 / 阻断 | 处置 |
| --- | ---: | ---: | ---: | --- |
| 2020-01 | 108 | 9 | `0 / 18` | 等待历史 Tushare 行情 |
| 2022-01 | 180 | 16 | `0 / 18` | 等待历史 Tushare 行情 |
| 2024-01 | 243 | 0 | `18 / 0` | 可进入年度质量审查 |
| 2025-01 | 263 | 0 | `18 / 0` | 可进入年度质量审查 |

因此当前不发布新增 14 因子，也不启动模型训练；保留原先 4 个四窗口通过因子作为候选，
同时把新增候选列入“公式已验证、历史 Tushare 覆盖未完成”的待复核集合。

证据：

- [`BJ 迁移审计脚本`](../../research_v2/scripts/audit_bj_code_migration.py)
- [`BJ 迁移审计单测`](../../research_v2/tests/unit/scripts/test_audit_bj_code_migration.py)
- [`2025-01 18 因子 alias canonical`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_2025-01_v1.json)
- [`四窗口 alias 汇总`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_bj_alias_window_summary_v1.json)

### 因子源与 YesLab 原始行情覆盖审计

补充审计了因子源有限值是否能在 YesLab `stock_factor_daily` 找到同一业务键。除 Alpha95
外，18 个候选在 2020/2022/2024/2025 四个窗口均没有有限值 orphan；Alpha95 存在：

| 窗口 | Alpha95 有限值 orphan 行 | orphan 证券数 | 解释 |
| --- | ---: | ---: | --- |
| 2020-01 | 750 | 119 | YesLab 原始行情缺行，但因子表仍有有限值 |
| 2022-01 | 875 | 117 | YesLab 原始行情缺行，但因子表仍有有限值 |
| 2024-01 | 128 | 18 | 同上 |
| 2025-01 | 215 | 38 | 同上 |

这项审计描述的是 YesLab 存储完整性，不替代 Tushare canonical 结果：2024/2025 的
Tushare alias replay 仍然是 `18/18 PASS`，说明这些 Alpha95 值可以由 Tushare 原始行情
重算；但如果未来不走 Tushare canonical、直接依赖 YesLab 原始表重算，Alpha95 必须先
阻断并修复缺行。

证据：

- [`2020 因子-原始行情覆盖`](../../research_v2/artifacts/diagnostics/factor_raw_market_coverage_oos18_2020-01_v1.json)
- [`2022 因子-原始行情覆盖`](../../research_v2/artifacts/diagnostics/factor_raw_market_coverage_oos18_2022-01_v1.json)
- [`2024 因子-原始行情覆盖`](../../research_v2/artifacts/diagnostics/factor_raw_market_coverage_oos18_2024-01_v1.json)
- [`2025 因子-原始行情覆盖`](../../research_v2/artifacts/diagnostics/factor_raw_market_coverage_oos18_2025-01_v1.json)

另外做了一个 V2 universe 对照：2020/2022 在 V2 限制下虽然可以把 Tushare 缺失代码降为
`0`，但横截面因子仍大面积 mismatch。这证明 V2 universe 不能用于公式重放，否则会把
“生成时完整股票池”改成“投资时股票池”，误判公式不一致。最终规则固定为：公式 replay 使用
源因子生成的完整证券集合；模型训练和组合约束才使用 V2 universe。

证据：

- [`2020 V2 universe 对照`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_v2universe_2020-01_v1.json)
- [`2022 V2 universe 对照`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_v2universe_2022-01_v1.json)

### Tushare canonical 源因子覆盖门禁

本轮又增加了一道独立门禁：回放前读取同一窗口的完整源因子有限值，检查其
`(trade_date, ts_code)` 是否都能在 canonical Tushare 行情集合中找到。这样可以把“行情
集合缺证券”和“公式数值不一致”分开记录，不能只看有限重叠样本的误差。

带 BJ 迁移映射的最新结果如下：

| 窗口 | Tushare 缺失代码 | 源因子覆盖缺口 | 18 个因子结果 | 说明 |
| --- | ---: | ---: | --- | --- |
| 2020-01 | 9 | `yeslab_163`：75 行、9 个代码 | `0 / 18 PASS` | 严格阻断，待历史 Tushare 行情 |
| 2022-01 | 16 | `yeslab_163`：174 行、16 个代码 | `0 / 18 PASS` | 严格阻断，待历史 Tushare 行情 |
| 2024-01 | 0 | 0 | `18 / 18 PASS` | 可进入年度质量审查 |
| 2025-01 | 0 | 0 | `18 / 18 PASS` | 可进入年度质量审查 |

缺口只在 `yeslab_163 / Alpha95` 出现，是因为该因子在两个历史窗口对缺失证券仍有有限
源值；其他因子的源有限值都落在当前 Tushare 可覆盖代码内，但仍因完整横截面缺失而整体
阻断。新增原因码为 `TUSHARE_SOURCE_MARKET_COVERAGE_GAP`，不会用 YesLab 行情或横截面
填充值绕过门禁。

证据：

- [`2020-01 最新 canonical 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_2020-01_v2.json)
- [`2022-01 最新 canonical 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_2022-01_v2.json)
- [`2024-01 最新 canonical 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_2024-01_v2.json)
- [`2025-01 最新 canonical 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_full_bj_alias_2025-01_v2.json)

### 历史 BJ 代码的 Tushare 端点可用性审计

为确认历史缺口不是单一 `daily` 接口造成，针对 2020-01 的 9 个缺失代码和 2022-01 的
16 个缺失代码，逐代码只读检查了 Tushare `daily`、`daily_basic`、`stk_factor` 三个端点。
结果是：25 个代码在目标回放窗口内的 `daily` 和 `daily_basic` 均为 0 行；`stk_factor`
虽能返回部分迁移/停牌前记录，但都在回放窗口开始前结束，不能覆盖因子源有限值所在日期。

| 窗口 | 缺失代码数 | `daily` 有完整窗口 | `daily_basic` 有完整窗口 | `stk_factor` 情况 | 结论 |
| --- | ---: | ---: | ---: | --- | --- |
| 2020-01 | 9 | 0 | 0 | 9 个均仅有 2019 年零散记录 | 无法补齐 2020-01 |
| 2022-01 | 16 | 0 | 0 | 16 个均最晚停在 2021-12 | 无法补齐 2022-01 |

因此本项目不会把 `stk_factor` 的不完整历史记录拼接成 canonical 行情，也不会用 YesLab
原始行情替代 Tushare。2020/2022 的严格阻断是当前 Tushare 可用历史数据的客观限制，不是
通过降低覆盖门槛或改变模型输入可以解决的问题；只有获得完整的 Tushare 历史快照后，才会
重新执行四窗口公式回放。

证据：

- [`2020-01 Tushare 端点覆盖审计`](../../research_v2/artifacts/diagnostics/tushare_bj_legacy_endpoint_coverage_2020-01_v1.json)
- [`2022-01 Tushare 端点覆盖审计`](../../research_v2/artifacts/diagnostics/tushare_bj_legacy_endpoint_coverage_2022-01_v1.json)
- [`端点覆盖审计脚本`](../../research_v2/scripts/audit_tushare_bj_legacy_coverage.py)

### 沪深范围的最新 canonical 结果

按新的研究范围，回放代码显式排除 `.BJ`，只保留 `.SH/.SZ`。此前的 18 因子四窗口文件是在
Tushare 行情拦截修正前生成的，只保留为历史对照，不能继续作为通过证据。修正后对正式九因子
重新执行了完整沪深单批横截面回放，覆盖门禁仍按 `(trade_date, ts_code)` 逐键检查：

| 窗口 | 沪深代码数 | Tushare 市场代码数 | 严格通过数 | 通过因子 |
| --- | ---: | ---: | ---: | --- |
| 2020-01 | 2,899 | 2,899 | 1 | `yeslab_437` |
| 2022-01 | 3,410 | 3,410 | 2 | `yeslab_380`, `yeslab_437` |
| 2024-01 | 4,125 | 4,125 | 0 | 无 |
| 2025-01 | 4,220 | 4,220 | 1 | `yeslab_437` |

严格四窗口交集暂时为空：`yeslab_437 / GP_068` 在 2024-01 有 2 个 mismatch，
`yeslab_380 / GP_011` 在 2020-01 和 2025-01 各有少量 mismatch，`yeslab_431 / GP_062`
四个窗口也有少量 mismatch。其余因子的大规模 mismatch 表明 YesLab 历史值与 Tushare
canonical 重算值不同，不能将旧表值直接混入新训练样本；它们的 predictive 结论应只引用
Tushare 重新物化后的质量文件。该回放是源值一致性诊断，不会自动撤销已经通过的
`formula_scope_gate`，也不代表发布或模型输入冻结。

证据：

- [`2020-01 九因子修正版回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos9_tushare_shsz_fixed_2020-01_v1.json)
- [`2022-01 九因子修正版回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos9_tushare_shsz_fixed_2022-01_v1.json)
- [`2024-01 九因子修正版回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos9_tushare_shsz_fixed_2024-01_v1.json)
- [`2025-01 九因子修正版回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos9_tushare_shsz_fixed_2025-01_v1.json)
- [`旧版沪深 canonical（历史对照）`](../../research_v2/artifacts/diagnostics/formula_replay_oos18_tushare_shsz_2025-01_v1.json)

### 沪深三因子的 predictive 与冗余复核

V2 训练宇宙实际只包含沪深代码（`2,868` 个 `.SZ`、`2,245` 个 `.SH`），因此现有训练/验证
RankIC 证据无需再混入 BJ。对四窗口 canonical 交集的三个因子，当前 2020–2024 TRAIN 和
2025 验证结果如下：

| 因子 | TRAIN mean RankIC | TRAIN ICIR | 2025 mean RankIC | 2025 ICIR | 方向日占比 | 年度 ICIR 通过年数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yeslab_380 / GP_011` | -0.081672 | -0.592950 | -0.088117 | -0.571801 | 0.708861 | 4 |
| `yeslab_431 / GP_062` | -0.069519 | -0.555636 | -0.088039 | -0.770018 | 0.776371 | 3 |
| `yeslab_437 / GP_068` | -0.066848 | -0.509068 | -0.088676 | -0.779731 | 0.780591 | 2 |

三个因子均通过主双重 predictive 门槛，固定方向都是 `-1`；但若采用“五年每年
`abs(ICIR) >= 0.50`”的严格分支，三者都不能直接冻结，需继续监控年度稳定性。TRAIN
全日相似度中三者两两最高为 `0.842817`，低于 `0.90` 冗余门槛，因此暂不需要再删减。
这三因子是当前沪深范围的最小可复现候选集合，不等于已发布模型输入。

证据：

- [`canonical 核心质量`](../../research_v2/artifacts/diagnostics/tushare_canonical_core_quality_v1.json)
- [`TRAIN45 候选质量明细`](../../research_v2/artifacts/diagnostics/train45_oos_representative_quality_v3.csv)
- [`三因子 TRAIN 相似度 pair`](../../research_v2/artifacts/diagnostics/train_similarity_pairs_oos18_recovered2_2020_2024_v1.csv)

### 横截面因子的沪深重新物化验证

由于 `yeslab_94` 和 `yeslab_98` 含有同日横截面波动判断，旧 YesLab 值可能是在包含 BJ 的
股票池上生成。本轮不再拿旧值做对照，而是直接用 Tushare 行情在 V2 沪深训练 universe 中
重新计算，并与 `forward_open_5d` 标签做 2025 验证：

| 因子 | 沪深代码数 | 有效日数 | mean RankIC | ICIR | 方向日占比 | 结果 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `yeslab_94` | 4,911 | 243 | -0.055005 | -0.698123 | 0.761316 | 验证 predictive 通过 |
| `yeslab_98` | 4,911 | 243 | -0.061364 | -0.759793 | 0.781893 | 验证 predictive 通过 |

这证明两因子在沪深范围重新定义后仍有验证期预测性。随后按年度分段完成了
2020–2024 TRAIN 重算：94 的 TRAIN mean RankIC/ICIR 为 `-0.038897/-0.508468`，5 年中
有 3 年达到 `abs(ICIR)>=0.50`；98 为 `-0.043933/-0.566906`，5 年全部达到该门槛。
两者都通过主双重 predictive 门槛，98 还通过严格五年年度 ICIR 分支，94 仍需年度监控。
这只是新沪深版本的质量证据，尚未完成与现有候选的全量相似度/准入比较，因此不改变当前
三因子 canonical 交集，也不发布新版本。

证据：

- [`沪深重物化验证脚本`](../../research_v2/scripts/evaluate_tushare_scope_validation.py)
- [`94/98 沪深 2025 质量摘要`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_94_98_v1/quality_summary.json)
- [`94/98 沪深 2025 日度 RankIC`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_94_98_v1/daily_rank_ic.csv)
- [`94/98 沪深 TRAIN/验证汇总`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_94_98_v1.json)

随后对严格年度分支优先级最高的 `yeslab_155 / 跳空去均值绝对值` 做了同口径沪深重新物化。
其 2020–2024 TRAIN mean RankIC/ICIR 为 `-0.042868/-0.648516`，5 个年度全部通过
`abs(ICIR)>=0.50`；2025 验证 mean RankIC/ICIR 为 `-0.048908/-0.661084`，方向日占比
`0.761317`。该因子满足当前双重 predictive 和严格年度分支，但还未与三因子 canonical
集合做全日相似度比较，状态为 `REMATERIALIZED_CANDIDATE_PENDING_SIMILARITY`，不发布新版本。

证据：[`155 沪深 TRAIN/验证汇总`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_155_v1.json)

### 沪深重新物化因子的 TRAIN 相似度

将 94、98、155 与已通过 canonical replay 的 431、437 在沪深 TRAIN 中逐日重算并比较，
五个年度的“年度中位数再取中位数”结果全部低于 `0.90`：

| 因子对 | 中位绝对 Spearman |
| --- | ---: |
| `155 / 431` | 0.196105 |
| `155 / 437` | 0.225461 |
| `94 / 431` | 0.129841 |
| `94 / 437` | 0.129120 |
| `98 / 431` | 0.120616 |
| `98 / 437` | 0.121247 |
| `94 / 98` | 0.741102 |

因此 94、98、155 暂未因冗余被淘汰；155 的预测性和严格年度稳定性证据最好，优先进入
下一轮 SH/SZ 准入提案。由于 380 依赖换手率且 2021 年 Tushare `daily_basic` 缓存仍不完整，
本轮先使用 431/437 作为已重算 canonical 代表。

证据：[`沪深 TRAIN 相似度汇总`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_train_5_v1.json)

补充 pair 审计已完成：使用沪深重新物化的 155 与已通过四窗口 Tushare replay 的 380
存储值逐日比较，五年聚合中位绝对 Spearman 为 `0.253011929`，年度最高仅 `0.294436289`。
因此 155 与 380 也未触发冗余门槛；380 在此处只作为已通过 canonical replay 的对照因子，
不是行情数据来源。

证据：[`155/380 沪深 TRAIN 相似度`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_155_380_train_v1.json)

### 沪深准入候选提案（未发布）

基于当前证据，形成 6 个沪深范围待准入候选：现有 canonical 的 `380、431、437`，以及
按沪深 universe 重新物化的 `155、98、94`。这不是把旧因子表改成新版本：前三个保持现有
canonical 版本，后三个必须使用新的 `new_shsz_rematerialized_v1` 版本标识，不能覆盖旧值。

其中 155/98 通过五年严格年度 ICIR 分支，94 为 3/5 年通过、保留监控；所有已审相似度
pair 均低于 0.90。提案仍未发布、未写入旧表或新 V2 表，也未开始模型训练；下一步只做
版本标识审批和 admission preflight。

证据：[`沪深准入候选提案`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_proposal_v1.json)

### 扩展 16 因子的沪深重新物化质量

为避免把前一阶段的 `UNSCREENED` 误认为失败，已将 expanded shortlist 中已有源码计算器的
16 个因子统一按沪深 universe、Tushare 行情和 `forward_open_5d` 标签重新物化。六个此前未在
replay registry 中登记但源码已经存在的计算器也已补登记：`GP_008/018/019/022/063/064`；
这只是让它们可审查，不代表公式一致性门禁自动通过。

结果为：16 个因子 TRAIN 和 2025 VALIDATION 均有完整有效日数（分别 `1212`、`243`），主
predictive 门槛全部通过。TRAIN 严格年度 `abs(ICIR)>=0.50` 的 5 年通过因子只有
`yeslab_376`、`yeslab_396`、`yeslab_442`；`yeslab_377/400/433` 为 `3/5`，其余为 `4/5`，
因此不能仅凭整体 ICIR 把 16 个因子全部冻结。

2025 验证期相似度只作为风险定位：`yeslab_387/388=0.979414`、`yeslab_377/380=0.975552`、
`yeslab_380/387=0.975030`，说明换手/成交额类因子存在明显冗余簇。正式代表选择仍需使用
TRAIN 全窗口 pair 证据；本轮没有用验证期相似度直接删因子。

该扩展质量结果仍未改变 6 因子提案：`yeslab_433/GP_064` 保留既有公式重叠 mismatch
阻断，其他因子仍需 TRAIN 相似度、PIT/FDR 和版本审查。没有发布新版本或启动模型训练。

证据：

- [`16 因子质量聚合`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_16_v1.json)
- [`16 因子 TRAIN 质量原始摘要`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_16_quality_v1/quality_summary.json)
- [`16 因子 2025 验证摘要`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_16_v1/quality_summary.json)
- [`16 因子 2025 验证相似度定位`](../../research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_16_v1/similarity_summary.json)
- [`扩展计算器注册测试`](../../research_v2/tests/unit/scripts/test_audit_formula_replay_consistency.py)

### 扩展 GP 的四窗口公式回放与训练冗余簇

对新增注册的六个 GP 因子执行了同一口径的四个 SH/SZ Tushare 窗口回放
（2020-01、2022-01、2024-01、2025-01）。结果不是“都能重算就都可用”：

| 因子 | 四窗口回放 | 处置 |
|---|---:|---|
| `yeslab_387 / GP_018` | 2/4 PASS | 2020/2025 各有少量金额精度 mismatch，监控 |
| `yeslab_388 / GP_019` | 4/4 PASS | 可继续做准入审查 |
| `yeslab_391 / GP_022` | 4/4 PASS | 可继续做准入审查 |
| `yeslab_377 / GP_008` | 0/4 PASS | 源值重叠 mismatch，阻断 |
| `yeslab_432 / GP_063` | 0/4 PASS | 源值重叠 mismatch，阻断 |
| `yeslab_433 / GP_064` | 0/4 PASS | 少量但稳定 mismatch，阻断 |

在此前六因子的完整 TRAIN 日相似度中，`GP_008/011/018/019` 形成强冗余簇：
`387/388=0.981729`、`380/388=0.979991`、`380/387=0.977751`、`377/380=0.970053`。
`GP_022/GP_064` 也达到 `0.911591`，但 GP_064 先因公式 mismatch 被阻断。这个结果支持
“簇内选代表、先解决公式门禁”的策略。修正后的三因子审查显示，验证期 `GP_019/GP_011`
中位绝对 Spearman 为 `0.977406`，应作为 GP_011 的冗余监控项；`GP_022/GP_011` 的验证期
数值为 `0.887810`，但已有 TRAIN 全日 pair 为 `0.904058`，仍超过 `0.90` 冗余门槛。因此
387/388/391 都不形成新的代表因子，当前没有把这些诊断结果写成新的模型输入集合。

证据：

- [`六因子 TRAIN 相似度`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_cluster_train_v1.json)
- [`六因子四窗口公式回放汇总`](../../research_v2/artifacts/diagnostics/tushare_shsz_registered_gp_replay_train_windows_v1.json)
- [`2025-01 注册 GP 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos6_tushare_shsz_registered_2025-01_v1.json)
- [`三因子修正版四窗口回放`](../../research_v2/artifacts/diagnostics/tushare_shsz_registered_gp_quality_review_v1.json)
- [`GP_008/063/064 修正版四窗口审查`](../../research_v2/artifacts/diagnostics/tushare_shsz_registered_gp_scope_precision_review_v1.json)

### 横截面范围与 GP_064 精度根因

进一步读取 2025-01-02 的 YesLab 源池后确认，`GP_008` 的源横截面有 `5,952` 个唯一代码，
其中 `551` 个为 `.BJ`；`GP_063` 有 `5,953` 个唯一代码，其中 `552` 个为 `.BJ`。当前 V2
沪深 universe 只有 `4,220` 个代码。因此这两个因子的旧源值 mismatch 主要反映生成股票池变化，
不是可以通过放宽误差门槛修复的普通数值误差；它们若继续研究，只能使用 Tushare 在沪深池上
重新物化的版本。即便不排除 BJ 的完整 Tushare 对照，源代码并不能全部覆盖：源池 `5,395`
只中 Tushare 只有 `5,147` 只，仍缺 `248` 只历史代码。

`GP_064` 的 mismatch 则不同：`600503.SH/2025-01-22` 的 YesLab `amount=126465.67`，
Tushare `amount=126465.674`，仅有小数精度差，但非线性指数项把结果差放大为 `7.569971`。
因此不能对 Tushare amount 做未经批准的四舍五入，也不能放宽 replay tolerance；GP_064 继续
保持 precision-sensitive 阻断。

证据：[`沪深范围 mismatch 根因诊断`](../../research_v2/artifacts/diagnostics/tushare_shsz_scope_mismatch_diagnosis_v1.json)

同样的 scope 问题也出现在 `GP_012/yeslab_381` 和 `GP_036/yeslab_405`：它们在四个窗口均为
`FORMULA_REPLAY_BLOCKED`，但不是 Tushare 缺行情；2025-01 源池分别含 `551`、`552` 只 BJ
代码，而 SH/SZ canonical universe 只有 `4,220` 只。两因子的 Tushare 重新物化 TRAIN/验证
RankIC 仍然良好，但旧值不能直接复用，需先生成 SH/SZ 版本再做相似度和准入审查。

证据：[`GP_012/GP_036 scope 诊断`](../../research_v2/artifacts/diagnostics/tushare_shsz_scope_diagnostic_gp012_gp036_v1.json)

同类结果还包括 `GP_031/yeslab_400` 与 `GP_056/yeslab_425`：四窗口修复后回放均为 `0/4`，
2025-01 源池分别有 `5,954` 只代码，其中 `551/552` 只为 BJ；它们的 Tushare 重物化质量
可以继续保留为研究记录，但旧值不能直接进入 SH/SZ 训练样本。剩余扩展因子
`yeslab_395/GP_026` 与 `yeslab_443/GP_074` 的源码也包含横截面 rank/mean，源池同样含
551/552 只 BJ。补完四窗口全市场回放后，两者均为 `0/4` 严格通过，但每个窗口的 Tushare
行情覆盖完整，因此结论是旧源 scope mismatch，必须重新物化 SH/SZ 版本，不是行情缺失。

证据：[`GP_031/GP_056 scope 审查`](../../research_v2/artifacts/diagnostics/tushare_shsz_remaining_gp_scope_review_v1.json)
以及 [`GP_026/GP_074 完成 scope 审查`](../../research_v2/artifacts/diagnostics/tushare_shsz_pending_gp_scope_completion_v1.json)。

### 三个严格年度稳定因子的扩展提案（未发布）

`yeslab_376/GP_007`、`yeslab_396/GP_027`、`yeslab_442/GP_073` 在 2020–2024 TRAIN 的
严格年度 ICIR 均为 `5/5`，2025 验证也通过。将它们与当前六因子提案的 SH/SZ 重新物化值
做完整 pair 后，所有 21 条新增边均低于 `0.90`，最高为 `yeslab_396/yeslab_442=0.880333`。

因此形成独立的 3 因子扩展提案，但不修改原 6 因子提案，也不发布新版本：这三个因子的
旧源值回放受原始横截面 scope 影响，`formula_scope_gate` 暂标
`PENDING_SCOPE_TRANSITION_REVIEW`。只有在确认新版本的公式、PIT 和可用性语义后，才可以
把 9 因子集合送入最终 admission preflight。

证据：

- [`扩展候选 21 条 TRAIN pair`](../../research_v2/artifacts/diagnostics/tushare_shsz_candidate_extension_similarity_v1.json)
- [`三个因子扩展提案`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_extension_proposal_v1.json)
- [`376/396/442 对 canonical TRAIN pair`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_strict_376_396_442_vs_canonical_train_v1/similarity_summary.json)
- [`376/396/442 对 155/94/98 TRAIN pair`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_strict_376_396_442_vs_rematerialized_155_94_98_train_v1/similarity_summary.json)

### 沪深准入预检结果

已补齐重新物化因子与 `yeslab_380` 的直接相似度证据。94/380 和 98/380 的五年聚合中位
绝对 Spearman 分别为 `0.161769`、`0.158912`，年度最高分别为 `0.185751`、`0.204554`，
均低于 `0.90`。补齐后，六个候选的 15 条两两相似度边全部具备证据，逐边结果全部 PASS。

对候选提案、三份质量摘要和四份相似度 artifact 执行只读 admission preflight，结果为
`PASS_WITH_MONITORING`：6 个候选的 predictive gate 均 PASS；155、98 的年度门槛 PASS；
380、431、437、94 保留年度 MONITOR。该状态只表示证据齐全并可进入人工版本审批，不表示
已发布或可以直接冻结为模型输入。预检输出仍明确记录 `NOT_PUBLISHED`、`database_writes=false`、
`model_training_started=false`、`locked_test_read=false`，没有写入任何已有表，也没有开始训练。

证据：

- [`94/98-380 沪深 TRAIN 相似度`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_94_98_380_train_v1.json)
- [`380/431/437 canonical 相似度边`](../../research_v2/artifacts/diagnostics/tushare_shsz_similarity_canonical_380_431_437_train_v1.json)
- [`沪深准入预检`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_preflight_v1.json)
- [`准入预检脚本`](../../research_v2/scripts/audit_shsz_admission_preflight.py)

### 九因子联合准入预检（范围门禁阻断）

在上述六因子证据基础上，已将 `yeslab_376/GP_007`、`yeslab_396/GP_027` 和
`yeslab_442/GP_073` 合并为九因子联合提案。预检共要求 `C(9,2)=36` 条 SH/SZ TRAIN
无序相似度边，实际观测到 `36/36` 条，所有边均低于 `0.90`，没有缺失边或冗余门槛失败。
九个候选的 predictive/质量门槛也均为通过；其中新增三个因子的严格年度 ICIR 均为
`5/5`。

联合预检仍判定为 `FAIL`，原因仅为三个新增因子的
`formula_scope_gate=PENDING_SCOPE_TRANSITION_REVIEW`。这表示它们的 SH/SZ 重新物化
版本在统计质量和去冗余方面已有证据，但旧源值回放受到股票池范围变化影响，尚未批准
新版本的公式、PIT 和可用性语义。该门禁是有意的硬阻断，不通过放宽 replay tolerance
或把旧源值直接当作新版本来绕过。

本次联合提案和预检均为只读证据：`publication_status=NOT_PUBLISHED`、
`database_writes=false`、`model_training_started=false`、`locked_test_read=false`。
因此当前可以确认的是“九因子证据已经齐全，但三因子范围语义尚未审批”，而不是九因子
已经进入模型输入。

证据：

- [`九因子联合提案`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_proposal_v2.json)
- [`九因子联合准入预检`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_preflight_v2.json)
- [`新增三因子范围扩展相似度`](../../research_v2/artifacts/diagnostics/tushare_shsz_candidate_extension_similarity_v1.json)

### 三个扩展因子的范围转换审计

为给 `formula_scope_gate` 提供可复核依据，已对 `yeslab_376/GP_007`、
`yeslab_396/GP_027` 和 `yeslab_442/GP_073` 执行只读范围转换审计。V2 的 2025 年实际
训练 universe 有 `4,912` 个唯一代码，其中 SH `2,180`、SZ `2,732`、BJ `0`；旧 YesLab
源池分别为 `5,952`、`5,952` 和 `5,955` 个代码，其中 BJ 分别为 `551`、`551` 和
`552`。这直接证明旧值回放的范围差异是真实存在的，不能把旧值 mismatch 当作新 SH/SZ
公式失败。

源码语义检查结果也已固化：三个因子都只读取 Tushare `pct_chg`，没有财务、现金流、资产负债表
或其他因子依赖；都包含当日横截面均值/标准差标准化；滚动窗口分别为 `10`、`5`、`5`；
没有负 `shift` 前视操作。新版本的范围契约明确为 `SH_SZ_ONLY`，并继续使用独立的 PIT
和可用性检查。

该审计把状态从“缺少范围证据”推进到“可人工审批”，但不会自动替代审批：
`formula_scope_gate` 仍为 `PENDING_SCOPE_TRANSITION_REVIEW`，联合九因子预检仍不发布、不
写库、不训练。审批通过后，下一步才是将这三个候选的 gate 改为 PASS 并重跑 36 边最终预检。

证据与脚本：

- [`范围转换审计结果`](../../research_v2/artifacts/diagnostics/tushare_shsz_scope_transition_audit_v1.json)
- [`范围转换审计脚本`](../../research_v2/scripts/audit_shsz_scope_transition.py)
- [`范围转换审计单元测试`](../../research_v2/tests/unit/scripts/test_audit_shsz_scope_transition.py)

### 九因子最终只读准入预检

基于上述范围审计和你已确认的 SH/SZ-only 研究范围，三个扩展因子的
`formula_scope_gate` 已在独立 v3 提案中标记为 `PASS`，没有修改 v2 历史提案。重新运行
九因子 admission preflight 后，`36/36` 条 TRAIN 相似度边全部具备证据且均低于 `0.90`，
没有缺失边、冗余失败或候选质量失败；最终状态为 `PASS_WITH_MONITORING`。

当前九因子集合为：
`yeslab_380, yeslab_431, yeslab_437, yeslab_155, yeslab_98, yeslab_94,
yeslab_376, yeslab_396, yeslab_442`。年度监控对象仍为 `yeslab_380/431/437/94`，其余
候选的年度门槛为 PASS。这个结果只说明集合已经具备进入人工发布审查的质量证据，不代表
已写入模型输入：`publication_status=NOT_PUBLISHED`、`database_writes=false`、
`model_training_started=false`、`locked_test_read=false`。

证据：

- [`九因子 v3 提案`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_proposal_v3.json)
- [`九因子最终准入预检`](../../research_v2/artifacts/diagnostics/tushare_shsz_admission_preflight_v3.json)

### 九因子 2025 验证期相似度完整性

对最终九因子集合补做了独立的 2025 验证期相似度完整性审计。结果为 `36/36` 条无序边
全部具备证据，每条边都有 `237` 个有效交易日（2025-01-02–2025-12-23），没有缺失边、
有效日不足或超过 `0.90` 的冗余边。最高中位绝对 Spearman 为
`yeslab_396/yeslab_442 = 0.878174`，低于训练期和验证期共同使用的 `0.90` 阈值。

因此当前九因子集合不仅通过 TRAIN 去冗余，也有完整 OOS 结构证据；但年度 MONITOR 因子
仍需在后续滚动窗口持续观察，不能因为一次验证期相似度通过就永久冻结。

证据与审计脚本：

- [`2025 验证期九因子相似度审计`](../../research_v2/artifacts/diagnostics/tushare_shsz_validation_similarity_admission_v1.json)
- [`验证期相似度审计脚本`](../../research_v2/scripts/audit_shsz_validation_similarity.py)
- [`验证期相似度审计测试`](../../research_v2/tests/unit/scripts/test_audit_shsz_validation_similarity.py)

### 九因子 TRAIN 到 2025 验证期性能漂移

对九因子的 TRAIN/2025 验证期 mean RankIC、ICIR 和方向做了固定方向漂移审计。结果为：

- `9/9` 因子方向一致，没有方向翻转；
- `9/9` 因子验证期 mean RankIC/ICIR predictive 门槛通过；
- 所有因子的验证期绝对 ICIR 都至少保留 TRAIN 的 `50%`，最低比例为
  `yeslab_380/GP_011 = 0.964`；
- `yeslab_94/GP_098` 等验证期绝对 RankIC 增强的因子仍保留年度监控，不把短期增强解释为
  永久 alpha。

该结果说明当前九因子集合没有明显 OOS 性能坍塌，但监控阈值已经明确：后续滚动验证若方向
翻转、验证期绝对 ICIR 低于 `0.50`，或低于 TRAIN 绝对 ICIR 的 `50%`，应自动退回人工复核，
而不是继续沿用已发布集合。

证据与脚本：

- [`九因子性能漂移审计`](../../research_v2/artifacts/diagnostics/tushare_shsz_performance_drift_audit_v1.json)
- [`性能漂移审计脚本`](../../research_v2/scripts/audit_shsz_performance_drift.py)
- [`性能漂移审计测试`](../../research_v2/tests/unit/scripts/test_audit_shsz_performance_drift.py)

### 九因子统一质量矩阵

为避免后续只读取单一 artifact，已将准入预检、年度稳定性、TRAIN/验证性能漂移、2025
验证期完整相似度和范围语义合并为统一 scoreboard。结果为 `PASS_WITH_MONITORING`，
9 个因子没有失败项；年度监控因子为 `yeslab_380/431/437/94`，其余 5 个因子的严格年度
ICIR 通过年数为 `5/5`。

该 scoreboard 是质量证据的汇总，不是发布动作：它明确记录
`publication_status=NOT_PUBLISHED`、`database_writes=false` 和
`model_training_started=false`。后续任何模型输入扩展都必须引用这张矩阵，并重新运行
相同的门槛检查，不能只引用某一个局部 RankIC 或相似度文件。

证据与实现：

- [`九因子统一质量矩阵`](../../research_v2/artifacts/diagnostics/tushare_shsz_factor_quality_scoreboard_v1.json)
- [`质量矩阵构建脚本`](../../research_v2/scripts/build_shsz_factor_quality_scoreboard.py)
- [`质量矩阵测试`](../../research_v2/tests/unit/scripts/test_build_shsz_factor_quality_scoreboard.py)

### Tushare 回放拦截修正与质量证据刷新

在结构覆盖复核中发现并修复了一个会影响质量结论的回放问题：
`TushareMarketSession` 原先只识别直接 `loadTable("dfs://YesLab", ...)` 形式，而计算器
实际使用 `database("dfs://YesLab"); loadTable(db, "stock_factor_daily")`。这会让部分重物化
计算静默回读 YesLab 行情表，尤其使 GP_011 在 2020-09-18 的有效样本错误降到 `1,354`。

修正后同一天的 Tushare SH/SZ 回放样本恢复为 `2,740`，完整 2020–2024 TRAIN 中九因子的
最小样本/中位数比例为 `0.809–0.812`，2025 验证期为 `0.990–0.995`，所有因子均通过
`>=0.70` 的结构覆盖门槛。修正后的 TRAIN/验证质量、性能漂移和统一 scoreboard 均保存为
v2/v3 版本；旧 v1 文件保留为历史对照，不再作为最终质量来源。

修正内容和证据：

- [`TushareMarketSession 修正`](../../research_v2/scripts/audit_formula_replay_tushare.py)
- [`回放拦截回归测试`](../../research_v2/tests/unit/scripts/test_audit_formula_replay_tushare.py)
- [`修正后样本覆盖审计`](../../research_v2/artifacts/diagnostics/tushare_shsz_sample_coverage_audit_v2.json)
- [`修正后性能漂移审计`](../../research_v2/artifacts/diagnostics/tushare_shsz_performance_drift_audit_v2.json)
- [`修正后统一质量矩阵`](../../research_v2/artifacts/diagnostics/tushare_shsz_factor_quality_scoreboard_v4.json)
- [`样本覆盖审计脚本`](../../research_v2/scripts/audit_shsz_sample_coverage.py)

修正后补做的四窗口九因子回放显示，Tushare canonical 计算与旧 YesLab 存量值并非同一
行情口径。`yeslab_155/376/396/442/94/98` 的旧值在四个窗口均出现大范围有限值差异，
因此这些因子只能使用已经生成的 Tushare SH/SZ 重新物化质量证据；`yeslab_380/431/437`
也需把少量键级差异保留在监控中。该结果不改变当前“未发布、未训练”的状态。

### 公式回放 mismatch 的精度根因

针对 GP_068 的两个大幅 mismatch 键，进一步抽取了 2023-10-09 至 2024-01-18 的输入序列。
`603500.SH` 的 73 个匹配交易日中有 65 天、`301230.SZ` 有 67 天的 `amount` 与 Tushare
不同，最大差异约 `0.005`；YesLab 的 `amount` 保留两位小数，Tushare 保留三位小数。
GP_068 对 `amount` 做 60 日 rank，这种精度差异会改变历史并列排序并放大为因子值差异，
例如 `603500.SH/2024-01-18` 的 GP_068 差异为 `333.893933`。这属于源数据精度不一致，
不是计算器实现错误。

正式 canonical 计算继续使用完整精度 Tushare 输入，不通过舍入来迎合旧 YesLab 值；旧表值
不再直接进入新训练样本。该诊断只解释 mismatch，不降低公式一致性门槛，也不触发发布。

证据：[`SH/SZ 公式精度诊断`](../../research_v2/artifacts/diagnostics/tushare_shsz_formula_precision_diagnostic_v1.json)
该诊断已纳入 [`统一质量矩阵 v4`](../../research_v2/artifacts/diagnostics/tushare_shsz_factor_quality_scoreboard_v4.json)
的 `formula_precision_diagnostic_status=MONITOR` 字段，但不改变因子准入 PASS/FAIL 计算。

### 因子库审计覆盖边界

本轮审计进一步固定了“已检验”和“已准入”的边界。当前因子目录共有 `487` 个条目；其中
`16` 个完成了 Tushare 沪深范围、5 个交易日收益标签的 TRAIN/验证质量检验，`9` 个进入现有
统一准入 scoreboard。另有 `62` 个因子在历史诊断文件中出现过，但这只代表存在局部公式、范围、
精度或候选证据，不能视为通过质量检验。目录中仍有 `425` 个因子没有任何诊断 artifact，必须继续
隔离，不能进入模型输入。

补充说明：历史代表性候选筛选文件中有 `28` 个因子带有验证期指标，但其中包含年度不稳定、相似度
待复核和公式回放阻断结果；因此这 `28` 个不能直接与当前 `16` 个 Tushare 重物化质量因子相加，
也不能据此扩大模型输入。

覆盖清单见 [`因子审计覆盖 v1`](../../research_v2/artifacts/diagnostics/tushare_shsz_factor_audit_coverage_v1.json)。
该清单不执行数据库写入、版本发布或模型训练；后续应按优先级对剩余因子做 Tushare 重物化质量检验，
再单独处理相似度、公式一致性、PIT 和准入门槛。

### GP_032 定向复核

对历史候选中唯一存在多窗口公式回放阻断、但阻断幅度很小的 `yeslab_401 / GP_032` 做了
一次独立的 Tushare canonical 验证重物化。2025-01-02–2025-12-23 共 `237` 个有效交易日、
`4,911` 只沪深股票，原始 5 日收益标签下 mean RankIC 为 `-0.09607`，方向取 `-1` 后
ICIR 为 `0.78321`，方向一致日占比 `78.48%`。

这说明 GP_032 的预测信号本身仍然有效；当前不能准入的原因不是预测能力，而是旧 YesLab
`amount` 两位小数与 Tushare 三位小数造成的局部公式值差异。继续使用 Tushare 重物化值进行研究，
暂不通过舍入或覆盖旧表来强行解除公式阻断。

证据：[`GP_032 Tushare 验证结果`](../../research_v2/artifacts/diagnostics/tushare_shsz_candidate_401_validation_2025_v1/quality_summary.json)

### 行业因子 yeslab_357 定向回放

为补齐代表性候选 `yeslab_357` 的证据链，回放器新增了显式全市场模式，并在 2025-01-02–01-31
对 `5,130` 只沪深股票执行单批次回放。结果为 `91,178` 个有限重叠值中有 `90,420` 个 mismatch，
最大绝对差 `5.1332`，因此不是数值精度级别的偶发差异。

根因是该计算器按静态 YesLab `stock_info_table` 的行业标签分组，而当前项目要求 Tushare canonical
且需要 PIT 行业归属；现有静态行业表无法证明历史时点行业集合一致。因此 `yeslab_357` 暂不准入，
下一步只有在建立 PIT 行业映射后，才有意义进行 Tushare 重物化 RankIC 复核。

证据：[`yeslab_357 行业范围回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos1_tushare_shsz_industry357_2025-01_v1.json)

同窗口补做 `yeslab_362` 和 `yeslab_363` 后，二者均为 `90,386 / 91,142` 个有限重叠值 mismatch，
最大绝对差约 `30,287.69`。这两个因子的阻断模式与 `yeslab_357` 一致，且差异更大；它们暂不进入
Tushare 因子准入候选，除非先补齐 PIT 行业分类映射。

证据：[`yeslab_362/363 行业范围回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos2_tushare_shsz_industry362_363_2025-01_v1.json)

### 剩余因子源码覆盖盘点

对 `数据库/factor_calculators` 做了只读源码匹配：目录中的 `487` 个因子有 `486` 个能匹配到
计算器（GP 因子按 `GP_NNN` 与 `GP_Factor_NNN_Calculator.py` 规则匹配，其余按
`FACTOR_CONFIG.factor_name`、显式 CSAD/Corwin-Schultz 输出适配器或 ROE/ROIC 目录别名匹配），
真正没有直接匹配的只有测试因子 `yeslab_0`。后续全库质检优先处理这些已有计算器但尚无
Tushare 质量证据的因子；“有计算器”只代表可重放，不代表通过准入。

此前 7 个 CSAD/价差因子已补齐复权字段回放支持；两个财务别名 `ROE/ROIC` 也已补齐公告日期
输入的回放定义，均仍暂不纳入模型输入。

随后补充了 Tushare `adj_factor` 适配，使 `high_hfq/low_hfq/close_hfq` 能在内存行情中生成，
并对 `yeslab_485`、`yeslab_487` 做了 2025-01-02–01-31 全市场回放。适配本身成功，但历史值仍有
较多差异：`yeslab_485` 为 `21,348 / 92,006` mismatch，`yeslab_487` 为 `12,281 / 91,332`
mismatch。说明源表的复权/价差实现口径仍未完全一致，不能因为字段已经补齐就解除准入阻断。

证据：[`CS 价差复权字段回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos2_tushare_shsz_csad_corwin_2025-01_v1.json)

对 2025-01-02 的 `5,106` 个沪深键做输入级抽查发现，YesLab `high_hfq/low_hfq` 与
Tushare `high/low * adj_factor` 并非完全相同：分别有 `1,218` 和 `1,166` 个键不同，最大差异约
`0.0052`。因此当前 mismatch 的主要原因已定位为复权高低价精度差异，不能把它误判成 CS 公式实现
错误；canonical 研究继续保留 Tushare 完整精度，旧值仍保持阻断。

在此基础上完成了首个全市场 CSAD 因子 `yeslab_481` 的单月回放。由于该因子需要 120 日滚动收益
相关矩阵，回放使用 2024-06-08 起的行情，最终在 `5,121` 只股票上产生 `92,094` 个有限重叠值，
其中 `38,059` 个 mismatch，另有 `38` 个历史源代码、`215` 个有效源值没有 Tushare 行情覆盖。
该因子的滚动同伴选择会放大复权价格精度差异，且存在历史代码覆盖缺口，故继续保持隔离。

证据：[`CSAD-20 回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos3_tushare_shsz_csad481_2025-01_v1.json)

同一进程复用 CSAD 核心后，`yeslab_482/483/484` 也完成了相同窗口的全市场回放，均被阻断：
`yeslab_482` 有 `15,803 / 92,149` mismatch 和 `44` 个未覆盖代码；`yeslab_483` 有
`37,063 / 92,094` mismatch 和 `38` 个未覆盖代码；`yeslab_484` 有 `28,013 / 92,095`
mismatch 和 `38` 个未覆盖代码。四个 CSAD 输出均不具备当前 Tushare canonical 准入证据。

证据：[`CSAD 其余三个输出回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos3_tushare_shsz_csad482_484_2025-01_v1.json)

`yeslab_222 / ROE` 和 `yeslab_224 / ROIC` 的有界回放则精确通过：在 512 只股票、2025-01
窗口中，分别有 `9,125` 和 `8,963` 个有限重叠值，均为零 mismatch。二者由
`financial_metrics_table` 的 `ann_date` 输入重建，说明当前存量值与公告日期前向填充逻辑一致。
但它们尚未完成 Tushare 标签下的 RankIC、PIT 完整性和去冗余检查，因此目前是“公式可重放、预测
质量待检验”，不是准入因子。

证据：[`ROE/ROIC 公告日期回放`](../../research_v2/artifacts/diagnostics/formula_replay_oos2_financial_aliases_2025-01_v1.json)

对两者补做的 2020 TRAIN 年度严格质量样本也为正向：ROE mean RankIC 为 `0.04874`、ICIR 为
`0.39782`，ROIC 为 `0.04781`、`0.39620`；有效日均为 `2,626` 和 `2,552` 个，整体有限覆盖率
为 `95.92%` 和 `93.24%`。这证明两者值得继续执行 2021–2024 稳定性和 2025 验证，尚不足以改变
“未准入”的结论。

证据：[`ROE/ROIC 2020 训练质量`](../../research_v2/artifacts/diagnostics/financial_aliases_train_2020_quality_v1.json)

继续复核 2021 TRAIN 后，两者均失去预测性：ROE mean RankIC 为 `-0.00363`、ICIR 为
`-0.02898`，ROIC 为 `-0.00326`、`-0.02689`；覆盖率仍保持 `97.14%` 和 `94.38%`。因此问题
不是样本缺失，而是跨年度信号不稳定。ROE/ROIC 的当前原始口径已满足“公式可重放”，但不满足年度
预测稳定性，停止将它们推进到 2025 验证或相似度阶段，并继续隔离。

证据：[`ROE/ROIC 2021 训练质量`](../../research_v2/artifacts/diagnostics/financial_aliases_train_2021_quality_v1.json)

对 `CFOA` 和 `ROA` 做的 2020 年双因子初筛显示预测性较好：CFOA mean RankIC `0.03234`、ICIR
`0.42935`，ROA `0.05206`、`0.43688`。但两者最低月度有限覆盖率仅 `0.6077` 和 `0.6023`，低于
当前 `0.70` 结构门槛，说明早期财务公告可用性存在明显缺口。它们暂不继续扩展年度稳定性，先诊断
财务数据可用日期和月度覆盖，再决定是否采用修复后的财务口径。

证据：[`CFOA/ROA 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_cfoa_roa_train_2020_quality_v1.json)

源数据月度诊断确认该覆盖失败来自 `factor_data5` 的财务可用性：在沪深范围内，CFOA 的月度有效值比例
从 2020-01 的 `76.90%` 升至 2020-12 的 `83.16%`，ROA 从 `77.84%` 升至 `83.80%`；有效股票数也
分别从 `4,016/4,051` 增至 `4,405/4,420`。因此不是 V2 标签或北交所范围问题，而是早期公告数据尚未
完整覆盖股票池。当前不应以事后数据回填来提高覆盖；需要建立财务数据的 PIT 可用日期修复后再评估。

证据：[`CFOA/ROA 可用性诊断`](../../research_v2/artifacts/diagnostics/financial_cfoa_roa_availability_2020_v1.json)

财务增长类初筛结果分化：`NP_Q_YOY` 的 mean RankIC 为 `-0.00354`、ICIR `-0.19439`，直接判定
预测失败；`ROE_YOY` 的 mean RankIC 为 `0.02403`、ICIR `0.40826`，但最低月度覆盖率只有 `61.86%`，
同样被结构门槛阻断。当前只有 ROE_YOY 值得等待 PIT 财务可用性修复后重测，NP_Q_YOY 不再推进。

证据：[`财务增长类 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_growth_train_2020_quality_v1.json)

`TA_YOY` 与 `TOE` 的初筛也没有形成可用候选：TA_YOY mean RankIC `0.01765`、ICIR `0.32825`，
但最低月度覆盖率 `62.44%`；TOE mean RankIC `-0.01029`、ICIR `-0.27550`，直接预测失败。TOE
不再推进，TA_YOY 保持在财务可用性修复后的重测队列。

证据：[`TA_YOY/TOE 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_ta_toe_train_2020_quality_v1.json)

`ROICD` 和 `CFOAD` 也呈现“预测性/覆盖性二选一”的结果：ROICD mean RankIC `0.03470`、ICIR
`0.43748`，但最低月度覆盖率 `61.84%`；CFOAD 覆盖率充足（最低 `96.03%`），但 mean RankIC
仅 `0.00188`、ICIR `0.04868`。CFOAD 直接淘汰，ROICD 保持在 PIT 财务可用性修复后的重测队列。

证据：[`ROICD/CFOAD 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_roicd_cfoad_train_2020_quality_v1.json)

`NP_QOQ` 和 `NP_Z` 的源数据覆盖接近完整，但进入固定股票池后仍都只有约 `62.5%` 的最低月度覆盖，
再次表明早期 PIT 财务可用性问题。NP_QOQ mean RankIC 仅 `0.00926`，其较高 ICIR 主要来自低日度
波动，不能作为单独准入依据；NP_Z mean RankIC `-0.00271`、ICIR `-0.14105`，直接淘汰。

证据：[`NP_QOQ/NP_Z 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_np_qoq_np_z_train_2020_quality_v1.json)

`OCF_Q_YOY` 与 `OCF_YOY` 均未通过：mean RankIC 分别为 `-0.00106` 和 `0.00029`，ICIR 为
`-0.05327` 和 `0.01520`，同时最低月度覆盖率约 `62.4%`。两者直接淘汰，不进入年度稳定性验证。

证据：[`OCF 增长类 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_ocf_growth_train_2020_quality_v1.json)

`OP_YOY` 与 `1个月振幅调整动量` 也均失败：mean RankIC 分别为 `-0.00157` 和 `-0.00649`。后者的
整体有限覆盖率仅 `33.93%`、最低月度覆盖率 `22.20%`，属于关键覆盖不足，需单独修复源数据才能
再次考虑；二者均不进入后续验证。

证据：[`OP_YOY/振幅调整动量 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_op_yoy_amplitude_train_2020_quality_v1.json)

`ROAD` 和 `ROED` 提供了本组中较强的预测信号：mean RankIC 分别为 `0.03837` 和 `0.03434`，ICIR
为 `0.43764` 和 `0.42503`，但最低月度覆盖率仅 `57.54%`、`61.86%`。它们与 CFOA/ROA/ROICD
同属“预测通过、早期 PIT 财务可用性阻断”队列，不能绕过结构门槛进入训练。

证据：[`ROAD/ROED 2020 初筛`](../../research_v2/artifacts/diagnostics/financial_road_roed_train_2020_quality_v1.json)

### 核心财务因子初筛汇总

`factor_data5` 中本轮覆盖的 `18` 个核心财务质量/增长/变化因子已经完成初筛：`8` 个因预测失败
直接拒绝，`2` 个（ROE、ROIC）因年度不稳定拒绝，剩余 `8` 个有一定 2020 预测性但都未达到早期
PIT 财务可用性的最低覆盖要求。当前没有任何一个可进入模型输入；后续修复优先级是建立 PIT
财务可用日期，再重测这 8 个条件候选，不能对已拒绝因子反复筛选。

统一状态见 [`核心财务因子初筛矩阵`](../../research_v2/artifacts/diagnostics/financial_core_screen_status_v1.json)。

### Alpha 库初筛

技术因子库的第一批 `Alpha1/Alpha2` 具有良好结构覆盖（最低月度覆盖率 `100%/93.75%`），但信号强度
不足：Alpha1 mean RankIC `0.01524`、ICIR `0.23253`，低于当前准入集合的基础强度；Alpha2 为
`0.00285/0.02889`。这说明覆盖充分不能替代预测性，二者均不进入年度稳定性验证。

证据：[`Alpha1/Alpha2 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_001_002_train_2020_quality_v1.json)

第二批 `Alpha3/Alpha4` 也未形成候选：Alpha3 mean RankIC `-0.02358`、ICIR `-0.19703`，Alpha4
`0.01161/0.16337`。二者覆盖率均接近完整，但预测强度不足，均在年度稳定性之前拒绝。

证据：[`Alpha3/Alpha4 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_003_004_train_2020_quality_v1.json)

第三批出现第一个 Alpha 库晋级候选：`Alpha5` mean RankIC `0.03460`、ICIR `0.47417`，最低月度覆盖率
`86.23%`，已满足 2020 初筛的预测和结构门槛，后续需要补公式回放与 2021–2024 年度稳定性。Alpha6
`0.00876/0.08063`，因信号偏弱拒绝。

证据：[`Alpha5/Alpha6 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_005_006_train_2020_quality_v1.json)

Alpha5 已完成 2025-01-02 至 2025-01-31 的 Tushare SH/SZ 全市场公式回放。92,196 个重叠观测中重算值与历史值
`0` 行不一致，说明当前 Alpha5 计算逻辑和输入字段可以精确复现；但仍有 49 行、26 个代码在 Tushare 市场源中没有对应
覆盖，严格公式门槛暂记为阻断。处理这些停牌/历史代码覆盖缺口后，才进入 2021–2024 年度稳定性检验。

证据：[`Alpha5 Tushare 公式回放`](../../research_v2/artifacts/diagnostics/tushare_shsz_alpha5_formula_replay_2025-01_v1.json)

第四批 `Alpha7/Alpha8` 的结构覆盖率均接近完整（最低月度覆盖率 `99.82%/99.83%`），但预测强度不足：
Alpha7 mean RankIC/ICIR 为 `0.01034/0.14647`，Alpha8 为 `0.01881/0.14628`。二者均未达到绝对
mean RankIC `0.02` 与绝对 ICIR `0.50` 的双重门槛，初筛拒绝，不进入年度稳定性检验。

证据：[`Alpha7/Alpha8 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_007_008_train_2020_quality_v1.json)

第五批 `Alpha9/Alpha10` 的覆盖率仍接近完整（最低月度覆盖率 `99.91%/100%`），但没有形成候选：Alpha9
mean RankIC/ICIR 为 `-0.01127/-0.09856`；Alpha10 为 `0.02624/0.16350`。Alpha10 虽超过 mean RankIC
门槛，但 ICIR 明显低于 `0.50`，因此二者均在初筛拒绝。

证据：[`Alpha9/Alpha10 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_009_010_train_2020_quality_v1.json)

第六批 `Alpha11/Alpha12` 也未形成候选。Alpha11 mean RankIC/ICIR 为 `-0.00745/-0.07414`，最低月度覆盖率
`81.63%`；Alpha12 为 `0.01290/0.11404`，覆盖率接近完整。二者均未达到双重预测门槛，初筛拒绝。

证据：[`Alpha11/Alpha12 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_011_012_train_2020_quality_v1.json)

第七批 `Alpha13/Alpha14` 的最低月度覆盖率为 `100%/99.80%`，但预测信号均不足：Alpha13
mean RankIC/ICIR 为 `0.00530/0.05765`，Alpha14 为 `-0.02180/-0.17115`。二者均未通过双重预测门槛，初筛拒绝。

证据：[`Alpha13/Alpha14 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_013_014_train_2020_quality_v1.json)

第八批 `Alpha15/Alpha16` 中，Alpha15 mean RankIC/ICIR 为 `0.01558/0.20366`，直接拒绝。Alpha16 为
`0.03230/0.49509`，覆盖率接近完整且 mean RankIC 达标，但 ICIR 比 `0.50` 门槛低 `0.00491`；严格流程暂不
进入年度稳定性，仅保留为隔离的边界复核候选。

证据：[`Alpha15/Alpha16 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_015_016_train_2020_quality_v1.json)

第九批 `Alpha17/Alpha18` 覆盖率接近完整（最低月度覆盖率 `99.99%/99.80%`），但预测强度仍不足：Alpha17
mean RankIC/ICIR 为 `0.01742/0.14355`，Alpha18 为 `-0.02501/-0.18852`。二者均未通过双重预测门槛，初筛拒绝。

证据：[`Alpha17/Alpha18 2020 初筛`](../../research_v2/artifacts/diagnostics/alpha_017_018_train_2020_quality_v1.json)
