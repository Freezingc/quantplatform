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
