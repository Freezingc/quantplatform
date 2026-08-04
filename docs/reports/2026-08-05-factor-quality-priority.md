# 因子质量补救优先级清单

日期：2026-08-05  
数据范围：`SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1`（2020-01-02–2024-12-31）  
锁定测试：未读取  
数据库写入：无；本报告只读取已有质量快照

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

候选批次目前是本地、可复核的 pickle 文件，包含 6 个因子的有限候选补行（分别为
1,386、1,386、1,385、6,853、1,385、8,204 行），不包含 `GP_026`，也没有写入
DolphinDB。每个批次都有 SHA-256；后续写入新 V2 隔离表时必须用该哈希和同一来源水印做幂等校验。

下一步是仅针对这 6 个候选物化到新的 V2 隔离对象，重新跑 compact coverage 和预测稳定性
门禁；`GP_026` 留在阻断清单，暂不进入候选表。任何正式发布仍需单独批准。
