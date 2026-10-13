# 日频非价量域可行性核查（2026-10-13，只读）

> **问题**：三个本项目从未触碰过的日频非价量域——`moneyflow`（个股资金流）、`cyq_perf`（筹码分布/胜率）、`margin_detail`（两融明细）——能否提升候选池的有效维度？
>
> **结论（分域）**：`moneyflow` 全域无信号（8 字段 ×2 对齐，|ICIR| ≤ 0.235）；`margin_detail` 有**越过准入门槛**的原始预测力，但增量信息很少（给定 dense9 的保留率仅 0.230–0.254）且横截面逐年漂移；`cyq_perf` 的 `high_dist` 预测力不达门槛（ICIR 0.351），却是唯一同时满足"稠密 + 近正交 + 高保留率"的候选。
>
> **方法学结论**：现行 `ICIR ≥ 0.50` 门槛并不是"过苛"，而是与"提升有效维度"这个目标**正交**。本次越过门槛的两个候选恰好是增量最少的，增量最多的候选恰好没过门槛。建议增设第二条准入轨道（见 §5）。

全程只读：无数据库写入、无生产工件写入、生产树零改动（`codex/publish-quant-platform @ 2502014df94a`）、2026/E9 锁定段未读未写。精确说明：`published_version` 指针查询不带日期过滤，会返回 `2026-*` 的分区标签（版本元数据），但所有行情/标签/因子数据查询都硬界定在 2022–2025，锁定段的任何数据行都未进入本次分析。已核查全部提交脚本，无任何 insert/update/delete/DDL；唯一的本地写操作是抓取脚本的进度日志。

明细数据表见 [`daily_domain_feasibility.md`](daily_domain_feasibility.md)（由 `make_report.py` 从工件程序化生成，无手抄数字）。

## 1. 先修了一个会让全部结论作废的口径 bug

`universe_daily` 与 `label_daily` 都是**按 YYYY-MM 逻辑分区版本化**的：同一 `(signal_date, ts_code)` 业务键在库里存着多份历史修订（2022–2025 段 universe 有 3,812,332 行、去重后 3,557,762 行；label 有 3,616,487 行、去重后同样 3,557,762 行）。未按已发布 `data_version` 过滤直接 inner merge，得到 **4,104,981 行——比任一输入都多**，`compute_daily_rank_ic` 随即以 `predictions and labels must have unique signal-date keys` 拒绝（第一次筛查因此崩溃，日志保留为 `logs/screen_run1_CRASH_dupkeys.log`）。

修法不是自己发明去重规则，而是复用生产口径：从 `published_version` 取每个逻辑分区的当前 `data_version`（同分区多行时按 `published_at` 最新者胜），只保留该版本的行——等价于 `research_v2/scripts/walk_forward_common.py` 的 `current_versions()` + `filter_current()`。修后两表都收敛到 3,557,762 行、0 重复键、且键集完全一致。

> 这条对**任何**直接读库算 IC 的研究脚本都适用，不只是本次。`factor_daily` 例外：已核查 2025 年每因子行数 = 唯一键数（如 `yeslab_96` 934,108 = 934,108），无需版本过滤；脚本内仍留了断言，一旦该表开始多版本就会立刻失败而不是静默膨胀。

## 2. 数据层：三源覆盖率与漂移

| 源 | 2022 | 2023 | 2024 | 2025 | 判读 |
|---|---|---|---|---|---|
| moneyflow | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 满覆盖、无漂移 |
| cyq_perf | 0.9696 | 0.9830 | 0.9950 | 0.9994 | 近满覆盖、轻微改善 |
| margin_detail | **0.5681** | 0.6570 | 0.7040 | **0.7254** | **两融标的名单逐年扩容，+15.7pp** |

`margin_detail` 的覆盖率漂移是实质性问题，不只是"覆盖低"：横截面本身在变，跨折不可直接比较；且缺失**非随机**（是否可两融与市值/流动性强相关），模型可能学到的是"是否两融标的"而非融资强度本身。它也意味着该域因子**永远无法作为独立 score 通过生产的 score-health 门**（要求 coverage ∈ [0.9, 1.0]），只能作为稀疏特征存在。

## 3. 预测力层：筛查与基准对照

口径：标签 `forward_open_5d:v1`、宇宙 `cn_liquid_a`、`minimum_cross_section=30`、ICIR = 日度 RankIC 的 mean/std（**未年化**），全部走 `research_v2.evaluation.rank_ic.compute_daily_rank_ic`（import 而非重写）。每个字段同时报告 `lag0`（源 T 日→信号 T 日）与 `lag1`（源 T-1 日→信号 T 日）两种对齐。

19 字段 ×2 对齐 = 38 次筛查，各域最强（完整表见明细报告 §B）：

| 域 | 最强字段 | lag | mean RankIC | ICIR(2022–2025) | ICIR(2022–2024) | 覆盖率 | 方向日占比 |
|---|---|---|---|---|---|---|---|
| margin_detail | buy_intensity | 0 | −0.0757 | **−0.605** | **−0.616** | 0.667 | 0.731 |
| margin_detail | buy_intensity | **1（保守）** | −0.0715 | **−0.557** | −0.564 | 0.666 | 0.718 |
| margin_detail | repay_intensity | 1（保守） | −0.0701 | **−0.543** | −0.561 | 0.666 | 0.711 |
| margin_detail | rzye_chg5 | 1（保守） | −0.0262 | −0.425 | −0.386 | 0.663 | 0.702 |
| cyq_perf | high_dist | 0 | +0.0422 | +0.351 | +0.330 | 0.987 | 0.660 |
| cyq_perf | winner_rate | 0 | −0.0328 | −0.285 | −0.276 | 0.987 | 0.606 |
| moneyflow | net_sm_ratio | 0 | +0.0170 | +0.235 | +0.207 | 1.000 | 0.611 |

基准来自库内 `factor_rank_ic_daily`（同标签、同 ICIR 口径；该表覆盖 2020Q1–2024Q4，故基准窗取 2022–2024，候选值按全方差定律从逐年 mean/std/天数精确池化到同窗）：

> 冻结 v2 可测的 13 个因子，2022–2024 |ICIR| **区间 0.497（yeslab_206，且它是 dense9 成员）– 1.106（yeslab_357），中位 0.656，dense9 均值 0.720**；|mean RankIC| 区间 0.033–0.091。

两点判读：

- `margin_detail` 的强度不是靠激进的 PIT 假设刷出来的：`margin_detail` 于 T+1 早 08:30 发布、A 股 09:30 开盘，`lag0` 严格来说可得但余量只有 1 小时；**换成保守的 `lag1`（T-1 数据）后 ICIR 仍有 −0.557 / −0.543，照样过门槛**。这是本次最硬的一个结果。
- `moneyflow` 八个字段（含大单/超大单/小单净额与其占比）在四年窗上全部 |ICIR| ≤ 0.235，约为最弱在册因子的一半，且方向日占比 ≤ 0.611。**该域判死，不建议再投入**。

## 4. 冗余与有效维度（核心结果）

窗口 2025（与 2026-10-13 的 alpha393 复检同口径，可直接比较）。阈值 |ICIR| ≥ 0.25 筛出 6 个幸存者，每个查三条腿：逐对 vs 冻结 15、**逐对 vs 其他幸存者**、给定 dense9 的集合增量保留率。

| 候选 | 源 | lag | ICIR | 覆盖率 | vs冻结最差 | vs候选最差 | 保留率 | 残差 RankIC | 判定 |
|---|---|---|---|---|---|---|---|---|---|
| high_dist | cyq_perf | 0 | +0.351 | 0.999 | 0.180 | 0.207 | **+0.622** | **+0.0276** | 唯一全过 |
| buy_intensity | margin | 1 | −0.557 | 0.722 | 0.574 | **0.881** | +0.254 | −0.0186 | 保留率不足 + 与 repay 近重复 |
| repay_intensity | margin | 1 | −0.543 | 0.722 | 0.576 | **0.881** | +0.230 | −0.0161 | 同上 |
| rzye_chg5 | margin | 1 | −0.425 | 0.722 | 0.198 | 0.261 | +0.370 | −0.0118 | 正交但残差 IC 偏弱、覆盖低 |
| winner_rate | cyq_perf | 0 | −0.285 | 0.999 | 0.506 | 0.264 | **−0.306** | +0.0097 | **条件化后符号翻转** |
| net_fin_ratio | margin | 0⚠ | −0.277 | 0.725 | 0.174 | 0.100 | +0.455 | −0.0085 | 保守 lag 未过阈，仅 lag0 可用 |

三个必须点名的方法学问题：

1. **只看"候选 vs 在册"会高估维度。** `buy_intensity` 与 `repay_intensity` 的日内 |Spearman| 中位数 **0.881**——它们是同一个经济信号（两融换手强度）的两种写法，`net_fin_ratio` 更是二者的差、`rzye_chg5` 是其 5 日平滑。仅按 0.90 BLOCK 上限做贪心会把这一个信号数成 4 个维度；按 0.80 MONITOR 上限降为 3 个；**再叠加有符号保留率门槛后只剩 3 个（`rzye_chg5`、`high_dist`、`net_fin_ratio`），而按残差 IC 排序真正值得推进的只有 1 个（`high_dist`，0.0276 ≥ 0.02）**。
2. **保留率必须按符号判定。** `winner_rate` 的 |保留率| = 0.306 看似达标，实际是原始 RankIC −0.0317 → 残差 **+0.0097**：dense9 不仅吸收了它的信息，还把它**反号**了。用绝对值比较会把它误判为通过——这类候选加进模型是有害的，不是"弱"。
3. **这复现了 2026-10-13 的 alpha393 教训**：逐对相关度门槛保护不了有效维度（alpha393 逐对全过 ≤0.66，集合保留率仅 10.2%）。本次 `buy_intensity` 逐对 vs 冻结最差仅 0.574（PASS），保留率却只有 0.254。**逐对门 + 集合保留率必须同时用。**

## 5. 对准入门槛的含义（回答"ICIR ≥ 0.50 是否过于苛刻"）

先澄清一个术语混淆：门槛是 **signed ICIR ≥ 0.50**（日度 RankIC 的 mean/std，未年化）**且 signed RankIC > 0.02**，不是"IC > 0.5"。在册因子的 |mean RankIC| 实际只有 0.033–0.091；在 5 日前向收益标签下 IC > 0.5 不存在。

0.50 这条线也不是拍脑袋：在册 13 个可测因子的 |ICIR| 最低就是 **0.497**（`yeslab_206`，还是 dense9 成员），中位 0.656。作为"独立 alpha 有多强"的标尺，它是 calibrated 的。

**问题在于它衡量的是"单独有多强"，不是"加进来有多少新信息"。** 本次证据把这两件事彻底分开了：

| | 越过 ICIR 门槛（buy/repay_intensity） | 未越过（high_dist / rzye_chg5 / net_fin_ratio） |
|---|---|---|
| 集合增量保留率 | 0.230 – 0.254 | 0.370 – 0.622 |
| 残差 RankIC | 0.016 – 0.019 | 0.012 – **0.028** |
| vs 冻结集最差逐对 | 0.574 – 0.576 | **0.174 – 0.198** |
| 覆盖率 | 0.72（逐年漂移） | 0.72 – **0.999** |

即：现行门槛**系统性地筛出"强但已被现有因子解释掉"的候选，同时丢弃"弱但正交"的候选**。对"提升有效维度"这个目标，它是反向选择。

建议改为**双轨准入**（不放宽 A 轨，只新增 B 轨）：

- **A 轨 · 独立 alpha**：维持现有全部门槛（signed RankIC > 0.02、signed ICIR ≥ 0.50、方向日占比 ≥ 0.55、年度严格 ≥4/5、逐对 <0.80/0.90）。产物可作为独立 score。
- **B 轨 · 维度补充**：有符号保留率 ≥ **+0.30**（负值直接判 SIGN_FLIP 拒绝）且 |残差 RankIC| ≥ 0.02 且 逐对 vs 在册与 vs 同批候选均 < 0.80 且 覆盖率 ≥ 0.90（低于则必须显式按稀疏特征管理并记录漂移）。**B 轨准入的因子只能作为模型特征，不得单独成为 score**——因为覆盖率先天的不足使它过不了 score-health 门。

按 B 轨，本次 38 次筛查里只有 **`high_dist`（cyq_perf）** 一个候选合格。这与"因子库里 383 个 SCREENED_ONLY 也许有遗珠"的直觉一致：遗珠更可能藏在"弱但正交"里，而不是"强"里——但**必须用保留率去捞，用 ICIR 捞不到**。

## 6. 若推进：前置条件与成本

本次全部数值来自 **tushare 源值直读**，属于**分诊级证据**，不能直接作为准入依据。要走正式流程：

1. **补数据窗口**：当前只抓了 2022–2025。F1–F4 的 train 段最早从 2017 开始，需要补 `cyq_perf`/`margin_detail` 的 2017–2021（`fetch_daily_domains.py` 支持断点续跑，改 `YEARS` 即可）。
2. **走管道物化 + 隔离重放**：以 PIT 语义写入（`available_date`：cyq_perf = T 日晚间可得，margin_detail 保守取 T+1），再由既有评测链复算，不能用本次的内存面板。
3. **保留率要按折估**：本次只在 2025 单年估计（242–243 天、日内覆盖 0.987，样本充足但仍是单年），需在 4 个 test 年各估一次并看稳定性。
4. **年度稳定性门未测**：≥4/5 年严格通过这条腿本次没跑。
5. **多重比较未校正**：38 次检验下 |ICIR| 0.557 的假阳性风险需量化。建议对 `high_dist`/`rzye_chg5` 沿用既有统计契约出区间（日均差、HAC lag=5、折内 circular block=10、5,000 次 bootstrap、seed=20260912、95% CI、分辨率 ±0.002）。
6. **机制重叠待查**：`high_dist = his_high / cost_50pct` 是"历史高点距离"，与 52 周高点/动量异象同源。若后续引入动量类因子，需重跑逐对冗余。
7. 2026/E9 锁定段禁读禁写，本次未触碰。

## 7. 文件清单

| 文件 | 说明 |
|---|---|
`fetch_daily_domains.py` | 三源 2022–2025 抓取（可断点续跑；交易日历取自 moneyflow 面板自身，避免 `trade_cal` 偶发空帧）
`screen_daily_domains.py` | IC 筛查；含 §1 的版本口径修复与唯一性硬断言
`pooled_window_icir.py` | 把逐年统计按全方差定律池化到基准窗 2022–2024（纯后处理，不重算）
`redundancy_survivors.py` | 冗余三腿 + 贪心维度计数
`greedy_caps.py` | 0.90 / 0.80 / 保留率感知三种贪心规则 + 决策表
`coverage_drift.py` | 逐年覆盖率与字段非空率
`make_report.py` | 从工件生成 `daily_domain_feasibility.md`
`daily_domain_feasibility.md` | 明细报告（全部数据表）
`analysis/` | 8 个 json/csv 摘要
`logs/` | 含两次崩溃日志（`screen_run1_CRASH_dupkeys.log`、`redundancy_run1_CRASH_colname.log`）与 `coverage_drift_run1_badkeycov.log`，失败记录按要求保留不覆盖

原始面板（594 MB / 12 个 parquet）与逐股票面板留在服务器 `/data/zhangyudong/qrv2_research_artifacts/daily_domain_feasibility_20261013/raw/`，不入 Git。

## 8. 复现

```bash
cd /data/zhangyudong/qrv2_research_artifacts/daily_domain_feasibility_20261013
python fetch_daily_domains.py        # -> raw/（已完成，可跳过）
python screen_daily_domains.py       # ~15 min -> analysis/ic_screen{.json,_summary.csv}
python pooled_window_icir.py         # <1 s
python redundancy_survivors.py 0.25  # ~7 min -> analysis/redundancy_survivors.json
python greedy_caps.py                # <1 s -> analysis/decision_table.csv, dimension_summary.json
python coverage_drift.py             # ~3 min -> analysis/coverage_drift.csv
python make_report.py                # <1 s -> daily_domain_feasibility.md
```

环境：python 3.12.2, pandas 2.3.3, numpy 2.5.1, pyarrow 25.0.1, dolphindb 3.0.4.2（服务器 conda 环境之外的 `~/.local` 栈；与 dense9 重放记录的 3.11.13/LightGBM 4.6.0 环境不同，本次不涉及模型训练，故无训练实现绑定问题）。生产树 `codex/publish-quant-platform @ 2502014df94a`，零改动。
