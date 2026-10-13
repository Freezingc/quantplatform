# 日频非价量域可行性核查 —— 明细报告（2026-10-13，只读）

- 生产树：codex/publish-quant-platform @ 2502014df94a（零改动）
- 环境：python 3.12.2, pandas 2.3.3, numpy 2.5.1, pyarrow 25.0.1, dolphindb 3.0.4.2, scipy 1.18.1
- 标签 / 宇宙：`forward_open_5d:v1` / `cn_liquid_a`，`minimum_cross_section=30`，ICIR = 日度 RankIC 的 mean/std（**未年化**）
- 上下文口径：`universe_daily` 与 `label_daily` 均按 `published_version` 指针保留当月已发布 `data_version`（等价 `walk_forward_common.filter_current`）；筛查窗 2022-2025 共 3,557,762 个成员日
- 2026 锁定段（E9）未读取、未写入；无数据库写入、无生产工件写入。**精确说明**：`published_version` 指针查询本身不带日期过滤，会返回 `2026-*` 的分区标签（版本元数据），但所有行情/标签/因子数据查询均以 `year(signal_date) between 2022 and 2025` 或 `between 2025.01.01 : 2025.12.31` 硬界定，锁定段的任何数据行都未进入本次分析

## A. 数据层：三源逐年覆盖率

`key_coverage` = 该源在当年有行记录的比例；`nonnull::<字段>` = 该字段在匹配行上的非空比例。

| source | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| cyq_perf | 0.9696 | 0.9830 | 0.9950 | 0.9994 |
| margin_detail | 0.5681 | 0.6570 | 0.7040 | 0.7254 |
| moneyflow | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**cyq_perf** 字段非空率

| year | winner_rate | cost_dispersion | cost_iqr_rel | high_dist | cost_50pct |
|---|---|---|---|---|---|
| 2022 | 0.9696 | 0.9696 | 0.9696 | 0.9696 | 0.9696 |
| 2023 | 0.9830 | 0.9830 | 0.9830 | 0.9830 | 0.9830 |
| 2024 | 0.9950 | 0.9950 | 0.9950 | 0.9950 | 0.9950 |
| 2025 | 0.9994 | 0.9994 | 0.9994 | 0.9994 | 0.9994 |

**margin_detail** 字段非空率

| year | buy_intensity | repay_intensity | net_fin_ratio | short_to_long | rzye_chg5 | rzye |
|---|---|---|---|---|---|---|
| 2022 | 0.5681 | 0.5681 | 0.5681 | 0.5681 | 0.5540 | 0.5681 |
| 2023 | 0.6570 | 0.6570 | 0.6570 | 0.6570 | 0.6569 | 0.6570 |
| 2024 | 0.7040 | 0.7040 | 0.7040 | 0.7040 | 0.7037 | 0.7040 |
| 2025 | 0.7254 | 0.7254 | 0.7254 | 0.7254 | 0.7252 | 0.7254 |

**moneyflow** 字段非空率

| year | net_mf_amount | net_lg | net_elg | net_sm | net_lg_ratio | net_elg_ratio | net_sm_ratio | net_mf_ratio |
|---|---|---|---|---|---|---|---|---|
| 2022 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2023 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2024 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2025 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

> 关键读法：`margin_detail` 的覆盖率从 2022 的 0.568 单调升到 2025 的 0.725（+15.7pp），因为两融标的名单逐年扩容。这意味着该域因子的横截面**本身在漂移**，跨折不可直接比较，且缺失非随机（是否可两融与市值/流动性强相关）。`cyq_perf` 从 0.970 升到 0.999，`moneyflow` 恒为 1.000。

## B. 预测力层：全字段 × 双对齐 IC 筛查

`lag0` = 源 T 日对信号 T 日；`lag1` = 源 T-1 日对信号 T 日。moneyflow/cyq_perf 于 T 日晚间发布，T+1 开盘建仓前可得，lag0 即为 PIT 安全；margin_detail 于 T+1 早 08:30 发布，早于 09:30 开盘但余量很小，故以 lag1 为保守决策口径，两者同时报告。

| source | candidate | lag | mean_rank_ic | icir | valid_days | coverage_of_universe_member_days | directional_day_share |
|---|---|---|---|---|---|---|---|
| cyq_perf | cost_50pct | 0 | -0.0394 | -0.2165 | 969 | 0.9874 | 0.5676 |
| cyq_perf | cost_50pct | 1 | -0.0384 | -0.2112 | 968 | 0.9863 | 0.5640 |
| cyq_perf | cost_dispersion | 0 | 0.0207 | 0.1935 | 969 | 0.9874 | 0.5707 |
| cyq_perf | cost_dispersion | 1 | 0.0206 | 0.1944 | 968 | 0.9863 | 0.5723 |
| cyq_perf | cost_iqr_rel | 0 | 0.0151 | 0.1462 | 969 | 0.9874 | 0.5490 |
| cyq_perf | cost_iqr_rel | 1 | 0.0161 | 0.1578 | 968 | 0.9863 | 0.5486 |
| cyq_perf | high_dist | 0 | 0.0422 | 0.3510 | 969 | 0.9874 | 0.6605 |
| cyq_perf | high_dist | 1 | 0.0408 | 0.3401 | 968 | 0.9863 | 0.6550 |
| cyq_perf | winner_rate | 0 | -0.0328 | -0.2845 | 969 | 0.9874 | 0.6058 |
| cyq_perf | winner_rate | 1 | -0.0294 | -0.2545 | 968 | 0.9863 | 0.6085 |
| margin_detail | buy_intensity | 0 | -0.0757 | -0.6053 | 951 | 0.6671 | 0.7308 |
| margin_detail | buy_intensity | 1 | -0.0715 | -0.5574 | 950 | 0.6665 | 0.7179 |
| margin_detail | net_fin_ratio | 0 | -0.0144 | -0.2775 | 951 | 0.6671 | 0.6267 |
| margin_detail | net_fin_ratio | 1 | -0.0121 | -0.2344 | 950 | 0.6665 | 0.6105 |
| margin_detail | repay_intensity | 0 | -0.0742 | -0.5843 | 951 | 0.6671 | 0.7277 |
| margin_detail | repay_intensity | 1 | -0.0701 | -0.5431 | 950 | 0.6665 | 0.7105 |
| margin_detail | rzye | 0 | 0.0062 | 0.0511 | 951 | 0.6671 | 0.5195 |
| margin_detail | rzye | 1 | 0.0066 | 0.0545 | 950 | 0.6665 | 0.5221 |
| margin_detail | rzye_chg5 | 0 | -0.0287 | -0.4652 | 946 | 0.6638 | 0.7061 |
| margin_detail | rzye_chg5 | 1 | -0.0262 | -0.4254 | 945 | 0.6631 | 0.7016 |
| margin_detail | short_to_long | 0 | -0.0181 | -0.1308 | 951 | 0.6671 | 0.5962 |
| margin_detail | short_to_long | 1 | -0.0177 | -0.1278 | 950 | 0.6664 | 0.5895 |
| moneyflow | net_elg | 0 | -0.0030 | -0.0551 | 969 | 1.0000 | 0.5222 |
| moneyflow | net_elg | 1 | 0.0013 | 0.0241 | 968 | 0.9989 | 0.5145 |
| moneyflow | net_elg_ratio | 0 | -0.0110 | -0.2316 | 969 | 1.0000 | 0.5893 |
| moneyflow | net_elg_ratio | 1 | -0.0063 | -0.1325 | 968 | 0.9989 | 0.5610 |
| moneyflow | net_lg | 0 | -0.0002 | -0.0034 | 969 | 1.0000 | 0.5015 |
| moneyflow | net_lg | 1 | 0.0044 | 0.0656 | 968 | 0.9989 | 0.5269 |
| moneyflow | net_lg_ratio | 0 | -0.0149 | -0.2268 | 969 | 1.0000 | 0.5944 |
| moneyflow | net_lg_ratio | 1 | -0.0096 | -0.1511 | 968 | 0.9989 | 0.5754 |
| moneyflow | net_mf_amount | 0 | 0.0159 | 0.1460 | 969 | 1.0000 | 0.5738 |
| moneyflow | net_mf_amount | 1 | 0.0153 | 0.1438 | 968 | 0.9989 | 0.5661 |
| moneyflow | net_mf_ratio | 0 | 0.0006 | 0.0054 | 969 | 1.0000 | 0.5108 |
| moneyflow | net_mf_ratio | 1 | -0.0002 | -0.0022 | 968 | 0.9989 | 0.4969 |
| moneyflow | net_sm | 0 | 0.0022 | 0.0308 | 969 | 1.0000 | 0.5201 |
| moneyflow | net_sm | 1 | -0.0018 | -0.0260 | 968 | 0.9989 | 0.4979 |
| moneyflow | net_sm_ratio | 0 | 0.0170 | 0.2350 | 969 | 1.0000 | 0.6109 |
| moneyflow | net_sm_ratio | 1 | 0.0119 | 0.1688 | 968 | 0.9989 | 0.5981 |

逐年 ICIR（同一口径，便于看年度稳定性）：

| source | candidate | lag | yearly_icir |
|---|---|---|---|
| cyq_perf | cost_50pct | 0 | {'2022': -0.22, '2023': -0.363, '2024': -0.272, '2025': -0.011} |
| cyq_perf | cost_50pct | 1 | {'2022': -0.211, '2023': -0.36, '2024': -0.268, '2025': -0.006} |
| cyq_perf | cost_dispersion | 0 | {'2022': 0.387, '2023': -0.034, '2024': 0.257, '2025': 0.161} |
| cyq_perf | cost_dispersion | 1 | {'2022': 0.383, '2023': -0.037, '2024': 0.273, '2025': 0.151} |
| cyq_perf | cost_iqr_rel | 0 | {'2022': 0.297, '2023': -0.037, '2024': 0.197, '2025': 0.103} |
| cyq_perf | cost_iqr_rel | 1 | {'2022': 0.303, '2023': -0.041, '2024': 0.234, '2025': 0.103} |
| cyq_perf | high_dist | 0 | {'2022': 0.302, '2023': 0.644, '2024': 0.204, '2025': 0.435} |
| cyq_perf | high_dist | 1 | {'2022': 0.288, '2023': 0.632, '2024': 0.197, '2025': 0.418} |
| cyq_perf | winner_rate | 0 | {'2022': -0.344, '2023': -0.24, '2024': -0.247, '2025': -0.319} |
| cyq_perf | winner_rate | 1 | {'2022': -0.295, '2023': -0.221, '2024': -0.236, '2025': -0.27} |
| margin_detail | buy_intensity | 0 | {'2022': -0.73, '2023': -0.589, '2024': -0.555, '2025': -0.577} |
| margin_detail | buy_intensity | 1 | {'2022': -0.681, '2023': -0.536, '2024': -0.498, '2025': -0.54} |
| margin_detail | net_fin_ratio | 0 | {'2022': -0.269, '2023': -0.269, '2024': -0.207, '2025': -0.409} |
| margin_detail | net_fin_ratio | 1 | {'2022': -0.231, '2023': -0.204, '2024': -0.135, '2025': -0.416} |
| margin_detail | repay_intensity | 0 | {'2022': -0.717, '2023': -0.571, '2024': -0.549, '2025': -0.534} |
| margin_detail | repay_intensity | 1 | {'2022': -0.667, '2023': -0.542, '2024': -0.494, '2025': -0.499} |
| margin_detail | rzye | 0 | {'2022': 0.033, '2023': -0.025, '2024': 0.129, '2025': 0.034} |
| margin_detail | rzye | 1 | {'2022': 0.036, '2023': -0.029, '2024': 0.134, '2025': 0.043} |
| margin_detail | rzye_chg5 | 0 | {'2022': -0.571, '2023': -0.445, '2024': -0.304, '2025': -0.628} |
| margin_detail | rzye_chg5 | 1 | {'2022': -0.563, '2023': -0.421, '2024': -0.266, '2025': -0.55} |
| margin_detail | short_to_long | 0 | {'2022': -0.088, '2023': -0.443, '2024': -0.068, '2025': -0.073} |
| margin_detail | short_to_long | 1 | {'2022': -0.083, '2023': -0.449, '2024': -0.062, '2025': -0.066} |
| moneyflow | net_elg | 0 | {'2022': -0.127, '2023': 0.144, '2024': -0.078, '2025': -0.153} |
| moneyflow | net_elg | 1 | {'2022': -0.009, '2023': 0.213, '2024': -0.003, '2025': -0.092} |
| moneyflow | net_elg_ratio | 0 | {'2022': -0.293, '2023': -0.05, '2024': -0.199, '2025': -0.396} |
| moneyflow | net_elg_ratio | 1 | {'2022': -0.148, '2023': 0.047, '2024': -0.114, '2025': -0.312} |
| moneyflow | net_lg | 0 | {'2022': -0.033, '2023': 0.145, '2024': -0.007, '2025': -0.139} |
| moneyflow | net_lg | 1 | {'2022': 0.049, '2023': 0.18, '2024': 0.081, '2025': -0.059} |
| moneyflow | net_lg_ratio | 0 | {'2022': -0.249, '2023': -0.175, '2024': -0.148, '2025': -0.403} |
| moneyflow | net_lg_ratio | 1 | {'2022': -0.166, '2023': -0.122, '2024': -0.057, '2025': -0.313} |
| moneyflow | net_mf_amount | 0 | {'2022': 0.094, '2023': 0.233, '2024': 0.137, '2025': 0.127} |
| moneyflow | net_mf_amount | 1 | {'2022': 0.096, '2023': 0.246, '2024': 0.099, '2025': 0.155} |
| moneyflow | net_mf_ratio | 0 | {'2022': -0.036, '2023': 0.043, '2024': 0.048, '2025': -0.042} |
| moneyflow | net_mf_ratio | 1 | {'2022': -0.026, '2023': 0.028, '2024': -0.008, '2025': 0.003} |
| moneyflow | net_sm | 0 | {'2022': 0.06, '2023': -0.079, '2024': 0.022, '2025': 0.149} |
| moneyflow | net_sm | 1 | {'2022': -0.005, '2023': -0.103, '2024': -0.046, '2025': 0.071} |
| moneyflow | net_sm_ratio | 0 | {'2022': 0.271, '2023': 0.221, '2024': 0.162, '2025': 0.367} |
| moneyflow | net_sm_ratio | 1 | {'2022': 0.208, '2023': 0.178, '2024': 0.077, '2025': 0.295} |

## C. 基准对照：冻结 v2 因子在同窗口的 |ICIR|

取自库内 `factor_rank_ic_daily`（同标签 `forward_open_5d:v1`、同 ICIR 口径），该表覆盖 2020Q1–2024Q4，故基准窗为 **2022–2024**；候选的 2022–2024 池化值由逐年 mean/std/天数按全方差定律精确还原（见 `pooled_window_icir.py`）。

| factor_id | role | days | mean_ic | icir_2224 | abs_icir |
|---|---|---|---|---|---|
| yeslab_357 | dense9 | 726 | -0.0550 | -1.1060 | 1.1060 |
| yeslab_471 | dense9 | 726 | 0.0575 | 0.8730 | 0.8730 |
| yeslab_390 | candidate_only | 726 | -0.0907 | -0.7550 | 0.7550 |
| yeslab_406 | candidate_only | 726 | -0.0791 | -0.7350 | 0.7350 |
| yeslab_464 | dense9 | 726 | -0.0795 | -0.7300 | 0.7300 |
| yeslab_381 | dense9 | 726 | -0.0879 | -0.7280 | 0.7280 |
| yeslab_389 | candidate_only | 726 | -0.0865 | -0.6560 | 0.6560 |
| yeslab_96 | dense9 | 726 | -0.0701 | -0.6390 | 0.6390 |
| yeslab_372 | candidate_only | 726 | -0.0904 | -0.6350 | 0.6350 |
| yeslab_484 | dense9 | 726 | -0.0766 | -0.5930 | 0.5930 |
| yeslab_425 | dense9 | 726 | -0.0769 | -0.5900 | 0.5900 |
| yeslab_384 | candidate_only | 726 | -0.0829 | -0.5480 | 0.5480 |
| yeslab_206 | dense9 | 726 | -0.0333 | -0.4970 | 0.4970 |

冻结集 |ICIR| 区间 **0.497（yeslab_206）– 1.106**，中位 0.656，dense9 均值 0.720。

| source | candidate | lag | icir_2022_2024 | icir_2022_2025 | coverage | vs_weakest_frozen_0.497 |
|---|---|---|---|---|---|---|
| margin_detail | buy_intensity | 0 | -0.6158 | -0.6053 | 0.6671 | 0.1190 |
| margin_detail | repay_intensity | 0 | -0.6045 | -0.5843 | 0.6671 | 0.1080 |
| margin_detail | buy_intensity | 1 | -0.5638 | -0.5574 | 0.6665 | 0.0670 |
| margin_detail | repay_intensity | 1 | -0.5606 | -0.5431 | 0.6665 | 0.0640 |
| margin_detail | rzye_chg5 | 0 | -0.4149 | -0.4652 | 0.6638 | -0.0820 |
| margin_detail | rzye_chg5 | 1 | -0.3864 | -0.4254 | 0.6631 | -0.1110 |
| cyq_perf | high_dist | 0 | 0.3301 | 0.3510 | 0.9874 | -0.1670 |
| cyq_perf | high_dist | 1 | 0.3206 | 0.3401 | 0.9863 | -0.1760 |
| cyq_perf | cost_50pct | 0 | -0.2814 | -0.2165 | 0.9874 | -0.2160 |
| cyq_perf | cost_50pct | 1 | -0.2763 | -0.2112 | 0.9863 | -0.2210 |
| cyq_perf | winner_rate | 0 | -0.2757 | -0.2845 | 0.9874 | -0.2210 |
| cyq_perf | winner_rate | 1 | -0.2504 | -0.2545 | 0.9863 | -0.2470 |
| margin_detail | net_fin_ratio | 0 | -0.2405 | -0.2775 | 0.6671 | -0.2560 |
| moneyflow | net_sm_ratio | 0 | 0.2072 | 0.2350 | 1.0000 | -0.2900 |
| cyq_perf | cost_dispersion | 1 | 0.2072 | 0.1944 | 0.9863 | -0.2900 |
| cyq_perf | cost_dispersion | 0 | 0.2033 | 0.1935 | 0.9874 | -0.2940 |
| moneyflow | net_lg_ratio | 0 | -0.1844 | -0.2268 | 1.0000 | -0.3130 |
| moneyflow | net_elg_ratio | 0 | -0.1842 | -0.2316 | 1.0000 | -0.3130 |
| margin_detail | net_fin_ratio | 1 | -0.1817 | -0.2344 | 0.6665 | -0.3150 |
| cyq_perf | cost_iqr_rel | 1 | 0.1741 | 0.1578 | 0.9863 | -0.3230 |
| cyq_perf | cost_iqr_rel | 0 | 0.1593 | 0.1462 | 0.9874 | -0.3380 |
| moneyflow | net_mf_amount | 0 | 0.1522 | 0.1460 | 1.0000 | -0.3450 |
| margin_detail | short_to_long | 0 | -0.1499 | -0.1308 | 0.6671 | -0.3470 |
| margin_detail | short_to_long | 1 | -0.1479 | -0.1278 | 0.6664 | -0.3490 |
| moneyflow | net_sm_ratio | 1 | 0.1409 | 0.1688 | 0.9989 | -0.3560 |
| moneyflow | net_mf_amount | 1 | 0.1400 | 0.1438 | 0.9989 | -0.3570 |
| moneyflow | net_lg_ratio | 1 | -0.1087 | -0.1511 | 0.9989 | -0.3880 |
| moneyflow | net_lg | 1 | 0.1016 | 0.0656 | 0.9989 | -0.3950 |
| moneyflow | net_elg_ratio | 1 | -0.0761 | -0.1325 | 0.9989 | -0.4210 |
| moneyflow | net_elg | 1 | 0.0625 | 0.0241 | 0.9989 | -0.4340 |
| margin_detail | rzye | 1 | 0.0579 | 0.0545 | 0.6665 | -0.4390 |
| margin_detail | rzye | 0 | 0.0561 | 0.0511 | 0.6671 | -0.4410 |
| moneyflow | net_sm | 1 | -0.0514 | -0.0260 | 0.9989 | -0.4460 |
| moneyflow | net_lg | 0 | 0.0327 | -0.0034 | 1.0000 | -0.4640 |
| moneyflow | net_elg | 0 | -0.0251 | -0.0551 | 1.0000 | -0.4720 |
| moneyflow | net_mf_ratio | 0 | 0.0199 | 0.0054 | 1.0000 | -0.4770 |
| moneyflow | net_mf_ratio | 1 | -0.0039 | -0.0022 | 0.9989 | -0.4930 |
| moneyflow | net_sm | 0 | 0.0015 | 0.0308 | 1.0000 | -0.4960 |

## D. 冗余与有效维度

窗口 2025（与 2026-10-13 alpha393 复检同口径，可直接比较）。三条腿：

1. **逐对 vs 冻结 15**：日内百分位秩的 |Spearman| 中位数，≥0.90 BLOCK、≥0.80 MONITOR
2. **逐对 vs 其他幸存者**：v1 脚本缺这条腿，而 margin 族在代数上互相纠缠（`net_fin_ratio = buy_intensity - repay_intensity`，`rzye_chg5` 是其 5 日平滑）
3. **给定 dense9 的集合增量保留率**：残差 RankIC / 原始 RankIC，门槛 ≥ +0.30（**有符号**）

| candidate | source | lag | used_safe_lag | icir_2022_2025 | coverage | worst_frozen_rho | worst_frozen_partner | worst_survivor_rho | worst_survivor_partner | retention_given_dense9 | retention_flag | residual_rank_ic | clears_icir_bar | clears_residual_ic_bar | dim_retention_aware |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high_dist | cyq_perf | 0 | True | 0.351 | 0.999 | 0.180 | yeslab_443_reg30_max20_v1 | 0.207 | repay_intensity | 0.622 | PASS | 0.028 | False | True | True |
| buy_intensity | margin_detail | 1 | True | -0.557 | 0.722 | 0.574 | yeslab_425 | 0.881 | repay_intensity | 0.254 | WEAK | -0.019 | True | False | False |
| repay_intensity | margin_detail | 1 | True | -0.543 | 0.722 | 0.576 | yeslab_425 | 0.881 | buy_intensity | 0.230 | WEAK | -0.016 | True | False | False |
| rzye_chg5 | margin_detail | 1 | True | -0.425 | 0.722 | 0.198 | yeslab_375 | 0.261 | buy_intensity | 0.370 | PASS | -0.012 | False | False | True |
| winner_rate | cyq_perf | 0 | True | -0.285 | 0.999 | 0.506 | yeslab_390 | 0.264 | buy_intensity | -0.306 | SIGN_FLIP | 0.010 | False | False | False |
| net_fin_ratio | margin_detail | 0 | False | -0.277 | 0.725 | 0.174 | yeslab_372 | 0.100 | winner_rate | 0.455 | PASS | -0.009 | False | False | True |

幸存者两两 |Spearman| 中位数矩阵：

| index | buy_intensity | high_dist | net_fin_ratio | repay_intensity | rzye_chg5 | winner_rate |
|---|---|---|---|---|---|---|
| buy_intensity | 0.000 | 0.201 | 0.065 | 0.881 | 0.261 | 0.264 |
| high_dist | 0.201 | 0.000 | 0.033 | 0.207 | 0.042 | 0.106 |
| net_fin_ratio | 0.065 | 0.033 | 0.000 | 0.064 | 0.026 | 0.100 |
| repay_intensity | 0.881 | 0.207 | 0.064 | 0.000 | 0.103 | 0.251 |
| rzye_chg5 | 0.261 | 0.042 | 0.026 | 0.103 | 0.000 | 0.092 |
| winner_rate | 0.264 | 0.106 | 0.100 | 0.251 | 0.092 | 0.000 |

三种贪心规则下的『有效新维度』计数：

- `cap_0.90` → **6**：['buy_intensity', 'repay_intensity', 'rzye_chg5', 'high_dist', 'winner_rate', 'net_fin_ratio']
- `cap_0.80` → **5**：['buy_intensity', 'rzye_chg5', 'high_dist', 'winner_rate', 'net_fin_ratio']
    - 拒 repay_intensity：['admitted:buy_intensity=0.881']
- `cap_0.80_retention_aware` → **3**：['rzye_chg5', 'high_dist', 'net_fin_ratio']
    - 拒 buy_intensity：['retention:0.254=WEAK']
    - 拒 repay_intensity：['retention:0.23=WEAK']
    - 拒 winner_rate：['retention:-0.306=SIGN_FLIP']

## E. 复现

```bash
cd /data/zhangyudong/qrv2_research_artifacts/daily_domain_feasibility_20261013
python fetch_daily_domains.py        # 抓取三源 2022-2025 -> raw/（可断点续跑）
python screen_daily_domains.py       # IC 筛查 -> analysis/ic_screen{.json,_summary.csv}
python pooled_window_icir.py         # 池化到基准窗 -> analysis/ic_screen_pooled_windows.csv
python redundancy_survivors.py 0.25  # 冗余三腿 -> analysis/redundancy_survivors.json
python greedy_caps.py                # 决策表与贪心规则 -> analysis/decision_table.csv
python coverage_drift.py             # 覆盖率漂移 -> analysis/coverage_drift.csv
python make_report.py                # 本报告
```

原始面板（594 MB，12 个 parquet）留在服务器 `raw/`，不入 Git；本目录只提交脚本与小件摘要。
