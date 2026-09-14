# M2 评价阶段的事前预测登记（在评价跑之前写下）

登记时刻（UTC）：2026-09-14T18:19:37+00:00。
来源测量：`dense_intersection_2022_2025_v2.json`（sha256 `0feb740b88d0ac12…`），
仪器 `screen_dense_intersection_coverage.py`，因子源 `dense_v2`（9 个），地板 0.9（从 `gates.py` 解析）。

**M2 发射时用的基线工件与 `m0r_v2` / `m1_full_v1` 是同一套**（`baseline_v2/baseline_oos` + `baseline_v2/report.json`），候选一侧给每个有效标签行都打分，
所以这个日集是 M2 **已经拿到的输入**的性质，可以在它的评价阶段跑完之前就写死。

## 预测

* **G6 = 945/969** = 0.9752321981424149
* 候选一侧逐日覆盖率恒为 **1.0000**；`days_dropped_by_the_baseline_side` = **24/24**

| 折 | TEST 年 | 天数 | 跌破地板 | **预测可评天** | 覆盖率 min | 覆盖率 median |
|---|---|---|---|---|---|---|
| F1 | 2022 | 242 | 1 | **241** | 0.8950819672131147 | 0.9253990284831323 |
| F2 | 2023 | 242 | 23 | **219** | 0.8902674386545354 | 0.9092936592336784 |
| F3 | 2024 | 242 | 0 | **242** | 0.9711145194274029 | 0.9924234358393046 |
| F4 | 2025 | 243 | 0 | **243** | 0.980854969840021 | 0.9859442882698697 |

不可评的 24 天：

```
2022-07-29, 2023-04-28, 2023-05-10, 2023-05-11, 2023-05-12, 2023-05-16, 2023-05-19, 2023-06-21, 2023-06-26, 2023-06-27, 2023-06-28, 2023-06-29, 2023-06-30, 2023-07-05, 2023-07-06, 2023-07-07, 2023-07-10, 2023-07-11, 2023-07-12, 2023-07-13, 2023-07-14, 2023-07-17, 2023-07-18, 2023-07-19
```

## 什么情况下这份预测算被推翻

If M2's signed report disagrees with any of these -- the 945/969 day count, a fold's valid-day count, a per-year coverage min or median, or candidate daily coverage of 1.0000 -- then an input changed between m0r_v2/m1_full_v1 and M2 (baseline artifact, labels, factor set, or the floor) and M2's numbers must not be accepted until the difference is explained. Agreement is evidence the evaluation consumed the same frozen inputs; it is not evidence that the candidate is good.

## 什么情况下它**不算**被验证

一致只说明 M2 的评价消费了同一套冻结输入（基线工件、标签、因子集、地板都没变），
**不说明候选好坏**，也不说明那 24 天该被丢掉——那 24 天的成因与处置见
`star_industry_coverage/root_cause.md` 与 `coverage_gates_vs_the_intersection_floor/finding.md`。
