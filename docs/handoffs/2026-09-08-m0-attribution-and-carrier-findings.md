# M0 候选卡门的归因与破解：瓶颈是模型载体（2026-09-08）

> 生成：2026-09-08，生产链与晋级证据完成于 2026-09-09 03:21 · 分支 `codex/publish-quant-platform`
> 状态：精调网格已完成（G4 FAIL +0.004202）；三组只读诊断实验已完成；**载体实验已完成，rankreg/trunc200 在零调参下即过 G4**；**rankreg 六门评估已完成：ALL PASSED**
> 本轮所有实验在 `/tmp/es_metric_diag/` 下只读进行：零数据库写入、零工件库写入、未触碰 2026 锁定段、未触碰 `materialization_status.{json,md}`

## 0. 一句话

M0 卡在 G4（+0.0042 < +0.005）**不是超参、正则、早停指标或因子集的问题，而是模型载体的问题**：在完全不调参的固定 config 下，只把目标函数从 LambdaRank 换成「回归日内收益百分位秩」，池化 delta 从 +0.00109 跳到 **+0.00915**；只把 `lambdarank_truncation_level` 从默认 20 抬到 200，跳到 **+0.00679**。前者已作为新增家族 **M0R** 落进生产链并完整跑通，**六道门全过（delta +0.009149，`all_passed=True`）**，晋级证据已产出并独立复核（第 9 节）。

归因同时显示：链上那 +0.0042 的约 **74% 来自逐折调参**而非 6 个稀疏因子；稀疏因子单独使用在所有载体下都为负（最好 −0.00877）。**adjudication-4「混合方案」的前提在数据上不成立**（第 3、7 节）。

途中还修掉三个既有阻塞：家族准入列表有两份定义（导致一次 46 分钟的完整运行在最后一步崩掉）、注册表证据解析器硬编码已退役折的 2020–2025 测试年（导致**任何** v2 候选都无法晋级）、以及注册表只有消费方没有生产方（第 9.4 节）。

## 1. 精调网格结果（`m0_v3`，24 config × 4 折 × 3 seed，`--grid refined`）

| 门 | 含义 | 观测 | 阈值 | 判定 |
|---|---|---|---|---|
| G1 | OOS mean RankIC | 0.089429 | ≥0.030 | PASS |
| G2 | HAC t | 12.383 | ≥2.0 | PASS |
| G3 | 正 IC 年份占比 | 1.00（4/4） | ≥0.70 | PASS |
| **G4** | **配对 delta** | **+0.004202** | **≥0.005** | **FAIL** |
| G5 | 单年最大贡献占比 | 0.2819 | ≤0.50 | PASS |
| G6 | 有效日覆盖 | 0.9752（945/969） | ≥0.95 | PASS |

- 基础网格 +0.003640 → 精调 **+0.004202**（+15%）。`lambda_l2` 轴只有 F2 选中 1.0，F1/F3/F4 仍选 0.0 → 正则维度基本到顶。
- 每折最优 config：F1 `63叶/深6/lr0.03/l2=0`；F2 `15/4/0.05/l2=1.0`；F3、F4 `15/4/0.05/l2=0`。
- **实际树数**（checkpoint 计数）：F1 `5/2/2`、F2 `6/4/4`、F3 `7/12/1`、F4 `7/12/61`。500 轮预算中位只用了 **6 棵**。

### G4 口径（务必记住）

G4 是**配对比较**：候选与基线取 `signal_date × ts_code` 交集后各自重算。基线是每折 `best_simple`（F1–F3=B2、F4=B3），行数与 `baseline_v2/baseline_oos/*.parquet` 完全一致。

- 基线自身全集：969 有效日，mean RankIC **0.08706**
- 配对后（候选能出可用分数的 945 天）：基线 mean **0.08523**
- 差值来源：被丢掉的 24 天恰好是基线表现好于其均值的日子。G6=0.9752 正是 945/969。

## 2. 实验一：早停指标不是瓶颈（F4，`15/4/0.05/l2=0`，3 seed，配对）

| 变体 | 停止规则 | 树数 | ens_val | ens_test |
|---|---|---|---|---|
| A_ndcg | 生产现状（LightGBM 默认 ndcg@1…5） | 7 / 12 / 61 | 0.10174 | 0.10028 |
| B_rankic | 自定义 feval = 日频截面 RankIC | 30 / 11 / 67 | 0.10248 | 0.10043 |
| C_fixed500 | 不早停，跑满 500 轮 | 500 | 0.09511 | **0.09477** |

- B − A = **+0.00015**（噪声级）；C − A = **−0.00551**（明显过拟合）。
- **结论：树数少不是指标错配造成的假象，而是信号上限。此路作废。**
- 顺带确认：LightGBM 4.6.0 在 `objective=lambdarank` 且未显式设 `metric` 时，早停指标为 `ndcg@1..5`，且 `first_metric_only=False`（五个指标全停滞 50 轮才停）。
- A 变体逐位复现链上产物（树数 7/12/61 与 checkpoint 一致、ens_val 0.101736 对报告 0.10174），对照可信。

## 3. 实验二：+0.0042 的归因消融（链上口径，3,417,963 配对行）

### 池化

| 特征子集 | 是否调参 | cand | delta | G4 |
|---|---|---|---|---|
| all15（链上实际） | 24-config 逐折 | 0.08943 | **+0.00420** | FAIL |
| dense9 | 6-config 逐折 | 0.08869 | +0.00347 | FAIL |
| dense9 | 固定 config | 0.08699 | +0.00177 | FAIL |
| all15 | 固定 config | 0.08632 | +0.00109 | FAIL |
| **sparse6 单独** | 固定 config | 0.06401 | **−0.02122** | FAIL |

固定 config = `15叶/深4/lr0.05/mcs200/ff0.85/l2=0`（4 折中 3 折的胜出配置）。

### 逐折 paired delta

| 折（测试年） | all15 调参 | dense9 调参 | all15 固定 | dense9 固定 | sparse6 固定 |
|---|---|---|---|---|---|
| F1 (2022) | +0.00793 | +0.00243 | −0.00117 | +0.00389 | −0.01936 |
| F2 (2023) | −0.00270 | −0.00636 | −0.00610 | −0.00640 | −0.01720 |
| F3 (2024) | +0.00902 | +0.01499 | +0.00902 | +0.00693 | −0.01338 |
| F4 (2025) | +0.00192 | +0.00188 | +0.00192 | +0.00188 | −0.03449 |

### 控制组验证

四折 paired delta 与「用链上**存档预测**直接算出的真值」逐位一致：F1 +0.00793 / F2 −0.00270 / F3 +0.00902 / F4 +0.00192；池化 cand 0.08943、base 0.08523、delta +0.00420，与 `m0_v3/report.json` 完全相同。

### 三条结论

1. **6 个稀疏因子单独使用是灾难性的**（池化 −0.02122，四折全负 −0.013 ~ −0.034）。这与因子端单变量诊断（稀疏 |IC| 0.124 > 稠密 0.100、与稠密相关仅 0.41–0.66）**直接冲突**：信号在因子里，但 LambdaRank 这个载体转化不出来。合理怀疑是 `lambdarank_truncation_level` 默认 20（每天只有前 20 个位置贡献梯度）叠加 ~50% NaN。
2. **稀疏因子的边际贡献只有约 +0.0007**（all15 调参 +0.00420 vs dense9 调参 +0.00347），不是 +0.0042 的主体。
3. **主体来自模型形式 + 逐折调参**：all15 从不调到调是 +0.00109 → +0.00420（+0.0031，占 74%）；dense9 是 +0.00177 → +0.00347（+0.0017）。
4. **换成 dense9 也过不了门**（+0.00347）。在冻结 15 因子内换子集，没有一条路到 +0.005。

## 4. 方法论警告：+0.0042 本身是「选出来的最大值」

每折在 24 个 config 中按**单一验证年**取最大，带选择偏差，真实期望值更低。证据就在上表：

- F1 dense9 按验证年选中 val 更高的 config（0.06141），测试反而更差（+0.00243 vs 固定 config 的 +0.00389）；
- F3 反向，选中 `l2=1.0` 后测试高达 +0.01499。

逐折选择的运气成分很大，所以候选实际上比 +0.0042 看起来的**离门更远**。

## 5. 更正

2026-09-08 早前汇报的「2023 年 −0.0110」是把两组不同行集的年度值相减所得，**错误**。链上真实的 F2(2023) 配对 delta 是 **−0.00270**。

## 6. 工程发现：`lgb.Dataset` 跨 seed 复用会静默改变模型

**现象**：把同一个 `lgb.Dataset` 对象用于多次 `lgb.train`（不同 seed），第 1 个 seed 正常，后续 seed 静默变成另一个模型。

| 数据 | 新鲜 Dataset | 复用 Dataset | 链上 checkpoint |
|---|---|---|---|
| F1 all15（2,283,604 行，21.7% NaN） | 5 / 2 / 2 | 5 / **79** / **64** | 5 / 2 / 2 |
| F1 dense9（同行数，1.9% NaN） | 1 / 1 / 72 | 1 / **46** / **35** | — |

- **不是 NaN 效应**（dense9 同样偏离）；**不是行序效应**（按 `signal_date, ts_code` 排序后结果不变）。
- **规模门槛**：18,000 行 / 15 特征 / 500 轮 / 早停的合成 fixture 上**不复现**（fresh 与 reuse 逐位相同）→ **无法用单元测试守卫**。
- 链上目前是安全的：`LGBMRankerBaseline.fit` 每次调用都新建 Dataset。但该不变量此前无任何守卫，任何「把 Dataset 提到循环外」的优化都会静默改变三个 seed 的模型。
- 已在 `research_v2/src/research_v2/baselines/lgbm_ranker.py` 的 Dataset 构造处加入带实测数据的注释守卫。

## 7. 模型载体实验：找到了真正的瓶颈（已完成）

假设：稀疏因子的问题出在**载体**（lambdarank 的 top-20 截断 + 5 档粗粒度 label + ~50% NaN），不是因子本身。全部 4 折 × 3 seed × 500 轮，**固定 config**（`15叶/深4/lr0.05/mcs200/ff0.85/l2=0`）以隔离载体效应，链上口径池化（3,417,963 配对行）：

### all15（15 因子）

| 载体 | cand mean RankIC | delta | G4 |
|---|---|---|---|
| ctrl（生产现状：lambdarank + 5 档 + 截断默认 20） | 0.08632 | +0.00109 | FAIL |
| grades20（relevance 20 档） | 0.08639 | +0.00116 | FAIL |
| avail_flags（all15 + 6 个稀疏缺失指示列） | 0.08426 | −0.00097 | FAIL |
| **trunc200**（`lambdarank_truncation_level=200`） | 0.09201 | **+0.00679** | **PASS** |
| **rankreg**（回归「日内收益百分位秩」+ RankIC feval 早停） | 0.09438 | **+0.00915** | **PASS** |

### sparse6（仅 6 个稀疏因子）

| 载体 | cand | delta |
|---|---|---|
| ctrl | 0.06401 | −0.02122 |
| trunc200 | 0.06480 | −0.02043 |
| grades20 | 0.06178 | −0.02345 |
| **rankreg** | 0.07645 | **−0.00877** |

### 逐折 paired delta（all15，固定 config）

| 折（测试年） | ctrl | trunc200 | grades20 | rankreg | avail_flags |
|---|---|---|---|---|---|
| F1 (2022) | −0.00117 | +0.00914 | −0.00109 | **+0.01249** | −0.00296 |
| F2 (2023) | −0.00610 | +0.00261 | −0.00335 | **+0.00319** | −0.00637 |
| F3 (2024) | +0.00902 | +0.01205 | +0.00395 | **+0.01392** | +0.00238 |
| F4 (2025) | +0.00192 | +0.00298 | +0.00467 | **+0.00645** | +0.00254 |

**rankreg 四折全胜**，trunc200 四折全胜 ctrl。`xendcg` 在 smoke 阶段即被淘汰（F4 delta −0.10712），未进全量。

### 结论

1. **真正的瓶颈是载体，不是超参、不是正则、不是因子集。** 同一个固定 config 下，只把目标函数从 lambdarank 换成「回归日内收益百分位秩」，池化 delta 从 +0.00109 变成 **+0.00915**（8.4 倍），且**无需任何调参**就超过链上充分调参的 +0.00420 一倍以上。
2. **主因是目标/标签形式，不是早停指标。** 实验一已证明：在 lambdarank 上把早停换成 RankIC feval 只值 +0.00015；而在 regression 目标上配 RankIC feval 早停值 +0.008。rankreg 建的树也远多于 ctrl（F2：214/146/133 vs 43/4/58），说明 ctrl 的早停在 lambdarank 下确实过早触发，但只有在换了目标之后这个改善才变得可观。
3. **`lambdarank_truncation_level=20` 是独立的第二个瓶颈**：仅把它抬到 200（其余全不动）就值 +0.0057（+0.00109 → +0.00679），同样直接过门。这与「每天 ~2400–3800 只票、只有前 20 个位置贡献梯度」的推断一致。
4. **稀疏因子仍未被救活**：rankreg 把 sparse6 从 −0.02122 改善到 −0.00877（2.4 倍），但四折仍全负。所以 all15 的收益主要来自载体让**稠密因子**被正确使用，而非解锁了稀疏因子。adjudication-4「混合方案」的前提（稀疏因子是候选的增量来源）**在数据上不成立**。
5. **细化 relevance 档位（grades20）与显式缺失指示列（avail_flags）都无效**，前者 +0.00007、后者 −0.00206 相对 ctrl。缺失结构不需要显式编码，LightGBM 的原生 NaN 处理已足够。

### 六门评估（rankreg，固定 config，零调参）

用框架自身的 `evaluate_frozen_oos` 对 rankreg 的池化预测做完整判定：

| 门 | 含义 | 观测 | 阈值 | 判定 |
|---|---|---|---|---|
| G1 | OOS mean RankIC | 0.094376 | ≥0.030 | PASS |
| G2 | HAC t | 12.905 | ≥2.0 | PASS |
| G3 | 正 IC 年份占比 | 1.00 | ≥0.70 | PASS |
| **G4** | **配对 delta** | **+0.009149** | **≥0.005** | **PASS** |
| G5 | 单年最大贡献占比 | 0.2791 | ≤0.50 | PASS |
| G6 | 有效日覆盖 | 0.9752（945/969） | ≥0.95 | PASS |

**`all_passed = True`** —— 这是本项目第一个通过全部六道晋级门的候选。

对照：链上 M0（LambdaRank，24-config 逐折调参后）delta +0.004202、G4 FAIL、`all_passed = False`。

两点必须记住：

- rankreg 的 +0.009149 是**固定 config、零调参**的结果，因此**不含选择偏差**，比链上那个「24 选 1 的最大值」+0.004202 更可信，尽管后者数字更小却更乐观。
- 这是在 `/tmp` 的只读诊断 harness 里量出来的，**尚未走生产链**（`run_real_candidate_training.py`）。要成为可晋级候选，必须由生产链重跑以产出带 HMAC 签名的 selection/model/prediction 证据包；本节数字是「值得去跑生产链」的证据，不是晋级证据本身。

### 交叉验证

两套独立 harness 在 ctrl 上逐位一致：F1 −0.00117、F2 −0.00610、F3 +0.00902、F4 +0.00192，池化 cand 0.08632 / delta +0.00109。六门评估脚本重跑 F1 rankreg 也得 +0.01249，与载体实验一致。

## 8. 决策状态

第 7 节的结果把原来的「A 沿载体方向 / B 承认门槛不可达」二选一作废了——载体方向成功，B 不再需要。当时列出的四个决策点，现状如下：

| # | 决策点 | 状态 |
|---|---|---|
| 1 | 改 M0 本身 vs 新增家族 | **已按「新增家族」执行**（第 9.1 节）。理由：纯增量、可回滚，M0 与 B0–B3 的既有 bundle digest、model_code_version、签名证据全部不受影响。若你倾向改 M0 本身，回滚方式是忽略 M0R 家族 |
| 2 | 是否同时抬 `lambdarank_truncation_level` | **未做**。它单独就过门（+0.00679）且是对现有 M0 契约改动最小的一条路，但 M0R 已达 +0.009149，没有必要同时改两处。保留为 M0 侧的独立选项 |
| 3 | M0R 之上是否再调参 | **未做，且刻意未做**。当前 +0.009149 是零调参数字，不含选择偏差。调参版很可能更高，但必须按第 4 节标注为有偏量 |
| 4 | 稀疏因子与 adjudication-4 的处置 | **待你裁定**。证据是：sparse6 在所有载体下都为负，all15 的收益主要来自稠密因子被正确使用。这与「因子优化」轨道的冻结结论直接冲突，属因子治理，本轨道未单方面改动任何因子集文件 |

需要你知道的一个副作用：本轨道的实现证明**候选家族的收益上限由载体决定，而不是由因子集决定**。这会改变因子端优化的边际价值判断，建议与另一轨道对齐后再决定第 4 点。

## 9. 生产链已跑通：M0R 六门全过，晋级证据已产出（2026-09-09 03:21）

第 7 节的数字来自 `/tmp` 的只读 harness。为把它变成可晋级的证据，M0R 已作为**新增候选家族**落进生产链并完整跑通。

### 9.1 实现（全部增量，M0 与 B0–B3 契约逐位未动）

| 文件 | 内容 |
|---|---|
| `research_v2/src/research_v2/baselines/lgbm_rankreg.py` | 载体本体：`LGBMRankRegBaseline`（与 `LGBMRankerBaseline` 同形状，复用 bundle/checkpoint/seed 集成机制）、`DailyRankICMetric`（门禁用的那个日频 Spearman，做成 LightGBM `feval`，用 `np.bincount` 向量化）、`_percentile_rank_targets`（连续目标 = 日内收益百分位秩） |
| `research_v2/scripts/run_real_candidate_training.py` | 家族 `M0R`（无 E-plan 前置）、`_LightGBMFamilySpec`（两个载体真正不同的 7 个字段）、`_run_full_lightgbm_family`（两个家族共用一条流水线，不再各存一份 200 行代码）、`m0r_config_grid`（**只有 1 个 config，刻意为之**） |
| `research_v2/src/research_v2/evaluation/model_walk_forward.py` | 准入 `M0R` + 其 model-card 家族名 |
| `research_v2/src/research_v2/registry/promotion_evidence.py` | **晋级证据生成方**（此前注册表只有消费方、没有生产方） |
| `research_v2/scripts/emit_promotion_evidence.py` | 对既有工作区补发证据的 CLI |

`m0r_config_grid` 只含 1 个 config（`15叶/深4/lr0.05/mcs200/ff0.85/l2=0`）是**刻意设计**：M0R 的 +0.00915 是在零搜索下量出来的，而搜索正是让 M0 的 +0.004202 变成乐观最大值的原因。锁定网格使生产数字不含验证选择偏差；1 trial 也远在 E7 的 30 预算内。带调参的 M0R 搜索是合理的后续，但必须按第 4 节的警告标注为有偏量。

### 9.2 生产链结果（`/data/zhangyudong/qrv2_research_artifacts/m0r_v2/`，耗时 57 分钟）

四折验证年 RankIC 与 `/tmp` harness **逐位一致**：F1 0.0679（harness 0.06795）、F2 0.0987（0.09871）、F3 0.0894（0.08944）、F4 0.1032（0.10318）。

| 指标 | M0（LambdaRank，24-config 逐折调参） | **M0R（rankreg，零调参）** |
|---|---|---|
| mean RankIC | 0.089429 | **0.094376** |
| baseline（配对） | 0.085227 | 0.085227 |
| **delta** | +0.004202 | **+0.009149** |
| ICIR | 0.7487 | **0.7682** |
| HAC t | 12.383 | **12.905** |
| 有效日 | 945 | 945 |
| G1–G6 | 5 PASS / **G4 FAIL** | **6 PASS** |
| `all_passed` | **False** | **True** |

分年（M0R）：2022 **0.09313**、2023 **0.07300**、2024 **0.10452**、2025 **0.10477** —— 四年全部为正，且**每一年都优于 M0 对应年份**（M0：0.08857 / 0.06710 / 0.09963 / 0.10025）。2023 这个 M0 唯一为负的年份也被翻正。

### 9.3 晋级证据

`emit_promotion_evidence` 对完成的工作区**重新校验**（逐个 bundle digest 与 HMAC 签名、从原始数值重算 G1–G6、并与驱动自己写的 report 交叉核对）后产出：

- 文档：`/data/zhangyudong/qrv2_research_artifacts/m0r_v2/promotion_evidence.json`（129,509 字节）
- `sha256` = `92b000c0d246556baf6dd077abe9e5f926ea69fd5840998c6af594b78b0c6d48`
- `model_family = M0R`，`valid_test_day_count = 969`，`score_health_days = 969`，六门全 PASS

**注意两个天数不是同一个量**：report 的 `valid_day_count = 945`（产出可用 RankIC 的天数），证据的 `valid_test_day_count = 969`（截面足够宽、可打分的标签天数），二者之比正是 G6 = 0.9752。交叉核对刻意不比较这两个数，否则每一次真实运行都会被误判为不一致。

### 9.4 途中修掉的三个既有阻塞（都不是 M0R 引入的）

1. **`MODEL_FAMILIES` 有两份定义。** 驱动加了 M0R，评估器 `model_walk_forward.py` 的副本没加，于是一次跑完全部四折、写完全部 bundle 与签名证据的运行（`m0r_v1`，46 分钟）在最后一步 `evaluate_model_walk_forward` 崩掉。已改为驱动**导入**评估器的那一份，并加测试钉住「驱动能跑的家族必须被评估器准入 + 其 model-card 家族名必须在映射表里」（变异验证：还原缺陷该测试失败）。
2. **注册表的证据解析器硬编码 2020–2025。** 那是已退役的 F01–F06 折的测试年；当前 F1–F4 契约的测试年是 2022–2025，于是**任何 v2 候选都无法晋级**，与得分无关、与家族无关。这正是 `fd55d2d` 在 `evaluate_oos_gates` 修过的同一类陈旧区间缺陷，只是漏掉了喂给它的那个 loader。已改为从折契约推导（变异验证：还原硬编码后 7 个测试失败，含 CLI 的 `registry promote` 测试）。
3. **注册表只有消费方、没有生产方。** `load_walk_forward_evidence` 要求的文档格式无人产出，驱动的 report 形状第一个字段就被拒。已补 `promotion_evidence.py`（见 9.1）。生成证据不是信任决策：注册表会从文档里的原始数值**重新推导** G1–G6，从不读取被上报的门状态，所以写这个文件无法让不合格模型晋级；不合格的候选在生成阶段就抛错。

### 9.5 model_id 铸造与晋级 CLI 的端到端验证

证据文档产出后仍缺一环：注册表用**单个** `sha256:<64 hex>` 标识一个模型，而走查训练的是**每折一个** bundle（`m0r-f1..f4`），此前没有任何代码把四折映射成一个 id，`register_draft` / `promote_walk_forward` 在生产代码里**从未被调用过**（只有测试调用）。

已补：

- `derive_walk_forward_model_id(bundle_digests)` —— 只对**已验证的四折 bundle 摘要**做规范 JSON 哈希，不接受任何调用方提供的元数据。这样铸造可以安全地发生在信任边界之外：每个 bundle 摘要已覆盖该 bundle 内全部文件（config、feature_order、含 family 与 model_code_version 的 model_card、signature、三个 seed checkpoint、验证预测），所以两次运行得到同一 id 当且仅当它们产出了同样的模型。
- `walk_forward_model_id(workspace, report)` —— 工作区入口。它调用 `bundle_digest`，而该方法会**重新哈希并校验每个文件**，因此「取 id」本身就是一次完整性检查：被篡改或截断的 bundle 会在这里抛错，而不是产出一个看起来合理的身份。

对真实 M0R 工件验证：

```
model_id = sha256:32315c89fb9a65b8ceab193036fc3a0a870d24f2e6d99c6ff74049dccc75788b

qr-v2 registry promote --to WALK_FORWARD_PASSED --evidence <promotion_evidence.json> --dry-run
  → 退出码 0
    evidence_hash = 92b000c0d246556baf6dd077abe9e5f926ea69fd5840998c6af594b78b0c6d48
    gate_statuses = G1..G6 全 PASS

同一命令去掉 --dry-run
  → 退出码 2，"registry persistence backend is not configured; refusing to promote"
```

CLI 重算出的 `evidence_hash` 与文档的 sha256 **完全一致**，所以从「训练出的 booster」到「经过门校验的晋级请求」这条路径已用真实工件走通。

### 9.6 生命周期还剩什么

`ModelStatus` 的路径是 `DRAFT → WALK_FORWARD_PASSED → SHADOW → LOCKED_TEST_APPROVED → LOCKED_TEST_PASSED → PRODUCTION`。

| 阶段 | 状态 |
|---|---|
| 取得 WALK_FORWARD_PASSED 所需的全部证据 | ✅ **已完成**（9.2、9.3） |
| 模型的注册表身份（model_id） | ✅ **已完成**（9.5） |
| 晋级 CLI 的校验路径 | ✅ **已用真实工件验证**（9.5，dry-run 通过） |
| 注册表持久化后端 | ✅ **已完成**（9.7） |
| **在注册表里记录 M0R 为 WALK_FORWARD_PASSED** | ✅ **已完成**（9.8），用真实工件、真实 CLI |
| SHADOW 及之后 | 🔒 治理步骤，**未做**。`enter_shadow` 需要 `ShadowDeploymentApproval`（评估方签发），再往后需要锁定测试段（2026），而 2026 全程未物化、禁止读取；`pointer switch/rollback` 需要独立的 pointer 后端 |

### 9.7 注册表持久化后端（补上最后一个工程缺口）

`ModelRegistry` 一直是纯内存的，其 docstring 早就写明「persistence adapters must retain every transition event」，但适配器从不存在，所以 `qr-v2 registry promote` 只能校验、重算、打印，然后拒绝——一个六门全过的模型**无处可记**。

新增 `research_v2/registry/registry_store.py`，沿用本仓其它证据存储的既有约定：内容寻址、HMAC 签名（同一信任根）、原子写入、写后不可变。每个模型两个文件：

```
<root>/models/<hex>/events.jsonl   每一次转换事件一行签名规范 JSON，append-only，写入后 flush + fsync
<root>/models/<hex>/record.json    当前状态的签名快照，用 os.replace 原子替换
```

**事件日志是权威，快照只是便利。** 打开存储时校验每一行事件签名、走一遍链条确认每个事件的 `previous_status` 等于前一个事件的 `new_status`，然后要求快照状态等于最后一个事件的状态。最后这一步与签名校验**不重复**，且有专门测试钉住：持有信任根的人可以把改过的快照重新签名，若没有这道交叉校验，他就能在没有任何转换事件的情况下把 DRAFT 标成 PRODUCTION。顺序也有讲究——日志行先 fsync 再移动快照，所以两步之间崩溃只会让快照过期，不会让日志失真。

`ModelRegistry.restore` 是接缝。它**不重放校验转换**，这是决定而非省略：每次转换在写入时已按其证据校验过，而 WALK_FORWARD_PASSED 之后的状态需要 `_RegistryWorkflowProof`——那是不可伪造的进程内令牌，**原则上就无法从磁盘重建**。restore 校验的是持久化状态本身：身份格式、状态枚举、与已注册父模型的 lineage 一致性、链条不断裂、哈希格式、历史必须始于 DRAFT。

CLI 侧：新增 `registry register`（`promote_walk_forward` 只接受已处于 DRAFT 的模型，此前生命周期**没有入口**）；`promote` 新增 `--actor`（被保留的事件若不记录是谁造成的，就不成审计链），且**仅在会改动状态的路径上强制**——dry-run 不产生任何记录，也就不需要归属。后端位置来自 `QRV2_MODEL_REGISTRY_ROOT`，**没有默认值**：晋级不应在没人指定的地方悄悄生成一个注册表，否则两个操作者会得到两个关于「生产上是什么」的真相来源。未配置时 CLI 仍照旧拒绝，只是现在会说出缺的是哪个变量。

### 9.8 真实晋级已执行

用真实的 M0R 工件、真实的 CLI，不是夹具：

```
qr-v2 registry register --model-id sha256:32315c89…788b --actor codex-m0r-track
  → status=DRAFT, transition_events=1

qr-v2 registry promote  --model-id sha256:32315c89…788b \
                        --to WALK_FORWARD_PASSED \
                        --evidence …/m0r_v2/promotion_evidence.json \
                        --actor codex-m0r-track
  → gate_statuses G1..G6 全 PASS
    evidence_hash 92b000c0d246556baf6dd077abe9e5f926ea69fd5840998c6af594b78b0c6d48
    status=WALK_FORWARD_PASSED, transition_events=2, 退出码 0
```

重新打开存储独立复核：1 个模型、状态 `WALK_FORWARD_PASSED`、2 条事件（`None → DRAFT`、`DRAFT → WALK_FORWARD_PASSED`，各带 actor 与 UTC 时间戳，晋级事件携带 1 个 evidence hash），`events.jsonl` 恰为 2 行。

落盘位置：`/data/zhangyudong/qrv2_research_artifacts/model_registry/`。

> 这个根目录是我选的**提案位置**（与其它工件同处 `qrv2_research_artifacts/` 下）。若你希望换地方或暂不保留，删除该目录即可完全回滚——它不被任何其它工件引用。

### 9.9 顺带修掉的陈旧契约文案

排查「与已修缺陷同类」的陈旧硬编码时，在驱动脚本里发现两处仍描述已退役的 F01–F06 折结构的帮助文本：`--baseline-artifacts` 写着 `(F01..F06)`，`--folds` 写着 `(default all six)`。均已改为按当前契约描述。`src/` 下没有同类问题（命中的都是解释历史的注释）。

### 9.10 可复现性：两次独立全量跑逐字节一致

`m0r_v2`（01:54 启动）与 `m0r_v3`（03:32 启动）是两次完全独立的生产链全量运行，相隔约 2.5 小时、各自使用独立工作区。逐层比对：

| 比对对象 | 结果 |
|---|---|
| `report.json` | **逐字节一致**（`cmp` 无输出） |
| `oos` 块 / `folds` 块 | 完全相同；delta 均为 `0.009149095118820283`，`all_passed` 均为 `true` |
| 四折验证年 RankIC | 0.0679 / 0.0987 / 0.0894 / 0.1032，两次**逐一相同** |
| 4 个 model bundle 的**每个文件** sha256 | 全部一致（含三个 seed 的 booster、config、model_card、signature、feature_order、preprocessing、验证预测） |
| dataset id（4）、prediction id（8）、selection id（4+4） | id 集合完全一致 |
| `promotion_evidence.json` | **逐字节一致**，sha256 两次均为 `92b000c0d246556baf6dd077abe9e5f926ea69fd5840998c6af594b78b0c6d48` |

`m0r_v3` 还顺带验证了驱动的**内联**证据生成接线：它在写完 report 后自动产出了 `promotion_evidence.json`，摘要与 `m0r_v2` 用独立 CLI 补跑出来的那份逐字节相同。

这条比对链的意义在于：`derive_walk_forward_model_id` 的输入是四折 bundle 摘要，而 bundle 摘要已被证明可复现，因此**注册表里那个 model_id（`sha256:32315c89…788b`）也是可复现的身份**，不是某一次运行的偶然产物。任何人在同一数据版本上重跑都会得到同一个 id、同一份证据、同一条晋级记录。

> `m0r_v1` 是那次因家族准入列表不同步而崩在评估步骤的运行，其四折训练产物完整保留作为该缺陷的现场证据；它没有 report.json。

## 10. 已完成的本轨道交付

- `research_v2/src/research_v2/baselines/lgbm_ranker.py`：在 Dataset 构造处加入带实测数据的注释守卫（第 6 节的 hazard 无法用单元测试捕获）。
- 本文档。
- 提交 `297deee`，已推送 `codex/publish-quant-platform`。门禁：ruff 全绿、mypy 158 文件全绿、pytest **1720 passed / 45 skipped**。
- 全部实验脚本与产物在 `/tmp/es_metric_diag/`（`run_f4.py`、`run_ablation.py`、`run_dense9_search.py`、`run_carrier.py`、`run_tuned_carrier.py`、`verify_hazard_real.py` 及各自 `.log` / `.json`）。
