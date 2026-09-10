# 发现：`model_status` 恒真，导致未过门的模型可被推理加载（2026-09-09）

> 生成：2026-09-09 · 分支 `codex/publish-quant-platform` · 提交 `e0e0f55`
> 严重性：**平台级完整性缺口**，早于 M0R，影响全部 5 个候选家族
> 状态：同类排查共找到 **4 处**恒真/可绕过的守卫——2 处已**显形**（`b9eb666`，§4.1）、2 处已**直接修严**（`17ece95` §4.2、`37b6c44` §4.3）；服务授权缺口已在**两侧闭合**（`e0e0f55`、`aa8dab6`）；并补上了证据链缺失的最后一环——**从「归属」到「复现」**（`04c5acd`，§4.4）。§4.1 两处的根源仍未除：需先由所有者定义 `split_id` 与 `code_version`（§5 第 5–6 步）

## 1. 发现过程与复现

尝试用推理加载器读取 `m0_v3` 的 bundle。`m0_v3` 是 M0（LambdaRank）的精调网格运行，其 `report.json` 明确写着：

```
delta      = 0.004202256886699618
G4         = {'status': 'FAIL', 'observed_value': 0.004202…, 'threshold': 0.005}
all_passed = False
```

而它的 model card 是：

```json
{"dataset_id": "6f9478d7…:train", "family": "LGBM_RANKER",
 "model_code_version": "lgbm-ranker-1", "model_status": "WALK_FORWARD_PASSED"}
```

**加载成功**，三个 seed 的 booster（7/12/61 棵树）全部返回。一个 G4 FAIL 的运行的模型可以被当作推理工件加载。

## 2. 根因：这个字段不可能为假

`run_real_candidate_training.py` 在 **5 处**把 `"model_status": "WALK_FORWARD_PASSED"` 作为字面量写入 model card：

| 行 | 路径 |
|---|---|
| 473 | torch 家族 smoke |
| 795 | M2 smoke |
| 1226 | LightGBM 家族共用 bundle 写入（M0 与 M0R） |
| 1835 | torch 交叉家族全量（M1/M3） |
| 2229 | M2 全量 |

而且它**必须**在 OOS 评估之前写：评估消费的正是这个 bundle 产出的冻结 TEST 预测。所以在写入的那一刻，门的结论根本还不存在。这个字段因此不是一个「可能为假」的断言——它对每一个运行的每一个 bundle 恒为真。

两个推理加载器都拿它当门禁：

- `src/research_v2/inference/artifact_loader.py:85`（torch，M1/M2/M3）
- `src/research_v2/inference/lgbm_artifact_loader.py`（LightGBM，M0/M0R）

**两处检查都恒真，因此都不提供任何保护。**

## 3. 已闭合的部分（LightGBM 侧）

新增 `verify_run_promotion(workspace, report_path, registry) -> PromotedRun`（`registry/promotion_evidence.py`）：

- 从四折**已验证** bundle 摘要派生该运行的注册表身份（派生过程本身会重哈希并校验每个文件）；
- 向**注册表**查询该身份的状态，只接受 `WALK_FORWARD_PASSED / SHADOW / LOCKED_TEST_APPROVED / LOCKED_TEST_PASSED / PRODUCTION`，拒绝 `DRAFT / RETIRED / LOCKED_TEST_FAILED`；
- 返回 `PromotedRun`，携带 `fold → bundle` 映射，使调用方能确认「我要加载的这个 bundle 确实属于那个被晋级的运行」。

注册表之所以能当权威，是因为它是**唯一**一处状态由重算的门导出、而非由工件生产者自行声明的地方。

`load_lgbm_inference_artifact` 现在**强制**要求 `PromotedRun`，缺失即拒；并要求被加载的 bundle 在授权运行的四折工件之内。card 上的 `model_status` 检查保留但**改名归位**为「bundle 形状标记」（证明它出自走查写入器），并在注释里写明它不是晋级证据。加载结果新增 `promotion_model_id` / `promotion_status`，使被服务的工件自带授权来源。

### 真实工件验证（非夹具）

| 场景 | 结果 |
|---|---|
| `m0r_v2`（`all_passed=True`，已注册） | ✅ 加载，记录 `WALK_FORWARD_PASSED` 与注册表 id |
| `m0_v3`（G4 FAIL，未注册） | ❌ 拒绝：`not registered in the model registry` |
| 未提供 `PromotedRun` | ❌ 拒绝 |
| **用 `m0r_v2` 的授权去加载 `m0_v3` 的 bundle** | ❌ 拒绝：`does not belong to promoted run` |

最后一条是关键：没有「折→工件」绑定，一个被晋级运行的授权就能服务任何工作区里的任何 bundle。

## 4. torch 侧（M1/M2/M3）：已闭合

`inference/artifact_loader.py` 曾有**完全相同**的漏洞。它没有和 LightGBM 侧一起修，是因为它有 **4 个调用点在驱动内部**（`run_real_candidate_training.py` 的 485、807、1867、2259 行）：驱动写完全 bundle 后立刻用它重新加载、以产出冻结 TEST 预测——**这是一次合法的「评估前加载」**，模型并没有被服务，而此时门也确实还没算。给服务路径设计的检查不能把它挡掉。

已按「把两种用途分开」实施（提交 `aa8dab6`），而不是放松任何一边：

| 函数 | 用途 | 晋级要求 | 返回的授权字段 |
|---|---|---|---|
| `load_bundle_for_evaluation` | 驱动内部：重读刚冻结的 bundle，使门所打分的 TEST 预测可证明来自持久化工件而非训练内存 | **无**（此刻门的结论尚不存在） | `promotion_model_id=""`、`promotion_status=""` |
| `load_inference_artifact` | **服务** | 强制要求 `verify_run_promotion` 产出的 `PromotedRun`，且被加载的 bundle 必须在该运行的四折工件之内 | 记录授权它的注册表 id 与状态 |

评估路径保留了原有的全部检查（bundle 完整性、allowlist、code version、dataset、feature order、preprocessor hash、推理契约、state_dict 重建），只是**不再声称也不携带**任何晋级判断——因此一次评估加载不可能被误认为一次已授权的服务加载。

公开的 `load_inference_artifact` 名字保持不变，所以原本就在「请求加载推理工件」的调用方现在得到了这个名字一直隐含的那道检查。card 字段在读取处被归位为「bundle 形状标记」，而那句误导性的报错 `inference requires a WALK_FORWARD_PASSED model` 已删除——它正是该字段无法支撑的那个断言。

调用点迁移：驱动 4 处，3 个测试文件共 14 处（每一处都是「建 bundle 再加载以检查重建」，不是服务）。

**未改变任何 bundle 摘要**，因此派生的 model_id 不变、已晋级的 M0R 注册表条目不受影响：这一步是拆分函数与归位检查，不重写任何工件。

超出测试套件的验证：改名后两个驱动 smoke 均端到端跑通（`--family M0R --smoke`、`--family M1 --smoke`，各退出码 0）。新增 4 个测试：无 `PromotedRun` 时服务被拒、bundle 不在授权运行内时服务被拒、服务成功时记录授权来源、评估加载不携带任何晋级声明。

### 也没有改 card 字段本身

把 `model_status` 改成真话（例如 `BUNDLE_FINALIZED`）在语义上更干净，但会改变**每个 bundle 的摘要** → 改变每个派生的 model_id → 使**已经晋级并记录在注册表里的那条 M0R 条目**成为孤儿，也会使我已记录的「两次全量跑逐字节一致」的可复现性证据与新契约下的重跑不可比。这应该是一次有意为之、协调好的契约变更，而不是顺带改。

## 4.1 同类排查又找到两处永不触发的守卫（提交 `b9eb666`）

`model_status` 是靠问「这个检查到底证明了什么」找到的。用同一个问题扫一遍晋级→服务路径，又找到两处，**都对任何真实工件从未生效过**：

| 守卫 | 代码意图 | 为何从不触发 |
|---|---|---|
| `factor_set_status` | 「inference requires a FROZEN factor set」 | 检查以字段存在为前提，而**没有任何工作区的 model card 含该字段**（已核 `m0_v3`、`m0r_v2`、`m0r_v3`）。驱动只写 `dataset_id`、`family`、`model_status`、`model_code_version` 四个字段 |
| `expected_inference_contract` | 绑定 factor set / universe / label 版本 / split / code version | **没有任何生产调用方传入**，只有测试传。对真实 bundle 传入契约会得到 `model card inference contract field is missing: factor_set_id`——说明校验逻辑本身有效，而链上产出的任何工件**都不可能满足它** |

### 为什么一直没被发现：夹具与生产的背离

torch 加载器自己的夹具 `_write_residual_bundle` 把 `**_INFERENCE_CONTRACT`（全部 9 个字段）写进它构造的每张 card。于是这些守卫在测试里「有覆盖」、全绿，而生产写入方一个字段都不写。**一个按系统从不产出的工件形状建模的测试套件，会对从不执行的检查报告通过。**

### 为什么没有直接改成强制

- 强制要求 `factor_set_status`：会拒绝**所有现存 bundle**，包括已晋级的 M0R。
- 填充推理契约：需要目前不存在的值。`build_inference_manifest` 是把 `universe_id`、`factor_set_id`、`input_versions` 作为**调用方传入**的参数，并不派生它们；而 `split_id` 与 `code_version` 在**全仓没有任何生产方**。把发明出来的值写进不可变的内容寻址工件比留空更糟——工件是永久的，而一个看起来合理的猜测与一个真正的决定无法区分。

### 实际做法：让缺口无法被忽略

新增单一共享定义 `unverified_bindings(model_card, expected_inference_contract)`，两个加载器都计算它，两个返回类型都携带它（**空元组表示每道守卫都真的触发了**）。调用方不再可能把一个未校验的绑定误认为已校验，审阅者从加载结果就能看出「哪些没有被确立」。

真实工件验证（非夹具）：在注册表授权下加载 `m0r_v2` 的 `m0r-f4`，返回

```
unverified_bindings == ('factor_set_status', 'inference_contract')
```

未改变任何 bundle 摘要，因此派生 model_id 与已晋级的注册表条目均不受影响。

新增 5 个测试，其中一个是**绊线**：加载由驱动自己的 bundle 写入器产出的 bundle，断言其 card 仍恰好是那四个字段、且两个绑定都被报告为未校验。若驱动哪天开始写契约，该测试会失败——届时应当**收紧**断言而不是删掉它，因为「夹具与生产的背离」正是需要持续可见的东西。

## 4.2 第三处：省略 `model_code_version` 即可跳过 allowlist（提交 `17ece95`）

torch 加载器的 `_allowlisted_factory` 原本写作：

```python
if model_code_version is not None and (not isinstance(...) or expected != model_code_version):
    raise ValueError("model_code_version is not allowlisted for model family")
```

于是**只要 card 里不含这个字段，整条 allowlist 就被跳过并成功加载**。已在真实夹具上验证。更糟的是随后 `audit.code_version` 报告的是 `research-code-1`——那是推理契约里的**研究代码版本**，不是模型代码版本，因为 `_model_card_code_version` 按 `("code_version", "model_code_version")` 顺序回退。**一个来源不明的工件不仅被接受，还被用另一个字段的身份描述了。**

已改为家族与版本**都强制**，并先查家族以保持报错准确（未知家族报 `model family is not allowlisted for inference`，而不是被说成版本问题）。LightGBM 加载器本来就是严格的——这正是两个加载器对同一种缺失行为不一致的原因。

**未改变任何生产工件**：驱动在全部 5 个 card 写入点都写了 `model_code_version`，所以链上产出的每个 bundle 都满足更严的检查。已用注册表授权加载 `m0r_v2` 的 `m0r-f4` 确认仍可加载，且 `code_version` 现在正确报告 `lgbm-rankreg-1`。摘要未变，派生 model_id 与注册表条目不受影响。

### 夹具与生产的背离是双向的

这次改动动到 3 个测试夹具，方向值得记下：它们比生产**更不完整**（card 只写 `dataset_id`、`family`、`model_status`）。这与 §4.1 的背离正好相反——那里夹具比生产**更完整**，于是测了永不触发的守卫。**夹具若不按系统真实写入的形状建模，两个方向都会误报覆盖情况。**

新增 4 个测试：省略 `model_code_version` 被拒、携带别的家族的版本被拒、未知家族按家族信息被拒、以及把 torch 侧的版本 allowlist 映射本身钉住（改映射必须是有意的行为）。

## 4.3 第四处：空契约伪装成「已校验」（提交 `37b6c44`）

`_validate_inference_contract` 对空映射的循环体不执行，于是 `expected_inference_contract={}` **通过校验**；而上一轮新增的 `unverified_bindings` 判据是 `is None`，所以空契约会把 `inference_contract` 从未校验列表里**移除**。

实测：把 card 的 8 个契约字段全部剔除后，传空契约加载成功，且报告 `unverified_bindings == ('factor_set_status',)` ——即**加载器主动声称它校验了一个它没校验的绑定**。这比原来的静默跳过更糟，因为新字段本该让缺口显形。

已改为：空契约直接拒绝（`inference contract must name at least one field to verify`），且 `unverified_bindings` **独立地**把空视为未校验——报告方不依赖校验方先跑过。

## 4.4 证据链证明的是「归属」，不是「复现」（提交 `04c5acd`）

审计证据链时确认它每一环都闭合：预测工件的分数 bytes 被签名；`_require_candidate_evidence` 把预测的 selection id / model id / fold / dataset 与冻结选择逐一对上；`_require_verified_model_bundle` 把该选择的 bundle 摘要与重新哈希的 bundle 对上。

**但这条链证明的是「这些分数被记录在这个 bundle 名下」，而不是「这些分数确由这个 bundle 算出」。** 持有信任根的人可以在正确的 metadata 下签署任意分数 bytes，而现有每一道检查都会通过。

唯一能闭合它的是**重算**，而重算的前提都成立：bundle 不可变、TEST 分片是不可变的已验证工件、打分是确定性的。新增 `verify_candidate_predictions_reproduce`：重新派生该运行的注册表身份（会重哈希并校验四折全部 bundle）→ 在**评估路径**下加载每折 bundle → 对该折冻结 TEST 分片打分 → 与签名的候选工件**逐值比对**。任何差异、缺行、多行都是拒绝，**且没有容差参数**（精确相等，一个 ulp 也算差异）。

**审计不要求该运行已晋级**：注册表状态只被读取并报告，不被强制执行。理由是要求晋级会把最该审计的运行排除在外——一个 G4 失败的运行照样用真 bundle 写出并签署了候选预测，而「这些分数是否确由这些 bundle 算出」正是失败运行留下的未决问题。服务路径（`load_inference_artifact` / `load_lgbm_inference_artifact`）仍然强制晋级，本模块不触碰它；LightGBM 侧为此新增了对称的 `load_lgbm_bundle_for_evaluation`——同样的完整性、白名单、code version、booster objective、签名与 metadata 校验，只少注册表授权，且返回空的 `promotion_model_id`/`promotion_status`，以免被误当成可服务模型。注册表**损坏**（`RegistryStoreError`）必须冒泡，不能被当成「未注册」，否则「打不开注册表」会被翻译成「这个运行没晋级」。

真实工件实跑，已晋级与未晋级的工作区都精确复现：

| 工作区 | 家族 | registry_status | F1 | F2 | F3 | F4 | 合计 |
|---|---|---|---|---|---|---|---|
| `m0r_v2` | LGBM_RANKREG | `WALK_FORWARD_PASSED` | 804,575 行 | 882,426 | 932,167 | 938,594 | **3,557,762 行，每折 max\|diff\| = 0.0** |
| `m0r_v3` | LGBM_RANKREG | 同上（与 `m0r_v2` 逐字节相同，派生同一 model_id） | 同上 | 同上 | 同上 | 同上 | 同上 |
| `m0_v3` | LGBM_RANKER | **`UNREGISTERED`**（G4 FAIL，delta +0.004202） | 同上 | 同上 | 同上 | 同上 | **3,557,762 行，每折 max\|diff\| = 0.0** |

`m0_v3` 那一行是这次放宽的直接产出：该运行从未进入注册表，在旧版本下**根本无法被审计**；现在它的四折存储分数与 bundle 重算逐值相同，派生身份 `sha256:fcd45b28…`。

**torch 家族也已在生产工件上验证（2026-09-10 23:36，对 `m1_v2` 跑到 F4 时已完成的三折做只读逐折探针）。** M1 是第一个 torch 载体，其 F1–F3 已冻结并签名，所以不必等整轮结束就能验证静态 torch 复现路径。dataset id 取自**签名的 model 选择证据**而不是 card 自述，并先把该证据的 `model_bundle_digest` 与重哈希的 bundle 摘要交叉校验，然后重算分数与签名的候选预测逐值比对：

| 折 | 家族 | 行数 | max\|diff\| | 耗时 |
|---|---|---|---|---|
| F1 | RESIDUAL_MLP | 803,572 | **0.0** | 217 s |
| F2 | RESIDUAL_MLP | 881,863 | **0.0** | 238 s |
| F3 | RESIDUAL_MLP | 931,411 | **0.0** | 262 s |

合计 **2,616,846 行逐位相同**。行集契约也同时对上了，而这一条是被检验过的、不是假设的：F1 的 test 分片有 804,575 行，重算与签名工件**同样**只覆盖 803,572 行，差掉的 1,003 行是 `label_valid=False`，由 batch 构造的同一道过滤在两侧一致地丢弃；比对按业务键做外连接，任何一侧多一行或少一行都会报错并说明是哪一侧。

**这三个 0.0 成立的前提需要写明（2026-09-11 补）**：探针当时是在**宿主默认 52 线程**下打分的，而 `m1_v2` 记录的 `torch_intraop_threads` 恰好也是 52，所以 0.0 是真的、但它是「记录值等于宿主默认值」的巧合结果，不是「审计与线程数无关」的证据。把同一折钉到 1 线程重测即得 4.923682914836913e-05（详见下方勘误）。审计现在自钉到运行记录的线程数，因此这个前提从「碰巧满足」变成了「被强制满足」。

意义在于把 torch 复现从「合成 bundle 上跑通」提升到「生产工件上跑通」，并且比整轮结束早约 8 小时——若它失败，就还有时间在 F4 完成之前修，而不是在判定归档时才发现审计路径本身不可用。
范围现在覆盖**链条能产出的每一个家族**（2026-09-10 补齐时序家族）。LightGBM 家族（M0、M0R）与静态 torch 家族（M1 `RESIDUAL_MLP`、M3 `SET_ATTENTION`）的分数是 bundle 与冻结帧的纯函数；torch 路径逐步复刻候选驱动的打分：以 bundle 自己记录的 `state_hash` 走评估加载 → `prepare_frame(test.assign(raw_return_5d=0.0))`（驱动同样把标签列置零：batch 构造要求存在 target 列，而打分不得能读到被预测的东西）→ 每个 signal_date 一个隔离 batch → 三个 seed 的日内百分位秩取均值。

时序家族（M2，card 依胜出配置写 `TEMPORAL_GRU` 或 `TEMPORAL_TCN`）此前是**按名拒绝**的，理由是它每行打分需要 20 天历史窗口。该缺口已闭合，做法不是让审计自己另建一套面板——那只会让审计与自己的假设一致——而是把面板组合抽成**一处库函数** `temporal_test_panel_batches`（train+validation+test 拼接后按 TEST 日期过滤），由**驱动的产出方与审计的重算方共同调用**；审计侧入口 `temporal_test_batches` 只从已发布的不可变分片与 bundle 自带的预处理状态重建。有一处家族差异必须照实复刻而非"顺手统一"：**M2 不置零标签列**，所以无可用标签的 TEST 行从未进入面板，存储分数覆盖的是有标签子集——那是被审计运行的属性，审计若替它补齐行数，就永远无法真正复现任何一次 M2 运行。

**勘误（2026-09-11 00:31，实测推翻本段）：精确相等在此**需要**钉线程。** 原文写「同一次前向在 1/4/16/52 intra-op 线程下逐位相同（已钉成测试，因为 torch 升级可能改变它，而那时的失败会长得像篡改）」——那次实测的对象是**未训练**模型（钉住该性质的那条测试就是这么建的，现已删除，因为它钉住的是一个假性质），它在未训练状态下确实干净；**为什么未训练就干净尚未定论**，能确证的只有「未训练干净、已训练不干净」这个对照。对**已训练的生产 bundle** 重测：`m1_v2` F1（`RESIDUAL_MLP`，803,572 行）在 **1/2/4/8 线程下与单线程逐位相同（差值全为 0.0）**，而在 **16 线程差 4.805844e-05、52 线程差 4.923683e-05，且 16 与 52 彼此也不相同**。三点结论：阈值在 8 与 16 之间（四档全 0 排除了「误差随线程数连续累积」的解释）；越过阈值后**不存在唯一的「正确分数」**，只有该运行当时记录的那个线程数下算出的分数；5e-5 的量级是近似并列股票之间的**秩反转**（会改变选股名次），不是末位舍入。同一 bundle 的 F2 差 4.6777060529579195e-05（881,863 行）、F3 差 1.2774655084313302e-04（931,411 行）。所以绑定线程数的**既有搜索记录、也有审计**：审计现在按 `recorded_run_threads()` 从 checkpoint manifest 读到的值把自己钉住再打分，并在结果里报 `scored_under_torch_threads`；若按宿主默认值打分而与记录值不符，就会报出一个**并非缺陷**的差异（实测 F1 钉到 1 线程 → `exit=1`、差 4.923682914836913e-05；自动钉到记录的 52 → `exit=0`、`max|diff|=0.0`）。训练则**从来不是**线程稳定的，上面第 4 节的谱系结论不变。

比对逻辑被抽成 `_compare_scores`，使其**非恒真性**可在无需签名工作区的前提下测试。这一点在此处比别处更要紧：**一个永远通过的验证器比没有验证器更糟，因为它把「从未检查」变成了「检查过且一致」**。当时 56 个测试钉住（2026-09-11 为 **74** 个）：原有的比对语义（正常路径、6 行中改 1 行、一个 ulp 的差异、缺行、多行、行序无关、4 种畸形帧拒绝、键类型对齐），授权与路由语义（未注册运行被报告而非拒绝、已注册运行报告注册表真实状态、注册表损坏必须冒泡、card 的 `dataset_id` 必须派生自该折报告项、family 路由到各自打分器且**不串道**、未白名单 family 按名拒绝、LightGBM 走评估加载器且不传 `promoted_run`、torch 路径四步的顺序与标签置零），以及**真实工件级**的两条路径：真三 seed `RESIDUAL_MLP` bundle（确定性逐位相同、打分不依赖标签列、三 seed 集成而非某个方便的 seed、NaN 标签帧仍打满全部行）与真三 seed `TemporalGRU` bundle + 真已发布折（只出现 TEST 日期、行集恰为有标签子集、两次重算逐位相同）。**勘误（2026-09-11）：原文此处的「另有前向的线程稳定性一条」已作废。** 那条测试断言的是未训练模型上的假性质，已删除；取而代之的是 `recorded_run_threads` 的两条（读出运行自己写入的值；没有 checkpoint 的工作区不报线程数）与审计侧自动钉线程的行为。测试数由 56 增至 74。

变异检验累计五处，每次都恰好杀死对应测试而非整片变红：把静态 torch 路由改指 LightGBM；去掉标签置零（NaN 标签帧那条随即以"无有效 target"失败）；只跑 `models[:1]`；把注册表属性访问挪进 `try`；把时序面板改成只拼 test 段（丢掉历史窗口，两条时序审计测试随即失败）。

另加 `scripts/verify_prediction_reproduction.py`，任何工作区都可审计，任何差异都非零退出，可直接当门禁用。**刻意没有接进训练驱动**：重算每折数据集哈希使其耗时以分钟计（四折约 25 分钟），那是审计成本，不该默认加到每次运行上。

## 5. 建议的处置顺序

1. **（已完成，`e0e0f55`）** LightGBM 侧闭合 + `verify_run_promotion` 原语 + 真实工件验证。
2. **（已完成，`aa8dab6`）** torch 加载器拆成 load-to-evaluate / load-to-serve，4 个驱动调用点与 14 处测试调用点改用前者，服务路径改用后者。
3. **（需协调，仍未做）** 重命名 card 的 `model_status` 为真话，并同步更新两个加载器、5 个写入点与相关测试；接受 bundle 摘要与派生 model_id 变化，重跑一次链并在注册表里登记新身份（旧条目标记为被取代）。
4. **（已完成，`b9eb666`）** 让两处永不触发的守卫显形：`unverified_bindings` 由两个加载器共同计算并随加载结果返回。
5. **（需所有者定义，仍未做）** 定义 `split_id` 与 `code_version` 的语义与权威来源，并决定 `factor_set_id` / `universe_id` / `label_id` 各由谁提供。这是**前置条件**：在这两个字段有定义之前，推理契约无法被写进工件，`inference_contract_hash` 也就无从取值。

   本轮排查已把可确定的部分确定下来，供决策时直接用：

   | 字段 | 仓内权威来源 | 结论 |
   |---|---|---|
   | `universe_id` | `configs/universe/cn_liquid_a.yaml` → `universe_id: cn_liquid_a` | ✅ 已有既定常量 |
   | `label_id` | `materializers/labels.py` 默认 `label_id="forward_open_5d"`；`quality/latest.py` 用 `"forward_open_5d:v1"` | ✅ 已有既定值 |
   | `universe_version` / `label_version` / `factor_daily` 版本 | 数据集 manifest 的 `metadata.data_versions`（三张表的逐分区内容哈希） | ✅ 已随工件持久化 |
   | `factor_set_id` | `factors/factor_sets.py:create_factor_set` 由成员身份算出的内容哈希 | ✅ 可派生（需成员四元组） |
   | `split_id` | 折契约 `datasets/splits.py:annual_folds()` 存在，但**无 id 字符串**；`factor_set_frozen_v2.json` 有 `fold_structure` 描述串 | ⚠️ 需决定：用描述串，还是对折契约取内容哈希 |
   | `code_version` | **全仓无生产方** | ⚠️ 需决定 |

   一条有约束力的推导：**`code_version` 不可取 git commit sha。** bundle 摘要是 card 字节的内容哈希，而派生的 model_id 又是四折 bundle 摘要的哈希；若 card 内含 git sha，则**每一次提交都会改变每个 bundle 摘要与 model_id**，跨运行的可复现性（§9.10 的「两次全量跑逐字节一致」）将不再成立——已记录的那两次之所以逐字节相同，正因为它们发生在同一个提交上。因此 `code_version` 应当是与 `model_code_version`（`lgbm-rankreg-1`）同类的**人工递增标签**，测试夹具里的 `"research-code-1"` 也正是这个形状。

   另外可参考：夹具 `_INFERENCE_CONTRACT` 里的 `universe_id="cn_liquid_a"` 与 `label_id="forward_open_5d"` 与上述真实来源**逐字相符**，说明该夹具当初是按真实意图值写的，其余三个（`factor-set-1`、`walk-forward-1`、`research-code-1`）则是同一「人工递增标签」约定的占位。
6. **（依赖第 5 步）** 把契约字段写入 model card（5 个写入点），并把 `factor_set_status` 与 `expected_inference_contract` 改为强制。与第 3 步同批做，因为它们同样会改变 bundle 摘要与派生 model_id，应一次重跑、一次重新登记，而不是分两次付这个代价。
7. **（文档）** `inference_contract_hash` 的契约文档应当把「服务授权来自注册表而非工件自述」与「加载结果必须报告未校验的绑定」写成显式要求——见 `2026-09-09-m0r-shadow-readiness-spec.md` §2.1。

## 6. 边界声明

本次改动只新增与收紧，未放松任何检查；未修改任何已有 bundle、摘要或注册表条目；未修改因子集契约；零数据库写入；未触碰 2026 锁定段与 `materialization_status.{json,md}`。

门禁：ruff clean、mypy 163 文件、pytest **1852 passed / 45 skipped**（六次修复合计 +32 测试）。
