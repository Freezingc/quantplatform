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

唯一能闭合它的是**重算**，而重算的前提都成立：bundle 不可变、TEST 分片是不可变的已验证工件、打分是确定性的。新增 `verify_candidate_predictions_reproduce`：重新派生该运行的注册表身份（会重哈希并校验四折全部 bundle）→ 要求该身份已晋级 → 在服务路径的授权下加载每折 bundle → 对该折冻结 TEST 分片打分 → 与签名的候选工件**逐值比对**。任何差异、缺行、多行都是拒绝，**且没有容差参数**（精确相等，一个 ulp 也算差异）。

真实工件实跑，两个已晋级工作区都精确复现：

| 工作区 | F1 | F2 | F3 | F4 | 合计 |
|---|---|---|---|---|---|
| `m0r_v2` | 804,575 行 | 882,426 | 932,167 | 938,594 | **3,557,762 行，每折 max\|diff\| = 0.0** |
| `m0r_v3` | 同上 | 同上 | 同上 | 同上 | 同上 |

范围限定在 LightGBM 家族（其打分是 bundle 与帧的纯函数）。torch 家族还需要该运行的 `FoldPreprocessor` 状态与 batch 构造，而那些住在候选驱动脚本里、不在库中，因此**明确拒绝并说明原因**，而不是静默跳过——一个悄悄少覆盖的验证器，与「报告了自己没检查的绑定」是同一个错误。

比对逻辑被抽成 `_compare_scores`，使其**非恒真性**可在无需签名工作区的前提下测试。这一点在此处比别处更要紧：**一个永远通过的验证器比没有验证器更糟，因为它把「从未检查」变成了「检查过且一致」**。11 个测试钉住：正常路径、6 行中改 1 行、一个 ulp 的差异、缺行、多行、行序无关、4 种畸形帧拒绝、以及让 join 真正成立（而非静默为空）的键类型对齐。

另加 `scripts/verify_prediction_reproduction.py`，任何工作区都可审计，任何差异都非零退出，可直接当门禁用。**刻意没有接进训练驱动**：重算每折数据集哈希使其耗时以分钟计（四折约 25 分钟），那是审计成本，不该默认加到每次运行上。

## 5. 建议的处置顺序

1. **（已完成，`e0e0f55`）** LightGBM 侧闭合 + `verify_run_promotion` 原语 + 真实工件验证。
2. **（已完成，`aa8dab6`）** torch 加载器拆成 load-to-evaluate / load-to-serve，4 个驱动调用点与 14 处测试调用点改用前者，服务路径改用后者。
3. **（需协调，仍未做）** 重命名 card 的 `model_status` 为真话，并同步更新两个加载器、5 个写入点与相关测试；接受 bundle 摘要与派生 model_id 变化，重跑一次链并在注册表里登记新身份（旧条目标记为被取代）。
4. **（已完成，`b9eb666`）** 让两处永不触发的守卫显形：`unverified_bindings` 由两个加载器共同计算并随加载结果返回。
5. **（需所有者定义，仍未做）** 定义 `split_id` 与 `code_version` 的语义与权威来源，并决定 `factor_set_id` / `universe_id` / `label_id` 各由谁提供。这是**前置条件**：在这两个字段有定义之前，推理契约无法被写进工件，`inference_contract_hash` 也就无从取值。
6. **（依赖第 5 步）** 把契约字段写入 model card（5 个写入点），并把 `factor_set_status` 与 `expected_inference_contract` 改为强制。与第 3 步同批做，因为它们同样会改变 bundle 摘要与派生 model_id，应一次重跑、一次重新登记，而不是分两次付这个代价。
7. **（文档）** `inference_contract_hash` 的契约文档应当把「服务授权来自注册表而非工件自述」与「加载结果必须报告未校验的绑定」写成显式要求——见 `2026-09-09-m0r-shadow-readiness-spec.md` §2.1。

## 6. 边界声明

本次改动只新增与收紧，未放松任何检查；未修改任何已有 bundle、摘要或注册表条目；未修改因子集契约；零数据库写入；未触碰 2026 锁定段与 `materialization_status.{json,md}`。

门禁：ruff clean、mypy 163 文件、pytest **1852 passed / 45 skipped**（六次修复合计 +32 测试）。
