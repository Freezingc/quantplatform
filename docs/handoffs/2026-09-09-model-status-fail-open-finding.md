# 发现：`model_status` 恒真，导致未过门的模型可被推理加载（2026-09-09）

> 生成：2026-09-09 · 分支 `codex/publish-quant-platform` · 提交 `e0e0f55`
> 严重性：**平台级完整性缺口**，早于 M0R，影响全部 5 个候选家族
> 状态：LightGBM 侧已闭合并用真实工件验证；**torch 侧（M1/M2/M3）仍未闭合**，需要一次协调的契约变更

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

## 4. 未闭合的部分（torch 侧，M1/M2/M3）

`inference/artifact_loader.py` 有**完全相同**的漏洞，但没有一并修，原因是它有 **4 个调用点在驱动内部**（`run_real_candidate_training.py` 的 485、807、1867、2259 行）：驱动写完全 bundle 后立刻用它重新加载、以产出冻结 TEST 预测——**这是一次合法的「评估前加载」**，模型并没有被服务，而此时门也确实还没算。给服务路径设计的检查不能把它挡掉。

正确修法是把两种用途分开：

- **load-to-evaluate**：驱动内部用，不要求晋级（模型正在被评估，不是被服务）；
- **load-to-serve**：要求注册表验证的 `PromotedRun`，与 LightGBM 侧同构。

这涉及 torch 加载器的公开签名、4 个驱动调用点、以及 `tests/unit/inference/test_artifact_loader.py` 等消费方，是一次**协调的契约变更**，不适合作为顺带改动塞进一个已经在改推理层的提交里。

### 也没有改 card 字段本身

把 `model_status` 改成真话（例如 `BUNDLE_FINALIZED`）在语义上更干净，但会改变**每个 bundle 的摘要** → 改变每个派生的 model_id → 使**已经晋级并记录在注册表里的那条 M0R 条目**成为孤儿，也会使我已记录的「两次全量跑逐字节一致」的可复现性证据与新契约下的重跑不可比。这应该是一次有意为之、协调好的契约变更，而不是顺带改。

## 5. 建议的处置顺序

1. **（已完成）** LightGBM 侧闭合 + `verify_run_promotion` 原语 + 真实工件验证。
2. **（建议下一步）** 把 torch 加载器拆成 load-to-evaluate / load-to-serve，4 个驱动调用点改用前者，服务路径改用后者。这一步不需要任何治理输入。
3. **（需协调）** 重命名 card 的 `model_status` 为真话，并同步更新两个加载器、5 个写入点与相关测试；接受 bundle 摘要与派生 model_id 变化，重跑一次链并在注册表里登记新身份（旧条目标记为被取代）。
4. **（文档）** `inference_contract_hash`（SHADOW 的声明工件之一）的契约文档应当把「服务授权来自注册表而非工件自述」写成一条显式要求——见 `2026-09-09-m0r-shadow-readiness-spec.md` §2.1。

## 6. 边界声明

本次改动只新增与收紧，未放松任何检查；未修改任何已有 bundle、摘要或注册表条目；未修改因子集契约；零数据库写入；未触碰 2026 锁定段与 `materialization_status.{json,md}`。

门禁：ruff clean、mypy 162 文件、pytest **1826 passed / 45 skipped**（本次 +6 测试）。
