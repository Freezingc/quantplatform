# M0R 进入 SHADOW 的准备度规格（2026-09-09）

> 生成：2026-09-09 · 分支 `codex/publish-quant-platform`
> 对象模型：`sha256:32315c89fb9a65b8ceab193036fc3a0a870d24f2e6d99c6ff74049dccc75788b`（M0R，当前状态 `WALK_FORWARD_PASSED`）
> 本文档**不签发任何批准**，也不代替任何人的判断。它把「还差什么」逐项列清，并把其中机械可导出的部分算好、验过可复现。

## 0. 一句话

从 `WALK_FORWARD_PASSED` 到 `SHADOW` 的工程通路已经铺好（`ModelRegistryStore.enter_shadow` + `qr-v2 registry shadow`），控制全部在领域层且一条未松；**剩下的是 2 项 provisioning、1 个必须由第二个人执行的职责分离、以及 5 个声明哈希中的 3 个在全仓没有任何格式定义**。5 个里有 2 个我已算出并验证可复现。

## 1. `enter_shadow` 实际要求什么（逐条读码确认）

`ModelRegistry.enter_shadow(model_id, *, approval, approval_root)`：

| # | 要求 | M0R 当前状态 |
|---|---|---|
| 1 | 模型状态必须是 `WALK_FORWARD_PASSED` | ✅ 已满足 |
| 2 | 批准书必须在 `approval_root` 下、用 `QRV2_SHADOW_APPROVAL_HMAC_KEY` 验签通过 | ❌ 该密钥**不存在**（`secrets/` 下只有 `selection_evidence_hmac.key`） |
| 3 | 批准书的 `model_id` 必须等于被晋级的模型 | ✅ 可满足（见 §2） |
| 4 | 批准书的 `walk_forward_evidence_hash` 必须出现在该模型**晋级事件**记录的哈希元组中 | ✅ **已机械验证吻合**（见 §2） |
| 5 | `QRV2_SHADOW_DEPLOYER_ID` 必须指明操作者 | ❌ 未设置 |
| 6 | **操作者不得等于批准者**（`actor != verified.actor`） | ❌ 需要**两个真人** |

第 4 条的实测证据：

```
晋级事件记录的哈希        = 64b413507bf453ab78633bba18f74a65de878af4b97221a79b663655b2da0e19
对 promotion_evidence.json 重算 _gate_evidence_hash
                            = 64b413507bf453ab78633bba18f74a65de878af4b97221a79b663655b2da0e19
吻合                      = True
```

所以治理那一步不会有工程意外等着——链条是接得上的。

## 2. 五个声明哈希：哪些能算、哪些不能

`ShadowDeploymentApprovalStore.issue` 需要 5 个哈希。**校验器只验格式（64 位十六进制）与签名，不验内容**——它们是签发人的**声明**，安全性来自评估方专属密钥 + 职责分离，而不是来自内容核对。

| 字段 | 含义 | 可否机械导出 | 值 / 状态 |
|---|---|---|---|
| `walk_forward_evidence_hash` | 本次晋级所依据的走查证据 | ✅ 可 | `64b413507bf453ab78633bba18f74a65de878af4b97221a79b663655b2da0e19` |
| `artifact_bundle_hash` | 部署用的模型工件包 | ✅ 可（定义见下） | `4c06e98a4b9e3372f98332d0e575dbf7f6bea9bbeb42b467bfbb43aa18dfe3c9` |
| `inference_contract_hash` | 推理契约 | ⚠️ **部分就绪**（见 §2.1） | 加载器已实现；契约文档格式仍需定义 |
| `v2test_drill_hash` | V2 测试演练 | ❌ **全仓无格式定义**，且演练是**活动**不是文档 | 需流程所有者定义并实际执行 |
| `rollback_plan_hash` | 回滚方案 | ❌ **全仓无格式定义** | 需流程所有者定义 |

后三项在 `research_v2/` 与 `docs/` 全库检索的结果是：**只作为 `monitoring/shadow.py` 里的待声明字段出现，没有任何模板、schema、先例或产出方**。因此我没有起草它们——凭空发明三类流程工件再去哈希，等于把「有人审阅过」这件事伪造出来，那正是这套签名机制要防止的。

### 2.1 为什么 `inference_contract_hash` 此前无定义：LightGBM 家族根本无法被推理加载

排查这一项时发现的不是「文档缺失」，而是一个真实的工程缺口：`inference/artifact_loader.py` 的 `_allowlisted_factory` 只认 `RESIDUAL_MLP` / `TEMPORAL_GRU` / `SET_ATTENTION`，对其它家族直接抛 `model family is not allowlisted for inference`。也就是说 **M0 与 M0R 都过不了推理加载**——模型过了六道门、进了注册表，却没有任何途径产出预测。这是既有缺口（M0 自加入起就如此），不是 M0R 引入的。

它也不是加一条 allowlist 就能通的：torch 加载器从头到尾是 torch 形状的（返回 `nn.Module`、用 `torch.load(weights_only=True)` 读 `seed-<seed>.pt`、靠重建原型模块比对 `state_dict` 来验证 bundle），而 LightGBM bundle 的同名文件是 **booster 文本**，其 signature 是 config 哈希上的 `ModelSignature` 而非模块的 `artifact_signature()`。

已补齐：新增 `research_v2/src/research_v2/inference/lgbm_artifact_loader.py`，作为**并列**加载器（不改动 torch 加载器，因此 M1/M2/M3 的契约与所有消费方不受影响），沿用同一套纪律：已验证 bundle、家族 allowlist、code version 匹配、dataset/feature_order/preprocessor 交叉校验、`WALK_FORWARD_PASSED` 与 `FROZEN` 要求、三 seed、`InferenceLoadAudit`。契约校验与 code version 读取**复用** torch 模块的实现，所以「推理契约」仍然只有一处定义。

两处校验超出照搬：

- bundle 存储的 config 带 `raw_factor_count`，而 signature 哈希的是**不含**它的搜索 config——照原样哈希永远匹配不上。加载器会剥离并在注释里说明原因。
- booster 自己的 dump 记录了 `objective`，加载器用它与家族交叉校验：card 写 `LGBM_RANKER` 而 booster 是 `objective=regression` 的 bundle 会被拒为「工件与 card 描述的是两个不同模型」。在两个载体共用同一 bundle 布局之后，这是最要紧的一种错配。

**用真实生产工件验证**：从 `m0r_v2` 加载 `m0r-f4`、对其冻结 TEST 帧打分，与链上存档的候选预测在全部 **938,594 行**上 **max|diff| = 0.0**。所以六门所依据的那些数值，正是这条推理路径会产出的数值。

打分语义（每 booster 先做日内百分位、再三 seed 取均值、按业务键 merge 而非按位置）此前只存在于训练驱动这个**脚本**里，推理侧无从调用；`score_lgbm_artifact` 是它的库内归属，并有测试钉住它与驱动 `_m0_seed_mean_scores` 逐值相等。

因此 `inference_contract_hash` 的现状是：**被声明的对象（可服务的推理路径）已经存在并经验证**，但「契约文档」本身的格式仍需流程所有者定义——那份文档应当描述消费方需要保证什么（特征顺序、宇宙、label 版本、打分语义、设备），而不是替代已实现的加载器。

该契约文档还应写入一条显式要求：**服务授权来自注册表，不来自工件自述**。原因是随后发现的一个平台级缺口——model card 的 `model_status` 由驱动在评估之前硬编码写入，对每个运行的每个 bundle 恒为真，因此拿它当晋级判据的两个加载器实际上不设防；一个 G4 FAIL 的运行（`m0_v3`）的 bundle 能被正常加载。LightGBM 侧已改为强制要求注册表验证的 `PromotedRun`，torch 侧仍待协调变更。完整记录见 `docs/handoffs/2026-09-09-model-status-fail-open-finding.md`。

### `artifact_bundle_hash` 的定义（提案，待流程所有者批准）

覆盖四折**已验证** bundle 的完整内容，而不只是它们的摘要：

```
payload = {
  "family": "M0R",
  "model_code_version": "lgbm-rankreg-1",
  "folds": {
    "<fold>": {
      "model_artifact_id": "m0r-<fold>",
      "bundle_digest": "<该 bundle 的快照摘要>",
      "artifact_hashes": { "<文件名>": "<sha256>", ... },   # 9 个文件，含 3 个 seed booster
      "file_count": 9
    }, ...
  }
}
artifact_bundle_hash = sha256(canonical_json(payload))       # sort_keys, 紧凑分隔符, ensure_ascii
```

bundle 通过 `ModelArtifactBundleStore.load_verified_bundle` 读取，因此**取哈希本身就是一次完整性校验**：被篡改或截断的 bundle 会在此抛错。

四折摘要：F1 `1525f36b0ab3476a…`、F2 `f8612901212c53fc…`、F3 `ca9cdf9821698973…`、F4 `5f50bf50c4bcb521…`，各 9 个文件。

**可复现性已验证**：从两次独立全量跑的工作区（`m0r_v2` 与 `m0r_v3`）分别计算，得到**同一个** `4c06e98a…fe3c9`。

## 3. 待办清单（按依赖顺序）

1. **provisioning**：生成并保管 `QRV2_SHADOW_APPROVAL_HMAC_KEY`（评估方专属，**不得**与信任根同值——否则「评估方独立」这个前提就没了）。
2. **定义三份工件的格式**：inference contract、v2test drill record、rollback plan。这是流程所有者的决定，不是工程决定。
3. **产出这三份工件**并各自取 sha256。
4. **由持有评估方密钥的人**签发批准书（`ShadowDeploymentApprovalStore.issue`），引用 §2 的 5 个哈希。
5. **由另一个人**设置 `QRV2_SHADOW_DEPLOYER_ID` 并执行 `qr-v2 registry shadow --model-id … --approval-root … --approval-id …`。第 6 条控制会拒绝同一个人。

第 4、5 步必须是两个不同的人。我不会用两个自造的身份串去完成它——那会把一个双人控制降级成一次形式检查。

## 4. 已经就绪的部分

| 组件 | 状态 |
|---|---|
| `ModelRegistryStore.enter_shadow` | ✅ 已实现，只持久化不决策，全部控制在领域层 |
| `qr-v2 registry shadow`（含 `--dry-run`） | ✅ 已实现 |
| 8 个测试（走真实 `ShadowDeploymentApprovalStore`，非桩） | ✅ 含职责分离、错误密钥、证据不匹配、模型不匹配、未过走查、被拒后日志不增长 |
| LightGBM 推理加载器（M0 与 M0R） | ✅ 已实现，真实工件上逐值复现链上预测（§2.1），16 个测试 |
| 哈希链衔接 | ✅ 已用真实工件验证（§1） |
| `artifact_bundle_hash` 定义与取值 | ✅ 已算出并跨两次独立运行验证可复现 |

门禁：ruff clean、mypy 162 文件、pytest **1820 passed / 45 skipped**。
