# V2 模型卡模板（model_card.json）

模型 artifact bundle（`ModelArtifactBundleStore`）是不可变的完整快照，目录结构：

```
artifacts/models/{model_id}/
  model_card.json                  # 本模板描述的模型卡
  config.json                      # 模型超参（族对应的精确字段集）
  signature.json                   # asdict(model.artifact_signature())
  feature_order.json               # {"feature_order": [...]}
  preprocessing.json               # 必须含 state_hash（训练拟合的预处理状态哈希）
  seed-20260801.pt                 # 三个固定种子的 checkpoint（仅 state_dict）
  seed-20260802.pt
  seed-20260803.pt
  validation_predictions.parquet   # signal_date, ts_code, percentile_rank
  checksums.sha256                 # 全文件 SHA-256 清单
```

`model_id` 必须匹配 `^[A-Za-z0-9][A-Za-z0-9._-]*$`；bundle 一经写入不可覆盖。

## model_card.json 字段

| 字段 | 必填 | 取值/约束 |
|---|---|---|
| `dataset_id` | 是 | 训练数据集的不可变标识；加载时与请求的 `expected_dataset_id` 精确比对 |
| `family` | 是 | `RESIDUAL_MLP` / `TEMPORAL_GRU` / `SET_ATTENTION`（推理白名单） |
| `model_status` | 是 | 状态机取值（见下）；推理加载要求 `WALK_FORWARD_PASSED` 及以上 |
| `model_code_version` | 推理必填 | 与族绑定的代码版本：`residual-mlp-1` / `temporal-gru-1` / `set-attention-1` |
| `factor_set_id` | 推理契约 | 冻结因子集标识 |
| `factor_set_status` | 推理契约 | 必须为 `FROZEN`（否则拒绝加载） |
| `universe_id` / `universe_version` | 推理契约 | 股票池定义与版本 |
| `label_id` / `label_version` | 推理契约 | 标签定义与版本 |
| `split_id` | 推理契约 | 训练/验证/测试划分标识 |
| `code_version` | 可选 | 管线代码版本记录 |

推理契约字段集以 `inference.artifact_loader._INFERENCE_CONTRACT_FIELDS` 为准；
传入未知字段会被拒绝。

## model_status 状态机

```
DRAFT → WALK_FORWARD_PASSED → SHADOW → LOCKED_TEST_APPROVED
→ LOCKED_TEST_PASSED → PRODUCTION → RETIRED
```

- 合法分支：`LOCKED_TEST_APPROVED → LOCKED_TEST_FAILED`；任意非 `RETIRED` → `RETIRED`。
- 禁止：`DRAFT` 直达 `SHADOW`/`PRODUCTION`；`SHADOW` 直达 `PRODUCTION`；
  `LOCKED_TEST_FAILED` 回到任何通过态。
- `WALK_FORWARD_PASSED` 晋级要求 G1–G6 门槛证据（`registry promote` 重算，
  不信任报告布尔字段）；`SHADOW`/`LOCKED_TEST_*`/`PRODUCTION` 需要 F 计划专用
  审批证据（部署审批、双人解锁等）。

## 加载校验链（`load_inference_artifact`）

1. 逐文件核对 `checksums.sha256`，计算 bundle 摘要；
2. `model_card.dataset_id` 与请求数据集精确一致；
3. `feature_order` 与请求一致；`preprocessing.state_hash` 与请求一致；
4. `signature.json` 必须等于按 `config.json` 重建模型后的 `artifact_signature()`；
5. 三个种子 `state_dict` 逐一通过结构校验（`weights_only` 反序列化，
   不加载任意 Python 对象）；
6. 全部通过后才实例化白名单模型类并加载权重。

## 模板示例

```json
{
  "dataset_id": "sha256:<training-dataset-manifest-id>",
  "family": "RESIDUAL_MLP",
  "model_status": "WALK_FORWARD_PASSED",
  "model_code_version": "residual-mlp-1",
  "factor_set_id": "fs-stable-v1",
  "factor_set_status": "FROZEN",
  "universe_id": "cn_liquid_a",
  "universe_version": 1,
  "label_id": "forward_open_5d",
  "label_version": 1,
  "split_id": "F01",
  "code_version": "research_v2@<git-sha>"
}
```
