# 2026-09-14：两个领先候选的预测复现审计跑完了——**四折全部逐位相同，最大绝对差 0.0**

**一句话**：签名证据链证明的是"这些分数被记录在这个 bundle 名下"（归属），
**不是**"这些分数确由这个 bundle 算出"（复现）；后者只有复现审计能证明。
此前审计只在 `m1_parallel_v1`（M1 载体）上跑过，而**被引用得最多的两个运行**——
`m0r_v2`（领先候选、已注册 `WALK_FORWARD_PASSED`）与 `m1_full_v1`（已归档的 M1 全网格判定）
——**都没有审计过**。本轮补上，**两者四折全部 `identical=true`、`max_absolute_difference=0.0`**，
合计复核 **3,557,762 + 3,553,163 = 7,110,925 行** OOS 打分。

**只读**：审计从已签名工件重算，不写任何东西；不连数据库；
跑的时候 M2 正在训练（跑前跑后 F1 都是 18/24、驱动 pid 3588838 存活），未受干扰。

---

## 1. 结果

### `m0r_v2`（M0R / LightGBM rank-regression，领先候选）

| 折 | 家族 | bundle | 行数 | identical | max abs diff |
|---|---|---|---|---|---|
| F1 | LGBM_RANKREG | `m0r-f1` | 804,575 | **true** | **0.0** |
| F2 | LGBM_RANKREG | `m0r-f2` | 882,426 | **true** | **0.0** |
| F3 | LGBM_RANKREG | `m0r-f3` | 932,167 | **true** | **0.0** |
| F4 | LGBM_RANKREG | `m0r-f4` | 938,594 | **true** | **0.0** |

`model_id = sha256:32315c89fb9a65b8ceab193036fc3a0a870d24f2e6d99c6ff74049dccc75788b`、
**`registry_status = WALK_FORWARD_PASSED`**、`reproduced = true`、`total_rows = 3,557,762`、
`scored_under_torch_threads = 52`（宿主默认；LightGBM 工作区没有 torch 检查点可钉）、
`threads_requested = null`。

### `m1_full_v1`（M1 / ResidualMLP 全网格，已归档判定）

| 折 | 家族 | bundle | 行数 | identical | max abs diff |
|---|---|---|---|---|---|
| F1 | RESIDUAL_MLP | `m1-f1` | 803,572 | **true** | **0.0** |
| F2 | RESIDUAL_MLP | `m1-f2` | 881,863 | **true** | **0.0** |
| F3 | RESIDUAL_MLP | `m1-f3` | 931,411 | **true** | **0.0** |
| F4 | RESIDUAL_MLP | `m1-f4` | 936,317 | **true** | **0.0** |

`model_id = sha256:5d01ec8f2d1fa5ed700b6a8c20cfdedf75634e7a9c16b1b5e3cd4c723976c832`、
**`registry_status = UNREGISTERED`**、`reproduced = true`、`total_rows = 3,553,163`、
**`scored_under_torch_threads = 1`（运行自己记录的值；宿主默认是 52）**、`threads_requested = 1`。

## 2. 三个必须一起说的细节

1. **"0.0" 之所以有意义，是因为审计把自己钉到了那次运行记录的线程数。**
   已训练的 torch bundle 的前向**不是线程稳定**的：同一份 `m1_v2` F1 bundle（803,572 行）
   在 1/2/4/8 线程下逐位相同，在 **16 线程差 4.805844e-05、52 线程差 4.923683e-05**，
   而且 16 与 52 彼此也不同；量级是**近似并列股票之间的秩反转**（会改变选股名次），不是末位舍入。
   所以审计从**该工作区自己的检查点 manifest** 读 `recorded_run_threads()` 并自钉
   （混合多个值的工作区直接拒绝，不猜）。`m1_full_v1` 因此钉到 1，日志明写
   `scoring under torch intra-op threads=1 (the run recorded 1; the host default was 52)`。
2. **M0R 没有可钉的 torch 线程数（LightGBM），于是在宿主默认 52 线程下打分，仍然逐位相同。**
   这与"逐行求和、无跨行归约"的树模型预测是一致的；
   **本文只陈述这次实测，不外推成"LightGBM 与线程数无关"的一般结论。**
3. **审计可以当门禁用**：`_compare_scores` 对**任何**分数差异或行集差异都抛
   `ReproductionError`，CLI 随之非零退出——`"reproduced": true` 只在四折全部相同的情况下才会被写出。

## 3. 顺带被机器确认的一件事：M1 全网格**不在册**

审计摘要里的 `registry_status` 是从签名 registry 读的，不是转述：
`m0r_v2` → **`WALK_FORWARD_PASSED`**，`m1_full_v1` → **`UNREGISTERED`**。
这与研究分支证据包 `problems_found.md` §H 记的治理未决项一致：
**M1 六道门全过、判定已归档，但从未注册**。
注册是治理动作（要写签名 registry），本轮未做；
但要注意 F6.1 的 `WALK_FORWARD_PASSED → SHADOW` 转换**以"已在册"为前提**，
所以若所有者打算让 M1 进影子，**注册是必须先走的一步**。

## 4. 哪些没有审计，以及为什么

| 工作区 | 是否审计 | 理由 |
|---|---|---|
| `m0r_v2` | **本轮已审计** | 领先候选、已注册，是影子审批的对象 |
| `m1_full_v1` | **本轮已审计** | 已归档的 M1 判定，G4 被反复引用 |
| `m1_parallel_v1` | 09-11 已审计（`exit=0`） | M1 载体，已被全网格取代 |
| `m0r_v3` | 未审计 | 与 `m0r_v2` 是**逐字节相同的独立复现**（两份 report.json 的 sha256 同为 `d16b5708c4cd…`）；若所有者想对复现件单独审计，同一条命令换工作区即可，约 18 分钟 |
| `m0r_resume_a` / `m0r_resume_c` | 未审计 | 验收 A 的 `--resume` 副本，不含独立训练 |
| `m0_v2` / `m0_v3` | 未审计 | 两者 **G4 FAIL**、不会晋级；审计它们不改变任何决定 |
| `m2_v1` | **还不能用整跑模式**（运行中） | 但 `--fold F1` 模式正是为此设计的：从签名选择取该折 bundle 与 TEST 数据集、不需要 report、不查 registry。**F1 落地后即可逐折审计**，比等四折早约三天 |

**建议的下一步（不是本轮做的）**：M2 每折落地后跑一次
`verify_prediction_reproduction.py --workspace …/m2_v1 --fold F1`（逐折，`--fold` 可重复）。
逐折预检看守查的是**证据链与可加载性**（bundle 验签、card family 白名单、loader 能否构造模型、
冻结 config 与搜索胜者是否一致），**它不重算分数**；
复现审计补的正是这一块，而 M2 是时序家族、TEST 面板要从三个已发布分片重建 20 日窗口，
**是最值得逐折提前验的一个**。

## 5. 复现

```bash
cd research_v2
A=/data/zhangyudong/qrv2_research_artifacts
K=$A/secrets/selection_evidence_hmac.key
for W in m0r_v2 m1_full_v1; do
  env QRV2_SELECTION_EVIDENCE_HMAC_KEY="$(cat $K)" \
      QRV2_PREDICTION_EVIDENCE_HMAC_KEY="$(cat $K)" \
      QRV2_MODEL_REGISTRY_ROOT=$A/model_registry \
    python scripts/verify_prediction_reproduction.py \
      --workspace $A/$W --report $A/$W/report.json
done
```

两个工作区并行跑约 **18 分钟**（单跑约 25 分钟；耗时主要在重算每折数据集哈希）。
本次原始输出已存为 `$A/reproduction_audit_m0r_v2.log` 与 `$A/reproduction_audit_m1_full_v1.log`
（各含完整的 JSON 摘要）。

## 6. 边界

* **没有**写任何工件、门值、registry 状态或数据库行；审计对工件树只读。
* 审计回答的是"**存下来的分数是否确由这个 bundle 算出**"，
  它**不**回答"这个 bundle 是否由那次搜索选出"（那是验收 C′ 的逐字节检查点比对）、
  也**不**回答"这次运行是否可从头重跑"（那是验收 A 的 `--resume` 逐字节报告比对）。
  三者合起来才是完整的可复现性主张，本文只补了其中此前缺的那一环。
