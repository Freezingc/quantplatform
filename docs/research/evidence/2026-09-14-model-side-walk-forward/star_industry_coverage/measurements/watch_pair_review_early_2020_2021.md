# 关注对组合评审（冻结集训练前置）

> 生成：2026-09-15T15:20:35 · 数据：factor_daily 2020–2025 × forward_open_5d:v1（label_valid）
> 判定规则：R1 近冗余 |median rho|≥0.90；R2 增量保留率 pooled≥0.50 且单年≥0.30；R3 体制 ON 桶保留率≥0.30 且无符号翻转

## yeslab_381<->yeslab_464 — 总判定：**REVIEW**

- 各年 median|rho| 最大值：0.673（R1 PASS）
- pooled 保留率：{'yeslab_381|yeslab_464': 0.526, 'yeslab_464|yeslab_381': 0.344}（R2 REVIEW，最差单年 0.163）
- R3 REVIEW

注：
- pooled retention 0.344 / worst-year 0.163 inside REVIEW band
- largecap_dominance-ON retention for resic__yeslab_464__given__yeslab_381 = 0.049 < 0.30
- liquidity_shock-ON retention for resic__yeslab_464__given__yeslab_381 = -0.057 < 0.30
- liquidity_shock-ON sign flip for resic__yeslab_464__given__yeslab_381 (raw -0.0620 vs resid +0.0035)

## yeslab_471<->yeslab_484 — 总判定：**REVIEW**

- 各年 median|rho| 最大值：0.727（R1 PASS）
- pooled 保留率：{'yeslab_471|yeslab_484': 0.354, 'yeslab_484|yeslab_471': 0.476}（R2 REVIEW，最差单年 0.266）
- R3 REVIEW

注：
- pooled retention 0.354 / worst-year 0.266 inside REVIEW band
- liquidity_shock-ON retention for resic__yeslab_484__given__yeslab_471 = -2.813 < 0.30
- liquidity_shock-ON sign flip for resic__yeslab_484__given__yeslab_471 (raw -0.0086 vs resid +0.0243)

## yeslab_381<->yeslab_443_reg30_max20_v1 — 总判定：**REVIEW**

- 各年 median|rho| 最大值：0.650（R1 PASS）
- pooled 保留率：{'yeslab_381|yeslab_443_reg30_max20_v1': 0.464, 'yeslab_443_reg30_max20_v1|yeslab_381': 0.493}（R2 REVIEW，最差单年 0.382）
- R3 REVIEW

注：
- pooled retention 0.464 / worst-year 0.382 inside REVIEW band
- liquidity_shock-ON retention for resic__yeslab_443_reg30_max20_v1__given__yeslab_381 = -1.520 < 0.30
- liquidity_shock-ON sign flip for resic__yeslab_443_reg30_max20_v1__given__yeslab_381 (raw -0.0249 vs resid +0.0378)

## 逐年明细

| 年 | 对 | mean rho | median|rho| | ≥0.85占比 | IC_A | IC_B | 残差IC A|B | 保留率 A|B | 残差IC B|A | 保留率 B|A |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020 | 381↔464 | +0.663 | 0.673 | 0.0% | -0.0424 | -0.0478 | -0.0117 | 0.28 | -0.0257 | 0.54 |
| 2020 | 471↔484 | -0.702 | 0.727 | 0.0% | +0.0557 | -0.0643 | +0.0148 | 0.27 | -0.0349 | 0.54 |
| 2020 | 381↔443_reg30_max20_v1 | +0.610 | 0.621 | 0.0% | -0.0424 | -0.0370 | -0.0250 | 0.59 | -0.0141 | 0.38 |
| 2021 | 381↔464 | +0.651 | 0.663 | 0.0% | -0.0692 | -0.0510 | -0.0469 | 0.68 | -0.0083 | 0.16 |
| 2021 | 471↔484 | -0.706 | 0.711 | 0.0% | +0.0585 | -0.0563 | +0.0256 | 0.44 | -0.0225 | 0.40 |
| 2021 | 381↔443_reg30_max20_v1 | +0.645 | 0.650 | 0.0% | -0.0692 | -0.0733 | -0.0268 | 0.39 | -0.0402 | 0.55 |

## 体制分桶（pooled 2020–2025）

### largecap_dominance

| 桶 | 天数 | rho 381↔464 | rho 471↔484 | rho 381↔443_reg30_max20_v1 | 保留率 381|464 | 保留率 464|381 | 保留率 471|484 | 保留率 484|471 | 保留率 381|443_reg30_max20_v1 | 保留率 443_reg30_max20_v1|381 |
|---|---|---|---|---|---|---|---|---|---|---|
| ON | 31 | +0.586 | -0.722 | +0.691 | 0.76 | 0.05 | 0.50 | 0.31 | 0.46 | 0.41 |
| OFF | 455 | +0.662 | -0.703 | +0.623 | 0.50 | 0.36 | 0.34 | 0.49 | 0.47 | 0.50 |

### liquidity_shock

| 桶 | 天数 | rho 381↔464 | rho 471↔484 | rho 381↔443_reg30_max20_v1 | 保留率 381|464 | 保留率 464|381 | 保留率 471|484 | 保留率 484|471 | 保留率 381|443_reg30_max20_v1 | 保留率 443_reg30_max20_v1|381 |
|---|---|---|---|---|---|---|---|---|---|---|
| ON | 15 | +0.648 | -0.676 | +0.560 | 0.77 | -0.06 | 1.13 | -2.81 | 0.97 | -1.52 |
| OFF | 471 | +0.658 | -0.705 | +0.630 | 0.51 | 0.36 | 0.34 | 0.49 | 0.43 | 0.52 |
