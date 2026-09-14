# Dense-factor coverage behind the dropped OOS days

Workspace `/data/zhangyudong/qrv2_research_artifacts/m0r_v2`. Dense set (parsed from the driver's own source): yeslab_206, yeslab_357, yeslab_381, yeslab_425, yeslab_443_reg30_max20_v1, yeslab_464, yeslab_471, yeslab_484, yeslab_96

| Year | Valid-label rows | Dense-incomplete | Share | Securities affected | STAR in frame | STAR affected | Days above 0.10 |
|---|---|---|---|---|---|---|---|
| 2022 | 803572 | 57051 | 0.0710 | 688 | 326 | 326 | 1 |
| 2023 | 881863 | 61554 | 0.0698 | 686 | 457 | 427 | 23 |
| 2024 | 931411 | 7666 | 0.0082 | 319 | 490 | 26 | 0 |
| 2025 | 936317 | 13332 | 0.0142 | 331 | 543 | 48 | 0 |

## Per-month incomplete share

| Year-month | Share |
|---|---|
| 2022-01 | 0.0565 |
| 2022-02 | 0.0598 |
| 2022-03 | 0.0559 |
| 2022-04 | 0.0532 |
| 2022-05 | 0.0590 |
| 2022-06 | 0.0744 |
| 2022-07 | 0.0774 |
| 2022-08 | 0.0805 |
| 2022-09 | 0.0748 |
| 2022-10 | 0.0750 |
| 2022-11 | 0.0890 |
| 2022-12 | 0.0850 |
| 2023-01 | 0.0878 |
| 2023-02 | 0.0907 |
| 2023-03 | 0.0920 |
| 2023-04 | 0.0964 |
| 2023-05 | 0.0984 |
| 2023-06 | 0.0979 |
| 2023-07 | 0.0996 |
| 2023-08 | 0.0872 |
| 2023-09 | 0.0739 |
| 2023-10 | 0.0054 |
| 2023-11 | 0.0064 |
| 2023-12 | 0.0072 |
| 2024-01 | 0.0081 |
| 2024-02 | 0.0062 |
| 2024-03 | 0.0053 |
| 2024-04 | 0.0040 |
| 2024-05 | 0.0051 |
| 2024-06 | 0.0075 |
| 2024-07 | 0.0097 |
| 2024-08 | 0.0088 |
| 2024-09 | 0.0061 |
| 2024-10 | 0.0068 |
| 2024-11 | 0.0118 |
| 2024-12 | 0.0171 |
| 2025-01 | 0.0143 |
| 2025-02 | 0.0132 |
| 2025-03 | 0.0125 |
| 2025-04 | 0.0129 |
| 2025-05 | 0.0131 |
| 2025-06 | 0.0144 |
| 2025-07 | 0.0154 |
| 2025-08 | 0.0168 |
| 2025-09 | 0.0143 |
| 2025-10 | 0.0133 |
| 2025-11 | 0.0135 |
| 2025-12 | 0.0165 |

## Which factors are missing together (top combinations per year)

* **2022**: f01,f12 = 43740; f00,f10 = 6261; f00 = 2175; f00,f10,f12,f14 = 1518; f00,f09,f10,f11,f12,f14 = 1166
  * two most-missing factors `f12` and `f01`: both missing on 44049 rows, f12 only on 4566, f01 only on 0 -> nested (one factor's gaps are a subset of the other's)
* **2023**: f01,f12 = 52400; f00,f10 = 4273; f00 = 1749; f00,f10,f12,f14 = 933; f00,f09,f10,f11,f12,f14 = 716
  * two most-missing factors `f12` and `f01`: both missing on 52725 rows, f12 only on 2807, f01 only on 0 -> nested (one factor's gaps are a subset of the other's)
* **2024**: f00,f10 = 3386; f00 = 1161; f00,f10,f12,f14 = 841; f00,f09,f10,f11,f12,f14 = 633; f00,f10,f11,f12,f14 = 508
  * two most-missing factors `f00` and `f10`: both missing on 6195 rows, f00 only on 1161, f10 only on 0 -> nested (one factor's gaps are a subset of the other's)
* **2025**: f00,f10 = 6119; f00 = 2266; f00,f10,f12,f14 = 1430; f00,f09,f10,f11,f12,f14 = 1188; f00,f10,f11,f12,f14 = 892
  * two most-missing factors `f00` and `f10`: both missing on 10806 rows, f00 only on 2266, f10 only on 0 -> nested (one factor's gaps are a subset of the other's)

## Per-factor missing share

| Factor | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| yeslab_206 (f00) | 0.0166 | 0.0104 | 0.0079 | 0.0140 |
| yeslab_357 (f01) | 0.0548 | 0.0598 | 0.0007 | 0.0005 |
| yeslab_381 (f04) | 0.0011 | 0.0006 | 0.0007 | 0.0010 |
| yeslab_425 (f09) | 0.0026 | 0.0015 | 0.0014 | 0.0023 |
| yeslab_443_reg30_max20_v1 (f10) | 0.0138 | 0.0083 | 0.0067 | 0.0115 |
| yeslab_464 (f11) | 0.0038 | 0.0021 | 0.0020 | 0.0032 |
| yeslab_471 (f12) | 0.0605 | 0.0630 | 0.0033 | 0.0053 |
| yeslab_484 (f13) | 0.0000 | 0.0000 | 0.0003 | 0.0000 |
| yeslab_96 (f14) | 0.0057 | 0.0032 | 0.0029 | 0.0047 |

Read this with `analyze_yearly_baseline_deltas.py`: that script shows the days these rows cost the evaluation, this one shows which factor and which securities they belong to.
