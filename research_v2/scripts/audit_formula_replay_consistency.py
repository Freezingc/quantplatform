"""Replay source-file calculators against stored YesLab factor observations.

This command is read-only.  It replays the registered candidate and blocked
factors against the current calculator and market snapshot.  A finite overlap
mismatch is evidence that the stored factor cannot be safely reconstructed
from the current calculator and market snapshot, even when the missing input
field is available from a secondary source.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
REPO_ROOT = Path(__file__).resolve().parents[2]
CALCULATOR_ROOT = REPO_ROOT / "数据库"


@dataclass(frozen=True)
class ReplayDefinition:
    factor_id: str
    factor_name: str
    store_factor: str
    calculator_module: str
    calculator_class: str
    lookback_days: int
    replay_lookback_days: int | None = None

    @property
    def effective_replay_lookback_days(self) -> int:
        return self.replay_lookback_days or self.lookback_days


REPLAY_DEFINITIONS = {
    "yeslab_5": ReplayDefinition(
        factor_id="yeslab_5",
        factor_name="Alpha5",
        store_factor="factor_data1",
        calculator_module="Alpha5Calculator",
        calculator_class="Alpha5Calculator",
        lookback_days=10,
        replay_lookback_days=10,
    ),
    "yeslab_31": ReplayDefinition(
        factor_id="yeslab_31",
        factor_name="Alpha32",
        store_factor="factor_data1",
        calculator_module="Alpha32Calculator",
        calculator_class="Alpha32Calculator",
        lookback_days=4,
        replay_lookback_days=4,
    ),
    "yeslab_71": ReplayDefinition(
        factor_id="yeslab_71",
        factor_name="Alpha62",
        store_factor="factor_data2",
        calculator_module="Alpha62Calculator",
        calculator_class="Alpha62Calculator",
        lookback_days=4,
        replay_lookback_days=4,
    ),
    "yeslab_105": ReplayDefinition(
        factor_id="yeslab_105",
        factor_name="Alpha83",
        store_factor="factor_data2",
        calculator_module="Alpha83Calculator",
        calculator_class="Alpha83Calculator",
        lookback_days=4,
        replay_lookback_days=4,
    ),
    "yeslab_94": ReplayDefinition(
        factor_id="yeslab_94",
        factor_name="日间反转-波动翻转",
        store_factor="factor_data3",
        calculator_module="DayReversalVolatilityCalculator",
        calculator_class="DayReversalVolatilityCalculator",
        lookback_days=20,
    ),
    "yeslab_95": ReplayDefinition(
        factor_id="yeslab_95",
        factor_name="日间反转-换手翻转",
        store_factor="factor_data3",
        calculator_module="DayReversalTurnoverCalculator",
        calculator_class="DayReversalTurnoverCalculator",
        lookback_days=20,
    ),
    "yeslab_96": ReplayDefinition(
        factor_id="yeslab_96",
        factor_name="修正日间反转",
        store_factor="factor_data3",
        calculator_module="ModifiedDayReversalCalculator",
        calculator_class="ModifiedDayReversalCalculator",
        lookback_days=20,
    ),
    "yeslab_98": ReplayDefinition(
        factor_id="yeslab_98",
        factor_name="日内反转-波动翻转",
        store_factor="factor_data3",
        calculator_module="IntradayReversalVolatilityCalculator",
        calculator_class="IntradayReversalVolatilityCalculator",
        lookback_days=20,
    ),
    "yeslab_114": ReplayDefinition(
        factor_id="yeslab_114",
        factor_name="惊恐波动",
        store_factor="factor_data3",
        calculator_module="TerriStdCalculator",
        calculator_class="TerriStdCalculator",
        lookback_days=20,
    ),
    "yeslab_213": ReplayDefinition(
        factor_id="yeslab_213",
        factor_name="量价齐飞加换手率",
        store_factor="factor_data5",
        calculator_module="FlyPlusTurnoverCalculator",
        calculator_class="FlyPlusTurnoverCalculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_143": ReplayDefinition(
        factor_id="yeslab_143",
        factor_name="日换手率波动偏离（1个月内）",
        store_factor="factor_data4",
        calculator_module="Bias_Std_Turn_1m_Calculator",
        calculator_class="Bias_Std_Turn_1m_Calculator",
        lookback_days=504,
        replay_lookback_days=504,
    ),
    "yeslab_155": ReplayDefinition(
        factor_id="yeslab_155",
        factor_name="跳空去均值绝对值",
        store_factor="factor_data4",
        calculator_module="SkipDemeanAbsCalculator",
        calculator_class="SkipDemeanAbsCalculator",
        lookback_days=1,
    ),
    "yeslab_163": ReplayDefinition(
        factor_id="yeslab_163",
        factor_name="Alpha95",
        store_factor="factor_data4",
        calculator_module="Alpha95Calculator",
        calculator_class="Alpha95Calculator",
        lookback_days=19,
    ),
    "yeslab_165": ReplayDefinition(
        factor_id="yeslab_165",
        factor_name="Alpha97",
        store_factor="factor_data4",
        calculator_module="Alpha97Calculator",
        calculator_class="Alpha97Calculator",
        lookback_days=10,
    ),
    "yeslab_167": ReplayDefinition(
        factor_id="yeslab_167",
        factor_name="Alpha99",
        store_factor="factor_data4",
        calculator_module="Alpha99Calculator",
        calculator_class="Alpha99Calculator",
        lookback_days=5,
    ),
    "yeslab_168": ReplayDefinition(
        factor_id="yeslab_168",
        factor_name="Alpha100",
        store_factor="factor_data4",
        calculator_module="Alpha100Calculator",
        calculator_class="Alpha100Calculator",
        lookback_days=20,
    ),
    "yeslab_206": ReplayDefinition(
        factor_id="yeslab_206",
        factor_name="60日成交量去均值相对偏离",
        store_factor="factor_data5",
        calculator_module="Bias_Vol_Demean_60_Calculator",
        calculator_class="Bias_Vol_Demean_60_Calculator",
        lookback_days=60,
    ),
    "yeslab_208": ReplayDefinition(
        factor_id="yeslab_208",
        factor_name="价格均值标准差乖离乘短期成交量波动",
        store_factor="factor_data5",
        calculator_module="Bias_Close_Mean_Std_Low_Vol_Std_60_Calculator",
        calculator_class="Bias_Close_Mean_Std_Low_Vol_Std_60_Calculator",
        lookback_days=60,
    ),
    "yeslab_222": ReplayDefinition(
        factor_id="yeslab_222",
        factor_name="ROE",
        store_factor="factor_data5",
        calculator_module="ROE_Calculator",
        calculator_class="ROE_Calculator",
        lookback_days=1,
    ),
    "yeslab_224": ReplayDefinition(
        factor_id="yeslab_224",
        factor_name="ROIC",
        store_factor="factor_data5",
        calculator_module="ROIC_Calculator",
        calculator_class="ROIC_Calculator",
        lookback_days=1,
    ),
    "yeslab_370": ReplayDefinition(
        factor_id="yeslab_370",
        factor_name="GP_001",
        store_factor="factor_data8",
        calculator_module="GP_Factor_001_Calculator",
        calculator_class="GP_Factor_001_Calculator",
        lookback_days=10,
        replay_lookback_days=180,
    ),
    "yeslab_371": ReplayDefinition(
        factor_id="yeslab_371",
        factor_name="GP_002",
        store_factor="factor_data8",
        calculator_module="GP_Factor_002_Calculator",
        calculator_class="GP_Factor_002_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_372": ReplayDefinition(
        factor_id="yeslab_372",
        factor_name="GP_003",
        store_factor="factor_data8",
        calculator_module="GP_Factor_003_Calculator",
        calculator_class="GP_Factor_003_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_379": ReplayDefinition(
        factor_id="yeslab_379",
        factor_name="GP_010",
        store_factor="factor_data8",
        calculator_module="GP_Factor_010_Calculator",
        calculator_class="GP_Factor_010_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_375": ReplayDefinition(
        factor_id="yeslab_375",
        factor_name="GP_006",
        store_factor="factor_data8",
        calculator_module="GP_Factor_006_Calculator",
        calculator_class="GP_Factor_006_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_384": ReplayDefinition(
        factor_id="yeslab_384",
        factor_name="GP_015",
        store_factor="factor_data8",
        calculator_module="GP_Factor_015_Calculator",
        calculator_class="GP_Factor_015_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_389": ReplayDefinition(
        factor_id="yeslab_389",
        factor_name="GP_020",
        store_factor="factor_data9",
        calculator_module="GP_Factor_020_Calculator",
        calculator_class="GP_Factor_020_Calculator",
        lookback_days=10,
        replay_lookback_days=180,
    ),
    "yeslab_390": ReplayDefinition(
        factor_id="yeslab_390",
        factor_name="GP_021",
        store_factor="factor_data9",
        calculator_module="GP_Factor_021_Calculator",
        calculator_class="GP_Factor_021_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_397": ReplayDefinition(
        factor_id="yeslab_397",
        factor_name="GP_028",
        store_factor="factor_data9",
        calculator_module="GP_Factor_028_Calculator",
        calculator_class="GP_Factor_028_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_404": ReplayDefinition(
        factor_id="yeslab_404",
        factor_name="GP_035",
        store_factor="factor_data9",
        calculator_module="GP_Factor_035_Calculator",
        calculator_class="GP_Factor_035_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_406": ReplayDefinition(
        factor_id="yeslab_406",
        factor_name="GP_037",
        store_factor="factor_data9",
        calculator_module="GP_Factor_037_Calculator",
        calculator_class="GP_Factor_037_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_411": ReplayDefinition(
        factor_id="yeslab_411",
        factor_name="GP_042",
        store_factor="factor_data9",
        calculator_module="GP_Factor_042_Calculator",
        calculator_class="GP_Factor_042_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_414": ReplayDefinition(
        factor_id="yeslab_414",
        factor_name="GP_045",
        store_factor="factor_data9",
        calculator_module="GP_Factor_045_Calculator",
        calculator_class="GP_Factor_045_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_415": ReplayDefinition(
        factor_id="yeslab_415",
        factor_name="GP_046",
        store_factor="factor_data9",
        calculator_module="GP_Factor_046_Calculator",
        calculator_class="GP_Factor_046_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_420": ReplayDefinition(
        factor_id="yeslab_420",
        factor_name="GP_051",
        store_factor="factor_data9",
        calculator_module="GP_Factor_051_Calculator",
        calculator_class="GP_Factor_051_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_423": ReplayDefinition(
        factor_id="yeslab_423",
        factor_name="GP_054",
        store_factor="factor_data9",
        calculator_module="GP_Factor_054_Calculator",
        calculator_class="GP_Factor_054_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_357": ReplayDefinition(
        factor_id="yeslab_357",
        factor_name="行业内收益标准化绝对值",
        store_factor="factor_data8",
        calculator_module="IndustryStandAbsCalculator",
        calculator_class="IndustryStandAbsCalculator",
        lookback_days=1,
    ),
    "yeslab_362": ReplayDefinition(
        factor_id="yeslab_362",
        factor_name="行业标准化绝对值除行业偏度5日均",
        store_factor="factor_data8",
        calculator_module="IndustryStandAbsDivSkewCalculator",
        calculator_class="IndustryStandAbsDivSkewCalculator",
        lookback_days=5,
    ),
    "yeslab_363": ReplayDefinition(
        factor_id="yeslab_363",
        factor_name="行业标准化绝对值除行业偏度绝对值",
        store_factor="factor_data8",
        calculator_module="IndustryStandAbsDivSkewAbsCalculator",
        calculator_class="IndustryStandAbsDivSkewAbsCalculator",
        lookback_days=5,
    ),
    "yeslab_365": ReplayDefinition(
        factor_id="yeslab_365",
        factor_name="行业标准化绝对值按正偏度过滤",
        store_factor="factor_data8",
        calculator_module="IndustryStandAbsCutBySkewCalculator",
        calculator_class="IndustryStandAbsCutBySkewCalculator",
        lookback_days=5,
    ),
    "yeslab_376": ReplayDefinition(
        factor_id="yeslab_376",
        factor_name="GP_007",
        store_factor="factor_data8",
        calculator_module="GP_Factor_007_Calculator",
        calculator_class="GP_Factor_007_Calculator",
        lookback_days=10,
    ),
    "yeslab_377": ReplayDefinition(
        factor_id="yeslab_377",
        factor_name="GP_008",
        store_factor="factor_data8",
        calculator_module="GP_Factor_008_Calculator",
        calculator_class="GP_Factor_008_Calculator",
        lookback_days=1,
    ),
    "yeslab_380": ReplayDefinition(
        factor_id="yeslab_380",
        factor_name="GP_011",
        store_factor="factor_data8",
        calculator_module="GP_Factor_011_Calculator",
        calculator_class="GP_Factor_011_Calculator",
        lookback_days=5,
    ),
    "yeslab_381": ReplayDefinition(
        factor_id="yeslab_381",
        factor_name="GP_012",
        store_factor="factor_data8",
        calculator_module="GP_Factor_012_Calculator",
        calculator_class="GP_Factor_012_Calculator",
        lookback_days=5,
    ),
    "yeslab_385": ReplayDefinition(
        factor_id="yeslab_385",
        factor_name="GP_016",
        store_factor="factor_data8",
        calculator_module="GP_Factor_016_Calculator",
        calculator_class="GP_Factor_016_Calculator",
        lookback_days=5,
    ),
    "yeslab_387": ReplayDefinition(
        factor_id="yeslab_387",
        factor_name="GP_018",
        store_factor="factor_data9",
        calculator_module="GP_Factor_018_Calculator",
        calculator_class="GP_Factor_018_Calculator",
        lookback_days=1,
    ),
    "yeslab_388": ReplayDefinition(
        factor_id="yeslab_388",
        factor_name="GP_019",
        store_factor="factor_data9",
        calculator_module="GP_Factor_019_Calculator",
        calculator_class="GP_Factor_019_Calculator",
        lookback_days=10,
    ),
    "yeslab_391": ReplayDefinition(
        factor_id="yeslab_391",
        factor_name="GP_022",
        store_factor="factor_data9",
        calculator_module="GP_Factor_022_Calculator",
        calculator_class="GP_Factor_022_Calculator",
        lookback_days=60,
    ),
    "yeslab_396": ReplayDefinition(
        factor_id="yeslab_396",
        factor_name="GP_027",
        store_factor="factor_data9",
        calculator_module="GP_Factor_027_Calculator",
        calculator_class="GP_Factor_027_Calculator",
        lookback_days=5,
    ),
    "yeslab_398": ReplayDefinition(
        factor_id="yeslab_398",
        factor_name="GP_029",
        store_factor="factor_data9",
        calculator_module="GP_Factor_029_Calculator",
        calculator_class="GP_Factor_029_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_400": ReplayDefinition(
        factor_id="yeslab_400",
        factor_name="GP_031",
        store_factor="factor_data9",
        calculator_module="GP_Factor_031_Calculator",
        calculator_class="GP_Factor_031_Calculator",
        lookback_days=60,
    ),
    "yeslab_401": ReplayDefinition(
        factor_id="yeslab_401",
        factor_name="GP_032",
        store_factor="factor_data9",
        calculator_module="GP_Factor_032_Calculator",
        calculator_class="GP_Factor_032_Calculator",
        lookback_days=60,
    ),
    "yeslab_405": ReplayDefinition(
        factor_id="yeslab_405",
        factor_name="GP_036",
        store_factor="factor_data9",
        calculator_module="GP_Factor_036_Calculator",
        calculator_class="GP_Factor_036_Calculator",
        lookback_days=5,
    ),
    "yeslab_425": ReplayDefinition(
        factor_id="yeslab_425",
        factor_name="GP_056",
        store_factor="factor_data9",
        calculator_module="GP_Factor_056_Calculator",
        calculator_class="GP_Factor_056_Calculator",
        lookback_days=30,
    ),
    "yeslab_429": ReplayDefinition(
        factor_id="yeslab_429",
        factor_name="GP_060",
        store_factor="factor_data9",
        calculator_module="GP_Factor_060_Calculator",
        calculator_class="GP_Factor_060_Calculator",
        lookback_days=10,
        replay_lookback_days=180,
    ),
    "yeslab_431": ReplayDefinition(
        factor_id="yeslab_431",
        factor_name="GP_062",
        store_factor="factor_data9",
        calculator_module="GP_Factor_062_Calculator",
        calculator_class="GP_Factor_062_Calculator",
        lookback_days=20,
    ),
    "yeslab_432": ReplayDefinition(
        factor_id="yeslab_432",
        factor_name="GP_063",
        store_factor="factor_data9",
        calculator_module="GP_Factor_063_Calculator",
        calculator_class="GP_Factor_063_Calculator",
        lookback_days=5,
    ),
    "yeslab_433": ReplayDefinition(
        factor_id="yeslab_433",
        factor_name="GP_064",
        store_factor="factor_data9",
        calculator_module="GP_Factor_064_Calculator",
        calculator_class="GP_Factor_064_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_436": ReplayDefinition(
        factor_id="yeslab_436",
        factor_name="GP_067",
        store_factor="factor_data9",
        calculator_module="GP_Factor_067_Calculator",
        calculator_class="GP_Factor_067_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_437": ReplayDefinition(
        factor_id="yeslab_437",
        factor_name="GP_068",
        store_factor="factor_data10",
        calculator_module="GP_Factor_068_Calculator",
        calculator_class="GP_Factor_068_Calculator",
        lookback_days=60,
    ),
    "yeslab_438": ReplayDefinition(
        factor_id="yeslab_438",
        factor_name="GP_069",
        store_factor="factor_data10",
        calculator_module="GP_Factor_069_Calculator",
        calculator_class="GP_Factor_069_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_442": ReplayDefinition(
        factor_id="yeslab_442",
        factor_name="GP_073",
        store_factor="factor_data10",
        calculator_module="GP_Factor_073_Calculator",
        calculator_class="GP_Factor_073_Calculator",
        lookback_days=60,
    ),
    "yeslab_443": ReplayDefinition(
        factor_id="yeslab_443",
        factor_name="GP_074",
        store_factor="factor_data10",
        calculator_module="GP_Factor_074_Calculator",
        calculator_class="GP_Factor_074_Calculator",
        lookback_days=30,
    ),
    "yeslab_446": ReplayDefinition(
        factor_id="yeslab_446",
        factor_name="GP_077",
        store_factor="factor_data10",
        calculator_module="GP_Factor_077_Calculator",
        calculator_class="GP_Factor_077_Calculator",
        lookback_days=60,
        replay_lookback_days=300,
    ),
    "yeslab_451": ReplayDefinition(
        factor_id="yeslab_451",
        factor_name="GP_082",
        store_factor="factor_data10",
        calculator_module="GP_Factor_082_Calculator",
        calculator_class="GP_Factor_082_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_464": ReplayDefinition(
        factor_id="yeslab_464",
        factor_name="GP_095",
        store_factor="factor_data10",
        calculator_module="GP_Factor_095_Calculator",
        calculator_class="GP_Factor_095_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_468": ReplayDefinition(
        factor_id="yeslab_468",
        factor_name="GP_099",
        store_factor="factor_data10",
        calculator_module="GP_Factor_099_Calculator",
        calculator_class="GP_Factor_099_Calculator",
        lookback_days=120,
        replay_lookback_days=180,
    ),
    "yeslab_378": ReplayDefinition(
        factor_id="yeslab_378",
        factor_name="GP_009",
        store_factor="factor_data8",
        calculator_module="GP_Factor_009_Calculator",
        calculator_class="GP_Factor_009_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_386": ReplayDefinition(
        factor_id="yeslab_386",
        factor_name="GP_017",
        store_factor="factor_data8",
        calculator_module="GP_Factor_017_Calculator",
        calculator_class="GP_Factor_017_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_409": ReplayDefinition(
        factor_id="yeslab_409",
        factor_name="GP_040",
        store_factor="factor_data9",
        calculator_module="GP_Factor_040_Calculator",
        calculator_class="GP_Factor_040_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_418": ReplayDefinition(
        factor_id="yeslab_418",
        factor_name="GP_049",
        store_factor="factor_data9",
        calculator_module="GP_Factor_049_Calculator",
        calculator_class="GP_Factor_049_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_419": ReplayDefinition(
        factor_id="yeslab_419",
        factor_name="GP_050",
        store_factor="factor_data9",
        calculator_module="GP_Factor_050_Calculator",
        calculator_class="GP_Factor_050_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_428": ReplayDefinition(
        factor_id="yeslab_428",
        factor_name="GP_059",
        store_factor="factor_data9",
        calculator_module="GP_Factor_059_Calculator",
        calculator_class="GP_Factor_059_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_434": ReplayDefinition(
        factor_id="yeslab_434",
        factor_name="GP_065",
        store_factor="factor_data9",
        calculator_module="GP_Factor_065_Calculator",
        calculator_class="GP_Factor_065_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_444": ReplayDefinition(
        factor_id="yeslab_444",
        factor_name="GP_075",
        store_factor="factor_data10",
        calculator_module="GP_Factor_075_Calculator",
        calculator_class="GP_Factor_075_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_461": ReplayDefinition(
        factor_id="yeslab_461",
        factor_name="GP_092",
        store_factor="factor_data10",
        calculator_module="GP_Factor_092_Calculator",
        calculator_class="GP_Factor_092_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_467": ReplayDefinition(
        factor_id="yeslab_467",
        factor_name="GP_098",
        store_factor="factor_data10",
        calculator_module="GP_Factor_098_Calculator",
        calculator_class="GP_Factor_098_Calculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_46": ReplayDefinition(
        factor_id="yeslab_46",
        factor_name="月换手波动",
        store_factor="factor_data3",
        calculator_module="turnover_monthly_std_Calculator",
        calculator_class="turnover_monthly_std_Calculator",
        lookback_days=60,
        replay_lookback_days=21,
    ),
    "yeslab_83": ReplayDefinition(
        factor_id="yeslab_83",
        factor_name="Alpha74",
        store_factor="factor_data2",
        calculator_module="Alpha74Calculator",
        calculator_class="Alpha74Calculator",
        lookback_days=20,
        replay_lookback_days=60,
    ),
    "yeslab_140": ReplayDefinition(
        factor_id="yeslab_140",
        factor_name="日换手率波动（1个月内）",
        store_factor="factor_data4",
        calculator_module="Std_Turn_1m_Calculator",
        calculator_class="Std_Turn_1m_Calculator",
        lookback_days=21,
        replay_lookback_days=60,
    ),
    "yeslab_209": ReplayDefinition(
        factor_id="yeslab_209",
        factor_name="量价齐飞",
        store_factor="factor_data5",
        calculator_module="FlyCalculator",
        calculator_class="FlyCalculator",
        lookback_days=20,
        replay_lookback_days=60,
    ),
    "yeslab_211": ReplayDefinition(
        factor_id="yeslab_211",
        factor_name="量价齐飞-开盘价",
        store_factor="factor_data5",
        calculator_module="FlyOpenCalculator",
        calculator_class="FlyOpenCalculator",
        lookback_days=20,
        replay_lookback_days=60,
    ),
    "yeslab_212": ReplayDefinition(
        factor_id="yeslab_212",
        factor_name="量价齐飞-排名",
        store_factor="factor_data5",
        calculator_module="FlyRankCalculator",
        calculator_class="FlyRankCalculator",
        lookback_days=20,
        replay_lookback_days=60,
    ),
    "yeslab_214": ReplayDefinition(
        factor_id="yeslab_214",
        factor_name="量价齐飞乘换手率5日均值",
        store_factor="factor_data5",
        calculator_module="FlyMultipleTurnover5Calculator",
        calculator_class="FlyMultipleTurnover5Calculator",
        lookback_days=25,
        replay_lookback_days=60,
    ),
    "yeslab_252": ReplayDefinition(
        factor_id="yeslab_252",
        factor_name="1个月上行波动率",
        store_factor="factor_data6",
        calculator_module="Vol_Up_Std_1M_Calculator",
        calculator_class="Vol_Up_Std_1M_Calculator",
        lookback_days=21,
        replay_lookback_days=80,
    ),
    "yeslab_260": ReplayDefinition(
        factor_id="yeslab_260",
        factor_name="3个月上行波动率",
        store_factor="factor_data6",
        calculator_module="Vol_Up_Std_3M_Calculator",
        calculator_class="Vol_Up_Std_3M_Calculator",
        lookback_days=63,
        replay_lookback_days=120,
    ),
    "yeslab_271": ReplayDefinition(
        factor_id="yeslab_271",
        factor_name="1个月日内振幅标准差",
        store_factor="factor_data6",
        calculator_module="Vol_HighLow_Std_1M_Calculator",
        calculator_class="Vol_HighLow_Std_1M_Calculator",
        lookback_days=21,
        replay_lookback_days=80,
    ),
    "yeslab_272": ReplayDefinition(
        factor_id="yeslab_272",
        factor_name="3个月日内振幅标准差",
        store_factor="factor_data6",
        calculator_module="Vol_HighLow_Std_3M_Calculator",
        calculator_class="Vol_HighLow_Std_3M_Calculator",
        lookback_days=63,
        replay_lookback_days=120,
    ),
    "yeslab_359": ReplayDefinition(
        factor_id="yeslab_359",
        factor_name="行业标准化绝对值乘行业波动率",
        store_factor="factor_data8",
        calculator_module="IndustryStandAbsMultiStdCalculator",
        calculator_class="IndustryStandAbsMultiStdCalculator",
        lookback_days=60,
        replay_lookback_days=180,
    ),
    "yeslab_392": ReplayDefinition(
        factor_id="yeslab_392",
        factor_name="GP_023",
        store_factor="factor_data9",
        calculator_module="GP_Factor_023_Calculator",
        calculator_class="GP_Factor_023_Calculator",
        lookback_days=20,
        replay_lookback_days=180,
    ),
    "yeslab_395": ReplayDefinition(
        factor_id="yeslab_395",
        factor_name="GP_026",
        store_factor="factor_data9",
        calculator_module="GP_Factor_026_Calculator",
        calculator_class="GP_Factor_026_Calculator",
        lookback_days=60,
    ),
    "yeslab_471": ReplayDefinition(
        factor_id="yeslab_471",
        factor_name="AttentionSpillover",
        store_factor="factor_data10",
        calculator_module="AttentionSpilloverCalculator",
        calculator_class="AttentionSpilloverCalculator",
        lookback_days=21,
    ),
    "yeslab_481": ReplayDefinition(
        factor_id="yeslab_481",
        factor_name="羊群效应_CSAD收益波动比20",
        store_factor="factor_data10",
        calculator_module="HerdingCSADRatio20Calculator",
        calculator_class="HerdingCSADRatio20Calculator",
        lookback_days=120,
    ),
    "yeslab_482": ReplayDefinition(
        factor_id="yeslab_482",
        factor_name="羊群效应_CSAD收益波动比120",
        store_factor="factor_data10",
        calculator_module="HerdingCSADRatio120Calculator",
        calculator_class="HerdingCSADRatio120Calculator",
        lookback_days=120,
    ),
    "yeslab_483": ReplayDefinition(
        factor_id="yeslab_483",
        factor_name="羊群效应_CSAD短长周期比",
        store_factor="factor_data10",
        calculator_module="HerdingCSADRatio20To120Calculator",
        calculator_class="HerdingCSADRatio20To120Calculator",
        lookback_days=120,
    ),
    "yeslab_484": ReplayDefinition(
        factor_id="yeslab_484",
        factor_name="羊群效应_CSAD波动率21",
        store_factor="factor_data10",
        calculator_module="HerdingCSADStd21Calculator",
        calculator_class="HerdingCSADStd21Calculator",
        lookback_days=120,
    ),
    "yeslab_485": ReplayDefinition(
        factor_id="yeslab_485",
        factor_name="CS价差_日频",
        store_factor="factor_data10",
        calculator_module="CorwinSchultzDailyCalculator",
        calculator_class="CorwinSchultzDailyCalculator",
        lookback_days=21,
    ),
    "yeslab_486": ReplayDefinition(
        factor_id="yeslab_486",
        factor_name="CS价差_21日均值",
        store_factor="factor_data10",
        calculator_module="CorwinSchultzMean21Calculator",
        calculator_class="CorwinSchultzMean21Calculator",
        lookback_days=21,
    ),
    "yeslab_487": ReplayDefinition(
        factor_id="yeslab_487",
        factor_name="CS价差_21日标准差",
        store_factor="factor_data11",
        calculator_module="CorwinSchultzStd21Calculator",
        calculator_class="CorwinSchultzStd21Calculator",
        lookback_days=21,
    ),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 9, 14))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2020, 9, 25))
    parser.add_argument(
        "--max-ts-codes",
        type=int,
        help="limit each replay query to the first N sorted securities for a bounded smoke audit",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(args.factor_id or REPLAY_DEFINITIONS)
    unsupported = sorted(set(args.factor_id).difference(REPLAY_DEFINITIONS))
    if unsupported:
        raise ValueError(f"unsupported formula replay factor IDs: {unsupported}")
    if not args.factor_id or len(set(args.factor_id)) != len(args.factor_id):
        raise ValueError("factor IDs must be non-empty and unique")
    if args.start > args.end:
        raise ValueError("start date must not be after end date")
    if args.end >= LOCKED_TEST_START:
        raise ValueError("formula replay cannot read 2026 locked-test data")
    if args.max_ts_codes is not None and args.max_ts_codes < 1:
        raise ValueError("max-ts-codes must be positive")
    return args


class _BoundedClient:
    """Limit stock-panel queries without changing the underlying read-only client."""

    def __init__(self, client: object, max_ts_codes: int | None) -> None:
        self._client = client
        self._max_ts_codes = max_ts_codes

    def run(self, script: str) -> object:
        result = self._client.run(script)
        if self._max_ts_codes is None or not isinstance(result, pd.DataFrame):
            return result
        if "ts_code" not in result.columns or result.empty:
            return result
        codes = sorted(result["ts_code"].astype(str).unique())[: self._max_ts_codes]
        return result.loc[result["ts_code"].astype(str).isin(codes)].copy()


def compare_replay_frames(source: pd.DataFrame, replay: pd.DataFrame) -> dict[str, object]:
    """Compare finite source observations with a calculator replay."""

    required_source = {"trade_date", "ts_code", "value"}
    required_replay = {"trade_date", "ts_code", "replay_value"}
    missing_source = sorted(required_source.difference(source.columns))
    missing_replay = sorted(required_replay.difference(replay.columns))
    if missing_source or missing_replay:
        raise ValueError(
            f"replay frames are missing columns: source={missing_source}, replay={missing_replay}"
        )
    left = source.loc[:, ["trade_date", "ts_code", "value"]].copy()
    right = replay.loc[:, ["trade_date", "ts_code", "replay_value"]].copy()
    for frame in (left, right):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
        frame["ts_code"] = frame["ts_code"].astype(str)
    key_columns = ["trade_date", "ts_code"]
    if left.duplicated(key_columns).any() or right.duplicated(key_columns).any():
        raise ValueError("formula replay frames contain duplicate business keys")
    merged = left.merge(right, on=key_columns, how="inner")
    source_values = pd.to_numeric(merged["value"], errors="coerce").to_numpy(float)
    replay_values = pd.to_numeric(merged["replay_value"], errors="coerce").to_numpy(float)
    finite = np.isfinite(source_values) & np.isfinite(replay_values)
    abs_diff = np.abs(source_values - replay_values)
    rel_diff = abs_diff / np.maximum(np.abs(source_values), 1e-12)
    mismatch = finite & (abs_diff > 1e-8) & (rel_diff > 1e-6)
    mismatch_dates = merged.loc[mismatch, "trade_date"]
    reason_codes = []
    if not finite.any():
        reason_codes.append("NO_FINITE_OVERLAP")
    if mismatch.any():
        reason_codes.append("FINITE_OVERLAP_MISMATCH")
    return {
        "source_rows": len(left),
        "replay_rows": len(right),
        "source_finite_rows": int(np.isfinite(left["value"].to_numpy(float)).sum()),
        "replay_finite_rows": int(
            np.isfinite(right["replay_value"].to_numpy(float)).sum()
        ),
        "overlap_rows": len(merged),
        "overlap_finite_rows": int(finite.sum()),
        "mismatch_rows": int(mismatch.sum()),
        "mismatch_date_count": int(mismatch_dates.nunique()),
        "mismatch_start": (
            mismatch_dates.min().date().isoformat() if not mismatch_dates.empty else None
        ),
        "mismatch_end": (
            mismatch_dates.max().date().isoformat() if not mismatch_dates.empty else None
        ),
        "max_abs_diff": float(abs_diff[finite].max()) if finite.any() else 0.0,
        "max_rel_diff": float(rel_diff[finite].max()) if finite.any() else 0.0,
        "decision_status": (
            "FORMULA_REPLAY_PASS"
            if finite.any() and not mismatch.any()
            else "FORMULA_REPLAY_BLOCKED"
        ),
        "reason_codes": reason_codes,
    }


def _query_source(
    client: object, definition: ReplayDefinition, start: date, end: date
) -> pd.DataFrame:
    result = client.run(
        "select trade_date,ts_code,value from "
        f"loadTable('dfs://YesLab','{definition.store_factor}') "
        f"where factor_name == '{definition.factor_name}' "
        f"and trade_date between date('{start:%Y.%m.%d}') "
        f"and date('{end:%Y.%m.%d}')"
    )
    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("stored factor replay query did not return a DataFrame")
    frame = result.loc[:, ["trade_date", "ts_code", "value"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.drop_duplicates(["trade_date", "ts_code"], keep="last").reset_index(drop=True)


def _replay_factor(
    client: object,
    definition: ReplayDefinition,
    start: date,
    end: date,
    *,
    industry_provider: object | None = None,
) -> pd.DataFrame:
    if str(CALCULATOR_ROOT) not in sys.path:
        sys.path.insert(0, str(CALCULATOR_ROOT))
    module = importlib.import_module(f"factor_calculators.{definition.calculator_module}")
    calculator_class = getattr(module, definition.calculator_class)
    calculator = calculator_class(client, definition.effective_replay_lookback_days)
    if industry_provider is not None:
        calculator.industry_provider = industry_provider
    wide = calculator.calculate(start.strftime("%Y.%m.%d"), end.strftime("%Y.%m.%d"))
    return (
        wide.rename_axis("trade_date")
        .reset_index()
        .melt(id_vars=["trade_date"], var_name="ts_code", value_name="replay_value")
    )


def run_audit(args: argparse.Namespace, *, client: object) -> dict[str, object]:
    bounded_client = _BoundedClient(client, args.max_ts_codes)
    rows: list[dict[str, object]] = []
    for factor_id in args.factor_id:
        definition = REPLAY_DEFINITIONS[factor_id]
        base_row = {
            "factor_id": factor_id,
            "factor_name": definition.factor_name,
            "calculator": f"{definition.calculator_module}.{definition.calculator_class}",
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "lookback_days": definition.lookback_days,
            "replay_lookback_days": definition.effective_replay_lookback_days,
            "max_ts_codes": args.max_ts_codes,
        }
        try:
            comparison = compare_replay_frames(
                _query_source(bounded_client, definition, args.start, args.end),
                _replay_factor(bounded_client, definition, args.start, args.end),
            )
            rows.append({**base_row, **comparison})
        except Exception as exc:
            rows.append(
                {
                    **base_row,
                    "decision_status": "FORMULA_REPLAY_ERROR",
                    "reason_codes": ["CALCULATOR_REPLAY_ERROR"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "scope": "FORMULA_REPLAY_CONSISTENCY_AUDIT_V1",
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "factor_ids": list(args.factor_id),
        "database_writes": False,
        "locked_test_read": False,
        "rows": rows,
    }


def _connect() -> object:
    import dolphindb as ddb  # type: ignore[import-untyped]

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    return client


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    payload = run_audit(args, client=_connect())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
