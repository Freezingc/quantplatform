import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from agent_gateway.dsl import ExpressionEngine
from agent_gateway.field_registry import FieldRegistry
from agent_gateway.ingest import FactorSourceIngestor
from agent_gateway.knowledge_base import KnowledgeBase
from agent_gateway.promote import CalculatorPromoter
from agent_gateway.validation import evaluate_factor_summary, placebo_test, rolling_validation, score_factor_summary


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.registry = FieldRegistry()
        dates = pd.date_range("2024-01-01", periods=6)
        columns = ["000001.SZ", "600000.SH"]
        self.close = pd.DataFrame(
            [[10, 20], [11, 19], [12, 18], [11, 17], [13, 18], [14, 19]],
            index=dates,
            columns=columns,
            dtype=float,
        )
        self.vol = pd.DataFrame(
            [[100, 120], [110, 100], [130, 90], [125, 95], [150, 80], [160, 85]],
            index=dates,
            columns=columns,
            dtype=float,
        )

    def test_validate_rejects_unknown_fields(self):
        result = ExpressionEngine(self.registry).validate("rank(fake_field)")
        self.assertFalse(result.ok)
        self.assertIn("fake_field", result.missing_fields)

    def test_evaluate_expression_to_wide(self):
        engine = ExpressionEngine(self.registry)
        result = engine.evaluate_wide(
            "rank(ts_delta(close, 2) / ts_shift(close, 2))",
            {"close": self.close},
        )
        self.assertEqual(result.shape, self.close.shape)
        self.assertTrue(result.iloc[-1].notna().all())

    def test_ingest_generates_registry_valid_candidate(self):
        ingestor = FactorSourceIngestor(self.registry)
        payload = ingestor.ingest_factor_source("5 day price reversal with volume confirmation", "text")
        self.assertGreaterEqual(len(payload["candidates"]), 1)
        valid = [candidate for candidate in payload["candidates"] if candidate["expression"]]
        self.assertGreaterEqual(len(valid), 1)

    def test_registry_reads_daily_task_fields(self):
        with TemporaryDirectory() as tmp:
            script = Path(tmp) / "update_stock_factor_daily.py"
            script.write_text(
                """
import dolphindb as ddb
appender = ddb.TableAppender(dbPath='dfs://YesLab', tableName="stock_factor_daily", ddbSession=s)
df = pro.stk_factor_pro(fields=[
    "ts_code", "trade_date", "turnover_rate_f", "volume_ratio", "ma_hfq_20"
])
""",
                encoding="utf-8",
            )
            registry = FieldRegistry()
            registry.enrich_from_daily_tasks([tmp])
            self.assertEqual(registry.get("turnover_rate_f").table, "stock_factor_daily")
            self.assertEqual(registry.get("ma_hfq_20").category, "market")

    def test_promote_generates_calculator_draft(self):
        promoter = CalculatorPromoter(self.registry)
        result = promoter.promote_to_calculator(
            "ReportReversal",
            "rank(-1 * ts_delta(close, 5) / ts_shift(close, 5))",
            write=False,
        )
        self.assertTrue(result["ok"])
        self.assertIn("class ReportReversalCalculator", result["content"])
        self.assertIn("'close'", result["content"])

    def test_score_and_validation_helpers(self):
        score = score_factor_summary({"ic_mean": 0.04, "ic_ir": 0.8, "annual_return": 0.12, "turnover": 0.2})
        self.assertGreater(score["score"], 0)
        self.assertIn("recommendation", score)
        self.assertIn("recommendation_code", score)
        self.assertIn("quality", score["components"])

        rows = []
        for date in pd.date_range("2023-01-01", periods=4):
            rows.append({"date": date, "instrument": "000001.SZ", "factor": 1.0, "forward_return": 0.02})
            rows.append({"date": date, "instrument": "600000.SH", "factor": 0.0, "forward_return": -0.01})
        df = pd.DataFrame(rows)
        rolling = rolling_validation(df)
        self.assertTrue(rolling["passed"])
        placebo = placebo_test(df)
        self.assertIn("passed", placebo)

    def test_unified_evaluation_recommendations(self):
        strong = evaluate_factor_summary(
            {
                "ic_mean": 0.05,
                "ic_ir": 1.1,
                "annual_return": 0.18,
                "sharpe": 1.4,
                "max_drawdown": 0.12,
                "win_rate": 0.58,
                "turnover": 0.25,
            },
            validation={"rolling_validation": {"passed": True}, "placebo": {"passed": True}},
        )
        self.assertIn(strong["recommendation"], {"\u9002\u5408\u5165\u6c60", "\u9002\u5408\u7ee7\u7eed\u4f18\u5316"})
        self.assertIn(strong["recommendation_code"], {"enter_pool", "continue_optimizing"})
        self.assertGreaterEqual(strong["components"]["out_of_sample"], 90)

        weak = evaluate_factor_summary(
            {
                "ic_mean": 0.003,
                "ic_ir": 0.1,
                "annual_return": -0.02,
                "sharpe": -0.2,
                "max_drawdown": 0.45,
                "turnover": 1.2,
            },
            validation={"rolling_validation": {"passed": False}},
        )
        self.assertEqual(weak["recommendation"], "\u4e0d\u5efa\u8bae\u7ee7\u7eed")
        self.assertEqual(weak["recommendation_code"], "stop")
        self.assertIn("high_turnover", weak["risk_flags"])

    def test_knowledge_base_distills_assets(self):
        with TemporaryDirectory() as tmp:
            kb = KnowledgeBase(tmp)
            evaluation = evaluate_factor_summary(
                {
                    "ic_mean": 0.05,
                    "ic_ir": 1.1,
                    "annual_return": 0.18,
                    "sharpe": 1.4,
                    "max_drawdown": 0.12,
                    "turnover": 0.25,
                },
                validation={"rolling_validation": {"passed": True}},
            )
            asset = kb.distill_from_event(
                "factor_eval",
                {"factor_name": "TestFactor", "expression": "rank(close)", "evaluation": evaluation},
            )
            self.assertIn(asset["asset_type"], {"rules", "findings"})
            self.assertTrue(Path(asset["path"]).exists())

            duplicate = kb.distill_from_event(
                "similarity_check",
                {"factor_name": "NearDuplicate", "max_similarity": 0.92, "threshold": 0.8},
            )
            self.assertEqual(duplicate["asset_type"], "failures")
            self.assertIn("high_similarity", duplicate["risk_flags"])
            self.assertGreaterEqual(len(kb.search("similarity")), 1)


if __name__ == "__main__":
    unittest.main()
