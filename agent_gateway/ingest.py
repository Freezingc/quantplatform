from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from .dsl import ExpressionEngine
from .field_registry import FieldRegistry
from .models import FactorCandidate


TERM_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("momentum", ("momentum", "动量", "趋势", "涨幅", "收益")),
    ("reversal", ("reversal", "反转", "回撤", "超跌", "均值回归")),
    ("volume", ("volume", "成交量", "放量", "缩量", "换手")),
    ("valuation", ("valuation", "估值", "市盈率", "市净率", "PE", "PB")),
    ("quality", ("quality", "质量", "盈利", "ROE", "ROA", "毛利率")),
    ("volatility", ("volatility", "波动", "标准差", "风险")),
)


class FactorSourceIngestor:
    """Extract factor candidates from URL, PDF/text file, or plain text.

    This is a deterministic local extractor. An external LLM can be layered on
    top later, but this class already enforces the field registry and produces
    auditable candidates without inventing fields.
    """

    def __init__(self, registry: FieldRegistry | None = None):
        self.registry = registry or FieldRegistry.from_project()
        self.engine = ExpressionEngine(self.registry)

    def ingest_factor_source(self, source: str, source_type: str = "auto", max_candidates: int = 6) -> dict:
        text, detected_type = self._read_source(source, source_type)
        extracted_terms = self.extract_terms(text)
        mapped, missing = self.registry.map_terms(extracted_terms)
        candidates = self.generate_candidates(text, mapped, missing, detected_type, max_candidates)
        return {
            "source_type": detected_type,
            "text_preview": text[:1200],
            "extracted_terms": extracted_terms,
            "mapped_fields": mapped,
            "missing_fields": missing,
            "candidates": [candidate.to_dict() for candidate in candidates],
        }

    def extract_terms(self, text: str) -> list[str]:
        lowered = text.lower()
        terms: set[str] = set()
        for spec in self.registry.list_fields():
            tokens = [spec.name, *spec.aliases, spec.description]
            for token in tokens:
                if token and (token.lower() in lowered or token in text):
                    terms.add(spec.name)

        for phrase in _window_phrases(text):
            terms.add(phrase)
        return sorted(terms)

    def generate_factor_expressions(self, description: str, max_candidates: int = 6) -> list[FactorCandidate]:
        extracted_terms = self.extract_terms(description)
        mapped, missing = self.registry.map_terms(extracted_terms)
        return self.generate_candidates(description, mapped, missing, "text", max_candidates)

    def generate_candidates(
        self,
        text: str,
        mapped_fields: dict[str, str],
        missing_fields: list[str],
        source_type: str,
        max_candidates: int,
    ) -> list[FactorCandidate]:
        signals = _detect_signals(text)
        fields = set(mapped_fields.values())
        candidates: list[FactorCandidate] = []

        def add(name: str, hypothesis: str, expression: str, required: list[str], confidence: float):
            validation = self.engine.validate(expression)
            expr_missing = validation.missing_fields + [field for field in required if self.registry.resolve_name(field) is None]
            candidates.append(FactorCandidate(
                factor_name=name,
                hypothesis=hypothesis,
                required_fields=sorted(set(required)),
                expression=expression if validation.ok else None,
                notes="; ".join(validation.errors) if not validation.ok else "Generated from source text and field registry.",
                source=source_type,
                missing_fields=sorted(set(missing_fields + expr_missing)),
                confidence=confidence if validation.ok else min(confidence, 0.25),
            ))

        if "momentum" in signals or {"close", "pct_chg"} & fields:
            add(
                "dsl_momentum_20d",
                "Stocks with stronger recent price momentum may outperform.",
                "rank(ts_delta(close, 20) / ts_shift(close, 20))",
                ["close"],
                0.72,
            )
            add(
                "dsl_decay_momentum_10d",
                "Recent returns should receive higher weight than older returns.",
                "rank(decay_linear(ts_delta(close, 5) / ts_shift(close, 5), 10))",
                ["close"],
                0.68,
            )

        if "reversal" in signals:
            add(
                "dsl_short_reversal_5d",
                "Short-term losers may mean-revert over the next holding period.",
                "rank(-1 * ts_delta(close, 5) / ts_shift(close, 5))",
                ["close"],
                0.74,
            )

        if "volume" in signals or {"vol", "turnover_rate"} & fields:
            add(
                "dsl_volume_price_confirmation",
                "Price moves confirmed by abnormal volume may carry stronger signal.",
                "rank(ts_delta(close, 5) / ts_shift(close, 5) * vol / ts_mean(vol, 20))",
                ["close", "vol"],
                0.7,
            )

        if "valuation" in signals or {"pe", "pb", "ps"} & fields:
            if "pb" in fields or self.registry.resolve_name("pb"):
                add("dsl_low_pb_value", "Lower price-to-book stocks may have value premium.", "rank(-1 * pb)", ["pb"], 0.66)
            if "pe" in fields or self.registry.resolve_name("pe"):
                add("dsl_low_pe_value", "Lower price-to-earnings stocks may have value premium.", "rank(-1 * pe)", ["pe"], 0.64)

        if "quality" in signals or {"roe", "roa"} & fields:
            if self.registry.resolve_name("roe"):
                add("dsl_quality_roe", "Higher profitability may indicate higher quality.", "rank(roe)", ["roe"], 0.66)
            if self.registry.resolve_name("roa"):
                add("dsl_quality_roa", "Higher asset profitability may indicate higher quality.", "rank(roa)", ["roa"], 0.63)

        if "volatility" in signals:
            add(
                "dsl_low_volatility",
                "Lower recent volatility may indicate more stable risk-adjusted returns.",
                "rank(-1 * ts_std(pct_chg, 20))",
                ["pct_chg"],
                0.62,
            )

        if not candidates:
            add(
                "dsl_price_mean_reversion",
                "Fallback candidate: price deviation from recent mean may revert.",
                "rank(-1 * (close / ts_mean(close, 20) - 1))",
                ["close"],
                0.45,
            )

        return candidates[:max_candidates]

    def _read_source(self, source: str, source_type: str) -> tuple[str, str]:
        detected = source_type
        if source_type == "auto":
            if re.match(r"^https?://", source):
                detected = "url"
            elif Path(source).exists():
                suffix = Path(source).suffix.lower()
                detected = "pdf" if suffix == ".pdf" else "file"
            else:
                detected = "text"

        if detected == "url":
            return _read_url(source), detected
        if detected == "pdf":
            return _read_pdf(Path(source)), detected
        if detected == "file":
            return Path(source).read_text(encoding="utf-8", errors="ignore"), detected
        return source, "text"


def _read_url(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "QuantPlatformAgentGateway/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read()
    text = raw.decode("utf-8", errors="ignore")
    text = re.sub(r"(?is)<(script|style|nav|footer).*?</\1>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_pdf(path: Path) -> str:
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise RuntimeError("PDF extraction requires pypdf or a readable text layer") from exc


def _detect_signals(text: str) -> set[str]:
    signals = set()
    lowered = text.lower()
    for name, patterns in TERM_PATTERNS:
        if any(pattern.lower() in lowered or pattern in text for pattern in patterns):
            signals.add(name)
    return signals


def _window_phrases(text: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"\b\d{1,3}\s*(?:day|days|d|日|天|月)\b", text, flags=re.I)]
