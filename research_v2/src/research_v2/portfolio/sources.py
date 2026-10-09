"""Score and index feeds for the portfolio policy layer.

All I/O lives here so that `policy.py` and `engine.py` stay pure and unit-testable. Two
score feeds are defined by an identical shape, which is the point of the design:

  * `FrozenArtifactSource` reads the signed TEST prediction artifacts. For 2022-2025 every
    trading day was already scored by the run that produced the gates, so "what does the
    model say today" is a lookup, not a re-inference.
  * `PredictionDailySource` reads the `prediction_daily` batches that
    `research_v2.inference` publishes. That is the feed for any date the artifacts do not
    cover -- in practice 2026 onward.

Both hand the engine ``{member: {signal_date: {ts_code: score}}}``, so the policy layer
does not know or care which one produced the numbers.

`prediction_daily` carries ``raw_score``/``score``/``rank``/``percentile_rank`` per
``(signal_date, model_id, model_version, ts_code)``, and the frozen artifacts carry
``(signal_date, ts_code, score)`` -- the same business shape, which is why one engine can
serve both.

Read-only. Nothing here writes to any database.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

import pandas as pd

DEFAULT_ARTIFACTS = Path("/data/zhangyudong/qrv2_research_artifacts")

#: model alias -> artifact workspace directory name
WORKSPACES = {"M0R": "m0r_v2", "M1": "m1_full_v1", "M2": "m2_v1", "M0-refined": "m0_v3"}

FOLD_YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}
YEAR_FOLD = {year: fold for fold, year in FOLD_YEAR.items()}

ScoreMap = dict[date, dict[str, float]]


class ScoreSource(Protocol):
    """Anything that can hand the engine per-member scores keyed by signal date."""

    def scores(self, member: str) -> ScoreMap: ...

    def folds_by_date(self) -> dict[date, str]: ...

    def returns(self) -> dict[date, dict[str, float]] | None: ...


@dataclass(frozen=True)
class ArtifactProvenance:
    """Which signed artifact each member's scores came from, so output is auditable."""

    member: str
    fold: str
    workspace: str
    model_id: str
    prediction_id: str
    dataset_id: str
    rows: int


@dataclass
class FrozenArtifactSource:
    """Scores from the signed frozen TEST prediction artifacts (2022-2025)."""

    artifacts: Path = DEFAULT_ARTIFACTS
    with_labels: bool = True

    _scores: dict[str, ScoreMap] | None = None
    _folds: dict[date, str] | None = None
    _returns: dict[date, dict[str, float]] | None = None
    _provenance: list[ArtifactProvenance] | None = None

    def load(self, members: Sequence[str]) -> None:
        unknown = [m for m in members if m not in WORKSPACES]
        if unknown:
            raise ValueError(f"unknown model aliases {unknown}; choose from {sorted(WORKSPACES)}")
        scores: dict[str, ScoreMap] = {}
        folds: dict[date, str] = {}
        returns: dict[date, dict[str, float]] = {}
        provenance: list[ArtifactProvenance] = []
        for member in members:
            workspace = self.artifacts / WORKSPACES[member]
            if not workspace.is_dir():
                raise FileNotFoundError(f"no artifact workspace for {member}: {workspace}")
            per_member: ScoreMap = {}
            for manifest_path in sorted((workspace / "predictions").glob("*/manifest.json")):
                manifest = json.loads(manifest_path.read_text())
                meta = manifest["metadata"]
                if meta.get("kind") != "candidate":
                    continue
                frame = pd.read_parquet(manifest_path.parent / "scores.parquet")
                frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
                fold = str(meta["fold_id"])
                for day, group in frame.groupby("signal_date"):
                    per_member.setdefault(day, {}).update(
                        dict(zip(group["ts_code"].astype(str), group["score"].astype(float),
                             strict=True))
                    )
                    folds.setdefault(day, fold)
                provenance.append(
                    ArtifactProvenance(
                        member=member, fold=fold, workspace=WORKSPACES[member],
                        model_id=str(meta.get("model_id", "")),
                        prediction_id=str(manifest.get("prediction_id", "")),
                        dataset_id=str(meta.get("dataset_id", "")), rows=int(len(frame)),
                    )
                )
                if self.with_labels:
                    returns.update(_load_labels(workspace, str(meta["dataset_id"])))
            if not per_member:
                raise ValueError(
                    f"{member} has no candidate prediction artifacts under {workspace}")
            scores[member] = per_member
        self._scores, self._folds, self._returns = scores, folds, (returns or None)
        self._provenance = provenance

    @property
    def provenance(self) -> list[ArtifactProvenance]:
        return list(self._provenance or [])

    def scores(self, member: str) -> ScoreMap:
        self._require(member)
        return (self._scores or {})[member]

    def folds_by_date(self) -> dict[date, str]:
        return dict(self._folds or {})

    def returns(self) -> dict[date, dict[str, float]] | None:
        return dict(self._returns) if self._returns else None

    def universe(self) -> dict[date, frozenset[str]] | None:
        """The label-valid names per date -- the backtest's own eligibility rule.

        A backtest can only rank names whose realised forward return exists, so this is the
        measurement's eligibility rule and not a look-ahead filter. A live date has no
        labels and therefore no restriction; the CLI passes None in that case.
        """
        if not self._returns:
            return None
        return {day: frozenset(codes) for day, codes in self._returns.items()}

    def signal_dates(self, members: Sequence[str]) -> list[date]:
        """Dates every requested member scored -- the ensemble's own coverage."""
        self._require(members[0] if isinstance(members, Sequence) else str(members))
        sets = [set((self._scores or {}).get(m, {})) for m in members]
        return sorted(set.intersection(*sets)) if sets else []

    def _require(self, member: str) -> None:
        if self._scores is None:
            raise RuntimeError("FrozenArtifactSource.load() must be called before use")


def _load_labels(workspace: Path, dataset_id: str) -> dict[date, dict[str, float]]:
    """signal_date -> ts_code -> realised raw_return_5d, from the frozen TEST split."""
    root = workspace / "datasets" / dataset_id / "split=test"
    shards = sorted(root.rglob("*.parquet"))
    out: dict[date, dict[str, float]] = {}
    for shard in shards:
        frame = pd.read_parquet(shard, columns=["signal_date", "ts_code", "raw_return_5d",
                                                "label_valid"])
        frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
        frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
        frame = frame.loc[frame["label_valid"]]
        frame["raw_return_5d"] = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
        frame = frame.loc[frame["raw_return_5d"].notna()]
        for day, group in frame.groupby("signal_date"):
            out.setdefault(day, {}).update(
                dict(zip(group["ts_code"].astype(str), group["raw_return_5d"].astype(float),
                         strict=True))
            )
    return out


def _same_columns(path: Path, frame: pd.DataFrame) -> bool:
    try:
        return list(pd.read_parquet(path).columns) == list(frame.columns)
    except Exception:  # noqa: BLE001 - an unreadable cache is simply not a cache
        return False


def fold_for_date(day: date) -> str | None:
    """The TEST fold a date belongs to, or None when it is outside 2022-2025."""
    return YEAR_FOLD.get(day.year)


def load_index_closes(
    index_code: str,
    *,
    start: date,
    end: date,
    cache: Path | None = None,
) -> list[tuple[date, float]]:
    """Oldest-first ``(trade_date, close)`` for one index, read-only.

    ``start`` should precede the first signal date by at least the MA window so the overlay
    is warm on day one; a cold start degrades to fully invested rather than to cash.
    """
    if cache is not None and cache.exists():
        frame = pd.read_parquet(cache)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        cached = frame.loc[frame["ts_code"] == index_code]
        # A cache that merely overlaps the request is worse than no cache: it silently
        # truncates the moving-average history, and a truncated history makes the overlay
        # return "fully invested" for every date it cannot see. Only accept a cache that
        # actually spans [start, end].
        spans = (
            not cached.empty
            and cached["trade_date"].min() <= start
            and cached["trade_date"].max() >= end
        )
        if spans:
            window = cached.loc[(cached["trade_date"] >= start) & (cached["trade_date"] <= end)]
            window = window.dropna(subset=["close"]).sort_values("trade_date")
            return [(d, float(c)) for d, c in
                    zip(window["trade_date"], window["close"], strict=True)]

    import sys

    sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/scripts")
    from walk_forward_common import new_holder, query  # noqa: PLC0415

    holder = new_holder()
    lo = start.strftime("%Y.%m.%d")
    hi = end.strftime("%Y.%m.%d")
    _, frame = query(
        holder,
        "select ts_code, trade_date, close from "
        "loadTable('dfs://YesLab','index_daily_data') "
        f"where ts_code = '{index_code}' and trade_date >= {lo} and trade_date <= {hi}",
    )
    if frame is None or frame.empty:
        raise ValueError(
            f"no index closes for {index_code} between {start} and {end}; the exposure "
            f"overlay cannot be evaluated without them"
        )
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"]).sort_values("trade_date")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        existing = (pd.read_parquet(cache)
                    if cache.exists() and _same_columns(cache, frame) else None)
        merged = (pd.concat([existing, frame], ignore_index=True)
                  .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
                  .sort_values(["ts_code", "trade_date"]) if existing is not None else frame)
        merged.to_parquet(cache, index=False)
    return [(d, float(c)) for d, c in zip(frame["trade_date"], frame["close"], strict=True)]
