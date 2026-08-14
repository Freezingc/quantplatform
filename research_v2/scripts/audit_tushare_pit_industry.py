"""Audit Tushare SW2021 level-1 point-in-time industry membership.

The command is read-only.  It compares historical Tushare membership intervals
with a YesLab security universe and the current static ``stock_info_table``
labels; it does not use static labels as a fallback for missing PIT members.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd

LOCKED_TEST_START = date(2026, 1, 1)
SHSZ_SUFFIXES = (".SH", ".SZ")
CLASSIFICATION_COLUMNS = ("index_code", "industry_name", "level")
MEMBER_COLUMNS = ("index_code", "con_code", "in_date", "out_date")


def parse_date(value: str) -> date:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("date must use YYYY-MM-DD") from exc
    if parsed >= LOCKED_TEST_START:
        raise ValueError("2026 locked-test data is not allowed")
    return parsed


def _require_columns(frame: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def normalize_classifications(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, CLASSIFICATION_COLUMNS, "classification frame")
    result = frame.loc[:, list(CLASSIFICATION_COLUMNS)].copy()
    result["index_code"] = result["index_code"].astype(str)
    result["industry_name"] = result["industry_name"].astype(str)
    result["level"] = result["level"].astype(str)
    result = result.loc[result["level"].eq("L1")]
    result = result.loc[result["industry_name"].str.len().gt(0)]
    return result.drop_duplicates("index_code", keep="last").reset_index(drop=True)


def normalize_members(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, MEMBER_COLUMNS, "membership frame")
    result = frame.loc[:, list(MEMBER_COLUMNS)].copy()
    result["index_code"] = result["index_code"].astype(str)
    result["con_code"] = result["con_code"].astype(str)
    for column in ("in_date", "out_date"):
        values = result[column]
        if pd.api.types.is_datetime64_any_dtype(values):
            result[column] = pd.to_datetime(values, errors="coerce")
        else:
            result[column] = pd.to_datetime(
                values.astype("string"), format="%Y%m%d", errors="coerce"
            )
    result = result.dropna(subset=["in_date", "con_code", "index_code"])
    result = result.loc[
        result["con_code"].str.endswith(SHSZ_SUFFIXES)
    ].copy()
    return result.reset_index(drop=True)


def build_pit_snapshot(
    members: pd.DataFrame,
    classifications: pd.DataFrame,
    *,
    as_of: date,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build one PIT snapshot; ``out_date`` is an exclusive boundary."""

    if as_of >= LOCKED_TEST_START:
        raise ValueError("2026 locked-test data is not allowed")
    classification_frame = normalize_classifications(classifications)
    member_frame = normalize_members(members)
    active = member_frame.loc[
        (member_frame["in_date"] <= pd.Timestamp(as_of))
        & (
            member_frame["out_date"].isna()
            | (pd.Timestamp(as_of) < member_frame["out_date"])
        )
    ].merge(classification_frame, on="index_code", how="inner")
    if codes is not None:
        code_set = set(map(str, codes))
        active = active.loc[active["con_code"].isin(code_set)]

    conflict_counts = active.groupby("con_code")["index_code"].nunique()
    conflict_code_values = tuple(
        sorted(conflict_counts.loc[conflict_counts > 1].index.astype(str))
    )
    conflict_codes = len(conflict_code_values)
    active = active.sort_values(
        ["con_code", "in_date", "index_code"], kind="stable"
    ).drop_duplicates("con_code", keep="last")
    snapshot = active.loc[
        :, ["con_code", "index_code", "industry_name", "in_date", "out_date"]
    ].rename(columns={"con_code": "ts_code", "index_code": "industry_code"})
    snapshot = snapshot.sort_values("ts_code").reset_index(drop=True)
    snapshot.attrs["overlap_conflict_code_count"] = conflict_codes
    snapshot.attrs["overlap_conflict_codes"] = conflict_code_values
    snapshot.attrs["as_of"] = pd.Timestamp(as_of).date().isoformat()
    return snapshot


class PITIndustryProvider:
    """Serve date-aware industry labels from Tushare membership intervals."""

    def __init__(
        self,
        members: pd.DataFrame,
        classifications: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> None:
        self.members = normalize_members(members)
        self.classifications = normalize_classifications(classifications)
        self.strict = bool(strict)
        self._snapshot_cache: dict[tuple[date, tuple[str, ...]], pd.DataFrame] = {}

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path,
        *,
        source: str = "SW2021",
        strict: bool = True,
    ) -> PITIndustryProvider:
        source_key = str(source).strip().lower()
        if not source_key:
            raise ValueError("industry source must be non-empty")
        classification_path = cache_dir / (
            f"pit_industry_{source_key}_l1_classification.json"
        )
        if not classification_path.is_file():
            raise FileNotFoundError(
                f"missing PIT industry classification cache: {classification_path}"
            )
        classifications = pd.read_json(classification_path, orient="table")
        normalized = normalize_classifications(classifications)
        member_frames: list[pd.DataFrame] = []
        missing: list[str] = []
        for index_code in normalized["index_code"]:
            member_path = cache_dir / (
                f"pit_industry_{source_key}_members_{str(index_code).replace('.', '_')}.json"
            )
            if source_key == "sw2021" and not member_path.is_file():
                member_path = cache_dir / (
                    f"pit_industry_members_{str(index_code).replace('.', '_')}.json"
                )
            if not member_path.is_file():
                missing.append(str(member_path))
                continue
            member_frames.append(pd.read_json(member_path, orient="table"))
        if missing:
            raise FileNotFoundError(f"missing PIT industry membership caches: {missing[:3]}")
        members = (
            pd.concat(member_frames, ignore_index=True)
            if member_frames
            else pd.DataFrame(columns=list(MEMBER_COLUMNS))
        )
        return cls(members, classifications, strict=strict)

    def snapshot(self, as_of: date, codes: Sequence[str]) -> pd.DataFrame:
        as_of_value = parse_date(pd.Timestamp(as_of).date().isoformat())
        code_values = tuple(dict.fromkeys(map(str, codes)))
        cache_key = (as_of_value, tuple(sorted(code_values)))
        if cache_key not in self._snapshot_cache:
            snapshot = build_pit_snapshot(
                self.members,
                self.classifications,
                as_of=as_of_value,
                codes=code_values,
            )
            conflict_count = int(snapshot.attrs.get("overlap_conflict_code_count", 0))
            if self.strict and conflict_count:
                raise ValueError(
                    f"PIT industry membership overlap on {as_of_value}: "
                    f"{conflict_count} code(s)"
                )
            self._snapshot_cache[cache_key] = snapshot
        return self._snapshot_cache[cache_key].copy(deep=True)

    def get_industry_labels(
        self, dates: Sequence[date | pd.Timestamp], codes: Sequence[str]
    ) -> pd.DataFrame:
        date_values = tuple(pd.Timestamp(value).normalize() for value in dates)
        if len(set(date_values)) != len(date_values):
            raise ValueError("industry label dates must be unique")
        code_values = tuple(dict.fromkeys(map(str, codes)))
        labels = pd.DataFrame(index=date_values, columns=code_values, dtype=object)
        for timestamp in date_values:
            snapshot = self.snapshot(timestamp.date(), code_values)
            if snapshot.empty:
                continue
            mapping = snapshot.set_index("ts_code")["industry_name"]
            common = [code for code in code_values if code in mapping.index]
            if common:
                labels.loc[timestamp, common] = mapping.loc[common].to_numpy(dtype=object)
        labels.index.name = "trade_date"
        return labels

    def coverage_report(
        self, dates: Sequence[date | pd.Timestamp], codes: Sequence[str]
    ) -> dict[str, object]:
        labels = self.get_industry_labels(dates, codes)
        mapped = labels.notna()
        code_values = tuple(dict.fromkeys(map(str, codes)))
        conflict_codes: set[str] = set()
        for timestamp in labels.index:
            snapshot = self.snapshot(timestamp.date(), code_values)
            conflict_codes.update(
                map(str, snapshot.attrs.get("overlap_conflict_codes", ()))
            )
        unmapped_codes = sorted(
            code for code in code_values if not bool(mapped.loc[:, code].any())
        )
        requested_rows = int(mapped.size)
        mapped_rows = int(mapped.to_numpy(dtype=bool).sum())
        return {
            "requested_rows": requested_rows,
            "mapped_rows": mapped_rows,
            "coverage_ratio": mapped_rows / requested_rows if requested_rows else 0.0,
            "conflict_code_count": len(conflict_codes),
            "unmapped_codes": unmapped_codes,
        }


class VersionedPITIndustryProvider:
    """Switch historical industry source at an explicit classification cutover."""

    def __init__(
        self,
        before: PITIndustryProvider,
        after: PITIndustryProvider,
        *,
        cutover: date,
    ) -> None:
        self.before = before
        self.after = after
        self.cutover = parse_date(cutover.isoformat())

    @classmethod
    def from_cache(
        cls,
        cache_dir: Path,
        *,
        cutover: date,
        strict: bool = True,
    ) -> VersionedPITIndustryProvider:
        return cls(
            PITIndustryProvider.from_cache(cache_dir, source="SW2014", strict=strict),
            PITIndustryProvider.from_cache(cache_dir, source="SW2021", strict=strict),
            cutover=cutover,
        )

    def _provider_for(self, value: pd.Timestamp) -> PITIndustryProvider:
        return self.before if value.date() < self.cutover else self.after

    def get_industry_labels(
        self, dates: Sequence[date | pd.Timestamp], codes: Sequence[str]
    ) -> pd.DataFrame:
        date_values = tuple(pd.Timestamp(value).normalize() for value in dates)
        if len(set(date_values)) != len(date_values):
            raise ValueError("industry label dates must be unique")
        code_values = tuple(dict.fromkeys(map(str, codes)))
        labels = pd.DataFrame(index=date_values, columns=code_values, dtype=object)
        groups: dict[PITIndustryProvider, list[pd.Timestamp]] = {}
        for timestamp in date_values:
            groups.setdefault(self._provider_for(timestamp), []).append(timestamp)
        for provider, provider_dates in groups.items():
            labels.loc[provider_dates, code_values] = provider.get_industry_labels(
                provider_dates, code_values
            ).to_numpy(dtype=object)
        labels.index.name = "trade_date"
        return labels

    def coverage_report(
        self, dates: Sequence[date | pd.Timestamp], codes: Sequence[str]
    ) -> dict[str, object]:
        labels = self.get_industry_labels(dates, codes)
        mapped = labels.notna()
        code_values = tuple(dict.fromkeys(map(str, codes)))
        conflict_count = 0
        for provider, provider_dates in (
            (self.before, [value for value in labels.index if value.date() < self.cutover]),
            (self.after, [value for value in labels.index if value.date() >= self.cutover]),
        ):
            if provider_dates:
                conflict_count += int(
                    provider.coverage_report(provider_dates, code_values)[
                        "conflict_code_count"
                    ]
                )
        requested_rows = int(mapped.size)
        mapped_rows = int(mapped.to_numpy(dtype=bool).sum())
        return {
            "requested_rows": requested_rows,
            "mapped_rows": mapped_rows,
            "coverage_ratio": mapped_rows / requested_rows if requested_rows else 0.0,
            "conflict_code_count": conflict_count,
            "unmapped_codes": sorted(
                code for code in code_values if not bool(mapped.loc[:, code].any())
            ),
        }


def coverage_for_keys(provider: object, keys: pd.DataFrame) -> dict[str, object]:
    """Measure PIT coverage on observed date/security keys only."""

    date_column = "signal_date" if "signal_date" in keys.columns else "trade_date"
    required = {date_column, "ts_code"}
    if missing := sorted(required.difference(keys.columns)):
        raise ValueError(f"PIT coverage keys are missing columns: {missing}")
    working = keys.loc[:, [date_column, "ts_code"]].copy()
    working = working.rename(columns={date_column: "trade_date"})
    working["trade_date"] = pd.to_datetime(
        working["trade_date"], errors="raise"
    ).dt.normalize()
    working["ts_code"] = working["ts_code"].astype(str)
    working = working.drop_duplicates(["trade_date", "ts_code"])
    dates = tuple(working["trade_date"].drop_duplicates())
    codes = tuple(working["ts_code"].drop_duplicates())
    labels = provider.get_industry_labels(dates, codes)
    label_frame = (
        labels.rename_axis("trade_date", axis=0)
        .rename_axis("ts_code", axis=1)
        .reset_index()
        .melt(
            id_vars=["trade_date"],
            var_name="ts_code",
            value_name="industry_name",
        )
    )
    merged = working.merge(label_frame, on=["trade_date", "ts_code"], how="left")
    mapped = merged["industry_name"].notna()
    conflict_report = provider.coverage_report(dates, codes)
    unmapped = merged.loc[~mapped, ["trade_date", "ts_code"]]
    requested_rows = int(len(merged))
    mapped_rows = int(mapped.sum())
    return {
        "requested_rows": requested_rows,
        "mapped_rows": mapped_rows,
        "coverage_ratio": mapped_rows / requested_rows if requested_rows else 0.0,
        "unmapped_key_count": int(len(unmapped)),
        "unmapped_code_count": int(unmapped["ts_code"].nunique()),
        "unmapped_codes": sorted(unmapped["ts_code"].unique()),
        "conflict_code_count": int(conflict_report["conflict_code_count"]),
    }


def summarize_snapshot(
    snapshot: pd.DataFrame,
    universe: pd.DataFrame,
    static_labels: pd.DataFrame | None = None,
) -> dict[str, object]:
    _require_columns(snapshot, ("ts_code", "industry_name"), "PIT snapshot")
    _require_columns(universe, ("ts_code",), "universe frame")
    universe_codes = {
        str(code)
        for code in universe["ts_code"].dropna()
        if str(code).endswith(SHSZ_SUFFIXES)
    }
    pit_codes = {
        str(code)
        for code in snapshot["ts_code"].dropna()
        if str(code).endswith(SHSZ_SUFFIXES)
    }
    covered_codes = universe_codes.intersection(pit_codes)
    summary: dict[str, object] = {
        "universe_shsz_count": len(universe_codes),
        "pit_shsz_count": len(pit_codes),
        "pit_shsz_coverage_count": len(covered_codes),
        "pit_shsz_coverage_ratio": (
            len(covered_codes) / len(universe_codes) if universe_codes else 0.0
        ),
        "pit_unmapped_universe_count": len(universe_codes.difference(pit_codes)),
        "overlap_conflict_code_count": int(
            snapshot.attrs.get("overlap_conflict_code_count", 0)
        ),
    }
    if static_labels is None:
        return summary
    _require_columns(static_labels, ("ts_code", "industry"), "static label frame")
    static = static_labels.loc[:, ["ts_code", "industry"]].copy()
    static["ts_code"] = static["ts_code"].astype(str)
    static = static.drop_duplicates("ts_code", keep="last")
    comparable = snapshot.merge(static, on="ts_code", how="inner")
    comparable = comparable.loc[comparable["ts_code"].isin(universe_codes)]
    summary["static_label_overlap_count"] = int(len(comparable))
    summary["static_label_exact_name_count"] = int(
        comparable["industry"].eq(comparable["industry_name"]).sum()
    )
    summary["static_label_exact_name_share"] = (
        float(comparable["industry"].eq(comparable["industry_name"]).mean())
        if len(comparable)
        else 0.0
    )
    return summary


def _read_or_fetch(
    path: Path | None,
    fetch: object,
) -> pd.DataFrame:
    if path is not None and path.is_file():
        return pd.read_json(path, orient="table")
    result = fetch()
    if result is None:
        result = pd.DataFrame()
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Tushare response is not a DataFrame")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result.to_json(orient="table", date_format="iso"), encoding="utf-8")
    return result


def fetch_pit_members(
    pro: object,
    *,
    source: str = "SW2021",
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_key = str(source).strip().upper()
    if source_key not in {"SW2014", "SW2021"}:
        raise ValueError("industry source must be SW2014 or SW2021")
    source_file_key = source_key.lower()
    classification_path = (
        cache_dir / f"pit_industry_{source_file_key}_l1_classification.json"
        if cache_dir is not None
        else None
    )
    classifications = _read_or_fetch(
        classification_path,
        lambda: pro.index_classify(
            src=source_key,
            level="L1",
            fields="index_code,industry_name,level,parent_code,src",
        ),
    )
    normalized_classifications = normalize_classifications(classifications)
    member_frames: list[pd.DataFrame] = []
    for index_code in normalized_classifications["index_code"]:
        member_path = (
            cache_dir
            / f"pit_industry_{source_file_key}_members_{index_code.replace('.', '_')}.json"
            if cache_dir is not None
            else None
        )
        if (
            source_file_key == "sw2021"
            and cache_dir is not None
            and not member_path.is_file()
        ):
            legacy_member_path = cache_dir / (
                f"pit_industry_members_{index_code.replace('.', '_')}.json"
            )
            if legacy_member_path.is_file():
                member_path = legacy_member_path
        member_frames.append(
            _read_or_fetch(
                member_path,
                lambda index_code=index_code: pro.index_member(
                    index_code=index_code,
                    fields="index_code,con_code,in_date,out_date,is_new",
                ),
            )
        )
    members = (
        pd.concat(member_frames, ignore_index=True)
        if member_frames
        else pd.DataFrame(columns=list(MEMBER_COLUMNS))
    )
    return normalized_classifications, normalize_members(members)


def fetch_sw2021_l1_members(
    pro: object, *, cache_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible SW2021 cache loader."""

    return fetch_pit_members(pro, source="SW2021", cache_dir=cache_dir)


def run_audit(
    *,
    pro: object,
    universe: pd.DataFrame,
    static_labels: pd.DataFrame,
    as_of_dates: Sequence[date],
    cache_dir: Path | None = None,
    source: str = "SW2021",
) -> dict[str, object]:
    classifications, members = fetch_pit_members(
        pro, source=source, cache_dir=cache_dir
    )
    universe_codes = universe["ts_code"].astype(str).dropna().unique()
    snapshots: dict[str, object] = {}
    for as_of in as_of_dates:
        snapshot = build_pit_snapshot(
            members, classifications, as_of=as_of, codes=universe_codes
        )
        snapshots[as_of.isoformat()] = summarize_snapshot(
            snapshot, universe, static_labels
        )
    return {
        "scope": f"TUSHARE_{str(source).upper()}_L1_PIT_INDUSTRY_AUDIT_V1",
        "industry_source": f"Tushare.index_classify({str(source).upper()},L1)+index_member",
        "market_scope": "SH/SZ",
        "excluded_exchanges": ["BJ"],
        "classification_count": int(len(classifications)),
        "membership_row_count": int(len(members)),
        "as_of_dates": [value.isoformat() for value in as_of_dates],
        "snapshots": snapshots,
        "database_writes": False,
        "locked_test_read": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", action="append", type=parse_date, required=True)
    parser.add_argument("--universe-start", type=parse_date, required=True)
    parser.add_argument("--universe-end", type=parse_date, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--source", choices=("SW2014", "SW2021"), default="SW2021")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.universe_start > args.universe_end:
        parser.error("--universe-start must be on or before --universe-end")
    args.as_of = tuple(sorted(set(args.as_of)))
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    args = _parse_args(argv)
    import dolphindb as ddb  # type: ignore[import-untyped]
    import tushare as ts  # type: ignore[import-untyped]

    token = os.getenv("QRV2_TUSHARE_TOKEN") or os.getenv("TUSHARE_TOKEN", "")
    if not token:
        raise RuntimeError("Tushare token is required")
    session = ddb.session()
    session.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    universe = session.run(
        "select distinct ts_code from loadTable('dfs://YesLab','stock_factor_daily') "
        f"where trade_date between date('{args.universe_start:%Y.%m.%d}') "
        f"and date('{args.universe_end:%Y.%m.%d}')"
    )
    static_labels = session.run(
        "select ts_code,industry from loadTable('dfs://YesLab','stock_info_table')"
    )
    payload = run_audit(
        pro=ts.pro_api(token),
        universe=universe,
        static_labels=static_labels,
        as_of_dates=args.as_of,
        cache_dir=args.cache_dir,
        source=args.source,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    main()
