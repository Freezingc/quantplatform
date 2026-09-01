"""Correlation-cluster analysis over the formal nine plus the 18 READY candidates.

Assembles the full 27-factor training-window median |Spearman| matrix from the
existing similarity artifacts (no new materialization), then runs average-linkage
hierarchical clustering on distance = 1 - median_abs_spearman.  The clusters are
information families: admitting factors from different clusters adds more
independent information than admitting within one cluster.

Read-only analysis; no database writes, no locked-test reads, no admission decisions.
"""

from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG = REPO_ROOT / "research_v2" / "artifacts" / "diagnostics"

FORMAL_NINE = [
    "yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155",
    "yeslab_98", "yeslab_94", "yeslab_376", "yeslab_396", "yeslab_442",
]
CANDIDATES = {
    "yeslab_114_tushare_shsz_v2": "yeslab_114",
    "yeslab_381_tushare_shsz_v1": "yeslab_381",
    "yeslab_405_tushare_shsz_v1": "yeslab_405",
    "yeslab_425_tushare_shsz_v2": "yeslab_425",
    "yeslab_484_tushare_shsz_v1": "yeslab_484",
    "yeslab_206_median_abs_v2": "yeslab_206_median_abs",
    "yeslab_363_tushare_shsz_v1": "yeslab_363",
    "yeslab_357_tushare_shsz_v1": "yeslab_357",
    "yeslab_96_canonical_replay_v1": "yeslab_96",
    "yeslab_443_tushare_shsz_v1": "yeslab_443",
    "yeslab_464_tushare_shsz_v1": "yeslab_464",
    "yeslab_370_tushare_shsz_v1": "yeslab_370",
    "yeslab_375_tushare_shsz_v1": "yeslab_375",
    "yeslab_389_tushare_shsz_v1": "yeslab_389",
    "yeslab_390_tushare_shsz_v1": "yeslab_390",
    "yeslab_471_tushare_pit_v1": "yeslab_471",
    "yeslab_398_tushare_shsz_v1": "yeslab_398",
    "yeslab_395_tushare_shsz_v1": "yeslab_395",
}
SOURCE_TO_NODE = {v: k for k, v in CANDIDATES.items()}
SOURCE_TO_NODE.update({f: f for f in FORMAL_NINE})

# candidate_id -> artifact(s) holding its nine-edge train similarity rows
NINE_EDGE_SOURCES = {
    "yeslab_114_tushare_shsz_v2": [("json_rows", "tushare_alpha114_canonical_train_similarity_v1/similarity_summary.json")],
    "yeslab_381_tushare_shsz_v1": [("json_rows", "tushare_alpha381_canonical_train_similarity_v1/similarity_summary.json")],
    "yeslab_405_tushare_shsz_v1": [("json_rows", "tushare_alpha405_canonical_train_similarity_v1/similarity_summary.json")],
    "yeslab_425_tushare_shsz_v2": [("json_rows", "tushare_alpha425_canonical_train_similarity_v1/similarity_summary.json")],
    "yeslab_484_tushare_shsz_v1": [("json_rows", "tushare_alpha484_canonical_v1/similarity_summary.json", "train")],
    "yeslab_206_median_abs_v2": [("json_pairs", "alpha206_median_abs_formal9_similarity_2020_2024_v1.json")],
    "yeslab_363_tushare_shsz_v1": [("json_rows", "tushare_shsz_alpha363_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_357_tushare_shsz_v1": [("json_rows", "tushare_shsz_alpha357_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_96_canonical_replay_v1": [("json_rows", "tushare_canonical_formal9_unique7_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_443_tushare_shsz_v1": [("json_rows", "tushare_shsz_alpha443_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_464_tushare_shsz_v1": [("json_rows", "tushare_shsz_gp_coverage_repair4_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_370_tushare_shsz_v1": [("json_rows", "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_375_tushare_shsz_v1": [("json_rows", "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_389_tushare_shsz_v1": [("json_rows", "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_390_tushare_shsz_v1": [("json_rows", "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_471_tushare_pit_v1": [("csv_pairs", "alpha471_vs_formal9_train_similarity_pairs_2020_2024_v1.csv")],
    "yeslab_398_tushare_shsz_v1": [("json_rows", "tushare_shsz_dq3_formal9_train_2020_2024_v1/similarity_summary.json")],
    "yeslab_395_tushare_shsz_v1": [("json_rows", "tushare_shsz_alpha395_formal9_train_2020_2024_v1/similarity_summary.json")],
}


def add_edge(edges, left, right, value):
    key = frozenset((left, right))
    if len(key) != 2:
        return
    edges.setdefault(key, []).append(float(value))


NODE_SET = set(SOURCE_TO_NODE.values())


def resolve(name):
    node = SOURCE_TO_NODE.get(name)
    if node is not None:
        return node
    if name in NODE_SET:
        return name
    return None


def load_json_rows(path, phase=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("pairs", [])
        if not rows and "train_pairs" in data:
            rows = data["train_pairs"]
    else:
        rows = data
    for row in rows:
        if phase is not None and row.get("phase") not in (None, phase):
            continue
        left = resolve(str(row.get("factor_id_left")))
        right = resolve(str(row.get("factor_id_right")))
        if left is None or right is None:
            left = resolve(str(row.get("candidate_factor_id")))
            right = resolve(str(row.get("reference_factor_id")))
        metric = row.get("median_abs_spearman", row.get("median_daily_abs_spearman"))
        if left and right and metric is not None:
            yield left, right, metric


def load_csv_pairs(path):
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            left = resolve(str(row.get("factor_id_left")))
            right = resolve(str(row.get("factor_id_right")))
            metric = row.get("median_abs_spearman") or row.get("median_daily_abs_spearman")
            if left and right and metric not in (None, ""):
                yield left, right, metric


def main():
    edges: dict[frozenset, list[float]] = {}

    # 1) formal nine mutual pairs from the nearmiss6 train batch
    for left, right, value in load_json_rows(DIAG / "tushare_shsz_nearmiss6_formal9_train_2020_2024_v1/similarity_summary.json"):
        if left in FORMAL_NINE and right in FORMAL_NINE:
            add_edge(edges, left, right, value)

    # 2) candidate vs formal nine edges from each candidate's source artifact
    for candidate_id, sources in NINE_EDGE_SOURCES.items():
        for spec in sources:
            kind, rel = spec[0], spec[1]
            phase = spec[2] if len(spec) > 2 else None
            path = DIAG / rel
            rows = load_csv_pairs(path) if kind == "csv_pairs" else load_json_rows(path, phase=phase)
            for left, right, value in rows:
                if candidate_id in (left, right):
                    other = right if left == candidate_id else left
                    if other in FORMAL_NINE:
                        add_edge(edges, candidate_id, other, value)

    # 3) candidate vs candidate edges from the pool18 preflight train CSV
    for left, right, value in load_csv_pairs(DIAG / "pool18_preflight_train_similarity_v1.csv"):
        add_edge(edges, left, right, value)

    nodes = FORMAL_NINE + list(CANDIDATES.keys())
    missing = []
    for a, b in combinations(nodes, 2):
        if frozenset((a, b)) not in edges:
            missing.append((a, b))
    if missing:
        raise SystemExit(f"missing {len(missing)} edges, e.g. {missing[:6]}")

    # collapse duplicate edge observations (take the mean; differences are run-level noise)
    matrix = {}
    dup_stats = []
    for key, values in edges.items():
        matrix[key] = float(np.mean(values))
        if len(values) > 1:
            dup_stats.append((sorted(key), values))
    n = len(nodes)
    expected = n * (n - 1) // 2
    assert len(matrix) == expected, f"matrix has {len(matrix)} pairs, expected {expected}"

    mat = np.zeros((n, n))
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if i == j:
                continue
            mat[i, j] = matrix[frozenset((a, b))]
    condensed = squareform(mat, checks=False)
    Z = linkage(condensed, method="average")
    for cut in (0.5, 0.4, 0.35):
        labels = fcluster(Z, t=cut, criterion="distance")
        k = len(set(labels.tolist()))
        print(f"cut distance {cut}: {k} clusters")

    labels = fcluster(Z, t=0.4, criterion="distance")
    clusters: dict[int, list[str]] = {}
    for node, lab in zip(nodes, labels):
        clusters.setdefault(int(lab), []).append(node)

    print("\nclusters at cut 0.4 (information families):")
    for lab, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  cluster {lab} ({len(members)}):")
        for m in members:
            tag = "FORMAL" if m in FORMAL_NINE else "candidate"
            print(f"    {m} [{tag}]")

    # substitution groups: connected components of the >=0.80 edge graph
    adj: dict[str, set] = {node: set() for node in nodes}
    strong_edges = []
    for key, value in matrix.items():
        if value >= 0.80:
            a, b = tuple(key)
            adj[a].add(b)
            adj[b].add(a)
            strong_edges.append([sorted(key), round(value, 4)])
    seen = set()
    components = []
    for node in nodes:
        if node in seen or not adj[node]:
            continue
        stack = [node]
        comp = []
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            comp.append(cur)
            stack.extend(adj[cur] - seen)
        components.append(sorted(comp))
    isolated = sorted(node for node in nodes if not adj[node])

    payload = {
        "scope": "POOL27_CORRELATION_CLUSTERS_V1",
        "market_scope": "SH/SZ",
        "window": "train 2020-01-02/2024-12-31",
        "metric": "median abs Spearman (training window), distance = 1 - metric, average linkage",
        "nodes": nodes,
        "pair_count": expected,
        "duplicate_edge_observations": len(dup_stats),
        "linkage": [list(map(float, row)) for row in Z],
        "clusters_at_cut_0.4": {str(k): v for k, v in clusters.items()},
        "strong_edges_ge_0.80": sorted(strong_edges),
        "substitution_groups": components,
        "isolated_nodes": isolated,
        "interpretation": ("substitution_groups are connected components of the >=0.80 median-abs-Spearman graph: "
            "factors inside a group are partial substitutes (admit at most one without separate combination review). "
            "isolated_nodes have no >=0.80 edge with any other factor in the pool and are mutually compatible on redundancy grounds."),
        "database_writes": False,
        "locked_test_read": False,
        "formal_factor_set_changed": False,
        "model_training_started": False,
    }
    out = DIAG / "pool27_correlation_clusters_v1.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    if dup_stats:
        print("sample duplicate observations:")
        for pair, values in dup_stats[:5]:
            print("  ", pair, [round(v, 6) for v in values])


if __name__ == "__main__":
    main()
