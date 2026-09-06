from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from research_v2.datasets.selection import (
    benjamini_hochberg,
    select_cluster_representatives,
    select_training_factors,
)


def test_benjamini_hochberg_is_deterministic_for_ties() -> None:
    adjusted = benjamini_hochberg(
        {("z", 1): 0.03, ("a", 1): 0.03, ("b", 1): 0.01}
    )

    assert adjusted[("b", 1)] == 0.03
    assert adjusted[("a", 1)] == 0.03
    assert adjusted[("z", 1)] == 0.03


def test_selector_uses_train_dates_and_freezes_unknown_direction() -> None:
    dates = [
        date(2015 + index // 26, 1, 1) + timedelta(days=index % 26)
        for index in range(130)
    ]
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for signal_date in dates:
        for number in range(30):
            value = float(number)
            labels.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{number:03d}",
                    "raw_return_5d": value,
                    "label_valid": True,
                }
            )
            rows.extend(
                [
                    {
                        "signal_date": signal_date,
                        "ts_code": f"S{number:03d}",
                        "factor_id": "positive",
                        "factor_version": 1,
                        "value": value,
                    },
                    {
                        "signal_date": signal_date,
                        "ts_code": f"S{number:03d}",
                        "factor_id": "unknown_negative",
                        "factor_version": 1,
                        "value": -value,
                    },
                ]
            )
    observations = pd.DataFrame(rows)
    label_frame = pd.DataFrame(labels)
    specs = pd.DataFrame(
        [
            {"factor_id": "positive", "factor_version": 1, "expected_direction": 1},
            {"factor_id": "unknown_negative", "factor_version": 1, "expected_direction": 0},
        ]
    )

    result = select_training_factors(
        observations,
        label_frame,
        factor_specs=specs,
        train_start=dates[0],
        train_end=dates[-1],
        inner_validation_ics={
            ("positive", 1): (0.1, 0.1, -0.1),
            ("unknown_negative", 1): (-0.1, -0.1, -0.1),
        },
    )

    selected = result.candidates.set_index("factor_id")
    assert selected.loc["positive", "status"] == "ELIGIBLE"
    assert selected.loc["positive", "frozen_direction"] == 1
    assert selected.loc["unknown_negative", "status"] == "ELIGIBLE"
    assert selected.loc["unknown_negative", "frozen_direction"] == -1

    future_date = date(2030, 1, 1)
    future_observations = pd.concat(
        [
            observations,
            pd.DataFrame(
                [
                    {
                        "signal_date": future_date,
                        "ts_code": "S000",
                        "factor_id": "positive",
                        "factor_version": 1,
                        "value": -999.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    future_labels = pd.concat(
        [
            label_frame,
            pd.DataFrame(
                [
                    {
                        "signal_date": future_date,
                        "ts_code": "S000",
                        "raw_return_5d": 999.0,
                        "label_valid": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    unchanged = select_training_factors(
        future_observations,
        future_labels,
        factor_specs=specs,
        train_start=dates[0],
        train_end=dates[-1],
        inner_validation_ics={
            ("positive", 1): (0.1, 0.1, -0.1),
            ("unknown_negative", 1): (-0.1, -0.1, -0.1),
        },
    )
    assert result.candidates.equals(unchanged.candidates)


def test_cluster_representative_uses_stable_business_tie_breakers() -> None:
    candidates = pd.DataFrame(
        [
            {
                "factor_id": "b",
                "factor_version": 1,
                "status": "ELIGIBLE",
                "fdr_q_value": 0.02,
                "signed_mean_rank_ic": 0.10,
                "positive_year_count": 4,
                "coverage": 0.95,
                "conflict_rate": 0.01,
            },
            {
                "factor_id": "a",
                "factor_version": 1,
                "status": "ELIGIBLE",
                "fdr_q_value": 0.01,
                "signed_mean_rank_ic": 0.09,
                "positive_year_count": 3,
                "coverage": 0.90,
                "conflict_rate": 0.02,
            },
        ]
    )
    similarity = pd.DataFrame(
        [
            {
                "factor_id_left": "a",
                "factor_version_left": 1,
                "factor_id_right": "b",
                "factor_version_right": 1,
                "median_daily_abs_spearman": 0.95,
                "status": "PASS",
            }
        ]
    )

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    assert chosen.selected_factor_keys == (("a", 1),)
    assert chosen.status == "READY"


# --- C7.3: clusters must be exposed with a content-addressed identity ------------


def _candidate(factor_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "factor_id": factor_id,
        "factor_version": 1,
        "status": "ELIGIBLE",
        "fdr_q_value": 0.01,
        "signed_mean_rank_ic": 0.05,
        "positive_year_count": 4,
        "coverage": 0.95,
        "conflict_rate": 0.01,
    }
    row.update(overrides)
    return row


def _edge(left: str, right: str, rho: float = 0.95, status: str = "PASS") -> dict[str, object]:
    return {
        "factor_id_left": left,
        "factor_version_left": 1,
        "factor_id_right": right,
        "factor_version_right": 1,
        "median_daily_abs_spearman": rho,
        "status": status,
    }


def _expected_cluster_id(members: list[tuple[str, int]]) -> str:
    import hashlib
    import json

    payload = json.dumps(
        [[factor_id, version] for factor_id, version in sorted(members)],
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_clusters_are_exposed_with_a_content_addressed_id() -> None:
    """C7.3: the cluster id is SHA-256 over the sorted member identities.

    Connected components were already computed internally to pick representatives, but the
    clusters themselves were never exposed and had no identity, so a clustering result could
    not be referenced in evidence or compared across folds. factor_set_id is content
    addressed (C8.2); a cluster that is not would be the one grouping in the pipeline whose
    identity depends on the order its members happened to arrive in.
    """
    candidates = pd.DataFrame([_candidate("a"), _candidate("b"), _candidate("c")])
    similarity = pd.DataFrame([_edge("a", "b")])

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    by_members = {cluster.members: cluster.cluster_id for cluster in chosen.clusters}
    assert by_members == {
        (("a", 1), ("b", 1)): _expected_cluster_id([("a", 1), ("b", 1)]),
        (("c", 1),): _expected_cluster_id([("c", 1)]),
    }


def test_cluster_id_is_invariant_to_the_order_of_both_inputs() -> None:
    candidates = pd.DataFrame([_candidate("a"), _candidate("b"), _candidate("c")])
    similarity = pd.DataFrame([_edge("a", "b"), _edge("b", "c")])

    forward = select_cluster_representatives(candidates, similarity, minimum_features=1)
    backward = select_cluster_representatives(
        candidates.iloc[::-1].reset_index(drop=True),
        similarity.iloc[::-1].reset_index(drop=True),
        minimum_features=1,
    )

    assert forward.clusters == backward.clusters
    assert [cluster.cluster_id for cluster in forward.clusters] == sorted(
        cluster.cluster_id for cluster in forward.clusters
    ), "clusters must come back in a deterministic order"


def test_chained_edges_form_one_cluster_through_transitivity() -> None:
    """C7.5's chained case: A-B and B-C above threshold means one cluster {A,B,C}.

    A per-pair grouping would split this into two clusters and could then keep two
    representatives that are transitively the same signal.
    """
    candidates = pd.DataFrame([_candidate("a"), _candidate("b"), _candidate("c")])
    similarity = pd.DataFrame([_edge("a", "b"), _edge("b", "c")])

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    assert len(chosen.clusters) == 1
    assert chosen.clusters[0].members == (("a", 1), ("b", 1), ("c", 1))
    assert len(chosen.selected_factor_keys) == 1


def test_edges_below_the_threshold_or_not_passing_do_not_join_clusters() -> None:
    candidates = pd.DataFrame([_candidate("a"), _candidate("b"), _candidate("c")])
    similarity = pd.DataFrame(
        [_edge("a", "b", rho=0.89), _edge("b", "c", rho=0.99, status="INSUFFICIENT")]
    )

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    assert len(chosen.clusters) == 3
    assert all(len(cluster.members) == 1 for cluster in chosen.clusters)


def test_ineligible_candidates_are_not_clustered() -> None:
    """Clustering happens over the eligible set, which is what a representative is drawn from."""
    candidates = pd.DataFrame(
        [_candidate("a"), _candidate("b", status="FDR_FAILED"), _candidate("c")]
    )
    similarity = pd.DataFrame([_edge("a", "b"), _edge("a", "c", rho=0.10)])

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    members = {member for cluster in chosen.clusters for member in cluster.members}
    assert members == {("a", 1), ("c", 1)}


def test_cluster_identity_is_order_independent_on_its_own() -> None:
    """Pins the sort inside ``cluster_identity``, which the end-to-end test cannot.

    ``select_cluster_representatives`` already sorts members before constructing a
    ``FactorCluster``, so an invariance test written against that function passes even if
    ``cluster_identity`` stops sorting -- confirmed by mutation. The function is public to
    the module, so its own contract needs its own test.
    """
    from research_v2.datasets.selection import cluster_identity

    members = [("c", 1), ("a", 2), ("a", 1), ("b", 1)]

    assert cluster_identity(members) == cluster_identity(list(reversed(members)))
    assert cluster_identity(members) == _expected_cluster_id(members)
    assert cluster_identity([("a", 1)]) != cluster_identity([("a", 2)]), (
        "a version change must produce a different cluster identity"
    )
