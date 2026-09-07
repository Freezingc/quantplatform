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
            ("positive", 1): (0.1, 0.1),
            ("unknown_negative", 1): (-0.1, -0.1),
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
            ("positive", 1): (0.1, 0.1),
            ("unknown_negative", 1): (-0.1, -0.1),
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


def test_identical_and_mirrored_factors_share_one_cluster_and_one_representative() -> None:
    """C7.5, end to end from raw values through similarity to clusters.

    Correlation is taken in absolute value, so a factor and its negation are the same signal
    and must collapse to one representative. Treating them as distinct would let a model be
    handed the same information twice with opposite signs, which is the duplication the
    clustering step exists to remove. An uncorrelated factor must stay in its own cluster.
    """
    from research_v2.factors.similarity import compute_factor_similarity

    days = [date(2025, 1, day) for day in (2, 3, 6, 7, 8)]
    codes = [f"{index:06d}.SZ" for index in range(40)]
    rows: list[dict[str, object]] = []
    for day in days:
        for index, code in enumerate(codes):
            dense = float(index) + 1.0
            scrambled = float((index * 7) % 40)
            for factor_id, value in (
                ("dense", dense),
                ("copy", dense),
                ("mirror", -dense),
                ("other", scrambled),
            ):
                rows.append(
                    {
                        "signal_date": day,
                        "ts_code": code,
                        "factor_id": factor_id,
                        "factor_version": 1,
                        "value": value,
                    }
                )

    similarity = compute_factor_similarity(
        pd.DataFrame(rows),
        start_date=days[0],
        end_date=days[-1],
        minimum_common_members=30,
        minimum_valid_days=3,
    )
    rho = {
        (str(row["factor_id_left"]), str(row["factor_id_right"])): float(
            row["median_daily_abs_spearman"]
        )
        for row in similarity.to_dict(orient="records")
    }
    # Premises, so the clustering assertion below cannot pass on a degenerate fixture.
    assert all(status == "PASS" for status in similarity["status"])
    assert rho[("copy", "dense")] == 1.0
    assert rho[("dense", "mirror")] == 1.0
    assert rho[("copy", "mirror")] == 1.0
    assert max(rho[pair] for pair in rho if "other" in pair) < 0.90

    candidates = pd.DataFrame(
        [
            _candidate("dense", fdr_q_value=0.001),
            _candidate("copy", fdr_q_value=0.010),
            _candidate("mirror", fdr_q_value=0.020),
            _candidate("other", fdr_q_value=0.030),
        ]
    )

    chosen = select_cluster_representatives(
        candidates, similarity, similarity_threshold=0.90, minimum_features=1
    )

    assert len(chosen.clusters) == 2
    large = max(chosen.clusters, key=lambda cluster: len(cluster.members))
    small = min(chosen.clusters, key=lambda cluster: len(cluster.members))
    assert large.members == (("copy", 1), ("dense", 1), ("mirror", 1))
    assert small.members == (("other", 1),)
    assert sorted(chosen.selected_factor_keys) == [("dense", 1), ("other", 1)]


def test_a_per_date_scalar_multiple_is_one_factor_however_much_its_values_move() -> None:
    """C7.5's known case, taken from a real in-repo diagnosis rather than constructed.

    ``artifacts/diagnostics/gp_turnover_scalar_degeneracy_diagnostic_v1.json`` records that
    yeslab_436 (GP_067), yeslab_446 (GP_077) and yeslab_468 (GP_099) all end in
    ``turnover.mul(<per-date Series>, axis=0)`` -- one multiplies by a per-date mean, another
    by a per-date max -- and that their daily RankIC series came out **exactly identical over
    all 36 signal days** of the smoke window. The admission decision rested on the reason:
    a factor equal to turnover times a per-date scalar has the same cross-sectional ranks as
    turnover every day, so no rank-based evaluation can tell them apart.

    That shape is not covered by the test above, and the difference is the point. ``copy`` is
    identical and ``mirror`` is a *constant* negation, so both are fixed multiples of ``dense``
    and any value-based comparison would catch them. A per-date scalar is neither: the values
    change every single day and the three factors are not scalar multiples of one another over
    the window as a whole. Only a rank-based, per-day similarity sees through it -- which is
    what ``compute_factor_similarity`` computes, and why this belongs in C7.5's set.

    The diagnostic also supplies the real contrast case: yeslab_464 (GP_095) ends in a
    per-stock regression residual, its RankIC series differs, and it was allowed to proceed to
    full evaluation. ``residual`` below plays that part, so the test cannot pass by putting
    everything in one cluster.

    Values are synthetic -- the diagnostic records the *structure*, not a sample of the source,
    and copying source rows into the tree is what C3.4 explicitly forbids. What is real is the
    mechanism and the contrast, both cited from a committed artifact.
    """
    from research_v2.factors.similarity import compute_factor_similarity

    days = [date(2025, 1, day) for day in (2, 3, 6, 7, 8)]
    codes = [f"{index:06d}.SZ" for index in range(40)]
    # Two different per-date scalars, neither constant across the window and not multiples of
    # one another, mirroring "per-date mean" versus "per-date max".
    mean_like = [1.0 + 0.25 * position for position in range(len(days))]
    max_like = [3.0 - 0.4 * position for position in range(len(days))]

    rows: list[dict[str, object]] = []
    for day_position, day in enumerate(days):
        for index, code in enumerate(codes):
            turnover = float(index) + 1.0
            for factor_id, value in (
                ("turnover", turnover),
                ("scalar_mean", turnover * mean_like[day_position]),
                ("scalar_max", turnover * max_like[day_position]),
                # per-stock and per-day, so its ranks are unrelated to turnover's
                ("residual", float((index * 7 + day_position * 13) % 40)),
            ):
                rows.append(
                    {
                        "signal_date": day,
                        "ts_code": code,
                        "factor_id": factor_id,
                        "factor_version": 1,
                        "value": value,
                    }
                )
    frame = pd.DataFrame(rows)

    # Premise: the mechanism the diagnostic names, asserted directly on the fixture. If the
    # three degenerate factors did not share a rank vector every day, the rho assertions below
    # would be measuring something else.
    wide = frame.pivot_table(index=["signal_date", "ts_code"], columns="factor_id", values="value")
    per_day_ranks = {
        factor: wide.groupby("signal_date")[factor].rank(method="average")
        for factor in ("turnover", "scalar_mean", "scalar_max")
    }
    assert per_day_ranks["scalar_mean"].equals(per_day_ranks["turnover"])
    assert per_day_ranks["scalar_max"].equals(per_day_ranks["turnover"])
    assert not wide.groupby("signal_date")["residual"].rank(method="average").equals(
        per_day_ranks["turnover"]
    )
    # And the values really do move day to day, so this is not the constant-multiple case.
    scalar_mean_rows = frame.loc[frame["factor_id"] == "scalar_mean"]
    daily_means = scalar_mean_rows.groupby("signal_date")["value"].mean()
    assert daily_means.nunique() == len(days), "the per-date scalar has to vary across the window"

    similarity = compute_factor_similarity(
        frame,
        start_date=days[0],
        end_date=days[-1],
        minimum_common_members=30,
        minimum_valid_days=3,
    )
    rho = {
        tuple(sorted((str(row["factor_id_left"]), str(row["factor_id_right"])))): float(
            row["median_daily_abs_spearman"]
        )
        for row in similarity.to_dict(orient="records")
    }
    assert all(status == "PASS" for status in similarity["status"])
    degenerate_pairs = [
        ("scalar_max", "scalar_mean"),
        ("scalar_max", "turnover"),
        ("scalar_mean", "turnover"),
    ]
    assert all(rho[pair] == 1.0 for pair in degenerate_pairs), rho
    assert max(rho[pair] for pair in rho if "residual" in pair) < 0.90, rho

    candidates = pd.DataFrame(
        [
            _candidate("turnover", fdr_q_value=0.001),
            _candidate("scalar_mean", fdr_q_value=0.010),
            _candidate("scalar_max", fdr_q_value=0.020),
            _candidate("residual", fdr_q_value=0.030),
        ]
    )

    chosen = select_cluster_representatives(
        candidates, similarity, similarity_threshold=0.90, minimum_features=1
    )

    assert len(chosen.clusters) == 2
    large = max(chosen.clusters, key=lambda cluster: len(cluster.members))
    small = min(chosen.clusters, key=lambda cluster: len(cluster.members))
    assert large.members == (("scalar_max", 1), ("scalar_mean", 1), ("turnover", 1))
    assert small.members == (("residual", 1),)
    assert sorted(chosen.selected_factor_keys) == [("residual", 1), ("turnover", 1)], (
        "one representative for three factors that are the same signal, and the contrast "
        "case keeps its own"
    )
