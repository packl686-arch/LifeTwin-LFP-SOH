from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_analysis as analysis


def _trajectory_rows(count: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "truth_family": analysis.TEST_FAMILIES[index % 8],
                "canonical_prefix_content_sha256": f"{index:064x}",
                "catastrophic": index % 3 == 0,
                "hard_eligible_visible_stress": True,
                "risk_prefix_only": float(index % 7),
                "risk_visible_stress": float((index * 3) % 11),
                "risk_placebo_8": float((index * 5) % 13),
                "risk_hash_prefix_only": f"{10_000 + index:064x}",
                "risk_hash_visible_stress": f"{20_000 + index:064x}",
                "risk_hash_placebo_8": f"{30_000 + index:064x}",
            }
        )
    return pd.DataFrame(rows)


def test_exact_binomial_bounds_handle_edges() -> None:
    assert analysis.clopper_pearson_lower(0, 10) == 0.0
    assert 0.0 < analysis.clopper_pearson_lower(8, 10) < 0.8
    lower, upper = analysis.clopper_pearson_two_sided(10, 10)
    assert 0.0 < lower < 1.0
    assert upper == 1.0


def test_risk_reduction_uses_same_pool_analytic_random_expectation() -> None:
    frame = pd.DataFrame(
        {
            "catastrophic": [True, True, False, False],
            "hard_eligible_visible_stress": [True] * 4,
            "issued_visible_stress": [False, False, True, True],
        }
    )
    result = analysis.risk_reduction(
        frame,
        issued_column="issued_visible_stress",
        expected_issue_count=2,
    )
    assert result.issued_catastrophic_rate == 0.0
    assert result.random_expected_catastrophic_rate == 0.5
    assert result.relative_risk_reduction == 1.0


def test_random_rankings_are_deterministic_and_do_not_change_pool() -> None:
    frame = _trajectory_rows(30)
    first = analysis.deterministic_random_rankings(frame, issue_count=15, rankings=9)
    repeated = analysis.deterministic_random_rankings(
        frame.sample(frac=1.0, random_state=9), issue_count=15, rankings=9
    )
    pd.testing.assert_frame_equal(first, repeated)
    assert set(first["issued_count"]) == {15}


def test_coverage_summary_uses_maximum_width_and_one_sided_bound() -> None:
    frame = pd.DataFrame(
        {
            "simultaneous_interval_covered": [True] * 90 + [False] * 10,
            "max_interval_width_pp": np.linspace(10.0, 30.0, 100),
        }
    )
    result = analysis.coverage_summary(frame)
    assert result.n == 100
    assert result.coverage == 0.9
    assert result.one_sided_95_lower < result.coverage
    assert result.median_max_width_pp == pytest.approx(20.0)


def test_bootstrap_draws_are_repeatable_with_occurrence_ordinals() -> None:
    rows = []
    for family_index, family in enumerate(analysis.TEST_FAMILIES):
        for index in range(6):
            rows.append(
                {
                    "truth_family": family,
                    "canonical_prefix_content_sha256": (
                        f"{family_index * 100 + index:064x}"
                    ),
                    "catastrophic": index < 3,
                    "hard_eligible_visible_stress": True,
                    "risk_prefix_only": float(index),
                    "risk_visible_stress": float(5 - index),
                    "risk_placebo_8": float((index * 2) % 5),
                    "risk_hash_prefix_only": (
                        f"{10_000 + family_index * 100 + index:064x}"
                    ),
                    "risk_hash_visible_stress": (
                        f"{20_000 + family_index * 100 + index:064x}"
                    ),
                    "risk_hash_placebo_8": (
                        f"{30_000 + family_index * 100 + index:064x}"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    first = analysis.bootstrap_risk_reductions(
        frame,
        protocol_id="fixture_protocol",
        issue_count=20,
        resamples=7,
    )
    second = analysis.bootstrap_risk_reductions(
        frame.sample(frac=1.0, random_state=3),
        protocol_id="fixture_protocol",
        issue_count=20,
        resamples=7,
    )
    pd.testing.assert_frame_equal(first, second)
    assert first["defined"].all()


def test_primary_flags_apply_width_and_placebo_rules() -> None:
    coverage = analysis.CoverageSummary(1500, 1350, 0.9, 0.87, 35.0, 45.0)
    intrinsic = analysis.CoverageSummary(250, 215, 0.86, 0.81, 38.0, 48.0)
    flags = analysis.primary_gate_flags(
        visible_reduction=0.35,
        increment=0.12,
        bootstrap_summary={
            "visible_one_sided_95_lower": 0.08,
            "increment_one_sided_95_lower": 0.02,
            "placebo_two_sided_95_lower": -0.03,
            "placebo_two_sided_95_upper": 0.02,
        },
        core_coverage=coverage,
        intrinsic_coverage=intrinsic,
        issued_center_minus_baseline_iae_pp=0.05,
    )
    assert all(flags.values())


def test_intrinsic_pairs_require_equal_outputs_and_cover_both_members() -> None:
    rows = []
    pairs = []
    for index in range(250):
        left = f"left_{index}"
        right = f"right_{index}"
        pairs.append(
            {
                "pair_id": f"pair_{index}",
                "left_cluster_id": left,
                "right_cluster_id": right,
            }
        )
        for cluster_id, covered in ((left, True), (right, index < 225)):
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "simultaneous_interval_covered": covered,
                    "max_interval_width_pp": 30.0,
                    "risk_prefix_only": float(index),
                    "risk_visible_stress": float(index + 1),
                }
            )
    scores, summary = analysis.evaluate_intrinsic_pairs(
        pd.DataFrame(rows), pd.DataFrame(pairs)
    )
    assert len(scores) == 250
    assert summary.covered == 225
    assert summary.coverage == 0.9


def test_stress_pairs_treat_score_or_error_ties_as_incorrect() -> None:
    rows = []
    pairs = []
    for index in range(250):
        left = f"left_{index}"
        right = f"right_{index}"
        pairs.append(
            {
                "pair_id": f"pair_{index}",
                "left_cluster_id": left,
                "right_cluster_id": right,
            }
        )
        rows.extend(
            [
                {
                    "cluster_id": left,
                    "center_endpoint_absolute_error_pp": 6.0,
                    "risk_prefix_only": 1.0,
                    "risk_visible_stress": 2.0 if index < 200 else 1.0,
                },
                {
                    "cluster_id": right,
                    "center_endpoint_absolute_error_pp": 2.0,
                    "risk_prefix_only": 1.0,
                    "risk_visible_stress": 1.0,
                },
            ]
        )
    scores, summary = analysis.evaluate_stress_plan_pairs(
        pd.DataFrame(rows), pd.DataFrame(pairs)
    )
    assert len(scores) == 250
    assert summary.arm_a_exact_tie_count == 250
    assert summary.arm_b_correct_order_count == 200
    assert summary.arm_b_correct_order_fraction == 0.8


def _score_bundle_fixture(
    *,
    eligible: bool = True,
    issued: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    days = np.asarray(analysis.FORECAST_DAYS)
    prediction = pd.DataFrame(
        {
            "protocol_id": "fixture",
            "partition": "test",
            "cluster_id": "opaque",
            "forecast_day": days,
            "center_forecast_pct": np.linspace(98.0, 90.0, 8),
            "sqrt_time_forecast_pct": np.linspace(98.0, 91.0, 8),
            "bounded_power_forecast_pct": np.linspace(98.0, 89.0, 8),
            "base_interval_lower_pct": np.linspace(92.0, 82.0, 8),
            "base_interval_upper_pct": np.linspace(101.0, 99.0, 8),
            "calibrated_interval_lower_pct": np.linspace(90.0, 80.0, 8),
            "calibrated_interval_upper_pct": np.linspace(102.0, 100.0, 8),
            "canonical_prefix_content_sha256": "0" * 64,
        }
    )
    truth = pd.DataFrame(
        {
            "protocol_id": "fixture",
            "partition": "test",
            "cluster_id": "opaque",
            "truth_family": "single_power",
            "forecast_day": days,
            "latent_retention_pct": np.linspace(98.0, 88.0, 8),
            "noisy_retention_pct": np.linspace(98.0, 88.0, 8),
        }
    )
    risk = pd.DataFrame(
        [
            {
                "protocol_id": "fixture",
                "partition": "test",
                "cluster_id": "opaque",
                "score_id": score_id,
                "raw_risk_score": value,
                "calibrated_catastrophic_probability": calibrated,
                "canonical_predictor_content_sha256": content_hash,
            }
            for score_id, value, calibrated, content_hash in (
                ("prefix_only", 0.1, 0.25, "0" * 64),
                ("visible_stress", 0.2, 0.30, "1" * 64),
                ("placebo_8", 0.15, np.nan, "2" * 64),
                ("arm_a_plus_s_plan", 0.12, np.nan, "3" * 64),
                ("strongest_single_feature", 0.11, np.nan, "4" * 64),
                ("planned_stress_only", 0.09, np.nan, "5" * 64),
                ("prefix_rmse_only", 0.08, np.nan, "6" * 64),
                ("v1_max_envelope_only", 0.07, np.nan, "7" * 64),
                (
                    "center_sqrt_abs_difference_only",
                    0.06,
                    np.nan,
                    "8" * 64,
                ),
            )
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "protocol_id": "fixture",
                "partition": "test",
                "cluster_id": "opaque",
                "arm": "prefix_only",
                "hard_eligible": eligible,
                "issued": issued,
                "issuance_rank": 1 if eligible else None,
                "raw_risk_score": 0.1,
                "canonical_predictor_content_sha256": "0" * 64,
            },
            {
                "protocol_id": "fixture",
                "partition": "test",
                "cluster_id": "opaque",
                "arm": "visible_stress",
                "hard_eligible": eligible,
                "issued": issued,
                "issuance_rank": 1 if eligible else None,
                "raw_risk_score": 0.2,
                "canonical_predictor_content_sha256": "1" * 64,
            },
        ]
    )
    return prediction, truth, risk, decision


def test_score_trajectory_table_rejects_arm_specific_eligibility() -> None:
    prediction, truth, risk, decision = _score_bundle_fixture()
    decision.loc[decision["arm"].eq("visible_stress"), "hard_eligible"] = False
    decision.loc[decision["arm"].eq("visible_stress"), "issued"] = False
    decision.loc[decision["arm"].eq("visible_stress"), "issuance_rank"] = None

    with pytest.raises(analysis.V015AnalysisError, match="eligibility pool"):
        analysis.score_trajectory_table(prediction, truth, risk, decision)


def test_score_trajectory_table_allows_ineligible_nan_risks() -> None:
    prediction, truth, risk, decision = _score_bundle_fixture(
        eligible=False,
        issued=False,
    )
    risk["raw_risk_score"] = np.nan
    risk["calibrated_catastrophic_probability"] = np.nan
    decision["raw_risk_score"] = np.nan

    points, trajectories = analysis.score_trajectory_table(
        prediction,
        truth,
        risk,
        decision,
    )

    assert len(points) == len(analysis.FORECAST_DAYS)
    assert len(trajectories) == 1
    assert not bool(trajectories.loc[0, "hard_eligible_visible_stress"])
    assert not bool(trajectories.loc[0, "issued_visible_stress"])
    assert np.isnan(trajectories.loc[0, "risk_visible_stress"])


def test_score_trajectory_table_rejects_issued_ineligible_member() -> None:
    prediction, truth, risk, decision = _score_bundle_fixture(
        eligible=False,
        issued=False,
    )
    decision.loc[decision["arm"].eq("visible_stress"), "issued"] = True

    with pytest.raises(
        analysis.V015AnalysisError,
        match="ineligible trajectory was issued",
    ):
        analysis.score_trajectory_table(prediction, truth, risk, decision)


def test_score_trajectory_table_rejects_ranked_ineligible_member() -> None:
    prediction, truth, risk, decision = _score_bundle_fixture(
        eligible=False,
        issued=False,
    )
    decision.loc[
        decision["arm"].eq("visible_stress"),
        "issuance_rank",
    ] = 1

    with pytest.raises(
        analysis.V015AnalysisError,
        match="ineligible trajectory has an issuance rank",
    ):
        analysis.score_trajectory_table(prediction, truth, risk, decision)


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ("raw_risk_score", "risk/visible_stress must be finite"),
        (
            "calibrated_catastrophic_probability",
            "calibrated risk/visible_stress must be finite",
        ),
    ],
)
def test_score_trajectory_table_rejects_eligible_nan_risk(
    column: str,
    match: str,
) -> None:
    prediction, truth, risk, decision = _score_bundle_fixture()
    risk.loc[risk["score_id"].eq("visible_stress"), column] = np.nan
    if column == "raw_risk_score":
        decision.loc[
            decision["arm"].eq("visible_stress"),
            "raw_risk_score",
        ] = np.nan

    with pytest.raises(analysis.V015AnalysisError, match=match):
        analysis.score_trajectory_table(prediction, truth, risk, decision)


def test_score_trajectory_table_keeps_ineligible_hashes_mandatory() -> None:
    prediction, truth, risk, decision = _score_bundle_fixture(
        eligible=False,
        issued=False,
    )
    risk.loc[risk["score_id"].eq("placebo_8"), "raw_risk_score"] = np.nan
    risk.loc[
        risk["score_id"].eq("placebo_8"),
        "canonical_predictor_content_sha256",
    ] = np.nan

    with pytest.raises(analysis.V015AnalysisError, match="Trajectory metadata"):
        analysis.score_trajectory_table(prediction, truth, risk, decision)
