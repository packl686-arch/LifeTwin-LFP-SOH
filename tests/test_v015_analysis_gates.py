from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_analysis as analysis
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    CORE_FAMILY_IDS,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
    OperatingCovariates,
    evaluate_intrinsic_pair_retention,
    evaluate_stress_plan_pair_retention,
    load_frozen_protocol_config,
)

_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2.json"
)


class _BlockCheckingVisibleState:
    feature_names = ("prefix_feature", *REAL_OPERATING_FIELDS)

    def __init__(self) -> None:
        self.calls = 0

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        operating = matrix[:, 1:]
        offsets = np.arange(len(REAL_OPERATING_FIELDS), dtype=float) * 100.0
        np.testing.assert_array_equal(
            operating - operating[:, [0]], np.broadcast_to(offsets, operating.shape)
        )
        self.calls += 1
        return operating[:, 0]


def _risk_rows(
    *,
    partition: str,
    family: str,
    count: int,
    eligible_count: int,
    catastrophic_count: int,
    hash_offset: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        catastrophic = index < catastrophic_count
        rows.append(
            {
                "partition": partition,
                "truth_family": family,
                "hard_eligible_visible_stress": index < eligible_count,
                "catastrophic": catastrophic,
                "risk_prefix_only": float(catastrophic),
                "risk_visible_stress": float(catastrophic),
                "risk_hash_prefix_only": f"{hash_offset + index:064x}",
                "risk_hash_visible_stress": (f"{1_000_000 + hash_offset + index:064x}"),
            }
        )
    return rows


def _gate_by_id(
    gates: tuple[analysis.GateEvaluation, ...],
) -> dict[str, analysis.GateEvaluation]:
    return {gate.gate_id: gate for gate in gates}


def test_random_ranking_point_uses_mean_not_pool_prevalence() -> None:
    trajectories = pd.DataFrame(
        {
            "catastrophic": [True, False, True, False],
            "hard_eligible_visible_stress": [True] * 4,
            "issued_visible_stress": [True, True, False, False],
        }
    )
    random_rankings = pd.DataFrame(
        {
            "issued_count": np.full(analysis.RANDOM_RANKING_COUNT, 2),
            "issued_catastrophic_rate": np.tile(
                [0.0, 0.5], analysis.RANDOM_RANKING_COUNT // 2
            ),
        }
    )

    result = analysis.risk_reduction_against_random_rankings(
        trajectories,
        random_rankings,
        issued_column="issued_visible_stress",
        expected_issue_count=2,
    )

    assert trajectories["catastrophic"].mean() == 0.5
    assert result.random_expected_catastrophic_rate == 0.25
    assert result.issued_catastrophic_rate == 0.5
    assert result.relative_risk_reduction == -1.0


def test_tied_policy_scores_sort_on_the_full_sha256_content() -> None:
    tail_zero = "0" * 64
    tail_one = "0" * 63 + "1"
    assert tail_zero[:63] == tail_one[:63]
    frame = pd.DataFrame(
        {
            "risk_visible_stress": [0.0, 0.0],
            "risk_hash_visible_stress": [tail_one, tail_zero],
            "hard_eligible_visible_stress": [True, True],
        },
        index=["first_row", "second_row"],
    )

    issued = analysis.rank_policy(
        frame,
        protocol_id="fixture",
        arm="visible_stress",
        score_column="risk_visible_stress",
        predictor_hash_column="risk_hash_visible_stress",
        issue_count=1,
    )
    expected = min(
        frame.index,
        key=lambda row_index: hashlib.sha256(
            (
                "fixture|visible_stress|"
                f"{frame.loc[row_index, 'risk_hash_visible_stress']}"
            ).encode("ascii")
        ).hexdigest(),
    )

    assert expected == "second_row"
    assert issued.to_dict() == {
        "first_row": False,
        "second_row": True,
    }


@pytest.mark.parametrize(
    ("increment", "expected_state"),
    [
        (0.049999999, "pass"),
        (0.05, "fail"),
        (-0.05, "fail"),
    ],
)
def test_placebo_point_gate_uses_strict_five_percent_boundary(
    increment: float,
    expected_state: str,
) -> None:
    point, interval = analysis.placebo_negative_control_gates(
        placebo_minus_prefix_increment=increment,
        bootstrap_summary={
            "placebo_two_sided_95_lower": -0.01,
            "placebo_two_sided_95_upper": 0.01,
        },
    )

    assert point.state == expected_state
    assert interval.state == "pass"


@pytest.mark.parametrize(
    ("lower", "upper", "expected_state"),
    [
        (-0.1, 0.0, "pass"),
        (0.0, 0.1, "pass"),
        (np.nextafter(0.0, 1.0), 0.1, "fail"),
    ],
)
def test_placebo_interval_gate_contains_zero_inclusively(
    lower: float,
    upper: float,
    expected_state: str,
) -> None:
    _, interval = analysis.placebo_negative_control_gates(
        placebo_minus_prefix_increment=0.0,
        bootstrap_summary={
            "placebo_two_sided_95_lower": lower,
            "placebo_two_sided_95_upper": upper,
        },
    )

    assert interval.state == expected_state


def test_stress_permutation_gate_has_exact_9900_strict_boundary() -> None:
    passing_metrics = pd.DataFrame(
        {
            "visible_minus_prefix_increment": np.concatenate(
                (np.zeros(9900), np.ones(100))
            )
        }
    )
    passing = analysis.summarize_stress_permutations(
        passing_metrics,
        observed_visible_minus_prefix_increment=1.0,
    )
    failing_metrics = pd.DataFrame(
        {
            "visible_minus_prefix_increment": np.concatenate(
                (np.zeros(9899), np.ones(101))
            )
        }
    )
    failing = analysis.summarize_stress_permutations(
        failing_metrics,
        observed_visible_minus_prefix_increment=1.0,
    )

    assert passing.strictly_lower_count == 9900
    assert passing.gate_passed is True
    assert failing.strictly_lower_count == 9899
    assert failing.gate_passed is False


def test_stress_permutation_moves_each_eight_field_block_jointly() -> None:
    rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    for family_index, family in enumerate(analysis.TEST_FAMILIES):
        for member_index in range(2):
            ordinal = family_index * 2 + member_index
            cluster_id = f"cluster-{ordinal:02d}"
            rows.append(
                {
                    "partition": "test",
                    "cluster_id": cluster_id,
                    "truth_family": family,
                    "canonical_prefix_content_sha256": f"{ordinal + 1:064x}",
                    "hard_eligible_visible_stress": True,
                    "catastrophic": ordinal % 3 == 0,
                    "prefix_feature": float(ordinal),
                }
            )
            base = float(ordinal + 1)
            operating_rows.append(
                {
                    "partition": "test",
                    "cluster_id": cluster_id,
                    **{
                        name: base + field_index * 100.0
                        for field_index, name in enumerate(REAL_OPERATING_FIELDS)
                    },
                }
            )
    state = _BlockCheckingVisibleState()
    result = analysis.stress_permutation_metrics(
        pd.DataFrame(rows),
        pd.DataFrame(operating_rows),
        pd.DataFrame(columns=["partition", "cluster_id"]),
        pd.DataFrame(columns=["partition", "cluster_id"]),
        visible_stress_state=state,
        protocol_id="fixture",
        random_expected_catastrophic_rate=0.5,
        observed_prefix_only_risk_reduction=0.0,
        issue_count=4,
        permutations=3,
    )

    assert state.calls == 3
    assert result["permutation_index"].tolist() == [0, 1, 2]
    assert result["issued_count"].tolist() == [4, 4, 4]
    assert np.isfinite(result["visible_minus_prefix_increment"]).all()


def test_policy_comparison_reports_all_nine_frozen_scores() -> None:
    count = 12
    trajectories = pd.DataFrame(
        {
            "catastrophic": [index < 6 for index in range(count)],
            "hard_eligible_visible_stress": [True] * count,
            **{
                f"risk_{score_id}": np.arange(count, dtype=float)
                for score_id in analysis.RISK_SCORE_IDS
            },
            **{
                f"risk_hash_{score_id}": [
                    f"{score_index * 100 + index + 1:064x}" for index in range(count)
                ]
                for score_index, score_id in enumerate(analysis.RISK_SCORE_IDS)
            },
        }
    )
    random_rankings = pd.DataFrame(
        {
            "issued_count": np.full(analysis.RANDOM_RANKING_COUNT, 4),
            "issued_catastrophic_rate": np.full(analysis.RANDOM_RANKING_COUNT, 0.5),
        }
    )

    result = analysis.evaluate_policy_rankings(
        trajectories,
        random_rankings,
        protocol_id="fixture",
        issue_count=4,
    )

    assert tuple(result["score_id"]) == analysis.RISK_SCORE_IDS
    assert (result["issued_count"] == 4).all()


def test_intrinsic_invariance_rejects_shifted_equal_width_interval() -> None:
    pair_rows: list[dict[str, str]] = []
    point_rows: list[dict[str, object]] = []
    trajectory_rows: list[dict[str, object]] = []
    for pair_index in range(250):
        left = f"left-{pair_index:03d}"
        right = f"right-{pair_index:03d}"
        pair_rows.append(
            {
                "pair_id": f"pair-{pair_index:03d}",
                "left_cluster_id": left,
                "right_cluster_id": right,
            }
        )
        for cluster_id in (left, right):
            for day_index, day in enumerate(analysis.FORECAST_DAYS):
                point_rows.append(
                    {
                        "cluster_id": cluster_id,
                        "forecast_day": day,
                        "center_forecast_pct": 95.0 - day_index,
                        "sqrt_time_forecast_pct": 94.0 - day_index,
                        "bounded_power_forecast_pct": 93.0 - day_index,
                        "base_interval_lower_pct": 85.0 - day_index,
                        "base_interval_upper_pct": 100.0 - day_index,
                        "calibrated_interval_lower_pct": 80.0 - day_index,
                        "calibrated_interval_upper_pct": 102.0 - day_index,
                        "canonical_prefix_content_sha256": f"{pair_index + 1:064x}",
                    }
                )
            trajectory_rows.append(
                {
                    "cluster_id": cluster_id,
                    **{
                        f"risk_{score_id}": float(pair_index)
                        for score_id in analysis.RISK_SCORE_IDS
                    },
                    **{
                        f"risk_hash_{score_id}": f"{pair_index + 1:064x}"
                        for score_id in analysis.RISK_SCORE_IDS
                    },
                    "hard_eligible_prefix_only": True,
                    "hard_eligible_visible_stress": True,
                    "issued_prefix_only": False,
                    "issued_visible_stress": False,
                    "issuance_rank_prefix_only": np.nan,
                    "issuance_rank_visible_stress": np.nan,
                }
            )
    points = pd.DataFrame(point_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    pairs = pd.DataFrame(pair_rows)
    analysis.validate_intrinsic_output_invariance(points, trajectories, pairs)

    shifted = points.copy()
    right_mask = shifted["cluster_id"].eq("right-000")
    shifted.loc[
        right_mask,
        [
            "base_interval_lower_pct",
            "base_interval_upper_pct",
            "calibrated_interval_lower_pct",
            "calibrated_interval_upper_pct",
        ],
    ] += 1.0
    with pytest.raises(analysis.V015AnalysisError, match="differs bitwise"):
        analysis.validate_intrinsic_output_invariance(shifted, trajectories, pairs)

    stress_shifted = trajectories.copy()
    stress_shifted.loc[
        stress_shifted["cluster_id"].eq("right-000"),
        "risk_placebo_8",
    ] = -0.0
    with pytest.raises(
        analysis.V015AnalysisError,
        match="risk_placebo_8",
    ):
        analysis.validate_stress_plan_arm_a_invariance(
            points,
            stress_shifted,
            pairs,
        )

    comparator_shifted = points.copy()
    comparator_shifted.loc[
        comparator_shifted["cluster_id"].eq("right-000"),
        "sqrt_time_forecast_pct",
    ] += 0.25
    with pytest.raises(
        analysis.V015AnalysisError,
        match="sqrt_time_forecast_pct",
    ):
        analysis.validate_stress_plan_arm_a_invariance(
            comparator_shifted,
            trajectories,
            pairs,
        )


def test_global_common_pool_minima_are_explicit_required_gates() -> None:
    rows: list[dict[str, object]] = []
    for partition, source_count, eligible_count, catastrophic_count in (
        ("test", 1900, 1805, 60),
        ("audit", 950, 903, 30),
    ):
        for index in range(source_count):
            rows.append(
                {
                    "partition": partition,
                    "hard_eligible_visible_stress": index < eligible_count,
                    "catastrophic": index < catastrophic_count,
                }
            )
    trajectories = pd.DataFrame(rows)
    assert analysis.common_pool_availability_reasons(trajectories) == ()
    gates = analysis.common_pool_gate_evaluations(trajectories)
    assert tuple(gate.gate_id for gate in gates) == (
        "test_common_pool_minimum_counts",
        "audit_common_pool_minimum_counts",
    )
    assert all(gate.state == "pass" for gate in gates)

    short = trajectories.copy()
    test_eligible = short.index[
        short["partition"].eq("test") & short["hard_eligible_visible_stress"]
    ]
    short.loc[test_eligible[-1], "hard_eligible_visible_stress"] = False
    reasons = analysis.common_pool_availability_reasons(short)
    assert any("test_common_eligible_count=1804" in reason for reason in reasons)
    assert analysis.common_pool_gate_evaluations(short)[0].state == "inconclusive"


def _pair_fixture_frames(
    *,
    cluster_ids: tuple[str, str],
    curves: tuple[np.ndarray, np.ndarray],
    operating: tuple[OperatingCovariates, OperatingCovariates],
    families: tuple[str, str],
    parameters: tuple[dict[str, float], dict[str, float]],
    gamma: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol = load_frozen_protocol_config(_CONFIG_PATH)
    prefix_rows: list[dict[str, object]] = []
    operating_rows: list[dict[str, object]] = []
    truth_rows: list[dict[str, object]] = []
    prefix_count = len(protocol.prefix_days)
    for cluster_id, curve, covariates, family, parameter in zip(
        cluster_ids,
        curves,
        operating,
        families,
        parameters,
        strict=True,
    ):
        prefix_rows.extend(
            {
                "cluster_id": cluster_id,
                "prefix_day": day,
                "observed_retention_pct": float(curve[index]),
            }
            for index, day in enumerate(protocol.prefix_days)
        )
        operating_rows.append({"cluster_id": cluster_id, **covariates.as_record()})
        parameter_text = json.dumps(
            parameter,
            sort_keys=True,
            separators=(",", ":"),
        )
        truth_rows.extend(
            {
                "cluster_id": cluster_id,
                "truth_family": family,
                "truth_parameters_json": parameter_text,
                "gamma": gamma,
                "forecast_day": day,
                "latent_retention_pct": float(curve[prefix_count + index]),
                "noisy_retention_pct": float(curve[prefix_count + index]),
            }
            for index, day in enumerate(protocol.forecast_days)
        )
    return (
        pd.DataFrame(prefix_rows),
        pd.DataFrame(operating_rows),
        pd.DataFrame(truth_rows),
    )


def test_intrinsic_construction_validator_recomputes_frozen_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_frozen_protocol_config(_CONFIG_PATH)
    midpoint = tuple(
        (lower + upper) / 2.0 for _, (lower, upper) in protocol.operating_support
    )
    covariates = OperatingCovariates(
        *midpoint,
        placebo_controls=(0.0,) * len(PLACEBO_FIELDS),
    )
    base = {"a": 0.4, "b": 0.5}
    mechanism = {
        "amplitude_pp": 6.75,
        "t_start_days": 1500.0,
        "duration_days": 800.0,
    }
    curves = evaluate_intrinsic_pair_retention(
        base,
        covariates,
        0.1,
        protocol.combined_days,
        mechanism="compact_smoothstep",
        mechanism_parameters=mechanism,
        time_scale_days=protocol.time_scale_days,
    )
    prefix, operating, truth = _pair_fixture_frames(
        cluster_ids=("left", "right"),
        curves=curves,
        operating=(covariates, covariates),
        families=("intrinsic_single_power", "intrinsic_compact_smoothstep"),
        parameters=(base, {**base, **mechanism}),
        gamma=0.1,
    )
    separation = float(abs(curves[0][-1] - curves[1][-1]))
    mapping_rows: list[dict[str, object]] = []
    for pair_index in range(250):
        construction_family = (
            "piecewise_linear_knee" if pair_index < 125 else "compact_smoothstep"
        )
        mapping_rows.append(
            {
                "pair_id": (
                    "intrinsic-pair"
                    if pair_index == 125
                    else f"intrinsic-{pair_index:03d}"
                ),
                "left_cluster_id": (
                    "left" if pair_index == 125 else f"left-{pair_index}"
                ),
                "right_cluster_id": (
                    "right" if pair_index == 125 else f"right-{pair_index}"
                ),
                "construction_family": construction_family,
                "left_side_code": "smooth_reference",
                "right_side_code": construction_family,
                "latent_prefix_rmse_pp": 0.0,
                "latent_prefix_max_abs_difference_pp": 0.0,
                "truth_separation_25y_pp": (separation if pair_index == 125 else 5.0),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    monkeypatch.setattr(
        analysis,
        "_validated_pair_members",
        lambda *_args, **_kwargs: [("intrinsic-pair", "left", "right")],
    )
    analysis.validate_intrinsic_pair_construction(
        prefix,
        operating,
        truth,
        mapping,
        protocol,
    )

    tampered = mapping.copy()
    target = tampered["pair_id"].eq("intrinsic-pair")
    tampered.loc[target, "truth_separation_25y_pp"] += 0.1
    with pytest.raises(analysis.V015AnalysisError, match="metadata is false"):
        analysis.validate_intrinsic_pair_construction(
            prefix,
            operating,
            truth,
            tampered,
            protocol,
        )

    wrong_allocation = mapping.copy()
    wrong_allocation.loc[0, "construction_family"] = "compact_smoothstep"
    wrong_allocation.loc[0, "right_side_code"] = "compact_smoothstep"
    with pytest.raises(analysis.V015AnalysisError, match="125 pairs"):
        analysis.validate_intrinsic_pair_construction(
            prefix,
            operating,
            truth,
            wrong_allocation,
            protocol,
        )


def test_stress_plan_construction_validator_checks_clean_contrast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_frozen_protocol_config(_CONFIG_PATH)
    support = protocol.support_map()
    past = tuple(sum(support[field]) / 2.0 for field in REAL_OPERATING_FIELDS[:4])
    low_plan = tuple(
        lower + 0.25 * (upper - lower)
        for lower, upper in (support[field] for field in REAL_OPERATING_FIELDS[4:])
    )
    high_plan = tuple(
        lower + 0.75 * (upper - lower)
        for lower, upper in (support[field] for field in REAL_OPERATING_FIELDS[4:])
    )
    low_covariates = OperatingCovariates(
        *past,
        *low_plan,
        placebo_controls=(0.0,) * len(PLACEBO_FIELDS),
    )
    high_covariates = OperatingCovariates(
        *past,
        *high_plan,
        placebo_controls=(0.0,) * len(PLACEBO_FIELDS),
    )
    parameters = {"a": 0.4, "b": 0.5}
    curves = evaluate_stress_plan_pair_retention(
        "single_power",
        parameters,
        low_covariates,
        high_covariates,
        0.1,
        protocol.combined_days,
        time_scale_days=protocol.time_scale_days,
    )
    prefix, operating, truth = _pair_fixture_frames(
        cluster_ids=("low", "high"),
        curves=curves,
        operating=(low_covariates, high_covariates),
        families=("single_power", "single_power"),
        parameters=(parameters, parameters),
        gamma=0.1,
    )
    mapping_rows: list[dict[str, object]] = []
    for pair_index in range(250):
        mapping_rows.append(
            {
                "pair_id": f"stress-{pair_index:03d}",
                "left_cluster_id": "low" if pair_index == 0 else f"low-{pair_index}",
                "right_cluster_id": (
                    "high" if pair_index == 0 else f"high-{pair_index}"
                ),
                "construction_family": CORE_FAMILY_IDS[
                    pair_index % len(CORE_FAMILY_IDS)
                ],
                "left_side_code": "low_plan",
                "right_side_code": "high_plan",
                "latent_prefix_rmse_pp": 0.0,
                "latent_prefix_max_abs_difference_pp": 0.0,
                "truth_separation_25y_pp": (
                    float(abs(curves[0][-1] - curves[1][-1]))
                    if pair_index == 0
                    else 0.0
                ),
            }
        )
    mapping = pd.DataFrame(mapping_rows)
    monkeypatch.setattr(
        analysis,
        "_validated_pair_members",
        lambda *_args, **_kwargs: [("stress-000", "low", "high")],
    )
    analysis.validate_stress_plan_pair_construction(
        prefix,
        operating,
        truth,
        mapping,
        protocol,
    )

    tampered = operating.copy()
    field = REAL_OPERATING_FIELDS[4]
    tampered.loc[tampered["cluster_id"].eq("high"), field] = low_plan[0]
    with pytest.raises(analysis.V015AnalysisError, match="upper support half"):
        analysis.validate_stress_plan_pair_construction(
            prefix,
            tampered,
            truth,
            mapping,
            protocol,
        )


def test_mean_baseline_uses_lexical_model_id_for_an_exact_tie() -> None:
    model_ids = (
        "target_prefix_persistence",
        "target_prefix_sqrt_time",
        "target_prefix_bounded_power_law",
    )
    metrics = pd.DataFrame(
        [
            {
                "cluster_id": f"cluster_{cluster_index}",
                "model_id": model_id,
                "trajectory_iae_pp": value,
                "finite_forecast": True,
            }
            for model_id in model_ids
            for cluster_index, value in enumerate((1.0, 2.0))
        ]
    )

    selected, table = analysis.select_strongest_mean_baseline(
        metrics,
        expected_clusters=2,
    )

    assert table["complete"].all()
    assert selected.mean_trajectory_iae_pp == 1.5
    assert selected.selected_model_id == "target_prefix_bounded_power_law"


def test_boolean_fields_reject_truthy_integers() -> None:
    frame = pd.DataFrame(
        {
            "simultaneous_interval_covered": [True, 1],
            "max_interval_width_pp": [10.0, 10.0],
        }
    )

    with pytest.raises(analysis.V015AnalysisError, match="must be boolean"):
        analysis.coverage_summary(frame)


def test_test_family_availability_uses_frozen_inclusive_minima() -> None:
    exact = pd.DataFrame(
        _risk_rows(
            partition="test",
            family="single_power",
            count=250,
            eligible_count=225,
            catastrophic_count=30,
            hash_offset=0,
        )
    )
    _, exact_gates = analysis.evaluate_test_safety_gates(
        exact,
        protocol_id="fixture",
    )
    target_id = "test_family_single_power_nonnegative_risk_reduction"
    assert _gate_by_id(exact_gates)[target_id].state != "inconclusive"

    low_eligibility = exact.copy()
    low_eligibility.loc[224, "hard_eligible_visible_stress"] = False
    _, low_eligibility_gates = analysis.evaluate_test_safety_gates(
        low_eligibility,
        protocol_id="fixture",
    )
    low_eligibility_gate = _gate_by_id(low_eligibility_gates)[target_id]
    assert low_eligibility_gate.state == "inconclusive"
    assert "eligible_count=224 minimum=225" in low_eligibility_gate.reasons

    low_catastrophe = exact.copy()
    low_catastrophe.loc[29, "catastrophic"] = False
    _, low_catastrophe_gates = analysis.evaluate_test_safety_gates(
        low_catastrophe,
        protocol_id="fixture",
    )
    low_catastrophe_gate = _gate_by_id(low_catastrophe_gates)[target_id]
    assert low_catastrophe_gate.state == "inconclusive"
    assert "catastrophic_count=29 minimum=30" in low_catastrophe_gate.reasons


def test_audit_availability_and_noninferiority_use_inclusive_boundaries() -> None:
    rows: list[dict[str, object]] = []
    rows.extend(
        _risk_rows(
            partition="audit",
            family="smooth_broken_power",
            count=100,
            eligible_count=100,
            catastrophic_count=20,
            hash_offset=0,
        )
    )
    rows.extend(
        _risk_rows(
            partition="audit",
            family="saturating_logistic_knee",
            count=100,
            eligible_count=80,
            catastrophic_count=0,
            hash_offset=1_000,
        )
    )
    rows.extend(
        _risk_rows(
            partition="audit",
            family="late_knee",
            count=200,
            eligible_count=180,
            catastrophic_count=20,
            hash_offset=2_000,
        )
    )
    rows.extend(
        _risk_rows(
            partition="audit",
            family="single_power",
            count=550,
            eligible_count=543,
            catastrophic_count=0,
            hash_offset=3_000,
        )
    )
    exact = pd.DataFrame(rows)

    _, exact_gates = analysis.evaluate_audit_directional_gates(
        exact,
        protocol_id="fixture",
        issued_center_minus_baseline_iae_pp=0.10,
    )
    by_id = _gate_by_id(exact_gates)
    assert all(gate.state != "inconclusive" for gate in exact_gates)
    assert by_id["audit_issued_center_iae_noninferiority"].state == "pass"

    low_overall = exact.copy()
    last_eligible = low_overall.index[
        (low_overall["truth_family"] == "single_power")
        & low_overall["hard_eligible_visible_stress"]
    ][-1]
    low_overall.loc[last_eligible, "hard_eligible_visible_stress"] = False
    _, low_gates = analysis.evaluate_audit_directional_gates(
        low_overall,
        protocol_id="fixture",
        issued_center_minus_baseline_iae_pp=0.10,
    )
    low_by_id = _gate_by_id(low_gates)

    assert (
        low_by_id["audit_visible_stress_positive_risk_reduction"].state
        == "inconclusive"
    )
    assert (
        "eligible_count=902 minimum=903"
        in low_by_id["audit_visible_stress_positive_risk_reduction"].reasons
    )
    assert low_by_id["audit_novel_nonnegative_risk_reduction"].state != ("inconclusive")
    assert low_by_id["audit_late_knee_nonnegative_risk_reduction"].state != (
        "inconclusive"
    )


def test_result_status_uses_frozen_conservative_precedence() -> None:
    passed = analysis.GateEvaluation("a", "pass", True, "fixture")
    failed = analysis.GateEvaluation("b", "fail", False, "fixture")
    inconclusive = analysis.GateEvaluation(
        "b",
        "inconclusive",
        None,
        "fixture",
        ("not available",),
    )

    success = analysis.resolve_result_status(
        (passed, analysis.GateEvaluation("b", "pass", True, "fixture")),
        required_gate_ids=("a", "b"),
    )
    failure = analysis.resolve_result_status(
        (passed, failed),
        required_gate_ids=("a", "b"),
    )
    unresolved = analysis.resolve_result_status(
        (analysis.GateEvaluation("a", "fail", False, "fixture"), inconclusive),
        required_gate_ids=("a", "b"),
    )
    void = analysis.resolve_result_status(
        (analysis.GateEvaluation("a", "fail", False, "fixture"), inconclusive),
        void_reasons=("commitment mismatch",),
        required_gate_ids=("a", "b"),
    )
    missing = analysis.resolve_result_status(
        (passed,),
        required_gate_ids=("a", "b"),
    )

    assert success["status"] == "success"
    assert failure["status"] == "failure"
    assert unresolved["status"] == "failure"
    assert void["status"] == "void"
    assert missing["status"] == "inconclusive_not_success"
