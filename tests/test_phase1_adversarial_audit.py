from __future__ import annotations

import pandas as pd
import pytest

from lifetwin.audits.phase1_adversarial import Phase1AuditBundle
from lifetwin.data.naumann import validate_naumann_calendar_observations


def test_data_identity_units_duplicates_and_missing_values(
    observations: pd.DataFrame,
    phase1_audit: Phase1AuditBundle,
) -> None:
    result = phase1_audit.summary["data_identity_units_duplicates_missing"]
    assert result["status"] == "passed"
    assert result["effective_independent_n"] == 17
    assert result["observation_count"] == 595
    assert result["published_physical_cell_count"] == 51
    assert result["exact_duplicate_row_count"] == 0
    assert result["duplicate_condition_checkup_count"] == 0
    assert result["missing_value_count"] == 0
    assert all(result["checks"].values())
    assert len(phase1_audit.data_conditions) == 17
    assert set(phase1_audit.data_conditions["observation_count"]) == {35}

    duplicated = pd.concat([observations, observations.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_naumann_calendar_observations(duplicated)

    missing = observations.copy()
    missing.loc[1, "source_cell_id"] = None
    with pytest.raises(ValueError, match="non-null"):
        validate_naumann_calendar_observations(missing)

    bad_hours = observations.copy()
    bad_hours.loc[1, "elapsed_hours"] += 1.0
    with pytest.raises(ValueError, match="Elapsed-hour conversion"):
        validate_naumann_calendar_observations(bad_hours)

    bad_capacity = observations.copy()
    bad_capacity.loc[1, "capacity_ah"] *= 0.99
    with pytest.raises(ValueError, match="Capacity retention"):
        validate_naumann_calendar_observations(bad_capacity)

    bad_license = observations.copy()
    bad_license.loc[1, "source_license"] = "UNKNOWN"
    with pytest.raises(ValueError, match="source_license"):
        validate_naumann_calendar_observations(bad_license)

    shuffled = observations.sample(frac=1.0, random_state=42).reset_index(drop=True)
    validate_naumann_calendar_observations(shuffled)

    swapped_identity = observations.copy()
    first = "NAUMANN_CAL_T40_SOC12.5"
    second = "NAUMANN_CAL_T40_SOC25"
    first_mask = observations["condition_id"] == first
    second_mask = observations["condition_id"] == second
    for column in ("condition_id", "cell_id", "test_id", "source_cell_id"):
        first_value = observations.loc[first_mask, column].iloc[0]
        second_value = observations.loc[second_mask, column].iloc[0]
        swapped_identity.loc[first_mask, column] = second_value
        swapped_identity.loc[second_mask, column] = first_value
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_naumann_calendar_observations(swapped_identity)

    bad_nominal_unit = observations.copy()
    bad_nominal_unit["capacity_ah"] *= 1000.0
    bad_nominal_unit["nominal_capacity_ah"] *= 1000.0
    with pytest.raises(ValueError, match="nominal capacity"):
        validate_naumann_calendar_observations(bad_nominal_unit)

    bad_resistance_unit = observations.copy()
    bad_resistance_unit["resistance_dc_ohm"] *= 1000.0
    with pytest.raises(ValueError, match="expressed in ohms"):
        validate_naumann_calendar_observations(bad_resistance_unit)

    bad_resistance_soc_unit = observations.copy()
    bad_resistance_soc_unit["resistance_dc_soc_pct"] /= 100.0
    with pytest.raises(ValueError, match="percent-valued"):
        validate_naumann_calendar_observations(bad_resistance_soc_unit)

    bad_time_unit = observations.copy()
    for column in ("elapsed_time_s", "elapsed_hours", "elapsed_days"):
        bad_time_unit[column] *= 1000.0
    with pytest.raises(ValueError, match="time axis mismatch"):
        validate_naumann_calendar_observations(bad_time_unit)


def test_future_capacity_attack_changes_scores_not_predictions(
    phase1_audit: Phase1AuditBundle,
) -> None:
    result = phase1_audit.summary["future_label_firewall_attack"]
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["attack"]["tested_prefix_checkups"] == [5, 8, 10, 14]
    assert result["attack"]["attack_count"] == 4
    assert result["maximum_absolute_prediction_delta_pp"] == 0.0
    assert result["maximum_absolute_score_change_pp"] > 0.0
    cases = phase1_audit.future_label_attacks
    assert list(cases["prefix_checkups"]) == [5, 8, 10, 14]
    assert cases["prediction_frame_identical"].all()
    assert cases["diagnostics_identical"].all()
    assert cases["parameters_identical"].all()
    assert cases["splits_identical"].all()
    assert cases["score_changed"].all()
    assert (
        cases["prediction_sha256_baseline"]
        == cases["prediction_sha256_attacked"]
    ).all()
    isolation = result["held_out_target_prefix_split_isolation"]
    assert isolation["status"] == "passed"
    assert all(isolation["checks"].values())


def test_every_condition_metric_is_independently_recomputed(
    phase1_audit: Phase1AuditBundle,
) -> None:
    result = phase1_audit.summary["independent_metric_recalculation"]
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["condition_method_group_count"] == len(
        phase1_audit.metric_recalculation
    )
    assert result["condition_method_group_count"] == 504
    assert max(result["maximum_absolute_metric_differences"].values()) <= 1e-10
    assert result["maximum_absolute_summary_difference"] <= 1e-10


def test_baselines_share_support_and_ablation_is_explicit(
    phase1_audit: Phase1AuditBundle,
) -> None:
    result = phase1_audit.summary["baseline_fairness_and_ablation"]
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["method_count"] == 6
    assert result["ready_future_coordinate_count"] > 0
    assert result["fallback_future_coordinate_count"] > 0
    assert max(result["maximum_routing_differences_pp"].values()) <= 1e-10

    ablations = phase1_audit.ablations
    assert len(ablations) == 12
    assert ablations["same_condition_support"].all()
    assert not ablations["confirmatory_claim_allowed"].any()
    assert set(ablations["condition_count"]) == {4, 17}
    assert set(ablations["single_factor_isolation"]) == {True, False}


def test_gate_boundaries_and_fallback_paths(
    phase1_audit: Phase1AuditBundle,
) -> None:
    result = phase1_audit.summary["gate_boundary_and_routing"]
    assert result["status"] == "passed"
    assert all(result["checks"].values())
    cases = phase1_audit.gate_boundaries.set_index("case")
    assert not cases.loc["six_points_with_negative_loss", "observed_gate_ready"]
    assert cases.loc["seven_points_with_negative_loss", "observed_gate_ready"]
    assert not cases.loc[
        "negative_loss_exactly_at_margin", "observed_gate_ready"
    ]
    assert cases.loc["negative_loss_beyond_margin", "observed_gate_ready"]
    assert cases["case_passed"].all()


def test_failure_condition_table_never_overstates_trust(
    phase1_audit: Phase1AuditBundle,
) -> None:
    summary = phase1_audit.summary["failure_condition_inventory"]
    table = phase1_audit.failure_conditions
    assert summary["status"] == "generated"
    assert len(table) == summary["row_count"] == 84
    assert summary["primary_prefix_row_count"] == 21
    assert summary["landmark_prefixes"] == [5, 8, 10, 14]
    assert not table["claim_allowed"].any()
    assert not table["deployment_trusted"].any()
    assert table["risk_flags"].str.len().gt(0).all()
    assert "trusted" not in set(table["trust_status"])
    assert table["activation_gate_ready"].any()
    assert (~table["activation_gate_ready"]).any()
    assert table["candidate_error_top_quartile"].any()
    assert table["duplicated_across_scenarios"].any()
    concentration = summary["improvement_concentration"]
    assert concentration["focus_scenario_occurrence_count"] == 2
    assert concentration["focus_share_of_primary_total_gain_fraction"] > 0.9
    assert concentration["focus_share_of_unseen_temperature_gain_fraction"] > 0.8
    assert phase1_audit.summary["model_validation_status"] == "not_confirmed"
    assert phase1_audit.summary["audit_execution_status"] == "passed"
