from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/experiments/v5_rcgp_literature_informed_development.json"


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def test_v5_plan_remains_development_only_and_future_label_free() -> None:
    plan = _plan()
    boundary = plan["evidence_boundary"]
    firewall = plan["data_firewall"]

    assert "development_plan" in plan["status"]
    assert boundary["eligible_for_existing_hithium_locked_test"] is False
    assert boundary["hithium_measurements_accessed"] is False
    assert boundary["fifteen_to_twenty_five_year_accuracy_claim_permitted"] is False
    assert boundary["realized_future_schedule_primary_evidence_eligible"] is False
    assert firewall["target_suffix_available_during_prediction"] is False
    assert firewall["target_suffix_available_during_hyperparameter_selection"] is False
    assert firewall["same_cell_rows_across_train_and_test_permitted"] is False
    assert firewall["random_row_split_permitted"] is False
    assert firewall["locked_test_available_to_active_acquisition"] is False


def test_v5_plan_prioritizes_small_data_models_before_deep_models() -> None:
    plan = _plan()
    components = plan["candidate_components"]
    selection = plan["model_selection"]

    assert components["pairwise_reference_residual"]["deep_siamese_or_cnn_in_phase_one"] is False
    assert components["pairwise_reference_residual"]["phase_one_models"] == [
        "ridge",
        "huber_or_robust_ridge",
        "extra_trees",
        "hist_gradient_boosting",
    ]
    assert selection["deep_model_minimum_training_cells"] >= 300
    assert selection["deep_model_seed_count"] >= 10
    assert selection["single_best_seed_reporting_permitted"] is False


def test_v5_plan_fails_closed_for_mechanisms_and_future_conditions() -> None:
    plan = _plan()
    components = plan["candidate_components"]
    gates = plan["gates"]

    assert components["frozen_center"][
        "candidate_branches_must_be_exact_zero_when_unsupported"
    ] is True
    assert components["mechanism_gate"]["enabled_by_default"] is False
    assert components["mechanism_gate"]["capacity_only_mechanism_names_permitted"] is False
    assert gates["H3_mechanism_gate"]["fallback_when_evidence_below_L1"] == (
        "disable_mechanism_gate"
    )
    assert gates["H4_future_conditions"]["oracle_primary_evidence_eligible"] is False
    assert components["active_experiment_selection"]["enabled_for_locked_test"] is False
    assert components["active_experiment_selection"][
        "acquisition_pool_and_locked_test_disjoint"
    ] is True


def test_v5_plan_has_falsifiable_accuracy_uncertainty_and_stopping_gates() -> None:
    plan = _plan()
    gates = plan["gates"]
    stopping = plan["stopping_rules"]

    assert gates["H1_pairwise_reference"][
        "minimum_relative_trajectory_mae_improvement"
    ] == 0.05
    assert gates["H1_pairwise_reference"][
        "paired_cell_bootstrap_delta_mae_upper_bound_must_be_below_pp"
    ] == 0.0
    assert gates["H2_gp_uncertainty"]["minimum_overall_90_interval_coverage"] >= 0.87
    assert gates["H2_gp_uncertainty"][
        "maximum_weighted_interval_score_ratio_vs_existing"
    ] <= 1.0
    assert gates["H5_active_experiment_selection"]["locked_test_query_permitted"] is False
    assert stopping["maximum_failed_outer_cv_rounds_before_branch_termination"] == 2
    assert stopping["same_exposed_cohort_architecture_search_after_termination_permitted"] is False
