from __future__ import annotations

from pathlib import Path

from lifetwin.validation.independent_intake import (
    compile_independent_lfp_intake,
    load_independent_candidate_config,
    load_independent_lfp_intake,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTAKE_PATH = PROJECT_ROOT / "configs/validation/calce_pln_storage_intake_v1.json"


def test_calce_pln_intake_is_honestly_blocked_from_lfp_confirmation() -> None:
    candidate = load_independent_candidate_config()
    intake = load_independent_lfp_intake(INTAKE_PATH, candidate=candidate)
    report, protocol = compile_independent_lfp_intake(intake, candidate)

    assert intake["dataset"]["cathode_chemistry"] == "LiCoO2"
    assert intake["dataset"]["anode_chemistry"] == "graphite"
    assert intake["structure_audit"]["maximum_observed_duration_days"] == 254
    assert intake["structure_audit"]["observed_physical_cell_count"] == 144
    assert (
        intake["structure_audit"]["observed_independent_scoring_cluster_count"] == 117
    )
    assert intake["outcome_exposure"]["classification"] == "development_only"

    assert report["readiness_status"] == "development_only_not_confirmation"
    assert report["maximum_evidence_tier_after_valid_execution"] == (
        "D0_ineligible_or_blocked"
    )
    assert report["allowed_claim_role_after_valid_execution"] == (
        "hypothesis_generation_only"
    )
    assert report["protocol_can_be_frozen_now"] is False
    assert {
        "data_license_not_explicit",
        "requested_use_not_confirmed",
        "rights_basis_missing",
        "cathode_not_lfp",
        "metadata_only_audit_not_preserved",
        "duration_below_730_days",
        "prefix_support_below_four",
        "future_support_below_two",
        "future_to_landmark_ratio_below_two",
        "outcome_values_exposed_before_freeze",
    }.issubset(report["failure_reasons"])
    assert protocol["status"] == "draft"
    assert protocol["eligibility"]["observed_result"] == "ineligible"
    assert protocol["eligibility"]["evidence_tier"] == ("D0_ineligible_or_blocked")


def test_calce_pln_artifact_inventory_is_complete_and_hash_shaped() -> None:
    intake = load_independent_lfp_intake(INTAKE_PATH)
    artifacts = intake["dataset"]["artifacts"]

    assert {artifact["logical_name"] for artifact in artifacts} == {
        "PLN_Number_SOC_Temp_StoragePeriod.zip",
        "Capacity Characterization_Initialization.zip",
        "Capacity_-40C.zip",
        "Capacity_-5C.zip",
        "Capacity_25C.zip",
        "Capacity_50C.zip",
    }
    assert all(artifact["byte_size"] > 0 for artifact in artifacts)
    assert all(len(artifact["sha256"]) == 64 for artifact in artifacts)
