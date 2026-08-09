from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from lifetwin.validation.independent_intake import (
    CANDIDATE_CONFIG_SHA256,
    IndependentLFPIntakeError,
    canonical_json_sha256,
    compile_independent_lfp_intake,
    load_independent_candidate_config,
    load_independent_lfp_intake,
    validate_independent_lfp_intake,
)
from lifetwin.validation.long_term_protocol import (
    validate_independent_long_term_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = (
    PROJECT_ROOT / "configs/validation/independent_safe_hard_candidate_v1.json"
)
INTAKE_TEMPLATE_PATH = (
    PROJECT_ROOT / "configs/validation/independent_lfp_dataset_intake.template.json"
)


def _template() -> dict[str, object]:
    return json.loads(INTAKE_TEMPLATE_PATH.read_text(encoding="utf-8"))


def _ready_intake() -> dict[str, object]:
    payload = _template()
    payload.update(
        {
            "intake_id": "independent_example_intake_v1",
            "created_at_utc": "2026-08-04T02:00:00Z",
        }
    )
    payload["dataset"].update(
        {
            "dataset_id": "independent_lfp_example_v1",
            "title": "Independent LFP example",
            "doi_or_persistent_url": "https://doi.org/10.0000/example",
            "repository_url": "https://example.org/datasets/lfp-v1",
            "repository_version": "v1.0.0",
            "access_mode": "public_download",
            "cathode_chemistry": "LFP",
            "anode_chemistry": "graphite",
            "physical_cell_id_field": "cell_id",
            "artifacts": [
                {
                    "logical_name": "observations.csv",
                    "source_url": "https://example.org/datasets/lfp-v1/data.csv",
                    "repository_version": "v1.0.0",
                    "byte_size": 1024,
                    "sha256": "a" * 64,
                    "retrieved_at_utc": "2026-08-04T01:00:00Z",
                }
            ],
        }
    )
    payload["dataset"]["data_license"].update(
        {
            "status": "explicit",
            "identifier": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "scope_note": "The machine-readable license permits the requested research use.",
        }
    )
    payload["structure_audit"].update(
        {
            "machine_readable_observations_verified": True,
            "physical_cell_ids_verified": True,
            "calendar_aging_separable_verified": True,
            "maximum_observed_duration_days": 1000.0,
            "observed_physical_cell_count": 32,
            "observed_independent_scoring_cluster_count": 8,
            "minimum_observed_positive_prefix_observations": 4,
            "minimum_observed_future_observations": 4,
            "minimum_observed_future_to_landmark_time_ratio": 2.0,
            "partition_identifiers_available": True,
            "candidate_landmarks_derived_without_outcome_values": True,
        }
    )
    payload["outcome_exposure"].update(
        {
            "classification": "public_but_project_blind",
            "classification_reason": (
                "Only source metadata and field names were inspected before this intake."
            ),
            "target_outcome_exposure_log": [
                {
                    "timestamp_utc": "2026-08-04T01:30:00Z",
                    "actor": "Jincheng Liu",
                    "material_accessed": "license and schema metadata",
                    "target_values_exposed": False,
                }
            ],
        }
    )
    rights = payload["data_rights_confirmation"]
    rights["basis"] = "machine_readable_license"
    for name, requested in payload["requested_use"].items():
        rights[f"{name}_confirmed"] = bool(requested)
    return payload


def test_candidate_nomination_is_semantically_frozen(tmp_path: Path) -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    assert canonical_json_sha256(candidate) == CANDIDATE_CONFIG_SHA256
    assert candidate["selection_provenance"][
        "selected_after_development_outcomes_were_exposed"
    ]
    assert candidate["selection_provenance"]["independent_confirmation_claim"] is False
    assert (
        candidate["fixed_structure"]["decision_rule"]["continuous_mixture_is_primary"]
        is False
    )

    attacked = deepcopy(candidate)
    attacked["fixed_structure"]["safe_pool"]["maximum_relative_iae_vs_persistence"] = (
        1.5
    )
    attacked_path = tmp_path / "attacked-candidate.json"
    attacked_path.write_text(json.dumps(attacked), encoding="utf-8")
    with pytest.raises(IndependentLFPIntakeError, match="config changed"):
        load_independent_candidate_config(attacked_path)
    with pytest.raises(IndependentLFPIntakeError, match="config changed"):
        validate_independent_lfp_intake(_template(), attacked)


def test_placeholder_intake_compiles_to_a_blocked_valid_draft() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = load_independent_lfp_intake(INTAKE_TEMPLATE_PATH, candidate=candidate)
    report, draft = compile_independent_lfp_intake(intake, candidate)
    assert report["readiness_status"] == "blocked_before_dataset_specific_freeze"
    assert report["protocol_can_be_frozen_now"] is False
    assert report["compiler_reads_measurement_values"] is False
    assert "data_not_acquired" in report["failure_reasons"]
    assert "data_license_not_explicit" in report["failure_reasons"]
    assert draft["status"] == "draft"
    assert draft["eligibility"]["observed_result"] == "ineligible"
    assert draft["outcome_blindness"]["classification"] == "unclassifiable"
    assert draft["claim_boundaries"]["allowed_claims"] == []
    assert validate_independent_long_term_protocol(draft) == draft


def test_qualified_metadata_reaches_manual_freeze_review_but_never_auto_freezes() -> (
    None
):
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = validate_independent_lfp_intake(_ready_intake(), candidate)
    first_report, first_draft = compile_independent_lfp_intake(intake, candidate)
    second_report, second_draft = compile_independent_lfp_intake(intake, candidate)
    assert first_report == second_report
    assert first_draft == second_draft
    assert first_report["failure_reasons"] == []
    assert first_report["readiness_status"] == (
        "ready_for_dataset_specific_freeze_review"
    )
    assert first_report["maximum_evidence_tier_after_valid_execution"] == (
        "D4_locked_external_trajectory"
    )
    assert first_report["manual_second_person_review_required"] is True
    assert first_report["protocol_can_be_frozen_now"] is False
    assert first_draft["status"] == "draft"
    assert first_draft["eligibility"]["observed_result"] == "pending"
    assert first_draft["eligibility"]["evidence_tier"] == (
        "D4_locked_external_trajectory"
    )
    assert first_draft["outcome_blindness"]["classification"] == (
        "public_but_project_blind"
    )
    assert first_draft["claim_boundaries"]["allowed_claims"] == [
        "public_data_project_blind_locked_confirmation"
    ]
    assert first_draft["candidate"]["fit_partition"] == "training_only"
    assert first_draft["candidate"]["hyperparameters_frozen"] is False
    assert first_draft["candidate"]["config_sha256"] == CANDIDATE_CONFIG_SHA256


def test_outcome_exposure_cannot_be_mislabeled_as_project_blind() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    attacked = _ready_intake()
    attacked["structure_audit"]["outcome_values_inspected"] = True
    report, draft = compile_independent_lfp_intake(attacked, candidate)
    assert report["readiness_status"] == "blocked_outcome_evidence_classification"
    assert "outcome_blindness_claim_inconsistent" in report["failure_reasons"]
    assert "outcome_values_exposed_before_freeze" in report["failure_reasons"]
    assert "outcomes_exposed_for_development" in draft["eligibility"]["failure_reasons"]
    assert draft["outcome_blindness"]["classification"] == "unclassifiable"
    assert draft["claim_boundaries"]["allowed_claims"] == []


def test_unknown_measurement_payload_and_invalid_hash_fail_closed() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    unknown = _ready_intake()
    unknown["target_capacity_values"] = [100.0, 99.0]
    with pytest.raises(IndependentLFPIntakeError, match="unknown"):
        validate_independent_lfp_intake(unknown, candidate)

    invalid_hash = _ready_intake()
    invalid_hash["dataset"]["artifacts"][0]["sha256"] = "not-a-hash"
    with pytest.raises(IndependentLFPIntakeError, match="SHA-256"):
        validate_independent_lfp_intake(invalid_hash, candidate)


def test_rights_scope_is_checked_separately_from_license_label() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = _ready_intake()
    intake["data_rights_confirmation"]["model_training_and_evaluation_confirmed"] = (
        False
    )
    report, _ = compile_independent_lfp_intake(intake, candidate)
    assert "requested_use_not_confirmed" in report["failure_reasons"]
    assert report["readiness_status"] == "blocked_before_dataset_specific_freeze"

    forbidden_scope = _ready_intake()
    forbidden_scope["requested_use"]["raw_data_redistribution"] = True
    with pytest.raises(IndependentLFPIntakeError, match="without raw redistribution"):
        validate_independent_lfp_intake(forbidden_scope, candidate)


def test_written_permission_requires_a_record_hash() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = _ready_intake()
    intake["dataset"]["data_license"]["status"] = "permission_granted"
    intake["data_rights_confirmation"]["basis"] = "written_permission"
    report, _ = compile_independent_lfp_intake(intake, candidate)
    assert "permission_record_hash_missing" in report["failure_reasons"]

    intake["data_rights_confirmation"]["permission_record_sha256"] = "b" * 64
    report, _ = compile_independent_lfp_intake(intake, candidate)
    assert report["failure_reasons"] == []


def test_structural_counts_are_integers_and_artifact_identities_are_unique() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    fractional = _ready_intake()
    fractional["structure_audit"]["observed_physical_cell_count"] = 8.5
    with pytest.raises(IndependentLFPIntakeError, match="integer or null"):
        validate_independent_lfp_intake(fractional, candidate)

    duplicate = _ready_intake()
    duplicate["dataset"]["artifacts"].append(
        deepcopy(duplicate["dataset"]["artifacts"][0])
    )
    with pytest.raises(IndependentLFPIntakeError, match="logical_name.*unique"):
        validate_independent_lfp_intake(duplicate, candidate)


def test_explicit_license_and_rights_basis_must_be_auditable() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    missing_license_identity = _ready_intake()
    missing_license_identity["dataset"]["data_license"]["identifier"] = None
    with pytest.raises(IndependentLFPIntakeError, match="requires identifier and URL"):
        validate_independent_lfp_intake(missing_license_identity, candidate)

    inconsistent = _ready_intake()
    inconsistent["dataset"]["data_license"]["status"] = "permission_granted"
    report, draft = compile_independent_lfp_intake(inconsistent, candidate)
    assert "rights_basis_inconsistent" in report["failure_reasons"]
    assert "permission_absent" in draft["eligibility"]["failure_reasons"]


def test_development_only_intake_does_not_offer_confirmation_freeze_steps() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = _ready_intake()
    intake["outcome_exposure"]["classification"] = "development_only"
    intake["outcome_exposure"]["classification_reason"] = (
        "The target outcomes were used during method development."
    )
    intake["outcome_exposure"]["target_outcome_exposure_log"][0][
        "target_values_exposed"
    ] = True
    intake["structure_audit"]["outcome_values_inspected"] = True
    report, draft = compile_independent_lfp_intake(intake, candidate)
    assert report["readiness_status"] == "development_only_not_confirmation"
    assert report["next_required_actions"] == [
        "retain_this_dataset_for_development_evidence_only",
        "use_a_new_unexposed_dataset_for_confirmation",
    ]
    assert draft["outcome_blindness"]["classification"] == "unclassifiable"


def test_exposed_outcomes_can_reach_locked_retrospective_freeze_review() -> None:
    candidate = load_independent_candidate_config(CANDIDATE_PATH)
    intake = _ready_intake()
    intake["outcome_exposure"]["classification"] = (
        "locked_retrospective_replication"
    )
    intake["outcome_exposure"]["classification_reason"] = (
        "Target outcomes were exposed before the dataset-specific protocol was frozen; "
        "the candidate and scorer are now locked for retrospective replication only."
    )
    intake["outcome_exposure"]["target_outcome_exposure_log"][0][
        "target_values_exposed"
    ] = True
    intake["structure_audit"]["outcome_values_inspected"] = True

    report, draft = compile_independent_lfp_intake(intake, candidate)

    assert report["failure_reasons"] == []
    assert report["readiness_status"] == (
        "ready_for_locked_retrospective_freeze_review"
    )
    assert report["maximum_evidence_tier_after_valid_execution"] == (
        "D3_long_horizon_trajectory_retrospective"
    )
    assert report["allowed_claim_role_after_valid_execution"] == (
        "retrospective_replication"
    )
    assert draft["outcome_blindness"]["classification"] == (
        "locked_retrospective_replication"
    )
    assert draft["claim_boundaries"]["allowed_claims"] == [
        "locked_retrospective_replication"
    ]


def test_intake_cli_blocks_template_and_refuses_overwrite(tmp_path: Path) -> None:
    output_directory = tmp_path / "compiled"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/compile_independent_lfp_intake.py"),
        str(INTAKE_TEMPLATE_PATH),
        "--output-directory",
        str(output_directory),
    ]
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
    first = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0
    assert json.loads(first.stdout)["status"] == "blocked_as_designed"
    assert (output_directory / "intake_report.json").is_file()
    assert (output_directory / "protocol_draft.json").is_file()

    second = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 1
    assert json.loads(second.stdout)["error"].startswith("Refusing to overwrite")
