"""Machine-readable evidence-role firewall for battery datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Mapping


EVIDENCE_MATRIX_SCHEMA = "lifetwin.dataset_evidence_matrix.v1"
DATASET_FIELDS = {
    "dataset_id",
    "title",
    "chemistry",
    "aging_mode",
    "access_state",
    "data_rights_state",
    "outcome_exposure",
    "evidence_role",
    "independent_confirmation_eligible",
    "allowed_actions",
    "prohibited_claims",
    "source_record",
}
OUTCOME_EXPOSURES = {
    "not_accessed",
    "outcome_exposed_development",
    "outcome_exposed_external_stress",
    "metadata_only_inspected",
}
EVIDENCE_ROLES = {
    "retrospective_method_development",
    "retrospective_external_stress",
    "software_portability_stress",
    "ineligible_chemistry_duration_screen",
    "permission_pending_unclassified",
    "access_blocked_unavailable",
    "future_enterprise_blind_validation",
}


class DatasetEvidenceMatrixError(ValueError):
    """Raised when evidence roles permit outcome or claim laundering."""


def _strings(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise DatasetEvidenceMatrixError(f"{label} must be a non-empty list")
    result = [str(item) for item in value]
    if any(not item.strip() for item in result) or len(result) != len(set(result)):
        raise DatasetEvidenceMatrixError(f"{label} contains empty or duplicate values")
    return result


def validate_dataset_evidence_matrix(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Validate dataset roles before any result is used as project evidence."""
    matrix = deepcopy(dict(value))
    expected_top_level = {
        "schema_version",
        "matrix_id",
        "frozen_date",
        "author",
        "project_outcome_exposure_state",
        "datasets",
        "global_rules",
    }
    if set(matrix) != expected_top_level:
        raise DatasetEvidenceMatrixError("Evidence matrix top-level fields changed")
    if matrix["schema_version"] != EVIDENCE_MATRIX_SCHEMA:
        raise DatasetEvidenceMatrixError("Evidence matrix schema changed")
    for field in ("matrix_id", "frozen_date", "author"):
        if not str(matrix[field]).strip():
            raise DatasetEvidenceMatrixError(f"{field} is empty")
    datasets = matrix["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise DatasetEvidenceMatrixError("Evidence matrix has no datasets")
    identities: set[str] = set()
    for raw in datasets:
        if not isinstance(raw, Mapping) or set(raw) != DATASET_FIELDS:
            raise DatasetEvidenceMatrixError("Dataset evidence fields changed")
        dataset_id = str(raw["dataset_id"])
        if not dataset_id or dataset_id in identities:
            raise DatasetEvidenceMatrixError(
                "Dataset identities are empty or duplicated"
            )
        identities.add(dataset_id)
        exposure = str(raw["outcome_exposure"])
        role = str(raw["evidence_role"])
        if exposure not in OUTCOME_EXPOSURES or role not in EVIDENCE_ROLES:
            raise DatasetEvidenceMatrixError("Dataset evidence enum is unsupported")
        independent = raw["independent_confirmation_eligible"]
        if not isinstance(independent, bool):
            raise DatasetEvidenceMatrixError(
                "Independent-confirmation eligibility must be boolean"
            )
        if independent and (
            exposure != "not_accessed" or role != "future_enterprise_blind_validation"
        ):
            raise DatasetEvidenceMatrixError(
                "Outcome-exposed data cannot be independent confirmation"
            )
        actions = _strings(raw["allowed_actions"], f"{dataset_id} allowed_actions")
        _strings(raw["prohibited_claims"], f"{dataset_id} prohibited_claims")
        rights = str(raw["data_rights_state"])
        if rights != "confirmed_for_requested_use" and "new_model_training" in actions:
            raise DatasetEvidenceMatrixError(
                "New model training requires confirmed data rights"
            )
        if exposure != "not_accessed" and "locked_test" in actions:
            raise DatasetEvidenceMatrixError(
                "Outcome-exposed data cannot be relabeled as a locked test"
            )
        if not str(raw["source_record"]).strip():
            raise DatasetEvidenceMatrixError("Dataset source record is empty")
    rules = matrix["global_rules"]
    if not isinstance(rules, Mapping) or set(rules) != {
        "outcome_exposed_data_can_raise_evidence_grade",
        "metadata_only_review_counts_as_outcome_access",
        "oracle_schedule_primary_evidence_eligible",
        "private_raw_data_public_release_permitted",
        "locked_test_open_count",
    }:
        raise DatasetEvidenceMatrixError("Evidence matrix global rules changed")
    if (
        rules["outcome_exposed_data_can_raise_evidence_grade"] is not False
        or rules["metadata_only_review_counts_as_outcome_access"] is not False
        or rules["oracle_schedule_primary_evidence_eligible"] is not False
        or rules["private_raw_data_public_release_permitted"] is not False
        or rules["locked_test_open_count"] != 1
    ):
        raise DatasetEvidenceMatrixError("Evidence firewall global rule weakened")
    return matrix


__all__ = [
    "EVIDENCE_MATRIX_SCHEMA",
    "DatasetEvidenceMatrixError",
    "validate_dataset_evidence_matrix",
]
