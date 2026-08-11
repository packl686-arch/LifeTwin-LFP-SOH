from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifetwin.validation.dataset_evidence_matrix import (
    DatasetEvidenceMatrixError,
    validate_dataset_evidence_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "configs/validation/dataset_evidence_matrix_2026_08.json"


def _matrix() -> dict[str, object]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_frozen_dataset_evidence_matrix_is_valid_and_hithium_only_is_independent() -> (
    None
):
    matrix = validate_dataset_evidence_matrix(_matrix())
    independent = [
        item["dataset_id"]
        for item in matrix["datasets"]
        if item["independent_confirmation_eligible"]
    ]
    assert independent == ["hithium_private_cycle_future_v1"]


def test_outcome_exposed_dataset_cannot_be_reclassified_as_independent() -> None:
    matrix = _matrix()
    matrix["datasets"][0]["independent_confirmation_eligible"] = True
    with pytest.raises(DatasetEvidenceMatrixError, match="Outcome-exposed"):
        validate_dataset_evidence_matrix(matrix)


def test_unclear_rights_cannot_authorize_new_model_training() -> None:
    matrix = _matrix()
    snl = next(
        item
        for item in matrix["datasets"]
        if item["dataset_id"] == "snl_battery_archive_lfp_v1"
    )
    snl["allowed_actions"].append("new_model_training")
    with pytest.raises(DatasetEvidenceMatrixError, match="confirmed data rights"):
        validate_dataset_evidence_matrix(matrix)


def test_exposed_dataset_cannot_be_called_a_locked_test() -> None:
    matrix = _matrix()
    matrix["datasets"][3]["allowed_actions"].append("locked_test")
    with pytest.raises(DatasetEvidenceMatrixError, match="locked test"):
        validate_dataset_evidence_matrix(matrix)


def test_global_evidence_firewall_cannot_be_weakened() -> None:
    matrix = _matrix()
    matrix["global_rules"]["oracle_schedule_primary_evidence_eligible"] = True
    with pytest.raises(DatasetEvidenceMatrixError, match="weakened"):
        validate_dataset_evidence_matrix(matrix)


def test_author_request_outcomes_are_not_treated_as_available_data() -> None:
    matrix = validate_dataset_evidence_matrix(_matrix())
    by_id = {item["dataset_id"]: item for item in matrix["datasets"]}

    vachenauer = by_id["vachenauer_ten_year_lfp_endpoint_v1"]
    assert vachenauer["evidence_role"] == "access_blocked_unavailable"
    assert "author_data_request" not in vachenauer["allowed_actions"]
    assert "model_training" in vachenauer["prohibited_claims"]

    yagci = by_id["yagci_stationary_storage_lfp_v1"]
    assert "primary_contact_bounced" in yagci["access_state"]
    assert "alternate_contact_discovery" in yagci["allowed_actions"]
    assert "model_training" in yagci["prohibited_claims"]
