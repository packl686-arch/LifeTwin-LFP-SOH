"""Metadata-only intake firewall for a future independent LFP dataset."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from lifetwin.validation.long_term_protocol import (
    validate_independent_long_term_protocol,
)


INTAKE_SCHEMA_VERSION = "lifetwin.independent_lfp_dataset_intake.v1"
CANDIDATE_SCHEMA_VERSION = "lifetwin.independent_safe_hard_candidate.v1"
CANDIDATE_ID = "lifetwin_safe_hard_transfer_candidate_v1"
CANDIDATE_CONFIG_SHA256 = (
    "596108e19ca0a8c7fb712bf82ca5be93817524f5f0c912f3b71b180a0fcba3af"
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANDIDATE_CONFIG = (
    PROJECT_ROOT / "configs/validation/independent_safe_hard_candidate_v1.json"
)
DEFAULT_PROTOCOL_TEMPLATE = (
    PROJECT_ROOT / "configs/validation/independent_long_term_lfp_protocol.template.json"
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "intake_id",
    "created_at_utc",
    "prepared_by",
    "candidate_id",
    "dataset",
    "structure_audit",
    "outcome_exposure",
    "data_rights_confirmation",
    "requested_use",
}
_DATASET_KEYS = {
    "dataset_id",
    "title",
    "doi_or_persistent_url",
    "repository_url",
    "repository_version",
    "access_mode",
    "paper_license",
    "data_license",
    "artifacts",
    "cathode_chemistry",
    "anode_chemistry",
    "aging_mode",
    "time_unit",
    "physical_cell_id_field",
    "condition_id_fields",
    "outcome_fields",
}
_LICENSE_KEYS = {"status", "identifier", "url", "scope_note"}
_ARTIFACT_KEYS = {
    "logical_name",
    "source_url",
    "repository_version",
    "byte_size",
    "sha256",
    "retrieved_at_utc",
}
_STRUCTURE_KEYS = {
    "metadata_only_audit",
    "outcome_values_inspected",
    "machine_readable_observations_verified",
    "physical_cell_ids_verified",
    "calendar_aging_separable_verified",
    "maximum_observed_duration_days",
    "observed_physical_cell_count",
    "observed_independent_scoring_cluster_count",
    "minimum_observed_positive_prefix_observations",
    "minimum_observed_future_observations",
    "minimum_observed_future_to_landmark_time_ratio",
    "partition_identifiers_available",
    "candidate_landmarks_derived_without_outcome_values",
}
_OUTCOME_EXPOSURE_KEYS = {
    "classification",
    "classification_reason",
    "target_outcome_exposure_log",
}
_EXPOSURE_ENTRY_KEYS = {
    "timestamp_utc",
    "actor",
    "material_accessed",
    "target_values_exposed",
}
_USE_KEYS = {
    "noncommercial_research",
    "competition_evaluation",
    "local_feature_extraction",
    "model_training_and_evaluation",
    "aggregate_metrics_and_figures",
    "derived_tables_without_raw_measurements",
    "raw_data_redistribution",
    "commercial_model_development",
}
_RIGHTS_KEYS = {
    "basis",
    "permission_record_sha256",
    "permission_record_publicly_redistributable",
    *(f"{name}_confirmed" for name in _USE_KEYS),
}
_LICENSE_STATUSES = {
    "explicit",
    "permission_granted",
    "unclear",
    "restricted",
    "not_published",
}
_ACCESS_MODES = {
    "not_acquired",
    "public_download",
    "permission_granted_copy",
    "custodian_scored",
    "restricted",
}
_AGING_MODES = {
    "controlled_calendar",
    "controlled_calendar_endpoint",
    "mixed_but_calendar_separable",
    "mixed_not_separable",
    "field_operation",
}
_OUTCOME_CLASSES = {
    "prospective_outcome_blind",
    "public_but_project_blind",
    "locked_retrospective_replication",
    "development_only",
    "unclassifiable",
}
_CLAIM_ROLE = {
    "prospective_outcome_blind": "confirmatory",
    "public_but_project_blind": (
        "locked_external_confirmation_with_public_data_caveat"
    ),
    "locked_retrospective_replication": "retrospective_replication",
    "development_only": "hypothesis_generation_only",
    "unclassifiable": "none",
}
_EVIDENCE_TIER = {
    "prospective_outcome_blind": "D5_prospective_operational",
    "public_but_project_blind": "D4_locked_external_trajectory",
    "locked_retrospective_replication": ("D3_long_horizon_trajectory_retrospective"),
    "development_only": "D0_ineligible_or_blocked",
    "unclassifiable": "D0_ineligible_or_blocked",
}
_PROTOCOL_FAILURE_MAP = {
    "data_license_not_explicit": "license_unclear",
    "requested_use_not_confirmed": "permission_absent",
    "rights_basis_inconsistent": "permission_absent",
    "cathode_not_lfp": "cathode_not_lfp",
    "anode_not_graphite": "anode_not_graphite",
    "physical_cell_ids_not_verified": "physical_cell_ids_absent",
    "duration_below_730_days": "duration_below_730_days",
    "calendar_aging_not_separable": "calendar_aging_not_separable",
    "machine_readable_observations_not_verified": (
        "machine_readable_observations_absent"
    ),
    "prefix_support_below_four": "prefix_too_short",
    "future_support_below_two": "future_support_too_short",
    "future_to_landmark_ratio_below_two": "future_support_too_short",
    "endpoint_only": "endpoint_only",
    "outcome_values_exposed_before_freeze": "outcomes_exposed_for_development",
    "physical_cell_count_below_eight": "statistical_units_insufficient",
    "independent_cluster_count_below_eight": "statistical_units_insufficient",
    "artifact_inventory_missing": "hashable_version_absent",
    "artifact_record_invalid": "hashable_version_absent",
}


class IndependentLFPIntakeError(ValueError):
    """Raised when an intake record violates the metadata-only contract."""


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IndependentLFPIntakeError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                IndependentLFPIntakeError(
                    f"Non-finite JSON constant is prohibited: {token}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentLFPIntakeError(f"Cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise IndependentLFPIntakeError("JSON root must be an object")
    return value


def _exact_keys(value: object, expected: set[str], *, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise IndependentLFPIntakeError(f"{path} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise IndependentLFPIntakeError(
            f"{path} keys changed: missing={missing}, unknown={unknown}"
        )
    return dict(value)


def _strict_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise IndependentLFPIntakeError(f"{path} must be a boolean")
    return value


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise IndependentLFPIntakeError(f"{path} must be a canonical non-empty string")
    return value


def _nullable_number(value: object, *, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IndependentLFPIntakeError(f"{path} must be numeric or null")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise IndependentLFPIntakeError(f"{path} must be finite and non-negative")
    return converted


def _nullable_integer(value: object, *, path: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IndependentLFPIntakeError(f"{path} must be an integer or null")
    return value


def _utc_timestamp(value: object, *, path: str, placeholders_allowed: bool) -> str:
    text = _string(value, path=path)
    if placeholders_allowed and text.startswith("REPLACE_"):
        return text
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IndependentLFPIntakeError(
            f"{path} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise IndependentLFPIntakeError(f"{path} must include the UTC timezone")
    return text


def _uri(value: object, *, path: str) -> str:
    text = _string(value, path=path)
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise IndependentLFPIntakeError(f"{path} must be an HTTP(S) URI")
    return text


def _sha256(value: object, *, path: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    text = _string(value, path=path)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise IndependentLFPIntakeError(f"{path} must be a lowercase SHA-256")
    return text


def _validate_candidate_config(candidate: Mapping[str, object]) -> dict[str, object]:
    parsed = dict(candidate)
    if canonical_json_sha256(parsed) != CANDIDATE_CONFIG_SHA256:
        raise IndependentLFPIntakeError("Independent candidate config changed")
    if parsed.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise IndependentLFPIntakeError("Independent candidate schema changed")
    if parsed.get("candidate_id") != CANDIDATE_ID:
        raise IndependentLFPIntakeError("Independent candidate identity changed")
    if parsed.get("status") != "nominated_before_next_independent_dataset":
        raise IndependentLFPIntakeError("Independent candidate status changed")
    return deepcopy(parsed)


def load_independent_candidate_config(
    path: str | Path = DEFAULT_CANDIDATE_CONFIG,
) -> dict[str, object]:
    return _validate_candidate_config(_load_json(path))


def validate_independent_lfp_intake(
    intake: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    frozen_candidate = _validate_candidate_config(candidate)
    parsed = _exact_keys(intake, _TOP_LEVEL_KEYS, path="$")
    if parsed["schema_version"] != INTAKE_SCHEMA_VERSION:
        raise IndependentLFPIntakeError("$.schema_version changed")
    _string(parsed["intake_id"], path="$.intake_id")
    _utc_timestamp(
        parsed["created_at_utc"],
        path="$.created_at_utc",
        placeholders_allowed=True,
    )
    _string(parsed["prepared_by"], path="$.prepared_by")
    if parsed["candidate_id"] != frozen_candidate["candidate_id"]:
        raise IndependentLFPIntakeError("$.candidate_id does not match the candidate")

    dataset = _exact_keys(parsed["dataset"], _DATASET_KEYS, path="$.dataset")
    for field in (
        "dataset_id",
        "title",
        "repository_version",
        "cathode_chemistry",
        "anode_chemistry",
        "physical_cell_id_field",
    ):
        _string(dataset[field], path=f"$.dataset.{field}")
    _uri(dataset["doi_or_persistent_url"], path="$.dataset.doi_or_persistent_url")
    _uri(dataset["repository_url"], path="$.dataset.repository_url")
    if dataset["access_mode"] not in _ACCESS_MODES:
        raise IndependentLFPIntakeError("$.dataset.access_mode is unsupported")
    if dataset["aging_mode"] not in _AGING_MODES:
        raise IndependentLFPIntakeError("$.dataset.aging_mode is unsupported")
    if dataset["time_unit"] not in {"day", "hour", "week"}:
        raise IndependentLFPIntakeError("$.dataset.time_unit is unsupported")
    for list_name in ("condition_id_fields", "outcome_fields"):
        values = dataset[list_name]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(item, str) or not item.strip() for item in values)
            or len(values) != len(set(values))
        ):
            raise IndependentLFPIntakeError(
                f"$.dataset.{list_name} must contain unique field names"
            )
    for license_name in ("paper_license", "data_license"):
        license_value = _exact_keys(
            dataset[license_name],
            _LICENSE_KEYS,
            path=f"$.dataset.{license_name}",
        )
        if license_value["status"] not in _LICENSE_STATUSES:
            raise IndependentLFPIntakeError(
                f"$.dataset.{license_name}.status is unsupported"
            )
        if license_value["identifier"] is not None:
            _string(
                license_value["identifier"],
                path=f"$.dataset.{license_name}.identifier",
            )
        if license_value["url"] is not None:
            _uri(license_value["url"], path=f"$.dataset.{license_name}.url")
        if license_value["status"] == "explicit" and (
            license_value["identifier"] is None or license_value["url"] is None
        ):
            raise IndependentLFPIntakeError(
                f"$.dataset.{license_name} explicit status requires identifier and URL"
            )
        _string(
            license_value["scope_note"],
            path=f"$.dataset.{license_name}.scope_note",
        )
    artifacts = dataset["artifacts"]
    if not isinstance(artifacts, list):
        raise IndependentLFPIntakeError("$.dataset.artifacts must be an array")
    artifact_names: set[str] = set()
    artifact_sources: set[tuple[str, str]] = set()
    for index, artifact in enumerate(artifacts):
        item = _exact_keys(
            artifact,
            _ARTIFACT_KEYS,
            path=f"$.dataset.artifacts[{index}]",
        )
        logical_name = _string(
            item["logical_name"], path=f"$.dataset.artifacts[{index}].logical_name"
        )
        source_url = _uri(
            item["source_url"], path=f"$.dataset.artifacts[{index}].source_url"
        )
        repository_version = _string(
            item["repository_version"],
            path=f"$.dataset.artifacts[{index}].repository_version",
        )
        if logical_name in artifact_names:
            raise IndependentLFPIntakeError(
                "$.dataset.artifacts logical_name values must be unique"
            )
        source_identity = (source_url, repository_version)
        if source_identity in artifact_sources:
            raise IndependentLFPIntakeError(
                "$.dataset.artifacts source URL/version pairs must be unique"
            )
        artifact_names.add(logical_name)
        artifact_sources.add(source_identity)
        if (
            isinstance(item["byte_size"], bool)
            or not isinstance(item["byte_size"], int)
            or item["byte_size"] <= 0
        ):
            raise IndependentLFPIntakeError(
                f"$.dataset.artifacts[{index}].byte_size must be positive"
            )
        _sha256(item["sha256"], path=f"$.dataset.artifacts[{index}].sha256")
        _utc_timestamp(
            item["retrieved_at_utc"],
            path=f"$.dataset.artifacts[{index}].retrieved_at_utc",
            placeholders_allowed=False,
        )

    structure = _exact_keys(
        parsed["structure_audit"],
        _STRUCTURE_KEYS,
        path="$.structure_audit",
    )
    for field in _STRUCTURE_KEYS - {
        "maximum_observed_duration_days",
        "observed_physical_cell_count",
        "observed_independent_scoring_cluster_count",
        "minimum_observed_positive_prefix_observations",
        "minimum_observed_future_observations",
        "minimum_observed_future_to_landmark_time_ratio",
    }:
        _strict_bool(structure[field], path=f"$.structure_audit.{field}")
    _nullable_number(
        structure["maximum_observed_duration_days"],
        path="$.structure_audit.maximum_observed_duration_days",
    )
    _nullable_number(
        structure["minimum_observed_future_to_landmark_time_ratio"],
        path=("$.structure_audit.minimum_observed_future_to_landmark_time_ratio"),
    )
    for field in (
        "observed_physical_cell_count",
        "observed_independent_scoring_cluster_count",
        "minimum_observed_positive_prefix_observations",
        "minimum_observed_future_observations",
    ):
        _nullable_integer(structure[field], path=f"$.structure_audit.{field}")

    exposure = _exact_keys(
        parsed["outcome_exposure"],
        _OUTCOME_EXPOSURE_KEYS,
        path="$.outcome_exposure",
    )
    if exposure["classification"] not in _OUTCOME_CLASSES:
        raise IndependentLFPIntakeError(
            "$.outcome_exposure.classification is unsupported"
        )
    _string(
        exposure["classification_reason"],
        path="$.outcome_exposure.classification_reason",
    )
    log = exposure["target_outcome_exposure_log"]
    if not isinstance(log, list):
        raise IndependentLFPIntakeError(
            "$.outcome_exposure.target_outcome_exposure_log must be an array"
        )
    for index, entry in enumerate(log):
        item = _exact_keys(
            entry,
            _EXPOSURE_ENTRY_KEYS,
            path=f"$.outcome_exposure.target_outcome_exposure_log[{index}]",
        )
        _utc_timestamp(
            item["timestamp_utc"],
            path=(
                f"$.outcome_exposure.target_outcome_exposure_log[{index}].timestamp_utc"
            ),
            placeholders_allowed=False,
        )
        _string(
            item["actor"],
            path=f"$.outcome_exposure.target_outcome_exposure_log[{index}].actor",
        )
        _string(
            item["material_accessed"],
            path=(
                "$.outcome_exposure.target_outcome_exposure_log"
                f"[{index}].material_accessed"
            ),
        )
        _strict_bool(
            item["target_values_exposed"],
            path=(
                "$.outcome_exposure.target_outcome_exposure_log"
                f"[{index}].target_values_exposed"
            ),
        )

    rights = _exact_keys(
        parsed["data_rights_confirmation"],
        _RIGHTS_KEYS,
        path="$.data_rights_confirmation",
    )
    if rights["basis"] not in {
        "none",
        "machine_readable_license",
        "written_permission",
        "custodian_agreement",
    }:
        raise IndependentLFPIntakeError(
            "$.data_rights_confirmation.basis is unsupported"
        )
    _sha256(
        rights["permission_record_sha256"],
        path="$.data_rights_confirmation.permission_record_sha256",
        nullable=True,
    )
    for field in _RIGHTS_KEYS - {"basis", "permission_record_sha256"}:
        _strict_bool(rights[field], path=f"$.data_rights_confirmation.{field}")

    requested = _exact_keys(
        parsed["requested_use"],
        _USE_KEYS,
        path="$.requested_use",
    )
    for field in _USE_KEYS:
        _strict_bool(requested[field], path=f"$.requested_use.{field}")
    if (
        requested["raw_data_redistribution"]
        or requested["commercial_model_development"]
    ):
        raise IndependentLFPIntakeError(
            "This intake version is restricted to noncommercial use without raw redistribution"
        )
    return deepcopy(parsed)


def load_independent_lfp_intake(
    path: str | Path,
    *,
    candidate: Mapping[str, object] | None = None,
) -> dict[str, object]:
    selected = (
        load_independent_candidate_config() if candidate is None else dict(candidate)
    )
    return validate_independent_lfp_intake(_load_json(path), selected)


def _below(value: object, threshold: float) -> bool:
    return value is None or float(value) < threshold


def _readiness_failures(intake: Mapping[str, object]) -> list[str]:
    dataset = intake["dataset"]
    structure = intake["structure_audit"]
    exposure = intake["outcome_exposure"]
    rights = intake["data_rights_confirmation"]
    requested = intake["requested_use"]
    failures: list[str] = []
    placeholder_fields = (
        intake["intake_id"],
        intake["created_at_utc"],
        dataset["dataset_id"],
        dataset["title"],
        dataset["repository_version"],
        dataset["cathode_chemistry"],
        dataset["anode_chemistry"],
        dataset["physical_cell_id_field"],
    )
    if any(str(value).startswith("REPLACE_") for value in placeholder_fields):
        failures.append("metadata_placeholders_present")
    if dataset["access_mode"] == "not_acquired":
        failures.append("data_not_acquired")
    dataset_uris = (
        dataset["doi_or_persistent_url"],
        dataset["repository_url"],
    )
    if any(
        (urlparse(str(value)).hostname or "").endswith(".invalid")
        for value in dataset_uris
    ):
        failures.append("placeholder_uri_present")
    if dataset["data_license"]["status"] not in {
        "explicit",
        "permission_granted",
    }:
        failures.append("data_license_not_explicit")
    if not dataset["artifacts"]:
        failures.append("artifact_inventory_missing")
    required_uses = [name for name, enabled in requested.items() if enabled]
    if any(not rights[f"{name}_confirmed"] for name in required_uses):
        failures.append("requested_use_not_confirmed")
    if rights["basis"] == "none":
        failures.append("rights_basis_missing")
    if (
        rights["basis"] == "machine_readable_license"
        and dataset["data_license"]["status"] != "explicit"
    ) or (
        dataset["data_license"]["status"] == "permission_granted"
        and rights["basis"] not in {"written_permission", "custodian_agreement"}
    ):
        failures.append("rights_basis_inconsistent")
    if (
        rights["basis"] in {"written_permission", "custodian_agreement"}
        and not rights["permission_record_sha256"]
    ):
        failures.append("permission_record_hash_missing")
    if str(dataset["cathode_chemistry"]).casefold() not in {
        "lfp",
        "lifepo4",
        "lithium_iron_phosphate",
    }:
        failures.append("cathode_not_lfp")
    if str(dataset["anode_chemistry"]).casefold() not in {
        "graphite",
        "natural_graphite",
        "synthetic_graphite",
    }:
        failures.append("anode_not_graphite")
    if dataset["aging_mode"] == "controlled_calendar_endpoint":
        failures.append("endpoint_only")
    if (
        dataset["aging_mode"]
        not in {
            "controlled_calendar",
            "mixed_but_calendar_separable",
        }
        or not structure["calendar_aging_separable_verified"]
    ):
        failures.append("calendar_aging_not_separable")
    if not structure["metadata_only_audit"]:
        failures.append("metadata_only_audit_not_preserved")
    if not structure["machine_readable_observations_verified"]:
        failures.append("machine_readable_observations_not_verified")
    if not structure["physical_cell_ids_verified"]:
        failures.append("physical_cell_ids_not_verified")
    if _below(structure["maximum_observed_duration_days"], 730.0):
        failures.append("duration_below_730_days")
    if _below(structure["observed_physical_cell_count"], 8.0):
        failures.append("physical_cell_count_below_eight")
    if _below(structure["observed_independent_scoring_cluster_count"], 8.0):
        failures.append("independent_cluster_count_below_eight")
    cells = structure["observed_physical_cell_count"]
    clusters = structure["observed_independent_scoring_cluster_count"]
    if cells is not None and clusters is not None and float(clusters) > float(cells):
        failures.append("independent_cluster_count_exceeds_cell_count")
    if _below(structure["minimum_observed_positive_prefix_observations"], 4.0):
        failures.append("prefix_support_below_four")
    if _below(structure["minimum_observed_future_observations"], 2.0):
        failures.append("future_support_below_two")
    if _below(structure["minimum_observed_future_to_landmark_time_ratio"], 2.0):
        failures.append("future_to_landmark_ratio_below_two")
    if not structure["partition_identifiers_available"]:
        failures.append("partition_identifiers_unavailable")
    if not structure["candidate_landmarks_derived_without_outcome_values"]:
        failures.append("outcome_free_landmarks_not_ready")
    outcome_class = exposure["classification"]
    if (
        outcome_class != "unclassifiable"
        and not exposure["target_outcome_exposure_log"]
    ):
        failures.append("outcome_exposure_log_missing")
    any_exposed = bool(structure["outcome_values_inspected"]) or any(
        bool(entry["target_values_exposed"])
        for entry in exposure["target_outcome_exposure_log"]
    )
    if (
        outcome_class
        in {
            "prospective_outcome_blind",
            "public_but_project_blind",
        }
        and any_exposed
    ):
        failures.append("outcome_blindness_claim_inconsistent")
    if any_exposed:
        failures.append("outcome_values_exposed_before_freeze")
    if outcome_class == "unclassifiable":
        failures.append("outcome_history_unclassifiable")
    return list(dict.fromkeys(failures))


def _readiness_status(failures: list[str], outcome_class: str) -> tuple[str, str, str]:
    if outcome_class == "development_only":
        return (
            "development_only_not_confirmation",
            _EVIDENCE_TIER[outcome_class],
            _CLAIM_ROLE[outcome_class],
        )
    evidence_failures = {
        "outcome_blindness_claim_inconsistent",
        "outcome_values_exposed_before_freeze",
        "outcome_exposure_log_missing",
        "outcome_history_unclassifiable",
    }
    substantive = [failure for failure in failures if failure not in evidence_failures]
    if substantive:
        return (
            "blocked_before_dataset_specific_freeze",
            "D0_ineligible_or_blocked",
            "none",
        )
    if failures:
        return (
            "blocked_outcome_evidence_classification",
            "D0_ineligible_or_blocked",
            "none",
        )
    if outcome_class == "locked_retrospective_replication":
        status = "ready_for_locked_retrospective_freeze_review"
    else:
        status = "ready_for_dataset_specific_freeze_review"
    return status, _EVIDENCE_TIER[outcome_class], _CLAIM_ROLE[outcome_class]


def _protocol_failure_reasons(failures: list[str]) -> list[str]:
    mapped = [
        _PROTOCOL_FAILURE_MAP[value]
        for value in failures
        if value in _PROTOCOL_FAILURE_MAP
    ]
    return list(dict.fromkeys(mapped))


def compile_independent_lfp_intake(
    intake: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    protocol_template_path: str | Path = DEFAULT_PROTOCOL_TEMPLATE,
) -> tuple[dict[str, object], dict[str, object]]:
    frozen_candidate = _validate_candidate_config(candidate)
    parsed = validate_independent_lfp_intake(intake, frozen_candidate)
    failures = _readiness_failures(parsed)
    outcome_class = str(parsed["outcome_exposure"]["classification"])
    readiness, evidence_tier, claim_role = _readiness_status(failures, outcome_class)
    if readiness in {
        "ready_for_dataset_specific_freeze_review",
        "ready_for_locked_retrospective_freeze_review",
    }:
        next_required_actions = [
            "freeze_physical_cell_partitions",
            "freeze_primary_landmark_and_common_future_window",
            "fit_training_only_nested_selector_parameters",
            "record_adapter_scorer_environment_and_model_hashes",
            "run_independent_protocol_validator",
            "freeze_prediction_bundle_before_truth_linkage",
        ]
    elif readiness == "development_only_not_confirmation":
        next_required_actions = [
            "retain_this_dataset_for_development_evidence_only",
            "use_a_new_unexposed_dataset_for_confirmation",
        ]
    else:
        next_required_actions = [
            "resolve_each_failure_reason_and_create_a_new_intake_version"
        ]
    report: dict[str, object] = {
        "schema_version": "lifetwin.independent_lfp_intake_report.v1",
        "intake_id": parsed["intake_id"],
        "dataset_id": parsed["dataset"]["dataset_id"],
        "candidate_id": frozen_candidate["candidate_id"],
        "candidate_config_sha256": canonical_json_sha256(frozen_candidate),
        "intake_sha256": canonical_json_sha256(parsed),
        "readiness_status": readiness,
        "failure_reasons": failures,
        "maximum_evidence_tier_after_valid_execution": evidence_tier,
        "allowed_claim_role_after_valid_execution": claim_role,
        "protocol_can_be_frozen_now": False,
        "manual_second_person_review_required": True,
        "compiler_reads_measurement_values": False,
        "compiler_accepts_measurement_value_fields": False,
        "next_required_actions": next_required_actions,
        "claim_boundary": (
            "Readiness compilation is not independent validation and cannot "
            "raise the evidence grade of any existing result."
        ),
    }
    report["report_content_sha256"] = canonical_json_sha256(report)

    template = validate_independent_long_term_protocol(
        _load_json(protocol_template_path)
    )
    draft = deepcopy(template)
    dataset = parsed["dataset"]
    structure = parsed["structure_audit"]
    safe_id = re.sub(r"[^a-z0-9]+", "_", str(dataset["dataset_id"]).casefold()).strip(
        "_"
    )
    draft["protocol_id"] = f"{safe_id or 'pending'}_independent_lfp_v1_draft"
    for field in (
        "dataset_id",
        "title",
        "doi_or_persistent_url",
        "repository_url",
        "repository_version",
        "access_mode",
        "paper_license",
        "data_license",
        "artifacts",
        "cathode_chemistry",
        "anode_chemistry",
        "aging_mode",
        "time_unit",
        "physical_cell_id_field",
        "condition_id_fields",
        "outcome_fields",
    ):
        draft["dataset"][field] = deepcopy(dataset[field])
    field_map = {
        "maximum_observed_duration_days": "maximum_observed_duration_days",
        "machine_readable_observations_verified": (
            "machine_readable_observations_verified"
        ),
        "physical_cell_ids_verified": "physical_cell_ids_verified",
        "calendar_aging_separable_verified": ("calendar_aging_separable_verified"),
        "observed_physical_cell_count": "observed_physical_cell_count",
        "observed_independent_scoring_cluster_count": (
            "observed_independent_scoring_cluster_count"
        ),
        "minimum_observed_positive_prefix_observations": (
            "minimum_observed_positive_prefix_observations"
        ),
        "minimum_observed_future_observations": (
            "minimum_observed_future_observations"
        ),
        "minimum_observed_future_to_landmark_time_ratio": (
            "minimum_observed_future_to_landmark_time_ratio"
        ),
    }
    for source, target in field_map.items():
        draft["dataset"][target] = structure[source]
    protocol_outcome_class = outcome_class if not failures else "unclassifiable"
    if failures:
        protocol_classification_reason = (
            "The metadata intake is blocked, so no outcome-blindness class or "
            "claim role is assigned to this protocol draft. See the intake report."
        )
    else:
        protocol_classification_reason = parsed["outcome_exposure"][
            "classification_reason"
        ]
    draft["outcome_blindness"] = {
        "classification": protocol_outcome_class,
        "classification_reason": protocol_classification_reason,
        "target_outcome_exposure_log": deepcopy(
            parsed["outcome_exposure"]["target_outcome_exposure_log"]
        ),
        "allowed_claim_role": _CLAIM_ROLE[protocol_outcome_class],
    }
    draft["candidate"].update(
        {
            "model_id": frozen_candidate["fixed_structure"]["primary_model_id"],
            "implementation_revision": frozen_candidate["candidate_id"],
            "config_sha256": canonical_json_sha256(frozen_candidate),
            "fit_partition": "training_only",
            "uses_target_future_outcomes": False,
            "hyperparameters_frozen": False,
        }
    )
    protocol_failures = _protocol_failure_reasons(failures)
    if failures:
        draft["eligibility"].update(
            {
                "observed_result": "ineligible",
                "evidence_tier": "D0_ineligible_or_blocked",
                "failure_reasons": protocol_failures or ["other"],
            }
        )
    else:
        draft["eligibility"].update(
            {
                "observed_result": "pending",
                "evidence_tier": _EVIDENCE_TIER[outcome_class],
                "failure_reasons": [],
            }
        )
        claim_for_class = {
            "prospective_outcome_blind": "prospective_outcome_blind_confirmation",
            "public_but_project_blind": (
                "public_data_project_blind_locked_confirmation"
            ),
            "locked_retrospective_replication": "locked_retrospective_replication",
        }.get(outcome_class)
        if claim_for_class is not None:
            draft["claim_boundaries"]["allowed_claims"] = [claim_for_class]
    draft["protocol_freeze"]["freeze_witnesses"] = [parsed["prepared_by"]]
    validated_draft = validate_independent_long_term_protocol(draft)
    return report, validated_draft


__all__ = [
    "CANDIDATE_CONFIG_SHA256",
    "CANDIDATE_ID",
    "IndependentLFPIntakeError",
    "canonical_json_sha256",
    "compile_independent_lfp_intake",
    "load_independent_candidate_config",
    "load_independent_lfp_intake",
    "validate_independent_lfp_intake",
]
