from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

import pytest

from lifetwin.validation.long_term_protocol import (
    IndependentLongTermProtocolValidationError,
    validate_independent_long_term_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_ROOT = PROJECT_ROOT / "configs" / "validation"
REGISTRY_PATH = VALIDATION_ROOT / "long_term_lfp_dataset_registry.json"
SCHEMA_PATH = VALIDATION_ROOT / "independent_long_term_lfp_protocol.schema.json"
TEMPLATE_PATH = VALIDATION_ROOT / "independent_long_term_lfp_protocol.template.json"
DOC_PATH = PROJECT_ROOT / "docs" / "independent_long_term_lfp_preregistration.md"
REGISTRY_DOC_PATH = PROJECT_ROOT / "docs" / "long_term_lfp_dataset_registry.md"


def _load_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    return json.loads(raw.decode("utf-8", errors="strict"))


def _dataset_by_id(registry: dict[str, object], dataset_id: str) -> dict[str, object]:
    datasets = registry["datasets"]
    assert isinstance(datasets, list)
    matches = [item for item in datasets if item["dataset_id"] == dataset_id]
    assert len(matches) == 1
    return matches[0]


def _resolve_local_ref(root: dict[str, object], ref: str) -> dict[str, object]:
    assert ref.startswith("#/")
    node: object = root
    for token in ref[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        assert isinstance(node, dict)
        node = node[token]
    assert isinstance(node, dict)
    return node


def _schema_errors(
    instance: object,
    schema: dict[str, object],
    root: dict[str, object],
    path: str = "$",
) -> list[str]:
    """Validate the schema subset used by this frozen protocol asset."""
    if "$ref" in schema:
        referenced = _resolve_local_ref(root, schema["$ref"])
        return _schema_errors(instance, referenced, root, path)

    errors: list[str] = []
    for subschema in schema.get("allOf", []):
        errors.extend(_schema_errors(instance, subschema, root, path))
    condition = schema.get("if")
    if condition is not None and not _schema_errors(instance, condition, root, path):
        errors.extend(_schema_errors(instance, schema["then"], root, path))

    if "oneOf" in schema:
        branch_errors = [
            _schema_errors(instance, branch, root, path) for branch in schema["oneOf"]
        ]
        if sum(not branch for branch in branch_errors) != 1:
            errors.append(f"{path}: expected exactly one oneOf branch")

    if "contains" in schema and isinstance(instance, list):
        matching_items = sum(
            not _schema_errors(item, schema["contains"], root, path)
            for item in instance
        )
        if matching_items < schema.get("minContains", 1):
            errors.append(f"{path}: required contained item is absent")

    if "not" in schema and not _schema_errors(instance, schema["not"], root, path):
        errors.append(f"{path}: matched prohibited schema")
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: const mismatch")
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: enum mismatch")

    allowed_types = schema.get("type")
    if allowed_types is not None:
        if isinstance(allowed_types, str):
            allowed_types = [allowed_types]
        type_matches = {
            "null": instance is None,
            "boolean": isinstance(instance, bool),
            "integer": isinstance(instance, int) and not isinstance(instance, bool),
            "number": isinstance(instance, (int, float))
            and not isinstance(instance, bool),
            "string": isinstance(instance, str),
            "array": isinstance(instance, list),
            "object": isinstance(instance, dict),
        }
        if not any(type_matches[name] for name in allowed_types):
            errors.append(f"{path}: type mismatch")
            return errors

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for name in required:
            if name not in instance:
                errors.append(f"{path}: missing required property {name}")
        properties = schema.get("properties", {})
        for name, value in instance.items():
            if name in properties:
                errors.extend(
                    _schema_errors(value, properties[name], root, f"{path}.{name}")
                )
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {name}")

    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            errors.append(f"{path}: too few items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            errors.append(f"{path}: too many items")
        if schema.get("uniqueItems"):
            normalized = [json.dumps(item, sort_keys=True) for item in instance]
            if len(normalized) != len(set(normalized)):
                errors.append(f"{path}: duplicate items")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, value in enumerate(instance):
                errors.extend(
                    _schema_errors(value, item_schema, root, f"{path}[{index}]")
                )

    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            errors.append(f"{path}: string too short")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            errors.append(f"{path}: pattern mismatch")
        if schema.get("format") == "uri" and not urlsplit(instance).scheme:
            errors.append(f"{path}: invalid URI")
        if schema.get("format") == "date-time":
            iso_value = instance[:-1] + "+00:00" if instance.endswith("Z") else instance
            try:
                from datetime import datetime

                datetime.fromisoformat(iso_value)
            except ValueError:
                errors.append(f"{path}: invalid date-time")

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            errors.append(f"{path}: above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: below exclusive minimum")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: above exclusive maximum")
    return errors


def test_protocol_assets_are_utf8_json_and_dataset_agnostic() -> None:
    schema = _load_json(SCHEMA_PATH)
    template = _load_json(TEMPLATE_PATH)
    _load_json(REGISTRY_PATH)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert template["schema_version"] == "1.0.0"
    assert template["status"] == "draft"
    assert validate_independent_long_term_protocol(template) == template

    generic_assets = (
        SCHEMA_PATH.read_text(encoding="utf-8")
        + TEMPLATE_PATH.read_text(encoding="utf-8")
    ).lower()
    assert "lam_joule" not in generic_assets
    assert "stanford long term" not in generic_assets
    assert "osf.io/ju325" not in generic_assets

    for path in (DOC_PATH, REGISTRY_DOC_PATH):
        text = path.read_bytes().decode("utf-8", errors="strict")
        assert "\ufffd" not in text


def test_template_validates_against_schema_and_freeze_condition_activates() -> None:
    schema = _load_json(SCHEMA_PATH)
    template = _load_json(TEMPLATE_PATH)
    assert _schema_errors(template, schema, schema) == []

    prematurely_frozen = {**template, "status": "frozen"}
    freeze_errors = _schema_errors(prematurely_frozen, schema, schema)
    assert freeze_errors
    assert any("protocol_freeze" in error for error in freeze_errors)

    unexpected_property = {**template, "dataset_specific_shortcut": True}
    assert any(
        "unexpected property" in error
        for error in _schema_errors(unexpected_property, schema, schema)
    )


def test_draft_template_preserves_blind_freeze_guards() -> None:
    template = _load_json(TEMPLATE_PATH)
    freeze = template["protocol_freeze"]
    execution = template["execution_record"]

    for name in (
        "frozen_at_utc",
        "git_commit_sha",
        "protocol_sha256",
        "model_bundle_sha256",
        "scorer_sha256",
        "adapter_sha256",
        "environment_lock_sha256",
    ):
        assert freeze[name] is None
    for name in (
        "prediction_bundle_sha256",
        "prediction_frozen_at_utc",
        "truth_linked_at_utc",
        "result_bundle_sha256",
    ):
        assert execution[name] is None

    assert template["dataset"]["access_mode"] == "not_acquired"
    assert template["dataset"]["data_license"]["status"] == "unclear"
    assert template["outcome_blindness"]["classification"] == "unclassifiable"
    assert template["outcome_blindness"]["allowed_claim_role"] == "none"
    assert template["partitions"]["cross_partition_cell_overlap_allowed"] is False
    assert template["partitions"]["pairwise_disjoint_verified"] is False
    assert template["partitions"]["assigned_unit_count"] is None
    assert template["partitions"]["partition_selection_uses_target_outcomes"] is False
    assert template["landmarks"]["future_outcomes_may_influence_prediction"] is False
    assert template["candidate"]["uses_target_future_outcomes"] is False
    assert (
        template["claim_boundaries"]["fifteen_to_twenty_five_year_claim_allowed"]
        is False
    )
    assert (
        template["claim_boundaries"]["claim_horizon_within_observed_support"] is False
    )
    assert execution["truth_linked_after_prediction_freeze"] is False
    assert execution["independent_scorer_verified"] is False


def test_eligibility_metrics_and_point_success_are_frozen_conservatively() -> None:
    template = _load_json(TEMPLATE_PATH)
    eligibility = template["eligibility"]
    assert eligibility["minimum_duration_days"] == 730
    assert eligibility["requires_lfp_cathode"] is True
    assert eligibility["requires_graphite_anode"] is True
    assert eligibility["requires_individual_physical_cell_ids"] is True
    assert eligibility["requires_separable_calendar_aging"] is True
    assert eligibility["minimum_physical_cells"] == 8
    assert eligibility["minimum_independent_scoring_clusters"] == 8
    assert eligibility["minimum_positive_prefix_observations"] == 4
    assert eligibility["minimum_future_observations"] == 2
    assert eligibility["minimum_future_to_landmark_time_ratio"] == 2.0

    assert template["metrics"]["statistical_unit"] == "physical_cell_trajectory"
    assert template["metrics"]["cluster_unit"] == "condition"
    assert template["metrics"]["primary_metric"] == (
        "trajectory_iae_pp_normalized_by_elapsed_horizon"
    )

    decision = template["decision_rules"]
    assert decision["minimum_mean_iae_improvement_pp"] == 0.1
    assert decision["minimum_relative_iae_improvement_fraction"] == 0.05
    assert decision["maximum_one_sided_cluster_randomization_p"] == 0.05
    assert decision["minimum_improved_cluster_fraction"] == 0.6
    assert decision["maximum_worst_cluster_regression_pp"] == 0.5
    assert decision["minimum_independent_scoring_clusters"] == 8
    assert decision["all_point_success_gates_required"] is True
    assert decision["underpowered_result"] == "inconclusive_not_success"


def test_mandatory_baselines_and_uncertainty_issuance_are_explicit() -> None:
    template = _load_json(TEMPLATE_PATH)
    baseline_ids = {baseline["model_id"] for baseline in template["baselines"]}
    assert baseline_ids == {
        "target_prefix_persistence",
        "target_prefix_sqrt_time",
        "target_prefix_bounded_power_law",
    }
    assert all(
        not baseline["uses_target_future_outcomes"]
        for baseline in template["baselines"]
    )

    issuance = template["uncertainty"]["operational_issuance"]
    assert template["uncertainty"]["calibration_unit"] == "independent_cluster"
    assert issuance["enabled_at_freeze"] is False
    assert issuance["minimum_calibration_clusters_per_route"] >= 20
    assert issuance["minimum_audit_clusters_per_route"] >= 20
    assert "never increase conformal n" in issuance["cluster_independence_definition"]
    assert issuance["requires_horizon_match"] is True
    assert issuance["requires_covariate_support"] is True
    assert issuance["requires_point_success"] is True
    assert issuance["requires_independent_calibration"] is True
    assert issuance["all_gates_required"] is True
    assert (
        "same_route_calibration_insufficient"
        in template["uncertainty"]["rejection_reasons"]
    )


def test_schema_enumerates_outcome_classes_failures_and_claim_boundaries() -> None:
    schema = _load_json(SCHEMA_PATH)
    definitions = schema["$defs"]
    blindness = definitions["outcome_blindness"]["properties"]["classification"]["enum"]
    assert set(blindness) == {
        "prospective_outcome_blind",
        "public_but_project_blind",
        "locked_retrospective_replication",
        "development_only",
        "unclassifiable",
    }

    failure_reasons = definitions["eligibility"]["properties"]["failure_reasons"][
        "items"
    ]["enum"]
    for reason in (
        "license_unclear",
        "physical_cell_ids_absent",
        "duration_below_730_days",
        "calendar_aging_not_separable",
        "endpoint_only",
        "source_overlap",
        "outcomes_exposed_for_development",
    ):
        assert reason in failure_reasons

    prohibited = set(
        template := _load_json(TEMPLATE_PATH)["claim_boundaries"]["prohibited_claims"]
    )
    assert "15_to_25_year_accuracy_without_observed_support" in prohibited
    assert "hithium_product_accuracy" in prohibited
    assert "utility_scale_station_validation_from_cell_only_data" in prohibited
    assert "formal_coverage_without_independent_route_calibration" in prohibited
    assert template == list(dict.fromkeys(template))


def _executed_success_payload() -> dict[str, object]:
    payload = deepcopy(_load_json(TEMPLATE_PATH))
    sha256 = "a" * 64
    payload["status"] = "executed"
    payload["protocol_freeze"].update(
        {
            "frozen_at_utc": "2026-07-21T00:00:00Z",
            "git_commit_sha": "b" * 40,
            "protocol_sha256": sha256,
            "model_bundle_sha256": sha256,
            "scorer_sha256": sha256,
            "adapter_sha256": sha256,
            "environment_lock_sha256": sha256,
        }
    )
    payload["dataset"].update(
        {
            "dataset_id": "independent_lfp_example",
            "title": "Independent long-term LFP example",
            "repository_version": "v1.0.0",
            "access_mode": "public_download",
            "artifacts": [
                {
                    "logical_name": "observations.csv",
                    "source_url": "https://example.org/observations.csv",
                    "repository_version": "v1.0.0",
                    "byte_size": 1024,
                    "sha256": sha256,
                    "retrieved_at_utc": "2026-07-20T00:00:00Z",
                }
            ],
            "cathode_chemistry": "LFP",
            "anode_chemistry": "graphite",
            "physical_cell_id_field": "cell_id",
            "maximum_observed_duration_days": 1000.0,
            "machine_readable_observations_verified": True,
            "physical_cell_ids_verified": True,
            "calendar_aging_separable_verified": True,
            "observed_physical_cell_count": 8,
            "observed_independent_scoring_cluster_count": 8,
            "minimum_observed_positive_prefix_observations": 4,
            "minimum_observed_future_observations": 2,
            "minimum_observed_future_to_landmark_time_ratio": 2.0,
        }
    )
    payload["dataset"]["data_license"].update(
        {
            "status": "explicit",
            "identifier": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        }
    )
    payload["eligibility"].update(
        {
            "observed_result": "eligible",
            "evidence_tier": "D4_locked_external_trajectory",
            "failure_reasons": [],
        }
    )
    payload["outcome_blindness"].update(
        {
            "classification": "public_but_project_blind",
            "classification_reason": (
                "Public outcomes existed, but the project exposure log is clean "
                "through prediction freeze."
            ),
            "allowed_claim_role": (
                "locked_external_confirmation_with_public_data_caveat"
            ),
        }
    )
    payload["partitions"].update(
        {
            "training_ids": ["cell_01", "cell_02"],
            "calibration_ids": ["cell_03", "cell_04"],
            "test_ids": ["cell_05", "cell_06"],
            "audit_ids": ["cell_07", "cell_08"],
            "assigned_unit_count": 8,
            "pairwise_disjoint_verified": True,
        }
    )
    payload["landmarks"].update(
        {"primary_value": 5.0, "common_score_start": 6.0, "common_score_end": 20.0}
    )
    payload["candidate"].update(
        {
            "implementation_revision": "candidate-v1",
            "config_sha256": sha256,
            "hyperparameters_frozen": True,
        }
    )
    for index, baseline in enumerate(payload["baselines"]):
        baseline.update(
            {
                "implementation_revision": f"baseline-v1-{index}",
                "config_sha256": sha256,
                "hyperparameters_frozen": True,
            }
        )
    payload["claim_boundaries"].update(
        {
            "maximum_claim_horizon_days": 730.0,
            "claim_horizon_within_observed_support": True,
            "allowed_claims": [
                "public_data_project_blind_locked_confirmation",
                "long_horizon_trajectory_performance_within_observed_support",
            ],
        }
    )
    payload["execution_record"].update(
        {
            "prediction_bundle_sha256": sha256,
            "prediction_frozen_at_utc": "2026-07-21T01:00:00Z",
            "truth_linked_at_utc": "2026-07-21T02:00:00Z",
            "result_bundle_sha256": sha256,
            "truth_linked_after_prediction_freeze": True,
            "independent_scorer_verified": True,
            "point_success_gates_passed": True,
            "landmark_consistency_gates_passed": True,
            "terminal_decision": "success",
        }
    )
    return payload


def test_malicious_executed_success_payload_is_rejected_by_cross_field_gates() -> None:
    schema = _load_json(SCHEMA_PATH)
    valid_success = _executed_success_payload()
    assert _schema_errors(valid_success, schema, schema) == []
    assert validate_independent_long_term_protocol(valid_success) == valid_success

    malicious = deepcopy(valid_success)
    malicious["eligibility"]["observed_result"] = "ineligible"
    malicious["eligibility"]["evidence_tier"] = "D5_prospective_operational"
    malicious["outcome_blindness"].update(
        {
            "classification": "locked_retrospective_replication",
            "allowed_claim_role": "confirmatory",
        }
    )
    malicious["claim_boundaries"].update(
        {
            "allowed_claims": ["prospective_outcome_blind_confirmation"],
            "fifteen_to_twenty_five_year_claim_allowed": True,
        }
    )
    malicious["candidate"]["hyperparameters_frozen"] = False
    malicious["baselines"][0]["hyperparameters_frozen"] = False
    malicious["execution_record"].update(
        {
            "truth_linked_after_prediction_freeze": False,
            "independent_scorer_verified": False,
            "point_success_gates_passed": False,
        }
    )

    errors = _schema_errors(malicious, schema, schema)
    assert errors
    with pytest.raises(IndependentLongTermProtocolValidationError):
        validate_independent_long_term_protocol(malicious)
    for field in (
        "eligibility.observed_result",
        "eligibility.evidence_tier",
        "outcome_blindness.allowed_claim_role",
        "candidate.hyperparameters_frozen",
        "baselines[0].hyperparameters_frozen",
        "claim_boundaries.fifteen_to_twenty_five_year_claim_allowed",
        "execution_record.truth_linked_after_prediction_freeze",
        "execution_record.independent_scorer_verified",
        "execution_record.point_success_gates_passed",
    ):
        assert any(field in error for error in errors), (field, errors)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("eligibility", "minimum_physical_cells", 7),
        ("eligibility", "minimum_independent_scoring_clusters", 7),
        ("eligibility", "minimum_positive_prefix_observations", 3),
        ("eligibility", "minimum_future_to_landmark_time_ratio", 1.5),
        ("decision_rules", "minimum_mean_iae_improvement_pp", 0.0),
        ("decision_rules", "minimum_relative_iae_improvement_fraction", 0.0),
        ("decision_rules", "minimum_improved_cluster_fraction", 0.5),
        ("decision_rules", "maximum_worst_cluster_regression_pp", 1.0),
        ("decision_rules", "minimum_independent_scoring_clusters", 4),
    ],
)
def test_production_validator_rejects_weakened_frozen_thresholds(
    section: str,
    field: str,
    value: object,
) -> None:
    payload = _executed_success_payload()
    payload[section][field] = value

    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match=re.escape(f"$.{section}.{field}"),
    ):
        validate_independent_long_term_protocol(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cathode_chemistry", "NMC811"),
        ("anode_chemistry", "silicon_graphite"),
        ("aging_mode", "mixed_not_separable"),
        ("machine_readable_observations_verified", False),
        ("physical_cell_ids_verified", False),
        ("calendar_aging_separable_verified", False),
        ("observed_physical_cell_count", 7),
        ("observed_independent_scoring_cluster_count", 7),
        ("minimum_observed_positive_prefix_observations", 3),
        ("minimum_observed_future_observations", 1),
        ("minimum_observed_future_to_landmark_time_ratio", 1.9),
    ],
)
def test_production_validator_rejects_false_eligible_dataset_claims(
    field: str,
    value: object,
) -> None:
    payload = _executed_success_payload()
    payload["dataset"][field] = value

    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match=re.escape(f"$.dataset.{field}"),
    ):
        validate_independent_long_term_protocol(payload)


@pytest.mark.parametrize(
    "partition", ["training_ids", "calibration_ids", "test_ids", "audit_ids"]
)
def test_production_validator_rejects_empty_required_partition(
    partition: str,
) -> None:
    payload = _executed_success_payload()
    payload["partitions"][partition] = []

    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match=re.escape(f"$.partitions.{partition}"),
    ):
        validate_independent_long_term_protocol(payload)


def test_production_validator_recomputes_partition_disjointness_and_count() -> None:
    overlap = _executed_success_payload()
    overlap["partitions"]["audit_ids"][0] = overlap["partitions"]["test_ids"][0]
    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match="pairwise disjoint",
    ):
        validate_independent_long_term_protocol(overlap)

    wrong_count = _executed_success_payload()
    wrong_count["partitions"]["assigned_unit_count"] = 7
    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match="assigned_unit_count",
    ):
        validate_independent_long_term_protocol(wrong_count)

    inconsistent_observed_count = _executed_success_payload()
    inconsistent_observed_count["dataset"]["observed_physical_cell_count"] = 9
    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match="assigned_unit_count",
    ):
        validate_independent_long_term_protocol(inconsistent_observed_count)


def test_production_validator_requires_exact_unique_mandatory_baselines() -> None:
    duplicate = _executed_success_payload()
    duplicate["baselines"][2] = deepcopy(duplicate["baselines"][1])
    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match="baselines",
    ):
        validate_independent_long_term_protocol(duplicate)

    wrong_role = _executed_success_payload()
    wrong_role["baselines"][0]["role"] = "secondary_baseline"
    with pytest.raises(
        IndependentLongTermProtocolValidationError,
        match="baselines",
    ):
        validate_independent_long_term_protocol(wrong_role)


def test_registry_records_zero_current_qualifiers_and_candidate_limits() -> None:
    registry = _load_json(REGISTRY_PATH)
    datasets = registry["datasets"]
    ids = [item["dataset_id"] for item in datasets]
    assert len(ids) == len(set(ids)) == 10
    assert (
        registry["qualification_summary"][
            "qualified_public_independent_trajectory_datasets"
        ]
        == 0
    )
    assert not any(item["qualifies_now"] for item in datasets)

    lam = _dataset_by_id(registry, "lam_joule_2025_osf")
    assert lam["data_license"]["status"] == "unclear"
    assert lam["license_audit"]["osf_node_license"] is None
    assert lam["license_audit"]["github_detected_license"] is None
    assert lam["maximum_possible_tier_after_action"] == (
        "D3_long_horizon_trajectory_retrospective"
    )

    vachenauer = _dataset_by_id(registry, "vachenauer_tum_2025_shelf")
    assert vachenauer["storage_temperature_c"] == 6.0
    assert vachenauer["storage_soc_pct"] == 50.0
    assert (
        "endpoint_only_no_dynamic_landmark_trajectory" in vachenauer["failure_reasons"]
    )
    assert vachenauer["maximum_possible_tier_after_action"] == (
        "D2_long_horizon_endpoint"
    )

    aeppli = _dataset_by_id(registry, "aeppli_empa_2025_second_life")
    assert "Nine selected physical cells" in aeppli["grain"]

    naumann = _dataset_by_id(registry, "naumann_mendeley_2021_calendar")
    assert "source_overlap_with_model_development" in naumann["failure_reasons"]
    geisbauer = _dataset_by_id(registry, "geisbauer_zenodo_2022_calendar")
    assert geisbauer["duration_days"] < 730
    wmg = _dataset_by_id(registry, "wmg_zenodo_2025_lgm50_calendar")
    assert "cathode_not_lfp" in wmg["failure_reasons"]
