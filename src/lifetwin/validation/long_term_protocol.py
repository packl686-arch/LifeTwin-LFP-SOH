from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker


DEFAULT_SCHEMA = (
    Path(__file__).resolve().parents[3]
    / "configs/validation/independent_long_term_lfp_protocol.schema.json"
)
MANDATORY_BASELINE_IDS = frozenset(
    {
        "target_prefix_persistence",
        "target_prefix_sqrt_time",
        "target_prefix_bounded_power_law",
    }
)
PARTITION_NAMES = ("training_ids", "calibration_ids", "test_ids", "audit_ids")


class IndependentLongTermProtocolValidationError(ValueError):
    """A long-term protocol failed its schema or cross-field gates."""


def _json_path(error: Any) -> str:
    parts = [str(part) for part in error.absolute_path]
    return "$" + "".join(
        f"[{part}]" if part.isdigit() else f".{part}" for part in parts
    )


def _schema_errors(
    protocol: Mapping[str, object], schema: Mapping[str, object]
) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    return [
        f"{_json_path(error)}: {error.message}"
        for error in sorted(
            validator.iter_errors(protocol),
            key=lambda item: (
                tuple(str(part) for part in item.absolute_path),
                item.message,
            ),
        )
    ]


def _required_mapping(payload: Mapping[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise IndependentLongTermProtocolValidationError(
            f"$.{name}: expected an object"
        )
    return dict(value)


def _partition_sets(partitions: Mapping[str, object]) -> dict[str, set[str]]:
    parsed: dict[str, set[str]] = {}
    for name in PARTITION_NAMES:
        values = partitions.get(name)
        if not isinstance(values, list) or not values:
            raise IndependentLongTermProtocolValidationError(
                f"$.partitions.{name}: eligible protocols require a non-empty array"
            )
        if not all(isinstance(value, str) and value for value in values):
            raise IndependentLongTermProtocolValidationError(
                f"$.partitions.{name}: every partition ID must be a non-empty string"
            )
        if len(values) != len(set(values)):
            raise IndependentLongTermProtocolValidationError(
                f"$.partitions.{name}: duplicate partition IDs are prohibited"
            )
        parsed[name] = set(values)
    return parsed


def _validate_partition_semantics(
    dataset: Mapping[str, object], partitions: Mapping[str, object]
) -> None:
    groups = _partition_sets(partitions)
    for left_index, left_name in enumerate(PARTITION_NAMES):
        for right_name in PARTITION_NAMES[left_index + 1 :]:
            overlap = sorted(groups[left_name] & groups[right_name])
            if overlap:
                raise IndependentLongTermProtocolValidationError(
                    "$.partitions: IDs must be pairwise disjoint; "
                    f"{left_name} and {right_name} overlap at {overlap}"
                )
    union = set().union(*groups.values())
    assigned_count = partitions.get("assigned_unit_count")
    if assigned_count != len(union):
        raise IndependentLongTermProtocolValidationError(
            "$.partitions.assigned_unit_count: must equal the number of unique "
            f"assigned IDs ({len(union)})"
        )
    if partitions.get("pairwise_disjoint_verified") is not True:
        raise IndependentLongTermProtocolValidationError(
            "$.partitions.pairwise_disjoint_verified: must be true after the "
            "validator recomputes disjointness"
        )
    if partitions.get("grouping_unit") == "physical_cell":
        observed_cells = dataset.get("observed_physical_cell_count")
        if not isinstance(observed_cells, int) or isinstance(observed_cells, bool):
            raise IndependentLongTermProtocolValidationError(
                "$.dataset.observed_physical_cell_count: must be an integer"
            )
        if assigned_count != observed_cells:
            raise IndependentLongTermProtocolValidationError(
                "$.partitions.assigned_unit_count: must equal the observed "
                "physical-cell count when grouping_unit is physical_cell"
            )
    else:
        observed_clusters = dataset.get("observed_independent_scoring_cluster_count")
        if assigned_count != observed_clusters:
            raise IndependentLongTermProtocolValidationError(
                "$.partitions.assigned_unit_count: must equal the observed "
                "independent-cluster count for condition, batch, or site grouping"
            )


def _validate_mandatory_baselines(protocol: Mapping[str, object]) -> None:
    baselines = protocol.get("baselines")
    if not isinstance(baselines, list):
        raise IndependentLongTermProtocolValidationError(
            "$.baselines: expected an array"
        )
    ids: list[str] = []
    for index, baseline in enumerate(baselines):
        if not isinstance(baseline, Mapping):
            raise IndependentLongTermProtocolValidationError(
                f"$.baselines[{index}]: expected an object"
            )
        model_id = baseline.get("model_id")
        if not isinstance(model_id, str):
            raise IndependentLongTermProtocolValidationError(
                f"$.baselines[{index}].model_id: expected a string"
            )
        ids.append(model_id)
        if baseline.get("role") != "mandatory_baseline":
            raise IndependentLongTermProtocolValidationError(
                f"$.baselines[{index}].role: must be mandatory_baseline"
            )
    if len(ids) != 3 or len(set(ids)) != 3 or set(ids) != MANDATORY_BASELINE_IDS:
        raise IndependentLongTermProtocolValidationError(
            "$.baselines: must contain each of persistence, sqrt-time, and "
            "bounded-power exactly once"
        )


def _validate_eligible_counts(
    dataset: Mapping[str, object], eligibility: Mapping[str, object]
) -> None:
    gates = (
        ("observed_physical_cell_count", "minimum_physical_cells"),
        (
            "observed_independent_scoring_cluster_count",
            "minimum_independent_scoring_clusters",
        ),
        (
            "minimum_observed_positive_prefix_observations",
            "minimum_positive_prefix_observations",
        ),
        (
            "minimum_observed_future_observations",
            "minimum_future_observations",
        ),
        (
            "minimum_observed_future_to_landmark_time_ratio",
            "minimum_future_to_landmark_time_ratio",
        ),
    )
    for observed_name, threshold_name in gates:
        observed = dataset.get(observed_name)
        threshold = eligibility.get(threshold_name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or observed < threshold
        ):
            raise IndependentLongTermProtocolValidationError(
                f"$.dataset.{observed_name}: must meet frozen threshold "
                f"$.eligibility.{threshold_name}={threshold}"
            )
    cells = dataset["observed_physical_cell_count"]
    clusters = dataset["observed_independent_scoring_cluster_count"]
    if clusters > cells:
        raise IndependentLongTermProtocolValidationError(
            "$.dataset.observed_independent_scoring_cluster_count: cannot exceed "
            "the observed physical-cell count"
        )


def validate_independent_long_term_protocol(
    protocol: Mapping[str, object],
    *,
    schema_path: Path = DEFAULT_SCHEMA,
) -> dict[str, object]:
    """Validate the JSON Schema and the cross-field evidence gates.

    JSON Schema handles types, fixed thresholds, and conditional fields. This
    function additionally recomputes partition disjointness and count
    consistency, which draft 2020-12 cannot express across four arbitrary ID
    arrays.
    """
    if not isinstance(protocol, Mapping):
        raise IndependentLongTermProtocolValidationError(
            "$: protocol must be a JSON object"
        )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = _schema_errors(protocol, schema)
    if errors:
        raise IndependentLongTermProtocolValidationError(
            "Protocol JSON Schema validation failed:\n" + "\n".join(errors)
        )

    parsed = deepcopy(dict(protocol))
    _validate_mandatory_baselines(parsed)
    eligibility = _required_mapping(parsed, "eligibility")
    status = parsed.get("status")
    eligible = eligibility.get("observed_result") == "eligible"
    if status in {"frozen", "executed"} and not eligible:
        raise IndependentLongTermProtocolValidationError(
            "$.eligibility.observed_result: frozen or executed protocols must "
            "be eligible"
        )
    if eligible:
        dataset = _required_mapping(parsed, "dataset")
        partitions = _required_mapping(parsed, "partitions")
        _validate_eligible_counts(dataset, eligibility)
        _validate_partition_semantics(dataset, partitions)
    return parsed
