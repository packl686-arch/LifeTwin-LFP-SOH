"""Fail-closed adapter for private enterprise LFP cycle-aging evaluations."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.private_cycle_adapter.config.v1"
PARTITION_SCHEMA_VERSION = "lifetwin.private_cycle_partitions.v1"
BUNDLE_SCHEMA_VERSION = "lifetwin.private_cycle_blind_bundle.v1"
PARTITIONS = ("development", "calibration", "locked_test")
PARTITION_METADATA_COLUMNS = ("cell_id", "batch_id", "condition_id")
PRIVATE_MEASUREMENT_COLUMNS = (
    "record_id",
    "cell_id",
    "batch_id",
    "condition_id",
    "cathode_chemistry",
    "temperature_c",
    "min_soc_pct",
    "max_soc_pct",
    "charge_c_rate",
    "discharge_c_rate",
    "visit_index",
    "elapsed_days",
    "equivalent_full_cycles",
    "capacity_ah",
    "reference_capacity_ah",
    "quality_status",
)
PARTITIONED_TRAJECTORY_COLUMNS = ("partition", *RPT_TRAJECTORY_COLUMNS)
PARTITIONED_PREFIX_COLUMNS = (
    "partition",
    "landmark_visit_count",
    *RPT_TRAJECTORY_COLUMNS,
)

_CONFIG_KEYS = {
    "schema_version",
    "adapter_id",
    "private_only",
    "dataset_id",
    "chemistry",
    "partition_policy",
    "trajectory_policy",
    "model_contract",
    "claim_boundary",
}
_PARTITION_POLICY_KEYS = {
    "grouping_field",
    "hash_seed",
    "minimum_group_count",
    "fractions",
}
_TRAJECTORY_POLICY_KEYS = {
    "landmark_visit_counts",
    "minimum_future_visits",
    "score_end_equivalent_full_cycles",
    "capacity_reference_rule",
    "quality_status_allowlist",
}
_MODEL_CONTRACT_KEYS = {
    "primary_model_id",
    "future_schedule_assumption",
    "explicit_forecast_elapsed_days_supported",
    "target_suffix_permitted_in_prediction_process",
}


class PrivateCycleAdapterError(ValueError):
    """Raised when a private-data firewall or schema invariant is violated."""


def _exact_keys(
    value: object, expected: set[str], *, path: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PrivateCycleAdapterError(f"{path} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise PrivateCycleAdapterError(
            f"{path} keys changed; missing={missing}, extra={extra}"
        )
    return value


def _nonempty_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PrivateCycleAdapterError(f"{path} must be a non-empty string")
    return value


def validate_private_cycle_adapter_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return a detached JSON-safe adapter configuration."""
    value = deepcopy(dict(_exact_keys(config, _CONFIG_KEYS, path="$")))
    if value["schema_version"] != SCHEMA_VERSION:
        raise PrivateCycleAdapterError("Private adapter schema changed")
    if value["private_only"] is not True:
        raise PrivateCycleAdapterError("Private adapter must remain private-only")
    for field in ("adapter_id", "dataset_id", "chemistry", "claim_boundary"):
        _nonempty_string(value[field], path=f"$.{field}")
    if str(value["chemistry"]).casefold() not in {
        "lfp",
        "lifepo4",
        "lithium_iron_phosphate",
    }:
        raise PrivateCycleAdapterError("Private adapter chemistry must be LFP")

    partition = _exact_keys(
        value["partition_policy"], _PARTITION_POLICY_KEYS, path="$.partition_policy"
    )
    if partition["grouping_field"] != "batch_id":
        raise PrivateCycleAdapterError("Partitions must be grouped by batch_id")
    _nonempty_string(partition["hash_seed"], path="$.partition_policy.hash_seed")
    minimum_groups = partition["minimum_group_count"]
    if isinstance(minimum_groups, bool) or int(minimum_groups) < 5:
        raise PrivateCycleAdapterError("At least five independent batches are required")
    fractions = _exact_keys(
        partition["fractions"], set(PARTITIONS), path="$.partition_policy.fractions"
    )
    fraction_values = [float(fractions[name]) for name in PARTITIONS]
    if not all(math.isfinite(item) and item > 0.0 for item in fraction_values):
        raise PrivateCycleAdapterError("Partition fractions must be finite and positive")
    if not math.isclose(sum(fraction_values), 1.0, abs_tol=1e-12):
        raise PrivateCycleAdapterError("Partition fractions must sum to one")

    trajectory = _exact_keys(
        value["trajectory_policy"],
        _TRAJECTORY_POLICY_KEYS,
        path="$.trajectory_policy",
    )
    landmarks = trajectory["landmark_visit_counts"]
    if not isinstance(landmarks, Sequence) or isinstance(landmarks, (str, bytes)):
        raise PrivateCycleAdapterError("Landmarks must be an array")
    landmark_values = tuple(int(item) for item in landmarks)
    if (
        not landmark_values
        or min(landmark_values) < 3
        or len(set(landmark_values)) != len(landmark_values)
    ):
        raise PrivateCycleAdapterError("Landmarks must be unique integers >= 3")
    minimum_future = trajectory["minimum_future_visits"]
    if isinstance(minimum_future, bool) or int(minimum_future) < 1:
        raise PrivateCycleAdapterError("At least one future RPT visit is required")
    if float(trajectory["score_end_equivalent_full_cycles"]) <= 0.0:
        raise PrivateCycleAdapterError("Score end must be positive")
    if trajectory["capacity_reference_rule"] != "explicit_per_row_reference_capacity":
        raise PrivateCycleAdapterError("Capacity reference rule changed")
    allowlist = trajectory["quality_status_allowlist"]
    if (
        not isinstance(allowlist, Sequence)
        or isinstance(allowlist, (str, bytes))
        or not allowlist
        or any(not isinstance(item, str) or not item for item in allowlist)
    ):
        raise PrivateCycleAdapterError("Quality-status allowlist is invalid")

    model = _exact_keys(
        value["model_contract"], _MODEL_CONTRACT_KEYS, path="$.model_contract"
    )
    _nonempty_string(model["primary_model_id"], path="$.model_contract.primary_model_id")
    if model["future_schedule_assumption"] != "constant_prefix_efc_per_day":
        raise PrivateCycleAdapterError("Future schedule assumption changed")
    if model["explicit_forecast_elapsed_days_supported"] is not True:
        raise PrivateCycleAdapterError("Explicit elapsed-day forecasts must remain supported")
    if model["target_suffix_permitted_in_prediction_process"] is not False:
        raise PrivateCycleAdapterError("Prediction process must reject target suffixes")
    return value


def _hashed_group_order(groups: Sequence[str], *, seed: str) -> list[str]:
    return sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}\0{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )


def _partition_group_counts(
    group_count: int, fractions: Mapping[str, object]
) -> dict[str, int]:
    calibration = max(1, int(round(group_count * float(fractions["calibration"]))))
    locked_test = max(1, int(round(group_count * float(fractions["locked_test"]))))
    development = group_count - calibration - locked_test
    if development < 2:
        deficit = 2 - development
        while deficit and calibration > 1:
            calibration -= 1
            deficit -= 1
        while deficit and locked_test > 1:
            locked_test -= 1
            deficit -= 1
        development = group_count - calibration - locked_test
    if development < 2:
        raise PrivateCycleAdapterError(
            "Partition policy cannot retain two development batches"
        )
    return {
        "development": development,
        "calibration": calibration,
        "locked_test": locked_test,
    }


def freeze_private_cycle_partitions(
    metadata: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    """Freeze batch-disjoint partitions without accepting any measurement values."""
    frozen = validate_private_cycle_adapter_config(config)
    if tuple(metadata.columns) != PARTITION_METADATA_COLUMNS:
        raise PrivateCycleAdapterError(
            "Partition freeze accepts only cell_id, batch_id, and condition_id"
        )
    if metadata.empty or metadata.isna().any().any():
        raise PrivateCycleAdapterError("Partition metadata is empty or incomplete")
    rows = metadata.astype(str).copy()
    if any((rows[column].str.strip() == "").any() for column in rows.columns):
        raise PrivateCycleAdapterError("Partition metadata contains empty identifiers")
    if rows["cell_id"].duplicated().any():
        raise PrivateCycleAdapterError("Each physical cell must appear exactly once")
    groups = sorted(rows["batch_id"].unique())
    minimum = int(frozen["partition_policy"]["minimum_group_count"])
    if len(groups) < minimum:
        raise PrivateCycleAdapterError(
            f"Observed {len(groups)} batches; at least {minimum} are required"
        )
    ordered_groups = _hashed_group_order(
        groups, seed=str(frozen["partition_policy"]["hash_seed"])
    )
    counts = _partition_group_counts(
        len(groups), frozen["partition_policy"]["fractions"]
    )
    group_partition: dict[str, str] = {}
    offset = 0
    for partition in PARTITIONS:
        selected = ordered_groups[offset : offset + counts[partition]]
        group_partition.update({group: partition for group in selected})
        offset += counts[partition]
    assignments = [
        {
            "cell_id": str(row.cell_id),
            "batch_id": str(row.batch_id),
            "condition_id": str(row.condition_id),
            "partition": group_partition[str(row.batch_id)],
        }
        for row in rows.sort_values("cell_id", kind="stable").itertuples(index=False)
    ]
    manifest: dict[str, object] = {
        "schema_version": PARTITION_SCHEMA_VERSION,
        "adapter_id": str(frozen["adapter_id"]),
        "dataset_id": str(frozen["dataset_id"]),
        "private_only": True,
        "adapter_config_sha256": canonical_json_sha256(frozen),
        "partition_grouping_field": "batch_id",
        "partition_selection_uses_measurement_values": False,
        "partition_selection_uses_target_outcomes": False,
        "measurement_value_fields_accepted": False,
        "group_counts": counts,
        "cell_count": len(assignments),
        "assignments": assignments,
        "public_release_permitted": False,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return manifest


def validate_private_cycle_partition_manifest(
    manifest: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    frozen = validate_private_cycle_adapter_config(config)
    value = deepcopy(dict(manifest))
    expected_hash = value.pop("manifest_content_sha256", None)
    if expected_hash != canonical_json_sha256(value):
        raise PrivateCycleAdapterError("Private partition manifest hash changed")
    value["manifest_content_sha256"] = expected_hash
    if value.get("schema_version") != PARTITION_SCHEMA_VERSION:
        raise PrivateCycleAdapterError("Private partition manifest schema changed")
    if value.get("adapter_config_sha256") != canonical_json_sha256(frozen):
        raise PrivateCycleAdapterError("Partition manifest config binding changed")
    if value.get("private_only") is not True or value.get(
        "public_release_permitted"
    ) is not False:
        raise PrivateCycleAdapterError("Partition manifest privacy flags changed")
    if value.get("partition_selection_uses_measurement_values") is not False:
        raise PrivateCycleAdapterError("Partition manifest used measurement values")
    assignments = value.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise PrivateCycleAdapterError("Partition assignments are missing")
    cells: set[str] = set()
    batch_partitions: dict[str, str] = {}
    for item in assignments:
        if not isinstance(item, Mapping) or set(item) != {
            "cell_id",
            "batch_id",
            "condition_id",
            "partition",
        }:
            raise PrivateCycleAdapterError("Partition assignment schema changed")
        cell = _nonempty_string(item["cell_id"], path="$.assignments[].cell_id")
        batch = _nonempty_string(item["batch_id"], path="$.assignments[].batch_id")
        partition = str(item["partition"])
        if cell in cells or partition not in PARTITIONS:
            raise PrivateCycleAdapterError("Partition assignments are invalid")
        cells.add(cell)
        if batch in batch_partitions and batch_partitions[batch] != partition:
            raise PrivateCycleAdapterError("A batch crosses private partitions")
        batch_partitions[batch] = partition
    if len(cells) != int(value.get("cell_count", -1)):
        raise PrivateCycleAdapterError("Partition cell count changed")
    return value


def validate_private_cycle_bundle_manifest(
    manifest: Mapping[str, object], config: Mapping[str, object]
) -> dict[str, object]:
    """Validate a blind-bundle seal without opening either truth vault."""
    frozen = validate_private_cycle_adapter_config(config)
    value = deepcopy(dict(manifest))
    expected_hash = value.pop("manifest_content_sha256", None)
    if expected_hash != canonical_json_sha256(value):
        raise PrivateCycleAdapterError("Private bundle manifest hash changed")
    value["manifest_content_sha256"] = expected_hash
    if value.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise PrivateCycleAdapterError("Private bundle manifest schema changed")
    if value.get("adapter_config_sha256") != canonical_json_sha256(frozen):
        raise PrivateCycleAdapterError("Private bundle config binding changed")
    if value.get("private_only") is not True or value.get(
        "public_release_permitted"
    ) is not False:
        raise PrivateCycleAdapterError("Private bundle privacy flags changed")
    if value.get("prediction_inputs_contain_target_suffix_outcomes") is not False:
        raise PrivateCycleAdapterError("Private prediction inputs contain suffix truth")
    if value.get("truth_vault_must_be_inaccessible_to_prediction_process") is not True:
        raise PrivateCycleAdapterError("Private truth-vault isolation flag changed")
    if value.get("partition_selection_uses_target_outcomes") is not False:
        raise PrivateCycleAdapterError("Private bundle partitions used target outcomes")
    expected_artifacts = {
        "development_trajectories",
        "calibration_prefixes",
        "calibration_truth_vault",
        "locked_test_prefixes",
        "locked_test_truth_vault",
    }
    counts = value.get("artifact_row_counts")
    hashes = value.get("artifact_canonical_sha256")
    if not isinstance(counts, Mapping) or set(counts) != expected_artifacts:
        raise PrivateCycleAdapterError("Private bundle artifact counts changed")
    if not isinstance(hashes, Mapping) or set(hashes) != expected_artifacts:
        raise PrivateCycleAdapterError("Private bundle artifact hashes changed")
    if any(isinstance(item, bool) or int(item) <= 0 for item in counts.values()):
        raise PrivateCycleAdapterError("Private bundle artifact count is invalid")
    if any(
        not isinstance(item, str)
        or len(item) != 64
        or any(character not in "0123456789abcdef" for character in item)
        for item in hashes.values()
    ):
        raise PrivateCycleAdapterError("Private bundle artifact hash is invalid")
    if value.get("landmark_visit_counts") != frozen["trajectory_policy"][
        "landmark_visit_counts"
    ]:
        raise PrivateCycleAdapterError("Private bundle landmarks changed")
    return value


def verify_private_cycle_bundle_frame(
    name: str,
    frame: pd.DataFrame,
    bundle_manifest: Mapping[str, object],
) -> None:
    """Verify one opened bundle member against the custodian's canonical hash."""
    hashes = bundle_manifest.get("artifact_canonical_sha256")
    counts = bundle_manifest.get("artifact_row_counts")
    if not isinstance(hashes, Mapping) or name not in hashes:
        raise PrivateCycleAdapterError(f"Unknown private bundle member: {name}")
    if not isinstance(counts, Mapping) or len(frame) != int(counts[name]):
        raise PrivateCycleAdapterError(f"Private bundle row count changed: {name}")
    observed = canonical_frame_sha256(frame, tuple(frame.columns))
    if observed != str(hashes[name]):
        raise PrivateCycleAdapterError(f"Private bundle frame changed: {name}")


def normalize_private_cycle_measurements(
    measurements: pd.DataFrame,
    partition_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Normalize private measurements after the metadata-only partition freeze."""
    frozen = validate_private_cycle_adapter_config(config)
    manifest = validate_private_cycle_partition_manifest(partition_manifest, frozen)
    if tuple(measurements.columns) != PRIVATE_MEASUREMENT_COLUMNS:
        raise PrivateCycleAdapterError("Private measurement columns changed")
    if measurements.empty or measurements.isna().any().any():
        raise PrivateCycleAdapterError("Private measurements are empty or incomplete")
    data = measurements.copy()
    string_columns = (
        "record_id",
        "cell_id",
        "batch_id",
        "condition_id",
        "cathode_chemistry",
        "quality_status",
    )
    for column in string_columns:
        data[column] = data[column].astype(str)
        if (data[column].str.strip() == "").any():
            raise PrivateCycleAdapterError(f"{column} contains empty values")
    if data["record_id"].duplicated().any():
        raise PrivateCycleAdapterError("Private record_id values must be unique")
    if not data["cathode_chemistry"].str.casefold().isin(
        {"lfp", "lifepo4", "lithium_iron_phosphate"}
    ).all():
        raise PrivateCycleAdapterError("Non-LFP measurements are not accepted")
    allowlist = set(frozen["trajectory_policy"]["quality_status_allowlist"])
    if not set(data["quality_status"]).issubset(allowlist):
        raise PrivateCycleAdapterError("Measurement quality status is not allowlisted")
    numeric_columns = tuple(
        column for column in PRIVATE_MEASUREMENT_COLUMNS if column not in string_columns
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise PrivateCycleAdapterError(f"{column} contains non-finite values")
    if (data[["elapsed_days", "equivalent_full_cycles"]] < 0.0).any().any():
        raise PrivateCycleAdapterError("Time and exposure coordinates must be non-negative")
    if (data[["capacity_ah", "reference_capacity_ah"]] <= 0.0).any().any():
        raise PrivateCycleAdapterError("Capacity values must be positive")
    if (
        (data["min_soc_pct"] < 0.0)
        | (data["max_soc_pct"] > 100.0)
        | (data["min_soc_pct"] >= data["max_soc_pct"])
    ).any():
        raise PrivateCycleAdapterError("SOC bounds are invalid")

    assignment_rows = manifest["assignments"]
    assignment = pd.DataFrame(assignment_rows).set_index("cell_id")
    observed_cells = set(data["cell_id"])
    if observed_cells != set(assignment.index):
        raise PrivateCycleAdapterError("Measurement cells differ from frozen partitions")
    for cell_id, cell in data.groupby("cell_id", sort=True):
        expected = assignment.loc[cell_id]
        if set(cell["batch_id"]) != {str(expected["batch_id"])} or set(
            cell["condition_id"]
        ) != {str(expected["condition_id"])}:
            raise PrivateCycleAdapterError("Cell identity changed after partition freeze")
        ordered = cell.sort_values("visit_index", kind="stable")
        visit = ordered["visit_index"].to_numpy(dtype=float)
        if not np.array_equal(visit, np.arange(len(ordered), dtype=float)):
            raise PrivateCycleAdapterError("visit_index must be contiguous from zero")
        if (np.diff(ordered["elapsed_days"].to_numpy(dtype=float)) <= 0.0).any():
            raise PrivateCycleAdapterError("elapsed_days must increase within each cell")
        if (
            np.diff(ordered["equivalent_full_cycles"].to_numpy(dtype=float))
            <= 0.0
        ).any():
            raise PrivateCycleAdapterError(
                "equivalent_full_cycles must increase within each cell"
            )
        condition_fields = (
            "temperature_c",
            "min_soc_pct",
            "max_soc_pct",
            "charge_c_rate",
            "discharge_c_rate",
            "reference_capacity_ah",
        )
        if any(cell[field].nunique() != 1 for field in condition_fields):
            raise PrivateCycleAdapterError("Cell condition fields must remain constant")

    rows = []
    for row in data.itertuples(index=False):
        partition = str(assignment.loc[str(row.cell_id), "partition"])
        rows.append(
            {
                "partition": partition,
                "dataset_id": str(frozen["dataset_id"]),
                "cell_id": str(row.cell_id),
                "condition_id": str(row.condition_id),
                "temperature_c": float(row.temperature_c),
                "min_soc_pct": float(row.min_soc_pct),
                "max_soc_pct": float(row.max_soc_pct),
                "dod_fraction": float((row.max_soc_pct - row.min_soc_pct) / 100.0),
                "charge_c_rate": float(row.charge_c_rate),
                "discharge_c_rate": float(row.discharge_c_rate),
                "visit_index": int(row.visit_index),
                "elapsed_days": float(row.elapsed_days),
                "equivalent_full_cycles": float(row.equivalent_full_cycles),
                "capacity_ah": float(row.capacity_ah),
                "capacity_retention_pct": float(
                    100.0 * row.capacity_ah / row.reference_capacity_ah
                ),
                "rpt_cycle_count": int(row.visit_index),
            }
        )
    return pd.DataFrame(rows, columns=PARTITIONED_TRAJECTORY_COLUMNS).sort_values(
        ["partition", "condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )


def build_private_cycle_blind_bundle(
    normalized: pd.DataFrame,
    partition_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    """Build separated prediction inputs and truth vaults after partition freeze."""
    frozen = validate_private_cycle_adapter_config(config)
    manifest = validate_private_cycle_partition_manifest(partition_manifest, frozen)
    if tuple(normalized.columns) != PARTITIONED_TRAJECTORY_COLUMNS:
        raise PrivateCycleAdapterError("Normalized private trajectory columns changed")
    landmarks = tuple(
        int(value) for value in frozen["trajectory_policy"]["landmark_visit_counts"]
    )
    minimum_future = int(frozen["trajectory_policy"]["minimum_future_visits"])
    score_end = float(
        frozen["trajectory_policy"]["score_end_equivalent_full_cycles"]
    )
    for cell_id, cell in normalized.groupby("cell_id", sort=True):
        ordered = cell.sort_values("visit_index", kind="stable")
        for landmark in landmarks:
            future = ordered.loc[
                (ordered["visit_index"] >= landmark)
                & (ordered["equivalent_full_cycles"] <= score_end)
            ]
            if len(future) < minimum_future:
                raise PrivateCycleAdapterError(
                    f"Cell {cell_id} lacks score-window future support at "
                    f"landmark {landmark}"
                )

    development = normalized.loc[
        normalized["partition"] == "development", RPT_TRAJECTORY_COLUMNS
    ].copy()
    outputs: dict[str, pd.DataFrame] = {
        "development_trajectories": development.reset_index(drop=True)
    }
    for partition in ("calibration", "locked_test"):
        selected = normalized.loc[normalized["partition"] == partition]
        prefix_rows = []
        truth_rows = []
        for _, cell in selected.groupby("cell_id", sort=True):
            ordered = cell.sort_values("visit_index", kind="stable")
            truth_rows.extend(ordered.to_dict("records"))
            for landmark in landmarks:
                for row in ordered.iloc[:landmark].to_dict("records"):
                    prefix_rows.append(
                        {
                            "partition": partition,
                            "landmark_visit_count": landmark,
                            **{column: row[column] for column in RPT_TRAJECTORY_COLUMNS},
                        }
                    )
        outputs[f"{partition}_prefixes"] = pd.DataFrame(
            prefix_rows, columns=PARTITIONED_PREFIX_COLUMNS
        )
        outputs[f"{partition}_truth_vault"] = pd.DataFrame(
            truth_rows, columns=PARTITIONED_TRAJECTORY_COLUMNS
        )

    artifact_hashes = {
        name: canonical_frame_sha256(frame, tuple(frame.columns))
        for name, frame in outputs.items()
    }
    bundle_manifest: dict[str, object] = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "adapter_id": str(frozen["adapter_id"]),
        "dataset_id": str(frozen["dataset_id"]),
        "private_only": True,
        "adapter_config_sha256": canonical_json_sha256(frozen),
        "partition_manifest_content_sha256": manifest["manifest_content_sha256"],
        "landmark_visit_counts": list(landmarks),
        "prediction_inputs_contain_target_suffix_outcomes": False,
        "truth_vault_must_be_inaccessible_to_prediction_process": True,
        "partition_selection_uses_target_outcomes": False,
        "artifact_row_counts": {
            name: len(frame) for name, frame in outputs.items()
        },
        "artifact_canonical_sha256": artifact_hashes,
        "claim_boundary": (
            "Bundle construction is a private custodian operation, not model "
            "validation. A blind claim requires a separate prediction-process "
            "identity that cannot read either truth vault."
        ),
        "public_release_permitted": False,
    }
    bundle_manifest["manifest_content_sha256"] = canonical_json_sha256(
        bundle_manifest
    )
    return outputs, bundle_manifest


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "PARTITIONED_PREFIX_COLUMNS",
    "PARTITIONED_TRAJECTORY_COLUMNS",
    "PARTITION_METADATA_COLUMNS",
    "PARTITION_SCHEMA_VERSION",
    "PRIVATE_MEASUREMENT_COLUMNS",
    "PrivateCycleAdapterError",
    "SCHEMA_VERSION",
    "build_private_cycle_blind_bundle",
    "freeze_private_cycle_partitions",
    "normalize_private_cycle_measurements",
    "validate_private_cycle_adapter_config",
    "validate_private_cycle_bundle_manifest",
    "validate_private_cycle_partition_manifest",
    "verify_private_cycle_bundle_frame",
]
