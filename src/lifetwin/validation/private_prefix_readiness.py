"""Truth-free readiness and domain-support audit for private LFP prefixes."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.validation.private_cycle_adapter import (
    PARTITIONED_PREFIX_COLUMNS,
    PrivateCycleAdapterError,
    validate_private_cycle_adapter_config,
    validate_private_cycle_partition_manifest,
)


READINESS_SCHEMA_VERSION = "lifetwin.private_prefix_readiness.config.v1"
SUPPORT_FEATURE_IDS = (
    "temperature_c",
    "dod_fraction",
    "charge_c_rate",
    "discharge_c_rate",
    "last_elapsed_days",
    "last_equivalent_full_cycles",
    "prefix_efc_per_day",
    "retention_change_pp",
    "retention_slope_pp_per_efc",
)
DRIFT_COLUMNS = (
    "partition",
    "cell_id",
    "batch_id",
    "landmark_visit_count",
    "nearest_development_cell_id",
    "support_distance",
    "support_threshold",
    "distance_ratio",
    "within_development_support",
)


def validate_private_prefix_readiness_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    value = deepcopy(dict(config))
    expected = {
        "schema_version",
        "audit_id",
        "private_only",
        "minimum_cells_by_partition",
        "support_distance",
        "decision_gate",
        "claim_boundary",
    }
    if set(value) != expected:
        raise PrivateCycleAdapterError("Private readiness config keys changed")
    if value["schema_version"] != READINESS_SCHEMA_VERSION:
        raise PrivateCycleAdapterError("Private readiness config schema changed")
    if value["private_only"] is not True:
        raise PrivateCycleAdapterError("Private readiness audit must remain private")
    minimums = value["minimum_cells_by_partition"]
    if not isinstance(minimums, Mapping) or set(minimums) != {
        "development",
        "calibration",
        "locked_test",
    }:
        raise PrivateCycleAdapterError("Private readiness cell minima changed")
    if any(isinstance(item, bool) or int(item) < 2 for item in minimums.values()):
        raise PrivateCycleAdapterError("Private readiness cell minima are invalid")
    support = value["support_distance"]
    if not isinstance(support, Mapping) or set(support) != {
        "feature_ids",
        "feature_scale_floors",
        "development_loo_quantile",
        "threshold_multiplier",
        "minimum_threshold",
    }:
        raise PrivateCycleAdapterError("Private readiness support config changed")
    if tuple(support["feature_ids"]) != SUPPORT_FEATURE_IDS:
        raise PrivateCycleAdapterError("Private readiness support features changed")
    floors = support["feature_scale_floors"]
    if not isinstance(floors, Mapping) or set(floors) != set(SUPPORT_FEATURE_IDS):
        raise PrivateCycleAdapterError("Private readiness scale floors changed")
    if any(
        not math.isfinite(float(item)) or float(item) <= 0 for item in floors.values()
    ):
        raise PrivateCycleAdapterError("Private readiness scale floors are invalid")
    quantile = float(support["development_loo_quantile"])
    multiplier = float(support["threshold_multiplier"])
    threshold = float(support["minimum_threshold"])
    if not 0.5 <= quantile < 1.0 or multiplier < 1.0 or threshold <= 0.0:
        raise PrivateCycleAdapterError(
            "Private readiness support thresholds are invalid"
        )
    gate = value["decision_gate"]
    if not isinstance(gate, Mapping) or set(gate) != {
        "maximum_out_of_support_fraction_by_partition",
        "maximum_distance_ratio",
        "require_complete_landmarks",
        "require_batch_disjoint_partitions",
        "truth_vault_inputs_permitted",
    }:
        raise PrivateCycleAdapterError("Private readiness decision gate changed")
    if not 0.0 <= float(gate["maximum_out_of_support_fraction_by_partition"]) < 1.0:
        raise PrivateCycleAdapterError("Private readiness OOD fraction is invalid")
    if float(gate["maximum_distance_ratio"]) < 1.0:
        raise PrivateCycleAdapterError("Private readiness distance ratio is invalid")
    if gate["require_complete_landmarks"] is not True:
        raise PrivateCycleAdapterError("Private readiness must require all landmarks")
    if gate["require_batch_disjoint_partitions"] is not True:
        raise PrivateCycleAdapterError("Private readiness must require batch isolation")
    if gate["truth_vault_inputs_permitted"] is not False:
        raise PrivateCycleAdapterError("Private readiness cannot accept truth vaults")
    return value


def _validated_development(frame: pd.DataFrame) -> pd.DataFrame:
    if tuple(frame.columns) != RPT_TRAJECTORY_COLUMNS:
        raise PrivateCycleAdapterError("Development trajectory columns changed")
    if frame.empty or frame.isna().any().any():
        raise PrivateCycleAdapterError("Development trajectories are incomplete")
    return frame.sort_values(
        ["condition_id", "cell_id", "visit_index"], kind="stable", ignore_index=True
    )


def _validated_prefixes(frame: pd.DataFrame, partition: str) -> pd.DataFrame:
    if tuple(frame.columns) != PARTITIONED_PREFIX_COLUMNS:
        raise PrivateCycleAdapterError(f"{partition} prefix columns changed")
    if frame.empty or frame.isna().any().any():
        raise PrivateCycleAdapterError(f"{partition} prefixes are incomplete")
    if set(frame["partition"].astype(str)) != {partition}:
        raise PrivateCycleAdapterError(f"{partition} prefix identity changed")
    return frame.sort_values(
        ["cell_id", "landmark_visit_count", "visit_index"],
        kind="stable",
        ignore_index=True,
    )


def _prefix_vector(group: pd.DataFrame) -> np.ndarray:
    ordered = group.sort_values("visit_index", kind="stable")
    for column in (
        "temperature_c",
        "dod_fraction",
        "charge_c_rate",
        "discharge_c_rate",
    ):
        if ordered[column].nunique() != 1:
            raise PrivateCycleAdapterError(
                f"Private prefix condition changed within cell: {column}"
            )
    elapsed = ordered["elapsed_days"].to_numpy(dtype=float)
    efc = ordered["equivalent_full_cycles"].to_numpy(dtype=float)
    retention = ordered["capacity_retention_pct"].to_numpy(dtype=float)
    if (
        len(ordered) < 3
        or not np.isfinite(np.column_stack([elapsed, efc, retention])).all()
        or (np.diff(elapsed) <= 0.0).any()
        or (np.diff(efc) <= 0.0).any()
    ):
        raise PrivateCycleAdapterError("Private prefix coordinates are invalid")
    elapsed_span = float(elapsed[-1] - elapsed[0])
    efc_span = float(efc[-1] - efc[0])
    if elapsed_span <= 0.0 or efc_span <= 0.0:
        raise PrivateCycleAdapterError("Private prefix exposure span is invalid")
    return np.asarray(
        [
            float(ordered["temperature_c"].iloc[0]),
            float(ordered["dod_fraction"].iloc[0]),
            float(ordered["charge_c_rate"].iloc[0]),
            float(ordered["discharge_c_rate"].iloc[0]),
            float(elapsed[-1]),
            float(efc[-1]),
            efc_span / elapsed_span,
            float(retention[-1] - retention[0]),
            float((retention[-1] - retention[0]) / efc_span),
        ],
        dtype=float,
    )


def _development_vectors(
    development: pd.DataFrame, landmarks: Sequence[int]
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for landmark in landmarks:
        values: dict[str, np.ndarray] = {}
        for cell_id, cell in development.groupby("cell_id", sort=True):
            ordered = cell.sort_values("visit_index", kind="stable")
            if len(ordered) < landmark:
                raise PrivateCycleAdapterError(
                    f"Development cell {cell_id} lacks landmark {landmark}"
                )
            values[str(cell_id)] = _prefix_vector(ordered.iloc[:landmark])
        result[int(landmark)] = values
    return result


def _target_vectors(
    prefixes: pd.DataFrame,
    partition: str,
    expected_cells: set[str],
    landmarks: Sequence[int],
) -> dict[tuple[str, int], np.ndarray]:
    observed_cells = set(prefixes["cell_id"].astype(str))
    if observed_cells != expected_cells:
        raise PrivateCycleAdapterError(
            f"{partition} prefix cells differ from frozen assignments"
        )
    expected_landmarks = set(int(value) for value in landmarks)
    result: dict[tuple[str, int], np.ndarray] = {}
    for cell_id, cell in prefixes.groupby("cell_id", sort=True):
        if set(cell["landmark_visit_count"].astype(int)) != expected_landmarks:
            raise PrivateCycleAdapterError(
                f"{partition} cell {cell_id} lacks a frozen landmark"
            )
        for landmark, group in cell.groupby("landmark_visit_count", sort=True):
            landmark = int(landmark)
            visits = group["visit_index"].to_numpy(dtype=int)
            if len(group) != landmark or not np.array_equal(
                visits, np.arange(landmark, dtype=int)
            ):
                raise PrivateCycleAdapterError(
                    f"{partition} prefix exposes incomplete or future rows"
                )
            result[(str(cell_id), landmark)] = _prefix_vector(group)
    return result


def _robust_scale(
    matrix: np.ndarray, config: Mapping[str, object]
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - median), axis=0)
    floors = np.asarray(
        [
            float(config["support_distance"]["feature_scale_floors"][feature])
            for feature in SUPPORT_FEATURE_IDS
        ],
        dtype=float,
    )
    return median, np.maximum(mad, floors)


def _nearest_distance(
    value: np.ndarray,
    references: Mapping[str, np.ndarray],
    scale: np.ndarray,
    *,
    excluded_cell_id: str | None = None,
) -> tuple[str, float]:
    distances = {
        cell_id: float(np.sqrt(np.mean(np.square((value - item) / scale))))
        for cell_id, item in references.items()
        if cell_id != excluded_cell_id
    }
    if not distances:
        raise PrivateCycleAdapterError("Private readiness has no reference neighbour")
    nearest = min(distances, key=lambda cell_id: (distances[cell_id], cell_id))
    return nearest, distances[nearest]


def audit_private_prefix_readiness(
    development_trajectories: pd.DataFrame,
    calibration_prefixes: pd.DataFrame,
    locked_test_prefixes: pd.DataFrame,
    partition_manifest: Mapping[str, object],
    adapter_config: Mapping[str, object],
    readiness_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    adapter = validate_private_cycle_adapter_config(adapter_config)
    manifest = validate_private_cycle_partition_manifest(partition_manifest, adapter)
    readiness = validate_private_prefix_readiness_config(readiness_config)
    development = _validated_development(development_trajectories)
    calibration = _validated_prefixes(calibration_prefixes, "calibration")
    locked = _validated_prefixes(locked_test_prefixes, "locked_test")
    assignments = pd.DataFrame(manifest["assignments"])
    cells_by_partition = {
        partition: set(
            assignments.loc[assignments["partition"] == partition, "cell_id"].astype(
                str
            )
        )
        for partition in ("development", "calibration", "locked_test")
    }
    if set(development["cell_id"].astype(str)) != cells_by_partition["development"]:
        raise PrivateCycleAdapterError(
            "Development cells differ from frozen assignments"
        )
    batch_sets = {
        partition: set(
            assignments.loc[assignments["partition"] == partition, "batch_id"].astype(
                str
            )
        )
        for partition in ("development", "calibration", "locked_test")
    }
    batch_disjoint = all(
        not batch_sets[left] & batch_sets[right]
        for left, right in (
            ("development", "calibration"),
            ("development", "locked_test"),
            ("calibration", "locked_test"),
        )
    )
    landmarks = tuple(
        int(value) for value in adapter["trajectory_policy"]["landmark_visit_counts"]
    )
    development_vectors = _development_vectors(development, landmarks)
    target_vectors = {
        "calibration": _target_vectors(
            calibration, "calibration", cells_by_partition["calibration"], landmarks
        ),
        "locked_test": _target_vectors(
            locked, "locked_test", cells_by_partition["locked_test"], landmarks
        ),
    }
    assignment_by_cell = assignments.set_index("cell_id")
    rows: list[dict[str, object]] = []
    for landmark in landmarks:
        reference = development_vectors[landmark]
        matrix = np.vstack([reference[cell_id] for cell_id in sorted(reference)])
        _, scale = _robust_scale(matrix, readiness)
        loo_distances = [
            _nearest_distance(
                reference[cell_id],
                reference,
                scale,
                excluded_cell_id=cell_id,
            )[1]
            for cell_id in sorted(reference)
        ]
        support = readiness["support_distance"]
        threshold = max(
            float(support["minimum_threshold"]),
            float(
                np.quantile(
                    loo_distances,
                    float(support["development_loo_quantile"]),
                    method="higher",
                )
            )
            * float(support["threshold_multiplier"]),
        )
        for partition in ("calibration", "locked_test"):
            for (cell_id, target_landmark), vector in target_vectors[partition].items():
                if target_landmark != landmark:
                    continue
                nearest, distance = _nearest_distance(vector, reference, scale)
                ratio = distance / threshold
                rows.append(
                    {
                        "partition": partition,
                        "cell_id": cell_id,
                        "batch_id": str(assignment_by_cell.loc[cell_id, "batch_id"]),
                        "landmark_visit_count": landmark,
                        "nearest_development_cell_id": nearest,
                        "support_distance": distance,
                        "support_threshold": threshold,
                        "distance_ratio": ratio,
                        "within_development_support": bool(distance <= threshold),
                    }
                )
    drift = pd.DataFrame(rows, columns=DRIFT_COLUMNS).sort_values(
        ["partition", "cell_id", "landmark_visit_count"],
        kind="stable",
        ignore_index=True,
    )
    minimums = readiness["minimum_cells_by_partition"]
    cell_count_checks = {
        partition: len(cells_by_partition[partition]) >= int(minimums[partition])
        for partition in ("development", "calibration", "locked_test")
    }
    out_fraction = {
        str(partition): float(1.0 - group["within_development_support"].mean())
        for partition, group in drift.groupby("partition", sort=True)
    }
    maximum_ratio = {
        str(partition): float(group["distance_ratio"].max())
        for partition, group in drift.groupby("partition", sort=True)
    }
    gate = readiness["decision_gate"]
    support_checks = {
        partition: out_fraction[partition]
        <= float(gate["maximum_out_of_support_fraction_by_partition"])
        and maximum_ratio[partition] <= float(gate["maximum_distance_ratio"])
        for partition in ("calibration", "locked_test")
    }
    checks = {
        "minimum_cell_counts": bool(all(cell_count_checks.values())),
        "batch_disjoint_partitions": bool(batch_disjoint),
        "complete_prefix_landmarks": True,
        "calibration_within_development_support": support_checks["calibration"],
        "locked_test_within_development_support": support_checks["locked_test"],
        "truth_vault_inputs_excluded": True,
    }
    ready = bool(all(checks.values()))
    decision: dict[str, object] = {
        "schema_version": "lifetwin.private_prefix_readiness.result.v1",
        "audit_id": readiness["audit_id"],
        "adapter_id": adapter["adapter_id"],
        "private_only": True,
        "cell_counts": {
            partition: len(cells_by_partition[partition])
            for partition in ("development", "calibration", "locked_test")
        },
        "batch_counts": {
            partition: len(batch_sets[partition])
            for partition in ("development", "calibration", "locked_test")
        },
        "landmark_visit_counts": list(landmarks),
        "out_of_support_fraction_by_partition": out_fraction,
        "maximum_distance_ratio_by_partition": maximum_ratio,
        "decision_checks": checks,
        "ready_to_issue_predictions": ready,
        "locked_test_truth_may_be_opened": False,
        "truth_vault_inputs_read": False,
        "target_suffix_outcomes_read": False,
        "model_accuracy_evidence_created": False,
        "public_release_permitted": False,
        "next_action": (
            "issue_calibration_prefix_predictions_without_truth_access"
            if ready
            else "stop_and_resolve_data_support_or_bundle_readiness_failures"
        ),
        "hashes": {
            "adapter_config_semantic_sha256": canonical_json_sha256(adapter),
            "partition_manifest_content_sha256": manifest["manifest_content_sha256"],
            "development_trajectories_sha256": canonical_frame_sha256(
                development, RPT_TRAJECTORY_COLUMNS
            ),
            "calibration_prefixes_sha256": canonical_frame_sha256(
                calibration, PARTITIONED_PREFIX_COLUMNS
            ),
            "locked_test_prefixes_sha256": canonical_frame_sha256(
                locked, PARTITIONED_PREFIX_COLUMNS
            ),
            "drift_table_sha256": canonical_frame_sha256(drift, DRIFT_COLUMNS),
        },
        "claim_boundary": readiness["claim_boundary"],
    }
    decision["decision_content_sha256"] = canonical_json_sha256(decision)
    return drift, decision


__all__ = [
    "DRIFT_COLUMNS",
    "READINESS_SCHEMA_VERSION",
    "SUPPORT_FEATURE_IDS",
    "audit_private_prefix_readiness",
    "validate_private_prefix_readiness_config",
]
