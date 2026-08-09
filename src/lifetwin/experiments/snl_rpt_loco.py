"""Grouped cross-condition development benchmark for SNL LFP RPT trajectories."""

from __future__ import annotations

from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.snl import DATASET_ID, RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.snl_lfp_rpt_loco.config.v1"
EXPERIMENT_ID = "snl_lfp_rpt_loco_v1"
CONFIG_SEMANTIC_SHA256 = (
    "502054492f33bbf78b6a35c624226308635d80a610f7b322ebe0d293b36b09c8"
)
BASE_MODEL_IDS = (
    "target_prefix_persistence",
    "target_prefix_constrained_linear_efc",
    "target_prefix_constrained_sqrt_efc",
    "target_prefix_bounded_power_efc",
    "condition_ridge_delta",
    "nearest_reference_delta_transfer",
)
SELECTOR_MODEL_ID = "snl_nested_safe_hard_local_risk_selector"
MODEL_IDS = (*BASE_MODEL_IDS, SELECTOR_MODEL_ID)
SIGNATURE_FEATURE_IDS = (
    "capacity_full_slope_pp_per_1000_efc",
    "capacity_sqrt_slope_pp_per_sqrt_1000_efc",
    "capacity_residual_std_pp",
    "capacity_last_retention_pct",
    "log_observation_density_per_1000_efc",
    "temperature_c",
    "dod_fraction",
    "discharge_c_rate",
)
REFERENCE_COLUMNS = ("outer_condition_id", *RPT_TRAJECTORY_COLUMNS)
TARGET_PREFIX_COLUMNS = (
    "outer_condition_id",
    *RPT_TRAJECTORY_COLUMNS,
    "landmark_visit_count",
)
TARGET_TRUTH_COLUMNS = ("outer_condition_id", *RPT_TRAJECTORY_COLUMNS)
PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "model_id",
    "forecast_equivalent_full_cycles",
    "predicted_capacity_retention_pct",
    "selected_expert_model_id",
    "selection_mode",
    "evidence_status",
)
DECISION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "issued",
    "selected_expert_model_id",
    "selection_mode",
    "evidence_status",
    "safe_pool_model_ids_json",
    "global_risks_json",
    "local_risks_json",
    "relative_local_risk_margin",
    "mean_neighbor_distance",
    "out_of_domain_threshold",
    "nearest_inner_validation_cells_json",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "model_id",
    "issued",
    "future_observation_count",
    "trajectory_iae_pp",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
)


class SNLRPTLOCOError(ValueError):
    """Raised when the frozen grouped SNL benchmark contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SNLRPTLOCOError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_snl_rpt_loco_config(config: Mapping[str, object]) -> dict[str, object]:
    detached = json.loads(
        json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    if canonical_json_sha256(detached) != CONFIG_SEMANTIC_SHA256:
        raise SNLRPTLOCOError("Frozen SNL RPT LOCO config changed")
    if detached.get("schema_version") != SCHEMA_VERSION:
        raise SNLRPTLOCOError("SNL RPT LOCO config schema changed")
    if detached.get("experiment_id") != EXPERIMENT_ID:
        raise SNLRPTLOCOError("SNL RPT LOCO experiment identity changed")
    if tuple(detached["base_experts"]["model_ids"]) != BASE_MODEL_IDS:
        raise SNLRPTLOCOError("SNL RPT LOCO model registry changed")
    if tuple(detached["prefix_signature"]["feature_ids"]) != SIGNATURE_FEATURE_IDS:
        raise SNLRPTLOCOError("SNL RPT LOCO signature registry changed")
    if detached["safe_hard_selector"]["model_id"] != SELECTOR_MODEL_ID:
        raise SNLRPTLOCOError("SNL RPT LOCO selector identity changed")
    if detached["dataset"]["dataset_id"] != DATASET_ID:
        raise SNLRPTLOCOError("SNL RPT LOCO dataset identity changed")
    return detached


def load_snl_rpt_loco_config(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SNLRPTLOCOError(f"Non-finite JSON constant: {token}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise SNLRPTLOCOError(f"Cannot load SNL RPT LOCO config: {path}") from exc
    if not isinstance(value, Mapping):
        raise SNLRPTLOCOError("SNL RPT LOCO config must be an object")
    return validate_snl_rpt_loco_config(value)


def _validated_trajectories(
    trajectories: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    if tuple(trajectories.columns) != RPT_TRAJECTORY_COLUMNS:
        raise SNLRPTLOCOError("SNL RPT trajectory columns changed")
    result = trajectories.copy()
    string_columns = {"dataset_id", "cell_id", "condition_id"}
    for column in string_columns:
        if result[column].isna().any():
            raise SNLRPTLOCOError(f"Null SNL trajectory identity: {column}")
        result[column] = result[column].astype(str)
    numeric_columns = [
        column for column in RPT_TRAJECTORY_COLUMNS if column not in string_columns
    ]
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(
        numeric.to_numpy(dtype=float)
    ).all():
        raise SNLRPTLOCOError("SNL RPT trajectories must contain finite values")
    for column in ("visit_index", "rpt_cycle_count"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise SNLRPTLOCOError(f"SNL RPT {column} must be integer-valued")
        result[column] = values.astype(np.int64)
    for column in set(numeric_columns) - {"visit_index", "rpt_cycle_count"}:
        result[column] = numeric[column].astype(float)
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise SNLRPTLOCOError("SNL RPT dataset identity changed")
    if result["cell_id"].nunique() != int(config["dataset"]["physical_cell_count"]):
        raise SNLRPTLOCOError("SNL RPT physical-cell count changed")
    if result["condition_id"].nunique() != int(
        config["dataset"]["condition_cluster_count"]
    ):
        raise SNLRPTLOCOError("SNL RPT condition-cluster count changed")
    if len(result) != int(config["dataset"]["rpt_trajectory_row_count"]):
        raise SNLRPTLOCOError("SNL RPT trajectory row count changed")
    for cell_id, cell in result.groupby("cell_id", sort=True):
        if cell["condition_id"].nunique() != 1:
            raise SNLRPTLOCOError(f"SNL condition changes within {cell_id}")
        ordered = cell.sort_values("visit_index")
        if ordered["visit_index"].tolist() != list(range(len(ordered))):
            raise SNLRPTLOCOError(f"SNL visit indices changed for {cell_id}")
        if not ordered["equivalent_full_cycles"].is_monotonic_increasing:
            raise SNLRPTLOCOError(f"SNL EFC order changed for {cell_id}")
        if ordered["equivalent_full_cycles"].duplicated().any():
            raise SNLRPTLOCOError(f"SNL EFC values are duplicated for {cell_id}")
        if abs(float(ordered.iloc[0]["capacity_retention_pct"]) - 100.0) > 1e-9:
            raise SNLRPTLOCOError(f"SNL initial retention changed for {cell_id}")
    result = result.sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    expected_hash = str(config["dataset"]["rpt_trajectory_sha256"])
    if canonical_frame_sha256(result, RPT_TRAJECTORY_COLUMNS) != expected_hash:
        raise SNLRPTLOCOError("Frozen SNL RPT trajectory hash changed")
    return result


def build_snl_rpt_loco_inputs(
    trajectories: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build outer-fold references, prefix-only targets, and separate truth."""
    frozen = validate_snl_rpt_loco_config(config)
    data = _validated_trajectories(trajectories, frozen)
    landmarks = tuple(
        int(value) for value in frozen["dynamic_landmarks"]["prefix_visit_counts"]
    )
    reference_frames: list[pd.DataFrame] = []
    prefix_frames: list[pd.DataFrame] = []
    truth_frames: list[pd.DataFrame] = []
    for condition_id in sorted(data["condition_id"].unique()):
        reference = data.loc[data["condition_id"] != condition_id].copy()
        reference.insert(0, "outer_condition_id", condition_id)
        reference_frames.append(reference.loc[:, REFERENCE_COLUMNS])
        target = data.loc[data["condition_id"] == condition_id].copy()
        truth = target.copy()
        truth.insert(0, "outer_condition_id", condition_id)
        truth_frames.append(truth.loc[:, TARGET_TRUTH_COLUMNS])
        for landmark in landmarks:
            prefix = target.loc[target["visit_index"] < landmark].copy()
            if prefix.groupby("cell_id").size().min() != landmark:
                raise SNLRPTLOCOError(
                    f"Target prefix support changed for {condition_id} N{landmark}"
                )
            prefix.insert(0, "outer_condition_id", condition_id)
            prefix["landmark_visit_count"] = landmark
            prefix_frames.append(prefix.loc[:, TARGET_PREFIX_COLUMNS])
    references = pd.concat(reference_frames, ignore_index=True).sort_values(
        ["outer_condition_id", "condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    prefixes = pd.concat(prefix_frames, ignore_index=True).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    truth = pd.concat(truth_frames, ignore_index=True).sort_values(
        ["outer_condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    for outer, frame in references.groupby("outer_condition_id", sort=True):
        if outer in set(frame["condition_id"]):
            raise SNLRPTLOCOError("Held-out condition leaked into reference rows")
    audit = {
        "schema_version": "lifetwin.snl_rpt_loco_input_audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "outer_fold_count": int(references["outer_condition_id"].nunique()),
        "target_cell_count": int(prefixes["cell_id"].nunique()),
        "landmark_visit_counts": list(landmarks),
        "reference_rows_sha256": canonical_frame_sha256(
            references, REFERENCE_COLUMNS
        ),
        "target_prefix_rows_sha256": canonical_frame_sha256(
            prefixes, TARGET_PREFIX_COLUMNS
        ),
        "target_truth_rows_sha256": canonical_frame_sha256(
            truth, TARGET_TRUTH_COLUMNS
        ),
        "target_suffix_rows_in_prefix_input": False,
        "target_condition_rows_in_its_reference_fold": False,
        "prediction_generated": False,
    }
    return references, prefixes, truth, audit


def _slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    centered = x - float(np.mean(x))
    denominator = float(np.sum(np.square(centered)))
    if denominator <= 0.0:
        raise SNLRPTLOCOError("Trajectory fit lacks distinct exposure values")
    slope = float(np.sum(centered * (y - float(np.mean(y)))) / denominator)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def _anchored_slope_prediction(
    prefix: pd.DataFrame,
    forecast_efc: np.ndarray,
    *,
    transform: str,
) -> np.ndarray:
    x = prefix["equivalent_full_cycles"].to_numpy(dtype=float)
    y = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    if transform == "linear":
        transformed = x / 1000.0
        future = forecast_efc / 1000.0
    elif transform == "sqrt":
        transformed = np.sqrt(x / 1000.0)
        future = np.sqrt(forecast_efc / 1000.0)
    else:
        raise SNLRPTLOCOError(f"Unknown exposure transform: {transform}")
    _, slope = _slope(transformed, y)
    slope = min(slope, 0.0)
    return y[-1] + slope * (future - transformed[-1])


def _bounded_power_prediction(
    prefix: pd.DataFrame,
    forecast_efc: np.ndarray,
    exponent_grid: Sequence[float],
) -> np.ndarray:
    x = prefix["equivalent_full_cycles"].to_numpy(dtype=float) / 1000.0
    y = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    best: tuple[float, float, float] | None = None
    for exponent in exponent_grid:
        transformed = np.power(x, float(exponent))
        intercept, slope = _slope(transformed, y)
        slope = min(slope, 0.0)
        fitted = intercept + slope * transformed
        error = float(np.mean(np.square(fitted - y)))
        candidate = (error, float(exponent), slope)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise SNLRPTLOCOError("Bounded power fit failed")
    _, exponent, slope = best
    transformed = np.power(x, exponent)
    future = np.power(forecast_efc / 1000.0, exponent)
    return y[-1] + slope * (future - transformed[-1])


def _condition_vector(cell: pd.DataFrame) -> np.ndarray:
    row = cell.iloc[0]
    return np.asarray(
        [
            float(row["temperature_c"]),
            float(row["dod_fraction"]),
            float(row["discharge_c_rate"]),
        ],
        dtype=float,
    )


def _condition_ridge_prediction(
    prefix: pd.DataFrame,
    references: pd.DataFrame,
    forecast_efc: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    features: list[np.ndarray] = []
    fade_rates: list[float] = []
    for _, cell in references.groupby("cell_id", sort=True):
        ordered = cell.sort_values("visit_index")
        exposure = np.sqrt(
            ordered["equivalent_full_cycles"].to_numpy(dtype=float) / 1000.0
        )
        fade = 100.0 - ordered["capacity_retention_pct"].to_numpy(dtype=float)
        denominator = float(np.dot(exposure, exposure))
        if denominator <= 0.0:
            continue
        fade_rates.append(max(0.0, float(np.dot(exposure, fade) / denominator)))
        features.append(_condition_vector(ordered))
    if len(features) < 2:
        raise SNLRPTLOCOError("Condition ridge lacks reference cells")
    matrix = np.vstack(features)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    penalty = np.eye(design.shape[1], dtype=float) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ np.asarray(fade_rates, dtype=float),
    )
    target_design = np.concatenate(
        [[1.0], (_condition_vector(prefix) - center) / scale]
    )
    fade_rate = max(0.0, float(target_design @ coefficients))
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    y0 = float(prefix.iloc[-1]["capacity_retention_pct"])
    return y0 - fade_rate * (
        np.sqrt(forecast_efc / 1000.0) - math.sqrt(x0 / 1000.0)
    )


def _signature(prefix: pd.DataFrame) -> dict[str, float]:
    ordered = prefix.sort_values("visit_index")
    x = ordered["equivalent_full_cycles"].to_numpy(dtype=float)
    y = ordered["capacity_retention_pct"].to_numpy(dtype=float)
    intercept, linear_slope = _slope(x / 1000.0, y)
    _, sqrt_slope = _slope(np.sqrt(x / 1000.0), y)
    residual = y - (intercept + linear_slope * (x / 1000.0))
    span = max(float(x[-1] - x[0]) / 1000.0, 1e-6)
    first = ordered.iloc[0]
    signature = {
        "capacity_full_slope_pp_per_1000_efc": float(linear_slope),
        "capacity_sqrt_slope_pp_per_sqrt_1000_efc": float(sqrt_slope),
        "capacity_residual_std_pp": float(np.sqrt(np.mean(np.square(residual)))),
        "capacity_last_retention_pct": float(y[-1]),
        "log_observation_density_per_1000_efc": float(math.log(len(y) / span)),
        "temperature_c": float(first["temperature_c"]),
        "dod_fraction": float(first["dod_fraction"]),
        "discharge_c_rate": float(first["discharge_c_rate"]),
    }
    if tuple(signature) != SIGNATURE_FEATURE_IDS or not all(
        math.isfinite(value) for value in signature.values()
    ):
        raise SNLRPTLOCOError("Prefix signature changed or became non-finite")
    return signature


def _robust_center_scale(
    signatures: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = signatures.loc[:, SIGNATURE_FEATURE_IDS].to_numpy(dtype=float)
    center = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - center), axis=0) * 1.4826
    scale = np.maximum(mad, 0.001)
    return center, scale


def _distances(
    target: Mapping[str, float],
    signatures: pd.DataFrame,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray([target[name] for name in SIGNATURE_FEATURE_IDS])
    matrix = signatures.loc[:, SIGNATURE_FEATURE_IDS].to_numpy(dtype=float)
    delta = (matrix - target_values) / scale
    return np.sqrt(np.mean(np.square(delta), axis=1))


def _reference_curve(cell: pd.DataFrame, forecast_efc: np.ndarray) -> np.ndarray:
    ordered = cell.sort_values("equivalent_full_cycles")
    x = ordered["equivalent_full_cycles"].to_numpy(dtype=float)
    y = ordered["capacity_retention_pct"].to_numpy(dtype=float)
    monotone = np.minimum.accumulate(y)
    clipped_x = np.minimum(forecast_efc, x[-1])
    result = np.interp(clipped_x, x, monotone)
    beyond = forecast_efc > x[-1]
    if beyond.any():
        window = min(4, len(x))
        _, slope = _slope(np.sqrt(x[-window:] / 1000.0), monotone[-window:])
        slope = min(slope, 0.0)
        result[beyond] = monotone[-1] + slope * (
            np.sqrt(forecast_efc[beyond] / 1000.0) - math.sqrt(x[-1] / 1000.0)
        )
    return result


def _nearest_reference_prediction(
    prefix: pd.DataFrame,
    references: pd.DataFrame,
    forecast_efc: np.ndarray,
    *,
    landmark: int,
    neighbor_count: int,
) -> tuple[np.ndarray, list[str], list[float]]:
    rows: list[dict[str, object]] = []
    reference_cells: dict[str, pd.DataFrame] = {}
    for cell_id, cell in references.groupby("cell_id", sort=True):
        ordered = cell.sort_values("visit_index")
        if len(ordered) < landmark:
            continue
        reference_prefix = ordered.iloc[:landmark]
        rows.append({"cell_id": str(cell_id), **_signature(reference_prefix)})
        reference_cells[str(cell_id)] = ordered
    signature_frame = pd.DataFrame(rows)
    if len(signature_frame) < neighbor_count:
        raise SNLRPTLOCOError("Nearest-reference expert lacks support")
    center, scale = _robust_center_scale(signature_frame)
    distance = _distances(_signature(prefix), signature_frame, center, scale)
    order = sorted(
        range(len(distance)),
        key=lambda index: (float(distance[index]), str(signature_frame.iloc[index]["cell_id"])),
    )[:neighbor_count]
    selected_ids = [str(signature_frame.iloc[index]["cell_id"]) for index in order]
    selected_distances = [float(distance[index]) for index in order]
    weights = 1.0 / np.maximum(np.asarray(selected_distances, dtype=float), 1e-6)
    weights /= float(np.sum(weights))
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    y0 = float(prefix.iloc[-1]["capacity_retention_pct"])
    values = []
    for cell_id in selected_ids:
        reference = reference_cells[cell_id]
        future = _reference_curve(reference, forecast_efc)
        at_landmark = float(_reference_curve(reference, np.asarray([x0]))[0])
        values.append(y0 + future - at_landmark)
    prediction = np.average(np.vstack(values), axis=0, weights=weights)
    return prediction, selected_ids, selected_distances


def _expert_predictions(
    prefix: pd.DataFrame,
    references: pd.DataFrame,
    forecast_efc: np.ndarray,
    config: Mapping[str, object],
    *,
    landmark: int,
) -> dict[str, np.ndarray]:
    y0 = float(prefix.iloc[-1]["capacity_retention_pct"])
    exponents = [float(value) for value in config["base_experts"]["bounded_power_exponent_grid"]]
    nearest, _, _ = _nearest_reference_prediction(
        prefix,
        references,
        forecast_efc,
        landmark=landmark,
        neighbor_count=int(config["base_experts"]["nearest_reference_count"]),
    )
    result = {
        "target_prefix_persistence": np.full_like(forecast_efc, y0, dtype=float),
        "target_prefix_constrained_linear_efc": _anchored_slope_prediction(
            prefix, forecast_efc, transform="linear"
        ),
        "target_prefix_constrained_sqrt_efc": _anchored_slope_prediction(
            prefix, forecast_efc, transform="sqrt"
        ),
        "target_prefix_bounded_power_efc": _bounded_power_prediction(
            prefix, forecast_efc, exponents
        ),
        "condition_ridge_delta": _condition_ridge_prediction(
            prefix,
            references,
            forecast_efc,
            alpha=float(config["base_experts"]["condition_ridge_alpha"]),
        ),
        "nearest_reference_delta_transfer": nearest,
    }
    lower, upper = (
        float(value) for value in config["base_experts"]["prediction_clip_pct"]
    )
    return {model: np.clip(values, lower, upper) for model, values in result.items()}


def _trajectory_iae(
    x0: float,
    actual_x: np.ndarray,
    actual_y: np.ndarray,
    predicted_y: np.ndarray,
) -> float:
    if len(actual_x) < 1 or not actual_x[-1] > x0:
        raise SNLRPTLOCOError("Trajectory IAE lacks future support")
    error = np.concatenate([[0.0], np.abs(predicted_y - actual_y)])
    x = np.concatenate([[x0], actual_x])
    return float(np.trapezoid(error, x) / (x[-1] - x0))


def _inner_validation_records(
    references: pd.DataFrame,
    config: Mapping[str, object],
    *,
    landmark: int,
) -> pd.DataFrame:
    score_end = float(config["dynamic_landmarks"]["score_end_equivalent_full_cycles"])
    records: list[dict[str, object]] = []
    conditions = sorted(references["condition_id"].unique())
    for held_condition in conditions:
        inner_training = references.loc[
            references["condition_id"] != held_condition,
            RPT_TRAJECTORY_COLUMNS,
        ].copy()
        inner_targets = references.loc[
            references["condition_id"] == held_condition,
            RPT_TRAJECTORY_COLUMNS,
        ].copy()
        for cell_id, cell in inner_targets.groupby("cell_id", sort=True):
            ordered = cell.sort_values("visit_index")
            prefix = ordered.iloc[:landmark]
            x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
            future = ordered.loc[
                (ordered["visit_index"] >= landmark)
                & (ordered["equivalent_full_cycles"] <= score_end)
            ]
            if len(future) < 1:
                raise SNLRPTLOCOError(
                    f"Inner validation lacks future support for {cell_id}"
                )
            future_x = future["equivalent_full_cycles"].to_numpy(dtype=float)
            future_y = future["capacity_retention_pct"].to_numpy(dtype=float)
            predictions = _expert_predictions(
                prefix,
                inner_training,
                future_x,
                config,
                landmark=landmark,
            )
            record: dict[str, object] = {
                "condition_id": held_condition,
                "cell_id": str(cell_id),
                **_signature(prefix),
            }
            for model_id in BASE_MODEL_IDS:
                record[f"risk__{model_id}"] = _trajectory_iae(
                    x0, future_x, future_y, predictions[model_id]
                )
            records.append(record)
    return pd.DataFrame(records).sort_values(
        ["condition_id", "cell_id"], kind="stable", ignore_index=True
    )


def _global_risks(inner: pd.DataFrame) -> dict[str, float]:
    risks: dict[str, float] = {}
    for model_id in BASE_MODEL_IDS:
        column = f"risk__{model_id}"
        by_condition = inner.groupby("condition_id", sort=True)[column].mean()
        risks[model_id] = float(by_condition.mean())
    return risks


def _selector_decision(
    prefix: pd.DataFrame,
    inner: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    selector = config["safe_hard_selector"]
    global_risk = _global_risks(inner)
    persistence = global_risk["target_prefix_persistence"]
    safe_pool = ["target_prefix_persistence"]
    for model_id in BASE_MODEL_IDS[1:]:
        risk = global_risk[model_id]
        if (
            risk <= float(selector["maximum_relative_risk_vs_persistence"]) * persistence
            and risk <= persistence + float(selector["maximum_absolute_risk_above_persistence_pp"])
        ):
            safe_pool.append(model_id)
    signature_frame = inner.loc[:, ["condition_id", "cell_id", *SIGNATURE_FEATURE_IDS]].copy()
    center, scale = _robust_center_scale(signature_frame)
    target_signature = _signature(prefix)
    distance = _distances(target_signature, signature_frame, center, scale)
    neighbor_count = min(int(selector["local_neighbor_count"]), len(inner))
    order = sorted(
        range(len(inner)),
        key=lambda index: (float(distance[index]), str(inner.iloc[index]["cell_id"])),
    )[:neighbor_count]
    neighbor_distance = np.asarray([distance[index] for index in order], dtype=float)
    weights = 1.0 / np.maximum(neighbor_distance, 1e-6)
    weights /= float(np.sum(weights))
    local_risk: dict[str, float] = {}
    penalty = float(selector["dispersion_penalty"])
    for model_id in safe_pool:
        values = inner.iloc[order][f"risk__{model_id}"].to_numpy(dtype=float)
        mean = float(np.sum(weights * values))
        variance = float(np.sum(weights * np.square(values - mean)))
        local_risk[model_id] = mean + penalty * math.sqrt(max(variance, 0.0))
    inner_nearest: list[float] = []
    for index in range(len(inner)):
        other = [
            float(value)
            for other_index, value in enumerate(
                _distances(
                    inner.iloc[index].to_dict(), signature_frame, center, scale
                )
            )
            if other_index != index
            and inner.iloc[other_index]["condition_id"]
            != inner.iloc[index]["condition_id"]
        ]
        if other:
            inner_nearest.append(min(other))
    if not inner_nearest:
        raise SNLRPTLOCOError("OOD calibration lacks cross-condition support")
    threshold = float(np.quantile(inner_nearest, 0.99, method="higher")) * 1.5
    mean_neighbor_distance = float(np.mean(neighbor_distance))
    ranked_local = sorted(local_risk, key=lambda model: (local_risk[model], model))
    best_local = ranked_local[0]
    if len(ranked_local) == 1:
        relative_margin = 1.0
    else:
        best = local_risk[ranked_local[0]]
        second = local_risk[ranked_local[1]]
        relative_margin = (second - best) / max(second, 1e-9)
    global_best = min(safe_pool, key=lambda model: (global_risk[model], model))
    if float(neighbor_distance[0]) > threshold:
        issued = False
        selected = "none"
        mode = "out_of_domain_refusal"
        evidence = "refused_out_of_domain"
    elif relative_margin < float(selector["minimum_relative_local_risk_margin"]):
        issued = True
        selected = global_best
        mode = "ambiguous_global_safe_fallback"
        evidence = "ambiguous_fallback"
    else:
        issued = True
        selected = best_local
        mode = "supported_local_safe_selection"
        evidence = "supported"
    return {
        "issued": issued,
        "selected_expert_model_id": selected,
        "selection_mode": mode,
        "evidence_status": evidence,
        "safe_pool_model_ids_json": json.dumps(safe_pool, separators=(",", ":")),
        "global_risks_json": json.dumps(global_risk, sort_keys=True, separators=(",", ":")),
        "local_risks_json": json.dumps(local_risk, sort_keys=True, separators=(",", ":")),
        "relative_local_risk_margin": relative_margin,
        "mean_neighbor_distance": mean_neighbor_distance,
        "out_of_domain_threshold": threshold,
        "nearest_inner_validation_cells_json": json.dumps(
            [str(inner.iloc[index]["cell_id"]) for index in order],
            separators=(",", ":"),
        ),
    }


def _forecast_grid(prefix: pd.DataFrame, config: Mapping[str, object]) -> np.ndarray:
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    end = float(config["dynamic_landmarks"]["score_end_equivalent_full_cycles"])
    step = float(
        config["dynamic_landmarks"]["forecast_grid_step_equivalent_full_cycles"]
    )
    first = math.ceil((x0 + 1e-12) / step) * step
    later = np.arange(first, end + step * 0.5, step, dtype=float)
    later = later[later > x0]
    return np.concatenate([[x0], later])


def predict_snl_rpt_loco(
    references: pd.DataFrame,
    prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Predict every outer fold without accepting target truth as an argument."""
    frozen = validate_snl_rpt_loco_config(config)
    if tuple(references.columns) != REFERENCE_COLUMNS:
        raise SNLRPTLOCOError("SNL LOCO reference columns changed")
    if tuple(prefixes.columns) != TARGET_PREFIX_COLUMNS:
        raise SNLRPTLOCOError("SNL LOCO target-prefix columns changed")
    prediction_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    inner_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for (outer, cell_id, landmark), prefix in prefixes.groupby(
        ["outer_condition_id", "cell_id", "landmark_visit_count"], sort=True
    ):
        ordered = prefix.sort_values("visit_index")
        landmark_int = int(landmark)
        if len(ordered) != landmark_int or ordered["visit_index"].tolist() != list(
            range(landmark_int)
        ):
            raise SNLRPTLOCOError("Target prefix is not exactly truncated")
        if set(ordered["condition_id"]) != {outer}:
            raise SNLRPTLOCOError("Target prefix condition identity changed")
        reference = references.loc[references["outer_condition_id"] == outer]
        if reference.empty or outer in set(reference["condition_id"]):
            raise SNLRPTLOCOError("Outer target condition leaked into references")
        reference_core = reference.loc[:, RPT_TRAJECTORY_COLUMNS]
        cache_key = (str(outer), landmark_int)
        if cache_key not in inner_cache:
            inner_cache[cache_key] = _inner_validation_records(
                reference_core, frozen, landmark=landmark_int
            )
        inner = inner_cache[cache_key]
        decision = _selector_decision(ordered, inner, frozen)
        decision_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "outer_condition_id": str(outer),
                "cell_id": str(cell_id),
                "landmark_visit_count": landmark_int,
                **decision,
            }
        )
        grid = _forecast_grid(ordered, frozen)
        experts = _expert_predictions(
            ordered,
            reference_core,
            grid,
            frozen,
            landmark=landmark_int,
        )
        for model_id, values in experts.items():
            for exposure, predicted in zip(grid, values, strict=True):
                prediction_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "dataset_id": DATASET_ID,
                        "outer_condition_id": str(outer),
                        "cell_id": str(cell_id),
                        "landmark_visit_count": landmark_int,
                        "model_id": model_id,
                        "forecast_equivalent_full_cycles": float(exposure),
                        "predicted_capacity_retention_pct": float(predicted),
                        "selected_expert_model_id": model_id,
                        "selection_mode": "diagnostic_base_expert",
                        "evidence_status": "diagnostic",
                    }
                )
        if bool(decision["issued"]):
            selected = str(decision["selected_expert_model_id"])
            for exposure, predicted in zip(grid, experts[selected], strict=True):
                prediction_rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "dataset_id": DATASET_ID,
                        "outer_condition_id": str(outer),
                        "cell_id": str(cell_id),
                        "landmark_visit_count": landmark_int,
                        "model_id": SELECTOR_MODEL_ID,
                        "forecast_equivalent_full_cycles": float(exposure),
                        "predicted_capacity_retention_pct": float(predicted),
                        "selected_expert_model_id": selected,
                        "selection_mode": str(decision["selection_mode"]),
                        "evidence_status": str(decision["evidence_status"]),
                    }
                )
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS).sort_values(
        [
            "outer_condition_id",
            "cell_id",
            "landmark_visit_count",
            "model_id",
            "forecast_equivalent_full_cycles",
        ],
        kind="stable",
        ignore_index=True,
    )
    decisions = pd.DataFrame(decision_rows, columns=DECISION_COLUMNS).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": "lifetwin.snl_rpt_loco_prediction_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "config_semantic_sha256": canonical_json_sha256(frozen),
        "reference_rows_sha256": canonical_frame_sha256(
            references, REFERENCE_COLUMNS
        ),
        "target_prefix_rows_sha256": canonical_frame_sha256(
            prefixes, TARGET_PREFIX_COLUMNS
        ),
        "prediction_rows_sha256": canonical_frame_sha256(
            predictions, PREDICTION_COLUMNS
        ),
        "selector_decision_rows_sha256": canonical_frame_sha256(
            decisions, DECISION_COLUMNS
        ),
        "prediction_row_count": len(predictions),
        "selector_decision_count": len(decisions),
        "issued_selector_decision_count": int(decisions["issued"].sum()),
        "target_truth_argument_accepted": False,
        "target_raw_zip_argument_accepted": False,
        "target_suffix_rows_present_in_prefix_input": False,
        "score_computed": False,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return predictions, decisions, manifest


def _validated_prediction_replay(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise SNLRPTLOCOError("SNL prediction columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise SNLRPTLOCOError("SNL selector-decision columns changed")
    frozen = validate_snl_rpt_loco_config(config)
    if manifest.get("config_semantic_sha256") != canonical_json_sha256(frozen):
        raise SNLRPTLOCOError("SNL prediction config hash changed")
    if manifest.get("prediction_rows_sha256") != canonical_frame_sha256(
        predictions, PREDICTION_COLUMNS
    ):
        raise SNLRPTLOCOError("SNL prediction rows changed after freeze")
    if manifest.get("selector_decision_rows_sha256") != canonical_frame_sha256(
        decisions, DECISION_COLUMNS
    ):
        raise SNLRPTLOCOError("SNL selector decisions changed after freeze")


def _score_prediction_curve(
    curve: pd.DataFrame,
    prefix_last: pd.Series,
    future: pd.DataFrame,
) -> tuple[float, float, float, float]:
    grid = curve["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
    predicted_grid = curve["predicted_capacity_retention_pct"].to_numpy(dtype=float)
    x = future["equivalent_full_cycles"].to_numpy(dtype=float)
    actual = future["capacity_retention_pct"].to_numpy(dtype=float)
    predicted = np.interp(x, grid, predicted_grid)
    error = predicted - actual
    x0 = float(prefix_last["equivalent_full_cycles"])
    iae = _trajectory_iae(x0, x, actual, predicted)
    return (
        iae,
        float(np.mean(np.abs(error))),
        float(np.sqrt(np.mean(np.square(error)))),
        float(abs(error[-1])),
    )


def _exact_one_sided_sign_flip_p(improvements: Sequence[float]) -> float | None:
    values = np.asarray(improvements, dtype=float)
    if len(values) == 0:
        return None
    observed = float(np.mean(values))
    count = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=len(values)):
        total += 1
        permuted = float(np.mean(values * np.asarray(signs)))
        if permuted >= observed - 1e-12:
            count += 1
    return count / total


def score_snl_rpt_loco(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Link frozen predictions to held-condition suffix truth and score."""
    frozen = validate_snl_rpt_loco_config(config)
    if tuple(truth.columns) != TARGET_TRUTH_COLUMNS:
        raise SNLRPTLOCOError("SNL target-truth columns changed")
    _validated_prediction_replay(predictions, decisions, manifest, frozen)
    score_end = float(frozen["dynamic_landmarks"]["score_end_equivalent_full_cycles"])
    score_rows: list[dict[str, object]] = []
    for (outer, cell_id), target in truth.groupby(
        ["outer_condition_id", "cell_id"], sort=True
    ):
        ordered = target.sort_values("visit_index")
        for landmark_count in (
            int(value)
            for value in frozen["dynamic_landmarks"]["prefix_visit_counts"]
        ):
            prefix_last = ordered.iloc[landmark_count - 1]
            future = ordered.loc[
                (ordered["visit_index"] >= landmark_count)
                & (ordered["equivalent_full_cycles"] <= score_end)
            ]
            minimum = int(
                frozen["dynamic_landmarks"]["minimum_future_visits_by_landmark"][
                    str(landmark_count)
                ]
            )
            if len(future) < minimum:
                raise SNLRPTLOCOError(
                    f"Future support changed for {cell_id} N{landmark_count}"
                )
            for model_id in MODEL_IDS:
                curve = predictions.loc[
                    (predictions["outer_condition_id"] == outer)
                    & (predictions["cell_id"] == cell_id)
                    & (predictions["landmark_visit_count"] == landmark_count)
                    & (predictions["model_id"] == model_id)
                ].sort_values("forecast_equivalent_full_cycles")
                issued = not curve.empty
                if not issued:
                    continue
                row: dict[str, object] = {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_id": DATASET_ID,
                    "outer_condition_id": str(outer),
                    "cell_id": str(cell_id),
                    "landmark_visit_count": landmark_count,
                    "model_id": model_id,
                    "issued": issued,
                    "future_observation_count": len(future),
                    "trajectory_iae_pp": 0.0,
                    "trajectory_mae_pp": 0.0,
                    "trajectory_rmse_pp": 0.0,
                    "endpoint_absolute_error_pp": 0.0,
                }
                iae, mae, rmse, endpoint = _score_prediction_curve(
                    curve, prefix_last, future
                )
                row.update(
                    {
                        "trajectory_iae_pp": iae,
                        "trajectory_mae_pp": mae,
                        "trajectory_rmse_pp": rmse,
                        "endpoint_absolute_error_pp": endpoint,
                    }
                )
                score_rows.append(row)
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    model_summary: dict[str, object] = {}
    for landmark in sorted(scores["landmark_visit_count"].unique()):
        landmark_scores = scores.loc[scores["landmark_visit_count"] == landmark]
        by_model: dict[str, object] = {}
        for model_id in MODEL_IDS:
            selected = landmark_scores.loc[landmark_scores["model_id"] == model_id]
            issued = selected.loc[selected["issued"]]
            cluster_means = issued.groupby("outer_condition_id", sort=True)[
                [
                    "trajectory_iae_pp",
                    "trajectory_mae_pp",
                    "endpoint_absolute_error_pp",
                ]
            ].mean()
            by_model[model_id] = {
                "issued_cell_fraction": (
                    float(
                        decisions.loc[
                            decisions["landmark_visit_count"] == landmark,
                            "issued",
                        ].mean()
                    )
                    if model_id == SELECTOR_MODEL_ID
                    else 1.0
                ),
                "issued_condition_cluster_count": int(len(cluster_means)),
                "condition_cluster_equal_trajectory_iae_pp": (
                    float(cluster_means["trajectory_iae_pp"].mean())
                    if len(cluster_means)
                    else None
                ),
                "condition_cluster_equal_trajectory_mae_pp": (
                    float(cluster_means["trajectory_mae_pp"].mean())
                    if len(cluster_means)
                    else None
                ),
                "condition_cluster_equal_endpoint_absolute_error_pp": (
                    float(cluster_means["endpoint_absolute_error_pp"].mean())
                    if len(cluster_means)
                    else None
                ),
            }
        model_summary[str(int(landmark))] = by_model

    primary_landmark = int(frozen["evaluation"]["primary_landmark_visit_count"])
    primary = scores.loc[
        (scores["landmark_visit_count"] == primary_landmark)
        & (scores["model_id"].isin([SELECTOR_MODEL_ID, "target_prefix_persistence"]))
    ]
    pivot = primary.pivot(
        index=["outer_condition_id", "cell_id"],
        columns="model_id",
        values="trajectory_iae_pp",
    )
    issued_pairs = pivot.dropna()
    cluster_pairs = issued_pairs.groupby(level="outer_condition_id").mean()
    improvements = (
        cluster_pairs["target_prefix_persistence"]
        - cluster_pairs[SELECTOR_MODEL_ID]
    )
    selector_iae = float(cluster_pairs[SELECTOR_MODEL_ID].mean())
    persistence_iae = float(cluster_pairs["target_prefix_persistence"].mean())
    absolute_improvement = persistence_iae - selector_iae
    relative_improvement = absolute_improvement / max(persistence_iae, 1e-12)
    issued_fraction = float(
        decisions.loc[
            decisions["landmark_visit_count"] == primary_landmark, "issued"
        ].mean()
    )
    improved_fraction = float((improvements > 0.0).mean())
    worst_regression = float(max(0.0, (-improvements).max()))
    p_value = _exact_one_sided_sign_flip_p(improvements.tolist())
    gate = frozen["evaluation"]["descriptive_success_gate"]
    gate_checks = {
        "relative_improvement": relative_improvement
        >= float(gate["minimum_relative_cluster_equal_IAE_improvement_vs_persistence"]),
        "absolute_improvement": absolute_improvement
        >= float(gate["minimum_absolute_cluster_equal_IAE_improvement_pp"]),
        "improved_cluster_fraction": improved_fraction
        >= float(gate["minimum_improved_condition_cluster_fraction"]),
        "worst_cluster_regression": worst_regression
        <= float(gate["maximum_worst_condition_cluster_regression_pp"]),
        "issued_cell_fraction": issued_fraction
        >= float(gate["minimum_issued_cell_fraction"]),
        "sign_flip_p_value": p_value is not None
        and p_value
        <= float(gate["maximum_exact_one_sided_condition_sign_flip_p_value"]),
    }
    summary: dict[str, object] = {
        "schema_version": "lifetwin.snl_rpt_loco_score_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": frozen["evidence_role"],
        "model_summary_by_landmark": model_summary,
        "primary_comparison": {
            "landmark_visit_count": primary_landmark,
            "selector_condition_cluster_equal_trajectory_iae_pp": selector_iae,
            "persistence_condition_cluster_equal_trajectory_iae_pp": persistence_iae,
            "absolute_improvement_pp": absolute_improvement,
            "relative_improvement_fraction": relative_improvement,
            "issued_cell_fraction": issued_fraction,
            "issued_condition_cluster_count": len(cluster_pairs),
            "improved_condition_cluster_fraction": improved_fraction,
            "worst_condition_cluster_regression_pp": worst_regression,
            "exact_one_sided_condition_sign_flip_p_value": p_value,
        },
        "descriptive_success_gate_checks": gate_checks,
        "descriptive_success_gate_passed": all(gate_checks.values()),
        "claim_boundary": (
            "Retrospective grouped development only. This is not outcome-blind, "
            "calendar-aging, field, Hithium-product, or 15-25 year validation."
        ),
        "public_release_status": frozen["rights_and_release"]["public_aggregate_results"],
        "prediction_manifest_content_sha256": manifest.get(
            "manifest_content_sha256"
        ),
    }
    summary["score_rows_sha256"] = canonical_frame_sha256(scores, SCORE_COLUMNS)
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return scores, summary


__all__ = [
    "BASE_MODEL_IDS",
    "CONFIG_SEMANTIC_SHA256",
    "DECISION_COLUMNS",
    "EXPERIMENT_ID",
    "MODEL_IDS",
    "PREDICTION_COLUMNS",
    "REFERENCE_COLUMNS",
    "SCORE_COLUMNS",
    "SELECTOR_MODEL_ID",
    "SNLRPTLOCOError",
    "TARGET_PREFIX_COLUMNS",
    "TARGET_TRUTH_COLUMNS",
    "build_snl_rpt_loco_inputs",
    "load_snl_rpt_loco_config",
    "predict_snl_rpt_loco",
    "score_snl_rpt_loco",
    "validate_snl_rpt_loco_config",
]
