"""Frozen held-out trajectory portability stress on the MATR LFP cohort."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from lifetwin.data.fastcharge_portability import (
    CANONICAL_CYCLE_COLUMNS,
    DATASET_ID,
    TARGET_PREFIX_COLUMNS,
    build_fastcharge_prediction_inputs,
)
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.fastcharge_lfp_trajectory_portability.config.v1"
EXPERIMENT_ID = "fastcharge_lfp_trajectory_portability_v1"
CONFIG_SEMANTIC_SHA256 = (
    "c8665bd57fccc9938207c0d4c16ae650825b16cbffc2fbd23903b07cb10f5bdb"
)
PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "lifetwin.fastcharge_trajectory_prediction_manifest.v1"
)
BASE_MODEL_IDS = (
    "target_prefix_persistence",
    "target_prefix_full_linear",
    "target_prefix_robust_recent_linear",
    "target_prefix_constrained_sqrt_linear",
    "nearest_neighbor_delta_transfer",
)
EQUAL_MODEL_ID = "equal_weight_base_mixture"
HARD_RISK_MODEL_ID = "hard_lowest_predicted_risk_expert"
MOE_MODEL_ID = "stability_shrunk_evidence_weighted_mixture"
MODEL_IDS = (*BASE_MODEL_IDS, EQUAL_MODEL_ID, HARD_RISK_MODEL_ID, MOE_MODEL_ID)
FEATURE_IDS = (
    "capacity_full_slope_pp_per_cycle",
    "capacity_recent_slope_pp_per_cycle",
    "capacity_slope_disagreement_pp_per_cycle",
    "capacity_residual_std_pp",
    "capacity_recovery_fraction",
    "log_internal_resistance_slope_per_cycle",
    "temperature_max_slope_c_per_cycle",
    "log_charge_duration_slope_per_cycle",
    "energy_efficiency_slope_per_cycle",
)
PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "paper_split",
    "cell_id",
    "training_split",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "forecast_cycle",
    "predicted_capacity_retention_pct",
    "interval_lower_pct",
    "interval_upper_pct",
    "calibration_half_width_pp",
    "normalization_capacity_ah",
    "prefix_row_count",
    "target_prefix_sha256",
    "dominant_expert_model_id",
    "expert_weights_json",
    "expert_risks_json",
    "nearest_training_cell_ids",
    "neighbor_distances_json",
    "risk_margin_fraction",
    "mean_neighbor_distance",
    "feature_jackknife_instability_l1",
    "margin_support",
    "distance_support",
    "stability_support",
    "selection_strength",
    "evidence_status",
    "operational_action",
)
CALIBRATION_COLUMNS = (
    "prefix_cycle",
    "model_id",
    "forecast_cycle",
    "calibration_cell_count",
    "absolute_residual_quantile_level",
    "calibration_half_width_pp",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "paper_split",
    "cell_id",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "dominant_expert_model_id",
    "evidence_status",
    "operational_action",
    "future_observation_count",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
    "empirical_interval_coverage_fraction",
    "mean_interval_width_pp",
    "endpoint_inside_interval",
)


class FastChargeTrajectoryPortabilityError(ValueError):
    """Raised when the frozen FastCharge trajectory contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FastChargeTrajectoryPortabilityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise FastChargeTrajectoryPortabilityError(
            "Value is not canonical finite JSON"
        ) from exc


def _streaming_frame_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    if tuple(frame.columns) != tuple(columns):
        raise FastChargeTrajectoryPortabilityError(
            "Frame column order does not match hash contract"
        )
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for values in frame.itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(columns, values, strict=True):
            if isinstance(value, np.integer):
                normalized: object = int(value)
            elif isinstance(value, np.floating):
                normalized = float(value)
            elif isinstance(value, np.bool_):
                normalized = bool(value)
            else:
                normalized = value
            record[column] = normalized
        if not first:
            digest.update(b",")
        digest.update(_json_text(record).encode("ascii"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def validate_fastcharge_trajectory_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise FastChargeTrajectoryPortabilityError("Config must be an object")
    if canonical_json_sha256(dict(config)) != CONFIG_SEMANTIC_SHA256:
        raise FastChargeTrajectoryPortabilityError(
            "Frozen FastCharge trajectory config changed"
        )
    detached = json.loads(_json_text(dict(config)))
    if detached.get("schema_version") != SCHEMA_VERSION:
        raise FastChargeTrajectoryPortabilityError("Config schema changed")
    if detached.get("experiment_id") != EXPERIMENT_ID:
        raise FastChargeTrajectoryPortabilityError("Experiment identity changed")
    if tuple(detached["base_experts"]["model_ids"]) != BASE_MODEL_IDS:
        raise FastChargeTrajectoryPortabilityError("Base model registry changed")
    if tuple(detached["similarity"]["feature_ids"]) != FEATURE_IDS:
        raise FastChargeTrajectoryPortabilityError("Feature registry changed")
    if detached["dataset"]["dataset_id"] != DATASET_ID:
        raise FastChargeTrajectoryPortabilityError("Dataset identity changed")
    return detached


def load_fastcharge_trajectory_config(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FastChargeTrajectoryPortabilityError(
                    f"Non-finite JSON constant: {value}"
                )
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FastChargeTrajectoryPortabilityError("Cannot load config") from exc
    if not isinstance(value, Mapping):
        raise FastChargeTrajectoryPortabilityError("Config must be an object")
    return validate_fastcharge_trajectory_config(value)


def _prefix_cycles(config: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["split_and_firewall"]["prefix_cycles"])


def _score_end(config: Mapping[str, object]) -> int:
    return int(config["split_and_firewall"]["score_end_cycle"])


def _validated_numeric_cycle_frame(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    if tuple(frame.columns) != tuple(columns):
        raise FastChargeTrajectoryPortabilityError("Cycle input columns changed")
    result = frame.copy()
    string_columns = {"dataset_id", "cell_id", "paper_split"}
    for column in string_columns:
        if result[column].isna().any():
            raise FastChargeTrajectoryPortabilityError(f"{column} cannot be null")
        result[column] = result[column].astype(str)
    numeric_columns = [column for column in columns if column not in string_columns]
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise FastChargeTrajectoryPortabilityError(
            "Cycle inputs must contain finite numbers"
        )
    for column in ("cycle_index", "prefix_cycle"):
        if column not in numeric:
            continue
        raw = numeric[column].to_numpy(dtype=float)
        if not np.equal(raw, np.floor(raw)).all():
            raise FastChargeTrajectoryPortabilityError(
                f"{column} must contain integers"
            )
        result[column] = raw.astype(np.int64)
    for column in set(numeric_columns) - {"cycle_index", "prefix_cycle"}:
        result[column] = numeric[column].astype(float)
    positive = (
        "discharge_capacity_ah",
        "temperature_max_c",
        "energy_efficiency",
    )
    if (result.loc[:, positive] <= 0.0).any().any():
        raise FastChargeTrajectoryPortabilityError(
            "Positive cycle inputs must remain positive"
        )
    if (result["internal_resistance_ohm"] < 0.0).any():
        raise FastChargeTrajectoryPortabilityError(
            "Internal resistance cannot be negative"
        )
    if (result["charge_time_s"] < 0.0).any():
        raise FastChargeTrajectoryPortabilityError("Charge duration cannot be negative")
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise FastChargeTrajectoryPortabilityError("Cycle dataset identity changed")
    return result


def _validated_training(
    training: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    result = _validated_numeric_cycle_frame(
        training,
        columns=CANONICAL_CYCLE_COLUMNS,
    )
    expected_split = str(config["split_and_firewall"]["training_split"])
    if set(result["paper_split"]) != {expected_split}:
        raise FastChargeTrajectoryPortabilityError("Training split changed")
    expected_cells = int(config["split_and_firewall"]["expected_training_cells"])
    if result["cell_id"].nunique() != expected_cells:
        raise FastChargeTrajectoryPortabilityError("Training cell count changed")
    score_end = _score_end(config)
    for cell_id, cell in result.groupby("cell_id", sort=True):
        if sorted(cell["cycle_index"].astype(int)) != list(range(1, score_end + 1)):
            raise FastChargeTrajectoryPortabilityError(
                f"Training history is incomplete for {cell_id}"
            )
    if result.duplicated(["cell_id", "cycle_index"]).any():
        raise FastChargeTrajectoryPortabilityError(
            "Training histories contain duplicate cycles"
        )
    return result.sort_values(
        ["cell_id", "cycle_index"], kind="stable", ignore_index=True
    )


def _validated_target_prefixes(
    prefixes: pd.DataFrame,
    training_cell_ids: set[str],
    config: Mapping[str, object],
) -> pd.DataFrame:
    result = _validated_numeric_cycle_frame(
        prefixes,
        columns=TARGET_PREFIX_COLUMNS,
    )
    evaluation_splits = {
        str(value) for value in config["split_and_firewall"]["evaluation_splits"]
    }
    if set(result["paper_split"]) != evaluation_splits:
        raise FastChargeTrajectoryPortabilityError("Evaluation splits changed")
    if set(result["prefix_cycle"].astype(int)) != set(_prefix_cycles(config)):
        raise FastChargeTrajectoryPortabilityError("Prefix registry changed")
    if set(result["cell_id"]) & training_cell_ids:
        raise FastChargeTrajectoryPortabilityError(
            "Target prefixes overlap training identities"
        )
    expected_by_split = {
        str(key): int(value)
        for key, value in config["split_and_firewall"][
            "expected_evaluation_cells_by_split"
        ].items()
    }
    actual_by_split = (
        result.groupby("paper_split", sort=True)["cell_id"].nunique().to_dict()
    )
    if actual_by_split != expected_by_split:
        raise FastChargeTrajectoryPortabilityError(
            "Evaluation split cell counts changed"
        )
    for (cell_id, prefix_cycle), group in result.groupby(
        ["cell_id", "prefix_cycle"], sort=True
    ):
        if group["paper_split"].nunique() != 1:
            raise FastChargeTrajectoryPortabilityError(
                f"Target split changes for {cell_id}"
            )
        if sorted(group["cycle_index"].astype(int)) != list(
            range(1, int(prefix_cycle) + 1)
        ):
            raise FastChargeTrajectoryPortabilityError(
                f"Target prefix is not exactly truncated for {cell_id} P{prefix_cycle}"
            )
    if result.duplicated(["cell_id", "prefix_cycle", "cycle_index"]).any():
        raise FastChargeTrajectoryPortabilityError(
            "Target prefixes contain duplicate cycles"
        )
    return result.sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def _normalization_capacity(cell: pd.DataFrame) -> float:
    initial = cell.loc[
        cell["cycle_index"].between(1, 5), "discharge_capacity_ah"
    ].astype(float)
    if len(initial) != 5:
        raise FastChargeTrajectoryPortabilityError(
            "Normalization requires cycles 1 through 5"
        )
    value = float(median(initial.tolist()))
    if not math.isfinite(value) or value <= 0.0:
        raise FastChargeTrajectoryPortabilityError(
            "Normalization capacity must be positive"
        )
    return value


def _retention(cell: pd.DataFrame, normalization: float) -> np.ndarray:
    return 100.0 * cell["discharge_capacity_ah"].to_numpy(dtype=float) / normalization


def _linear_parameters(
    cycle: np.ndarray,
    values: np.ndarray,
    *,
    constrain_non_positive: bool,
) -> tuple[float, float]:
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(values, dtype=float)
    centered = x - float(np.mean(x))
    denominator = float(np.sum(np.square(centered)))
    if denominator <= 0.0:
        raise FastChargeTrajectoryPortabilityError("Linear fit lacks support")
    slope = float(np.sum(centered * (y - float(np.mean(y)))) / denominator)
    if constrain_non_positive:
        slope = min(slope, 0.0)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def _robust_recent_parameters(
    cycle: np.ndarray,
    values: np.ndarray,
    *,
    window_cycles: int,
) -> tuple[float, float]:
    x = np.asarray(cycle, dtype=float)[-window_cycles:]
    y = np.asarray(values, dtype=float)[-window_cycles:]
    slopes: list[float] = []
    for left in range(len(x) - 1):
        delta_x = x[left + 1 :] - x[left]
        slopes.extend(((y[left + 1 :] - y[left]) / delta_x).tolist())
    if not slopes:
        raise FastChargeTrajectoryPortabilityError("Robust fit lacks support")
    slope = min(float(median(slopes)), 0.0)
    intercept = float(median((y - slope * x).tolist()))
    return intercept, slope


def _constrained_sqrt_linear_parameters(
    cycle: np.ndarray,
    retention: np.ndarray,
    config: Mapping[str, object],
) -> tuple[float, float]:
    shifted = np.maximum(np.asarray(cycle, dtype=float) - 1.0, 0.0)
    loss = 100.0 - np.asarray(retention, dtype=float)
    design = np.column_stack((np.sqrt(shifted), shifted))
    weights = np.ones(len(shifted), dtype=float)
    coefficients = np.zeros(2, dtype=float)
    iterations = int(config["base_experts"]["huber_iterations"])
    huber_delta = float(config["base_experts"]["huber_delta_mad"])
    for _ in range(iterations):
        root_weight = np.sqrt(weights)
        result = lsq_linear(
            design * root_weight[:, None],
            loss * root_weight,
            bounds=(0.0, np.inf),
            method="trf",
            lsmr_tol="auto",
        )
        if not result.success or not np.isfinite(result.x).all():
            raise FastChargeTrajectoryPortabilityError(
                "Constrained sqrt-linear fit failed"
            )
        coefficients = result.x.astype(float)
        residual = loss - design @ coefficients
        centered = residual - float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(centered)))
        cutoff = huber_delta * max(scale, 1e-9)
        magnitude = np.abs(centered)
        weights = np.ones(len(shifted), dtype=float)
        large = magnitude > cutoff
        weights[large] = cutoff / magnitude[large]
    return float(coefficients[0]), float(coefficients[1])


def _target_only_predictions(
    prefix: pd.DataFrame,
    forecast_cycles: np.ndarray,
    config: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], float]:
    normalization = _normalization_capacity(prefix)
    cycle = prefix["cycle_index"].to_numpy(dtype=float)
    retention = _retention(prefix, normalization)
    linear_intercept, linear_slope = _linear_parameters(
        cycle,
        retention,
        constrain_non_positive=True,
    )
    recent_intercept, recent_slope = _robust_recent_parameters(
        cycle,
        retention,
        window_cycles=int(config["base_experts"]["robust_recent_window_cycles"]),
    )
    sqrt_coefficient, loss_slope = _constrained_sqrt_linear_parameters(
        cycle,
        retention,
        config,
    )
    shifted = np.maximum(forecast_cycles.astype(float) - 1.0, 0.0)
    predictions = {
        "target_prefix_persistence": np.full(
            len(forecast_cycles), float(retention[-1])
        ),
        "target_prefix_full_linear": (
            linear_intercept + linear_slope * forecast_cycles
        ),
        "target_prefix_robust_recent_linear": (
            recent_intercept + recent_slope * forecast_cycles
        ),
        "target_prefix_constrained_sqrt_linear": (
            100.0 - sqrt_coefficient * np.sqrt(shifted) - loss_slope * shifted
        ),
    }
    return (
        {
            model_id: np.clip(values.astype(float), 0.0, 110.0)
            for model_id, values in predictions.items()
        },
        normalization,
    )


def _unconstrained_slope(cycle: np.ndarray, values: np.ndarray) -> float:
    return _linear_parameters(
        cycle,
        values,
        constrain_non_positive=False,
    )[1]


def _trajectory_signature(
    prefix: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, float]:
    normalization = _normalization_capacity(prefix)
    cycle = prefix["cycle_index"].to_numpy(dtype=float)
    retention = _retention(prefix, normalization)
    intercept, full_slope = _linear_parameters(
        cycle,
        retention,
        constrain_non_positive=False,
    )
    recent_window = int(config["base_experts"]["robust_recent_window_cycles"])
    recent_x = cycle[-recent_window:]
    recent_y = retention[-recent_window:]
    recent_slope = _unconstrained_slope(recent_x, recent_y)
    residual = retention - (intercept + full_slope * cycle)
    recovery_fraction = float(np.mean(np.diff(retention) > 0.1))
    log_floors = config["similarity"]["positive_log_floors"]
    resistance = np.log(
        np.maximum(
            prefix["internal_resistance_ohm"].to_numpy(dtype=float),
            float(log_floors["internal_resistance_ohm"]),
        )
    )
    charge_duration = np.log(
        np.maximum(
            prefix["charge_time_s"].to_numpy(dtype=float),
            float(log_floors["charge_time_s"]),
        )
    )
    signature = {
        "capacity_full_slope_pp_per_cycle": float(full_slope),
        "capacity_recent_slope_pp_per_cycle": float(recent_slope),
        "capacity_slope_disagreement_pp_per_cycle": float(
            abs(full_slope - recent_slope)
        ),
        "capacity_residual_std_pp": float(np.sqrt(np.mean(np.square(residual)))),
        "capacity_recovery_fraction": recovery_fraction,
        "log_internal_resistance_slope_per_cycle": _unconstrained_slope(
            cycle, resistance
        ),
        "temperature_max_slope_c_per_cycle": _unconstrained_slope(
            cycle,
            prefix["temperature_max_c"].to_numpy(dtype=float),
        ),
        "log_charge_duration_slope_per_cycle": _unconstrained_slope(
            cycle, charge_duration
        ),
        "energy_efficiency_slope_per_cycle": _unconstrained_slope(
            cycle,
            prefix["energy_efficiency"].to_numpy(dtype=float),
        ),
    }
    if tuple(signature) != FEATURE_IDS or not all(
        math.isfinite(value) for value in signature.values()
    ):
        raise FastChargeTrajectoryPortabilityError("Trajectory signature is invalid")
    return signature


def _distance_map(
    target_signature: Mapping[str, float],
    reference_signatures: Mapping[str, Mapping[str, float]],
    feature_ids: Sequence[str],
    config: Mapping[str, object],
) -> dict[str, float]:
    reference_ids = sorted(reference_signatures)
    if len(reference_ids) < int(config["mixture"]["risk_reference_neighbor_count"]):
        raise FastChargeTrajectoryPortabilityError(
            "Too few reference cells for neighbor risk"
        )
    matrix = np.asarray(
        [
            [reference_signatures[cell_id][feature_id] for feature_id in feature_ids]
            for cell_id in reference_ids
        ],
        dtype=float,
    )
    target = np.asarray(
        [target_signature[feature_id] for feature_id in feature_ids], dtype=float
    )
    center = np.median(matrix, axis=0)
    scale = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    floors = np.asarray(
        [
            float(config["similarity"]["feature_scale_floors"][feature_id])
            for feature_id in feature_ids
        ]
    )
    scale = np.maximum(scale, floors)
    distance = np.sqrt(np.mean(np.square((matrix - target) / scale), axis=1))
    return {
        cell_id: float(value)
        for cell_id, value in zip(reference_ids, distance, strict=True)
    }


def _nearest_cells(
    distances: Mapping[str, float],
    count: int,
) -> list[str]:
    return sorted(distances, key=lambda cell_id: (distances[cell_id], cell_id))[:count]


def _analog_prediction(
    target_prefix: pd.DataFrame,
    reference_cells: Mapping[str, pd.DataFrame],
    target_signature: Mapping[str, float],
    reference_signatures: Mapping[str, Mapping[str, float]],
    forecast_cycles: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    distances = _distance_map(
        target_signature,
        reference_signatures,
        FEATURE_IDS,
        config,
    )
    analog_config = config["base_experts"]["nearest_neighbor_delta_transfer"]
    nearest = _nearest_cells(distances, int(analog_config["neighbor_count"]))
    epsilon = float(analog_config["distance_epsilon"])
    raw_weights = np.asarray(
        [1.0 / (distances[cell_id] + epsilon) for cell_id in nearest], dtype=float
    )
    weights = raw_weights / float(np.sum(raw_weights))
    target_normalization = _normalization_capacity(target_prefix)
    target_last = float(_retention(target_prefix, target_normalization)[-1])
    prefix_cycle = int(target_prefix["cycle_index"].max())
    deltas: list[np.ndarray] = []
    for cell_id in nearest:
        reference = reference_cells[cell_id].sort_values("cycle_index", kind="stable")
        normalization = _normalization_capacity(reference)
        retention = _retention(reference, normalization)
        deltas.append(
            retention[forecast_cycles.astype(int) - 1] - retention[prefix_cycle - 1]
        )
    prediction = target_last + np.sum(weights[:, None] * np.vstack(deltas), axis=0)
    return np.clip(prediction, 0.0, 110.0)


def _piecewise_increasing(
    value: float,
    zero_below: float,
    full_above: float,
) -> float:
    if full_above <= zero_below:
        raise FastChargeTrajectoryPortabilityError("Invalid support thresholds")
    return float(np.clip((value - zero_below) / (full_above - zero_below), 0, 1))


def _piecewise_decreasing(
    value: float,
    full_below: float,
    zero_above: float,
) -> float:
    if zero_above <= full_below:
        raise FastChargeTrajectoryPortabilityError("Invalid support thresholds")
    return float(np.clip((zero_above - value) / (zero_above - full_below), 0, 1))


def _risk_weights(
    risks: Mapping[str, float],
    config: Mapping[str, object],
) -> dict[str, float]:
    epsilon = float(config["mixture"]["risk_epsilon_pp"])
    power = float(config["mixture"]["risk_inverse_power"])
    raw = np.asarray(
        [(risks[model_id] + epsilon) ** (-power) for model_id in BASE_MODEL_IDS]
    )
    normalized = raw / float(np.sum(raw))
    return {
        model_id: float(value)
        for model_id, value in zip(BASE_MODEL_IDS, normalized, strict=True)
    }


def _weight_diagnostics(
    target_signature: Mapping[str, float],
    reference_signatures: Mapping[str, Mapping[str, float]],
    reference_errors: Mapping[str, Mapping[str, float]],
    distance_thresholds: Mapping[str, float],
    config: Mapping[str, object],
) -> dict[str, object]:
    count = int(config["mixture"]["risk_reference_neighbor_count"])
    epsilon = float(
        config["base_experts"]["nearest_neighbor_delta_transfer"]["distance_epsilon"]
    )
    penalty = float(config["mixture"]["risk_dispersion_penalty"])
    variants: list[tuple[tuple[str, ...], dict[str, float]]] = []
    feature_variants = [FEATURE_IDS] + [
        tuple(feature_id for feature_id in FEATURE_IDS if feature_id != omitted)
        for omitted in FEATURE_IDS
    ]
    full_distances: dict[str, float] | None = None
    full_nearest: list[str] | None = None
    variant_weights: list[dict[str, float]] = []
    for variant_index, feature_ids in enumerate(feature_variants):
        distances = _distance_map(
            target_signature,
            reference_signatures,
            feature_ids,
            config,
        )
        nearest = _nearest_cells(distances, count)
        if variant_index == 0:
            full_distances = distances
            full_nearest = nearest
        raw_neighbor_weights = np.asarray(
            [1.0 / (distances[cell_id] + epsilon) for cell_id in nearest],
            dtype=float,
        )
        neighbor_weights = raw_neighbor_weights / float(np.sum(raw_neighbor_weights))
        risks: dict[str, float] = {}
        for model_id in BASE_MODEL_IDS:
            errors = np.asarray(
                [reference_errors[cell_id][model_id] for cell_id in nearest],
                dtype=float,
            )
            mean_error = float(np.sum(neighbor_weights * errors))
            dispersion = float(np.sum(neighbor_weights * np.abs(errors - mean_error)))
            risks[model_id] = mean_error + penalty * dispersion
        variants.append((feature_ids, risks))
        variant_weights.append(_risk_weights(risks, config))
    if full_distances is None or full_nearest is None:
        raise FastChargeTrajectoryPortabilityError("No similarity variants built")

    mean_risks = {
        model_id: float(np.mean([risks[model_id] for _, risks in variants]))
        for model_id in BASE_MODEL_IDS
    }
    evidence_weights = _risk_weights(mean_risks, config)
    variant_matrix = np.asarray(
        [
            [weights[model_id] for model_id in BASE_MODEL_IDS]
            for weights in variant_weights
        ],
        dtype=float,
    )
    variant_mean = np.mean(variant_matrix, axis=0)
    instability = float(
        np.mean(np.sum(np.abs(variant_matrix - variant_mean[None, :]), axis=1))
    )
    ordered = sorted(
        mean_risks,
        key=lambda model_id: (mean_risks[model_id], BASE_MODEL_IDS.index(model_id)),
    )
    best_risk = mean_risks[ordered[0]]
    second_risk = mean_risks[ordered[1]]
    risk_margin = float(
        (second_risk - best_risk)
        / max(best_risk, float(config["mixture"]["risk_epsilon_pp"]))
    )
    margin_support = _piecewise_increasing(
        risk_margin,
        float(config["mixture"]["relative_margin_full_uniform_below"]),
        float(config["mixture"]["relative_margin_full_evidence_above"]),
    )
    mean_distance = float(
        np.mean([full_distances[cell_id] for cell_id in full_nearest])
    )
    distance_support = _piecewise_decreasing(
        mean_distance,
        float(distance_thresholds["full_evidence_below"]),
        float(distance_thresholds["full_uniform_above"]),
    )
    stability_config = config["mixture"]["feature_jackknife_stability"]
    stability_support = _piecewise_decreasing(
        instability,
        float(stability_config["full_stability_below"]),
        float(stability_config["zero_stability_above"]),
    )
    selection_strength = margin_support * distance_support * stability_support
    uniform_value = 1.0 / len(BASE_MODEL_IDS)
    final_weights = {
        model_id: float(
            uniform_value
            + selection_strength * (evidence_weights[model_id] - uniform_value)
        )
        for model_id in BASE_MODEL_IDS
    }
    total = float(sum(final_weights.values()))
    final_weights = {
        model_id: value / total for model_id, value in final_weights.items()
    }
    dominant = min(
        BASE_MODEL_IDS,
        key=lambda model_id: (
            -final_weights[model_id],
            BASE_MODEL_IDS.index(model_id),
        ),
    )
    refusal_threshold = float(distance_thresholds["refusal_at_or_above"])
    if mean_distance >= refusal_threshold:
        evidence_status = "out_of_domain"
        action = "refuse_recommended"
    elif margin_support == 0.0:
        evidence_status = "risk_ambiguous_equal_blend"
        action = "predict_with_warning"
    elif distance_support == 0.0:
        evidence_status = "distance_unsupported_equal_blend"
        action = "predict_with_warning"
    elif stability_support == 0.0:
        evidence_status = "feature_unstable_equal_blend"
        action = "predict_with_warning"
    elif selection_strength < 1.0:
        evidence_status = "partial_evidence_blend"
        action = "predict_with_warning"
    else:
        evidence_status = "supported_stable_risk_weighted_blend"
        action = "predict"
    return {
        "risks": mean_risks,
        "weights": final_weights,
        "evidence_weights": evidence_weights,
        "dominant": dominant,
        "hard_risk_model": ordered[0],
        "nearest": full_nearest,
        "distances": {cell_id: full_distances[cell_id] for cell_id in full_nearest},
        "risk_margin": risk_margin,
        "mean_distance": mean_distance,
        "instability": instability,
        "margin_support": margin_support,
        "distance_support": distance_support,
        "stability_support": stability_support,
        "selection_strength": selection_strength,
        "evidence_status": evidence_status,
        "operational_action": action,
    }


def _training_resources(
    training: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    score_end = _score_end(config)
    cells = {
        str(cell_id): group.sort_values("cycle_index", kind="stable").reset_index(
            drop=True
        )
        for cell_id, group in training.groupby("cell_id", sort=True)
    }
    signatures: dict[int, dict[str, dict[str, float]]] = {}
    target_only: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    base_errors: dict[int, dict[str, dict[str, float]]] = {}
    normalizations: dict[str, float] = {
        cell_id: _normalization_capacity(cell) for cell_id, cell in cells.items()
    }
    retentions: dict[str, np.ndarray] = {
        cell_id: _retention(cell, normalizations[cell_id])
        for cell_id, cell in cells.items()
    }
    distance_thresholds: dict[int, dict[str, float]] = {}
    threshold_config = config["mixture"]["distance_support_thresholds"]
    neighbor_count = int(config["mixture"]["risk_reference_neighbor_count"])
    for prefix_cycle in _prefix_cycles(config):
        forecast = np.arange(prefix_cycle + 1, score_end + 1, dtype=int)
        signatures[prefix_cycle] = {}
        target_only[prefix_cycle] = {}
        base_errors[prefix_cycle] = {}
        for cell_id, cell in cells.items():
            prefix = cell.loc[cell["cycle_index"] <= prefix_cycle]
            signatures[prefix_cycle][cell_id] = _trajectory_signature(prefix, config)
            predictions, _ = _target_only_predictions(prefix, forecast, config)
            target_only[prefix_cycle][cell_id] = predictions
            observed = retentions[cell_id][forecast - 1]
            base_errors[prefix_cycle][cell_id] = {
                model_id: float(np.mean(np.abs(values - observed)))
                for model_id, values in predictions.items()
            }
        loo_mean_distances: list[float] = []
        for target_cell_id in sorted(cells):
            references = {
                cell_id: signature
                for cell_id, signature in signatures[prefix_cycle].items()
                if cell_id != target_cell_id
            }
            distances = _distance_map(
                signatures[prefix_cycle][target_cell_id],
                references,
                FEATURE_IDS,
                config,
            )
            nearest = _nearest_cells(distances, neighbor_count)
            loo_mean_distances.append(
                float(np.mean([distances[cell_id] for cell_id in nearest]))
            )
        full_below = float(
            np.quantile(
                loo_mean_distances,
                float(threshold_config["full_evidence_training_loo_quantile"]),
                method="linear",
            )
        )
        uniform_above = float(
            np.quantile(
                loo_mean_distances,
                float(threshold_config["full_uniform_training_loo_quantile"]),
                method="linear",
            )
        )
        refusal = float(
            np.quantile(
                loo_mean_distances,
                float(threshold_config["refusal_training_loo_quantile"]),
                method="linear",
            )
            * float(threshold_config["refusal_multiplier"])
        )
        if not full_below < uniform_above < refusal:
            raise FastChargeTrajectoryPortabilityError(
                "Training-derived distance thresholds are not ordered"
            )
        distance_thresholds[prefix_cycle] = {
            "full_evidence_below": full_below,
            "full_uniform_above": uniform_above,
            "refusal_at_or_above": refusal,
            "training_loo_minimum": float(np.min(loo_mean_distances)),
            "training_loo_median": float(np.median(loo_mean_distances)),
            "training_loo_maximum": float(np.max(loo_mean_distances)),
        }
    return {
        "cells": cells,
        "signatures": signatures,
        "target_only": target_only,
        "base_errors": base_errors,
        "normalizations": normalizations,
        "retentions": retentions,
        "distance_thresholds": distance_thresholds,
    }


def _reference_error_table(
    prefix_cycle: int,
    reference_ids: Sequence[str],
    resources: Mapping[str, object],
    config: Mapping[str, object],
) -> dict[str, dict[str, float]]:
    score_end = _score_end(config)
    forecast = np.arange(prefix_cycle + 1, score_end + 1, dtype=int)
    cells = resources["cells"]
    signatures = resources["signatures"][prefix_cycle]
    retentions = resources["retentions"]
    base_errors = resources["base_errors"][prefix_cycle]
    result: dict[str, dict[str, float]] = {}
    reference_set = set(reference_ids)
    for cell_id in sorted(reference_set):
        analog_reference_ids = sorted(reference_set - {cell_id})
        reference_cells = {
            reference_id: cells[reference_id] for reference_id in analog_reference_ids
        }
        reference_signatures = {
            reference_id: signatures[reference_id]
            for reference_id in analog_reference_ids
        }
        target = cells[cell_id].loc[cells[cell_id]["cycle_index"] <= prefix_cycle]
        analog = _analog_prediction(
            target,
            reference_cells,
            signatures[cell_id],
            reference_signatures,
            forecast,
            config,
        )
        observed = retentions[cell_id][forecast - 1]
        result[cell_id] = {
            **base_errors[cell_id],
            "nearest_neighbor_delta_transfer": float(
                np.mean(np.abs(analog - observed))
            ),
        }
    return result


def _expert_predictions_for_target(
    target_prefix: pd.DataFrame,
    target_signature: Mapping[str, float],
    reference_ids: Sequence[str],
    prefix_cycle: int,
    resources: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], float]:
    forecast = np.arange(prefix_cycle + 1, _score_end(config) + 1, dtype=int)
    target_only, normalization = _target_only_predictions(
        target_prefix,
        forecast,
        config,
    )
    cells = resources["cells"]
    signatures = resources["signatures"][prefix_cycle]
    reference_cells = {cell_id: cells[cell_id] for cell_id in reference_ids}
    reference_signatures = {cell_id: signatures[cell_id] for cell_id in reference_ids}
    analog = _analog_prediction(
        target_prefix,
        reference_cells,
        target_signature,
        reference_signatures,
        forecast,
        config,
    )
    experts = {**target_only, "nearest_neighbor_delta_transfer": analog}
    if tuple(experts) != BASE_MODEL_IDS:
        raise FastChargeTrajectoryPortabilityError("Expert prediction registry changed")
    return experts, normalization


def _all_model_predictions(
    experts: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, object],
) -> dict[str, np.ndarray]:
    matrix = np.vstack([experts[model_id] for model_id in BASE_MODEL_IDS])
    uniform = np.full(len(BASE_MODEL_IDS), 1.0 / len(BASE_MODEL_IDS))
    weights = np.asarray(
        [diagnostics["weights"][model_id] for model_id in BASE_MODEL_IDS]
    )
    hard_model = str(diagnostics["hard_risk_model"])
    result = {
        **experts,
        EQUAL_MODEL_ID: np.sum(uniform[:, None] * matrix, axis=0),
        HARD_RISK_MODEL_ID: experts[hard_model],
        MOE_MODEL_ID: np.sum(weights[:, None] * matrix, axis=0),
    }
    if tuple(result) != MODEL_IDS:
        raise FastChargeTrajectoryPortabilityError("Prediction model registry changed")
    return {
        model_id: np.clip(values.astype(float), 0.0, 110.0)
        for model_id, values in result.items()
    }


def _calibration_table(
    resources: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    cell_ids = sorted(resources["cells"])
    calibration_rows: list[dict[str, object]] = []
    quantile_arrays: dict[tuple[int, str], np.ndarray] = {}
    coverage = float(config["uncertainty"]["nominal_pointwise_coverage"])
    n_calibration = len(cell_ids)
    quantile_level = min(
        1.0,
        math.ceil((n_calibration + 1) * coverage) / n_calibration,
    )
    for prefix_cycle in _prefix_cycles(config):
        forecast = np.arange(prefix_cycle + 1, _score_end(config) + 1, dtype=int)
        residuals: dict[str, list[np.ndarray]] = {
            model_id: [] for model_id in MODEL_IDS
        }
        for target_cell_id in cell_ids:
            reference_ids = [
                cell_id for cell_id in cell_ids if cell_id != target_cell_id
            ]
            reference_errors = _reference_error_table(
                prefix_cycle,
                reference_ids,
                resources,
                config,
            )
            signatures = resources["signatures"][prefix_cycle]
            target_signature = signatures[target_cell_id]
            diagnostics = _weight_diagnostics(
                target_signature,
                {cell_id: signatures[cell_id] for cell_id in reference_ids},
                reference_errors,
                resources["distance_thresholds"][prefix_cycle],
                config,
            )
            target_prefix = resources["cells"][target_cell_id].loc[
                resources["cells"][target_cell_id]["cycle_index"] <= prefix_cycle
            ]
            experts, _ = _expert_predictions_for_target(
                target_prefix,
                target_signature,
                reference_ids,
                prefix_cycle,
                resources,
                config,
            )
            model_predictions = _all_model_predictions(experts, diagnostics)
            observed = resources["retentions"][target_cell_id][forecast - 1]
            for model_id in MODEL_IDS:
                residuals[model_id].append(
                    np.abs(model_predictions[model_id] - observed)
                )
        for model_id in MODEL_IDS:
            matrix = np.vstack(residuals[model_id])
            quantiles = np.quantile(
                matrix,
                quantile_level,
                axis=0,
                method="higher",
            ).astype(float)
            quantile_arrays[(prefix_cycle, model_id)] = quantiles
            for index, forecast_cycle in enumerate(forecast):
                calibration_rows.append(
                    {
                        "prefix_cycle": prefix_cycle,
                        "model_id": model_id,
                        "forecast_cycle": int(forecast_cycle),
                        "calibration_cell_count": n_calibration,
                        "absolute_residual_quantile_level": quantile_level,
                        "calibration_half_width_pp": float(quantiles[index]),
                    }
                )
    table = pd.DataFrame(calibration_rows, columns=CALIBRATION_COLUMNS).sort_values(
        ["prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    return table, quantile_arrays


def _one_hot_weights(model_id: str) -> dict[str, float]:
    return {candidate: float(candidate == model_id) for candidate in BASE_MODEL_IDS}


def _prediction_metadata(
    model_id: str,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    if model_id in BASE_MODEL_IDS:
        return {
            "dominant": model_id,
            "weights": _one_hot_weights(model_id),
            "risks": {},
            "nearest": "not_applicable",
            "distances": {},
            "risk_margin": 0.0,
            "mean_distance": 0.0,
            "instability": 0.0,
            "margin_support": 0.0,
            "distance_support": 0.0,
            "stability_support": 0.0,
            "selection_strength": 0.0,
            "evidence_status": "base_expert_comparator",
            "operational_action": "predict",
        }
    if model_id == EQUAL_MODEL_ID:
        uniform = {candidate: 1.0 / len(BASE_MODEL_IDS) for candidate in BASE_MODEL_IDS}
        return {
            **diagnostics,
            "dominant": "equal_weight_no_dominant",
            "weights": uniform,
            "evidence_status": "equal_weight_comparator",
            "operational_action": "predict",
        }
    if model_id == HARD_RISK_MODEL_ID:
        hard = str(diagnostics["hard_risk_model"])
        return {
            **diagnostics,
            "dominant": hard,
            "weights": _one_hot_weights(hard),
            "evidence_status": "hard_risk_comparator",
            "operational_action": "predict",
        }
    if model_id != MOE_MODEL_ID:
        raise FastChargeTrajectoryPortabilityError("Unknown prediction model")
    return dict(diagnostics)


def predict_fastcharge_trajectory_portability(
    training_cycles: pd.DataFrame,
    target_prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Predict all held-out target suffixes using only frozen input surfaces."""
    parsed = validate_fastcharge_trajectory_config(config)
    training = _validated_training(training_cycles, parsed)
    prefixes = _validated_target_prefixes(
        target_prefixes,
        set(training["cell_id"]),
        parsed,
    )
    resources = _training_resources(training, parsed)
    calibration, calibration_quantiles = _calibration_table(resources, parsed)
    training_ids = sorted(resources["cells"])
    reference_errors = {
        prefix_cycle: _reference_error_table(
            prefix_cycle,
            training_ids,
            resources,
            parsed,
        )
        for prefix_cycle in _prefix_cycles(parsed)
    }
    rows: list[dict[str, object]] = []
    for (paper_split, cell_id, prefix_cycle), target_prefix in prefixes.groupby(
        ["paper_split", "cell_id", "prefix_cycle"], sort=True
    ):
        prefix_cycle = int(prefix_cycle)
        target_prefix = target_prefix.sort_values("cycle_index", kind="stable")
        target_signature = _trajectory_signature(target_prefix, parsed)
        training_signatures = resources["signatures"][prefix_cycle]
        diagnostics = _weight_diagnostics(
            target_signature,
            training_signatures,
            reference_errors[prefix_cycle],
            resources["distance_thresholds"][prefix_cycle],
            parsed,
        )
        experts, normalization = _expert_predictions_for_target(
            target_prefix,
            target_signature,
            training_ids,
            prefix_cycle,
            resources,
            parsed,
        )
        model_predictions = _all_model_predictions(experts, diagnostics)
        forecast = np.arange(prefix_cycle + 1, _score_end(parsed) + 1, dtype=int)
        prefix_hash = canonical_frame_sha256(
            target_prefix.loc[:, TARGET_PREFIX_COLUMNS].reset_index(drop=True),
            TARGET_PREFIX_COLUMNS,
        )
        for model_id in MODEL_IDS:
            metadata = _prediction_metadata(model_id, diagnostics)
            half_width = calibration_quantiles[(prefix_cycle, model_id)]
            center = model_predictions[model_id]
            lower = np.clip(center - half_width, 0.0, 110.0)
            upper = np.clip(center + half_width, 0.0, 110.0)
            weights_json = _json_text(metadata["weights"])
            risks_json = _json_text(metadata["risks"])
            distances_json = _json_text(metadata["distances"])
            nearest = metadata["nearest"]
            nearest_text = (
                ";".join(str(value) for value in nearest)
                if isinstance(nearest, list)
                else str(nearest)
            )
            shared = {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "paper_split": str(paper_split),
                "cell_id": str(cell_id),
                "training_split": str(parsed["split_and_firewall"]["training_split"]),
                "prefix_cycle": prefix_cycle,
                "score_end_cycle": _score_end(parsed),
                "model_id": model_id,
                "normalization_capacity_ah": normalization,
                "prefix_row_count": len(target_prefix),
                "target_prefix_sha256": prefix_hash,
                "dominant_expert_model_id": str(metadata["dominant"]),
                "expert_weights_json": weights_json,
                "expert_risks_json": risks_json,
                "nearest_training_cell_ids": nearest_text,
                "neighbor_distances_json": distances_json,
                "risk_margin_fraction": float(metadata["risk_margin"]),
                "mean_neighbor_distance": float(metadata["mean_distance"]),
                "feature_jackknife_instability_l1": float(metadata["instability"]),
                "margin_support": float(metadata["margin_support"]),
                "distance_support": float(metadata["distance_support"]),
                "stability_support": float(metadata["stability_support"]),
                "selection_strength": float(metadata["selection_strength"]),
                "evidence_status": str(metadata["evidence_status"]),
                "operational_action": str(metadata["operational_action"]),
            }
            for index, forecast_cycle in enumerate(forecast):
                rows.append(
                    {
                        **shared,
                        "forecast_cycle": int(forecast_cycle),
                        "predicted_capacity_retention_pct": float(center[index]),
                        "interval_lower_pct": float(lower[index]),
                        "interval_upper_pct": float(upper[index]),
                        "calibration_half_width_pp": float(half_width[index]),
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": parsed["evidence_role"],
        "config_semantic_sha256": canonical_json_sha256(parsed),
        "training_cycle_sha256": canonical_frame_sha256(
            training,
            CANONICAL_CYCLE_COLUMNS,
        ),
        "target_prefix_sha256": canonical_frame_sha256(
            prefixes,
            TARGET_PREFIX_COLUMNS,
        ),
        "calibration_table_sha256": canonical_frame_sha256(
            calibration,
            CALIBRATION_COLUMNS,
        ),
        "prediction_sha256": _streaming_frame_sha256(
            predictions,
            PREDICTION_COLUMNS,
        ),
        "prediction_row_count": len(predictions),
        "training_cell_count": int(training["cell_id"].nunique()),
        "target_cell_count": int(prefixes["cell_id"].nunique()),
        "target_cells_by_split": {
            str(key): int(value)
            for key, value in prefixes.groupby("paper_split")["cell_id"]
            .nunique()
            .items()
        },
        "prefix_cycles": list(_prefix_cycles(parsed)),
        "score_end_cycle": _score_end(parsed),
        "model_ids": list(MODEL_IDS),
        "training_derived_distance_thresholds": {
            str(key): value for key, value in resources["distance_thresholds"].items()
        },
        "evaluation_target_future_outcomes_used": False,
        "complete_training_histories_used": True,
        "interval_calibration_target_suffix_used": False,
        "outcome_exposed_cohort": True,
        "inference_scope": "held_out_portability_stress_not_independent_confirmation",
    }
    return predictions, manifest, calibration


def _normalized_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise FastChargeTrajectoryPortabilityError("Prediction columns changed")
    result = predictions.copy()
    integer_columns = {
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    }
    float_columns = {
        "predicted_capacity_retention_pct",
        "interval_lower_pct",
        "interval_upper_pct",
        "calibration_half_width_pp",
        "normalization_capacity_ah",
        "risk_margin_fraction",
        "mean_neighbor_distance",
        "feature_jackknife_instability_l1",
        "margin_support",
        "distance_support",
        "stability_support",
        "selection_strength",
    }
    numeric_columns = integer_columns | float_columns
    string_columns = set(PREDICTION_COLUMNS) - numeric_columns
    for column in string_columns:
        if result[column].isna().any():
            raise FastChargeTrajectoryPortabilityError(
                f"Prediction string {column} cannot be null"
            )
        result[column] = result[column].astype(str)
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FastChargeTrajectoryPortabilityError(
                f"Prediction numeric {column} must be finite"
            )
        if column in integer_columns:
            raw = values.to_numpy(dtype=float)
            if not np.equal(raw, np.floor(raw)).all():
                raise FastChargeTrajectoryPortabilityError(
                    f"Prediction {column} must be integral"
                )
            result[column] = raw.astype(np.int64)
        else:
            result[column] = values.astype(float)
    if set(result["experiment_id"]) != {EXPERIMENT_ID}:
        raise FastChargeTrajectoryPortabilityError(
            "Prediction experiment identity changed"
        )
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise FastChargeTrajectoryPortabilityError(
            "Prediction dataset identity changed"
        )
    if set(result["model_id"]) != set(MODEL_IDS):
        raise FastChargeTrajectoryPortabilityError("Prediction model registry changed")
    for column in (
        "predicted_capacity_retention_pct",
        "interval_lower_pct",
        "interval_upper_pct",
    ):
        if not result[column].between(0.0, 110.0).all():
            raise FastChargeTrajectoryPortabilityError(
                f"Prediction {column} exceeds physical output bounds"
            )
    if not (
        (result["interval_lower_pct"] <= result["predicted_capacity_retention_pct"])
        & (result["predicted_capacity_retention_pct"] <= result["interval_upper_pct"])
    ).all():
        raise FastChargeTrajectoryPortabilityError(
            "Prediction interval excludes its center"
        )
    if (result["calibration_half_width_pp"] < 0.0).any():
        raise FastChargeTrajectoryPortabilityError(
            "Calibration half-width cannot be negative"
        )
    for column in (
        "margin_support",
        "distance_support",
        "stability_support",
        "selection_strength",
    ):
        if not result[column].between(0.0, 1.0).all():
            raise FastChargeTrajectoryPortabilityError(
                f"Prediction {column} must lie in [0, 1]"
            )
    if result.duplicated(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"]
    ).any():
        raise FastChargeTrajectoryPortabilityError(
            "Prediction coordinates are duplicated"
        )
    for value in result["expert_weights_json"].unique():
        try:
            weights = json.loads(value)
        except json.JSONDecodeError as exc:
            raise FastChargeTrajectoryPortabilityError(
                "Expert weights are not valid JSON"
            ) from exc
        if _json_text(weights) != value or set(weights) != set(BASE_MODEL_IDS):
            raise FastChargeTrajectoryPortabilityError(
                "Expert-weight registry or canonical encoding changed"
            )
        numeric_weights = np.asarray(list(weights.values()), dtype=float)
        if (numeric_weights < 0.0).any() or not math.isclose(
            float(np.sum(numeric_weights)),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FastChargeTrajectoryPortabilityError(
                "Expert weights must be a probability vector"
            )
    for column in ("expert_risks_json", "neighbor_distances_json"):
        for value in result[column].unique():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise FastChargeTrajectoryPortabilityError(
                    f"{column} is not valid JSON"
                ) from exc
            if _json_text(decoded) != value:
                raise FastChargeTrajectoryPortabilityError(
                    f"{column} is not canonical JSON"
                )
    return result.sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def _validate_prediction_replay(
    predictions: pd.DataFrame,
    manifest: Mapping[str, object],
    training: pd.DataFrame,
    prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    expected_predictions, expected_manifest, _ = (
        predict_fastcharge_trajectory_portability(
            training,
            prefixes,
            config,
        )
    )
    actual_hash = _streaming_frame_sha256(predictions, PREDICTION_COLUMNS)
    expected_hash = _streaming_frame_sha256(
        expected_predictions,
        PREDICTION_COLUMNS,
    )
    if actual_hash != expected_hash:
        raise FastChargeTrajectoryPortabilityError(
            "Predictions differ from the deterministic frozen replay"
        )
    if set(manifest) != set(expected_manifest):
        raise FastChargeTrajectoryPortabilityError("Prediction manifest keys changed")
    for key, expected_value in expected_manifest.items():
        if manifest[key] != expected_value:
            raise FastChargeTrajectoryPortabilityError(
                f"Prediction manifest mismatch for {key}"
            )


def _validated_full_cycles(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    result = _validated_numeric_cycle_frame(
        cycles,
        columns=CANONICAL_CYCLE_COLUMNS,
    )
    score_end = _score_end(config)
    expected_count = int(config["dataset"]["fixed_horizon_included_cell_count"])
    if result["cell_id"].nunique() != expected_count:
        raise FastChargeTrajectoryPortabilityError(
            "Full scoring cohort cell count changed"
        )
    for cell_id, cell in result.groupby("cell_id", sort=True):
        if sorted(cell["cycle_index"].astype(int)) != list(range(1, score_end + 1)):
            raise FastChargeTrajectoryPortabilityError(
                f"Full scoring trajectory is incomplete for {cell_id}"
            )
    if result.duplicated(["cell_id", "cycle_index"]).any():
        raise FastChargeTrajectoryPortabilityError(
            "Full scoring trajectories contain duplicate cycles"
        )
    return result.sort_values(
        ["paper_split", "cell_id", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def score_fastcharge_trajectory_portability(
    full_cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score committed target predictions after reconstructing prefix inputs."""
    parsed = validate_fastcharge_trajectory_config(config)
    cycles = _validated_full_cycles(full_cycles, parsed)
    training, prefixes, _ = build_fastcharge_prediction_inputs(cycles, parsed)
    training = _validated_training(training, parsed)
    prefixes = _validated_target_prefixes(
        prefixes,
        set(training["cell_id"]),
        parsed,
    )
    ordered_predictions = _normalized_predictions(predictions)
    expected_rows = (
        int(parsed["split_and_firewall"]["expected_total_evaluation_cells"])
        * len(MODEL_IDS)
        * sum(_score_end(parsed) - prefix for prefix in _prefix_cycles(parsed))
    )
    if len(ordered_predictions) != expected_rows:
        raise FastChargeTrajectoryPortabilityError("Prediction row count changed")
    _validate_prediction_replay(
        ordered_predictions,
        prediction_manifest,
        training,
        prefixes,
        parsed,
    )

    evaluation_splits = set(parsed["split_and_firewall"]["evaluation_splits"])
    truth_rows: list[pd.DataFrame] = []
    for cell_id, cell in cycles.loc[
        cycles["paper_split"].isin(evaluation_splits)
    ].groupby("cell_id", sort=True):
        cell = cell.sort_values("cycle_index", kind="stable")
        normalization = _normalization_capacity(cell)
        truth_rows.append(
            pd.DataFrame(
                {
                    "paper_split": str(cell["paper_split"].iloc[0]),
                    "cell_id": str(cell_id),
                    "forecast_cycle": cell["cycle_index"].astype(int),
                    "observed_capacity_retention_pct": _retention(cell, normalization),
                }
            )
        )
    truth = pd.concat(truth_rows, ignore_index=True)
    linked = ordered_predictions.merge(
        truth,
        on=["paper_split", "cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise FastChargeTrajectoryPortabilityError(
            "Predictions cannot link to scoring truth"
        )
    score_rows: list[dict[str, object]] = []
    for (paper_split, cell_id, prefix_cycle, model_id), group in linked.groupby(
        ["paper_split", "cell_id", "prefix_cycle", "model_id"], sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        observed = group["observed_capacity_retention_pct"].to_numpy(dtype=float)
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        lower = group["interval_lower_pct"].to_numpy(dtype=float)
        upper = group["interval_upper_pct"].to_numpy(dtype=float)
        error = predicted - observed
        absolute = np.abs(error)
        inside = (observed >= lower) & (observed <= upper)
        score_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "paper_split": str(paper_split),
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix_cycle),
                "score_end_cycle": _score_end(parsed),
                "model_id": str(model_id),
                "dominant_expert_model_id": str(
                    group["dominant_expert_model_id"].iloc[0]
                ),
                "evidence_status": str(group["evidence_status"].iloc[0]),
                "operational_action": str(group["operational_action"].iloc[0]),
                "future_observation_count": len(group),
                "trajectory_mae_pp": float(np.mean(absolute)),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(absolute[-1]),
                "empirical_interval_coverage_fraction": float(np.mean(inside)),
                "mean_interval_width_pp": float(np.mean(upper - lower)),
                "endpoint_inside_interval": float(inside[-1]),
            }
        )
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    metric_columns = list(SCORE_COLUMNS[-6:])
    by_split_prefix = (
        scores.groupby(["paper_split", "prefix_cycle", "model_id"], sort=True)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )
    by_split = (
        scores.groupby(["paper_split", "model_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    overall = scores.groupby("model_id", sort=True)[metric_columns].mean().reset_index()

    def overall_metric(model_id: str, metric: str) -> float:
        return float(overall.loc[overall["model_id"] == model_id, metric].iloc[0])

    moe_mae = overall_metric(MOE_MODEL_ID, "trajectory_mae_pp")
    equal_mae = overall_metric(EQUAL_MODEL_ID, "trajectory_mae_pp")
    persistence_mae = overall_metric("target_prefix_persistence", "trajectory_mae_pp")
    gate = parsed["evaluation"]["frozen_portability_gate"]
    overall_equal_pass = (moe_mae - equal_mae) <= float(
        gate["maximum_overall_mae_degradation_vs_equal_weight_pp"]
    )
    split_deltas: list[dict[str, object]] = []
    split_pass = True
    for paper_split in parsed["split_and_firewall"]["evaluation_splits"]:
        moe_value = float(
            by_split.loc[
                (by_split["paper_split"] == paper_split)
                & (by_split["model_id"] == MOE_MODEL_ID),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        equal_value = float(
            by_split.loc[
                (by_split["paper_split"] == paper_split)
                & (by_split["model_id"] == EQUAL_MODEL_ID),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        delta = moe_value - equal_value
        passed = delta <= float(
            gate["maximum_each_split_mae_degradation_vs_equal_weight_pp"]
        )
        split_pass = split_pass and passed
        split_deltas.append(
            {
                "paper_split": paper_split,
                "moe_mae_pp": moe_value,
                "equal_weight_mae_pp": equal_value,
                "delta_pp": delta,
                "passed": passed,
            }
        )
    persistence_improvement = (persistence_mae - moe_mae) / persistence_mae
    persistence_pass = persistence_improvement >= float(
        gate["minimum_relative_mae_improvement_vs_persistence"]
    )
    moe_coverage = overall_metric(MOE_MODEL_ID, "empirical_interval_coverage_fraction")
    moe_width = overall_metric(MOE_MODEL_ID, "mean_interval_width_pp")
    coverage_pass = moe_coverage >= float(
        gate["minimum_empirical_interval_coverage_fraction"]
    )
    width_pass = moe_width <= float(gate["maximum_mean_interval_width_pp"])
    passed_all = all(
        (
            overall_equal_pass,
            split_pass,
            persistence_pass,
            coverage_pass,
            width_pass,
        )
    )

    decisions = (
        ordered_predictions.loc[ordered_predictions["model_id"] == MOE_MODEL_ID]
        .groupby(["paper_split", "cell_id", "prefix_cycle"], sort=True)
        .first()
    )
    action_counts = {
        str(key): int(value)
        for key, value in decisions["operational_action"].value_counts().items()
    }
    evidence_counts = {
        str(key): int(value)
        for key, value in decisions["evidence_status"].value_counts().items()
    }
    summary: dict[str, object] = {
        "schema_version": "lifetwin.fastcharge_trajectory_score_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": parsed["evidence_role"],
        "prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "score_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "training_cell_count": int(training["cell_id"].nunique()),
        "evaluation_cell_count": int(prefixes["cell_id"].nunique()),
        "evaluation_cells_by_split": {
            str(key): int(value)
            for key, value in prefixes.groupby("paper_split")["cell_id"]
            .nunique()
            .items()
        },
        "prefix_cycles": list(_prefix_cycles(parsed)),
        "score_end_cycle": _score_end(parsed),
        "model_ids": list(MODEL_IDS),
        "by_split_prefix_metrics": json.loads(
            _json_text(by_split_prefix.to_dict(orient="records"))
        ),
        "by_split_metrics": json.loads(_json_text(by_split.to_dict(orient="records"))),
        "overall_metrics": json.loads(_json_text(overall.to_dict(orient="records"))),
        "primary_comparison": {
            "moe_overall_mae_pp": moe_mae,
            "equal_weight_overall_mae_pp": equal_mae,
            "delta_vs_equal_weight_pp": moe_mae - equal_mae,
            "persistence_overall_mae_pp": persistence_mae,
            "relative_improvement_vs_persistence": persistence_improvement,
            "split_deltas_vs_equal_weight": split_deltas,
        },
        "interval_diagnostic": {
            "moe_empirical_coverage_fraction": moe_coverage,
            "moe_mean_width_pp": moe_width,
            "nominal_pointwise_coverage": parsed["uncertainty"][
                "nominal_pointwise_coverage"
            ],
            "formal_exchangeable_coverage_claim": False,
        },
        "frozen_portability_gate": {
            "status": "passed" if passed_all else "failed",
            "overall_equal_weight_gate_passed": overall_equal_pass,
            "all_split_equal_weight_gates_passed": split_pass,
            "persistence_improvement_gate_passed": persistence_pass,
            "interval_coverage_gate_passed": coverage_pass,
            "interval_width_gate_passed": width_pass,
            "thresholds": gate,
            "interpretation": (
                "held_out_portability_stress_not_independent_confirmation"
            ),
        },
        "operational_action_counts": action_counts,
        "evidence_status_counts": evidence_counts,
        "allowed_claims": list(parsed["claim_boundaries"]["allowed_claims"]),
        "prohibited_claims": list(parsed["claim_boundaries"]["prohibited_claims"]),
    }
    return scores, summary


__all__ = [
    "BASE_MODEL_IDS",
    "CALIBRATION_COLUMNS",
    "CONFIG_SEMANTIC_SHA256",
    "EQUAL_MODEL_ID",
    "EXPERIMENT_ID",
    "FEATURE_IDS",
    "FastChargeTrajectoryPortabilityError",
    "HARD_RISK_MODEL_ID",
    "MODEL_IDS",
    "MOE_MODEL_ID",
    "PREDICTION_COLUMNS",
    "SCORE_COLUMNS",
    "load_fastcharge_trajectory_config",
    "predict_fastcharge_trajectory_portability",
    "score_fastcharge_trajectory_portability",
    "validate_fastcharge_trajectory_config",
]
