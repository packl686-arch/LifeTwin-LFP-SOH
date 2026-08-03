"""Nested-LOCO dynamic model selection for the NASA accelerated-cycle stress set.

The prediction boundary is explicit: the held-out cell contributes only its
prefix, while the other three cells may contribute complete histories through
the common scoring endpoint. The scorer receives held-out suffix outcomes only
after the prediction artifact and its manifest have been committed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    EVIDENCE_ROLE,
    PREFIX_CYCLES,
    PRIMARY_PREFIX_CYCLE,
    SCORE_END_CYCLE,
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.nasa_dynamic_gate.config.v2"
EXPERIMENT_ID = "nasa_pcoe_dynamic_gate_accelerated_cycling_stress_v2"
CONFIG_SEMANTIC_SHA256 = (
    "3a227b2e3fa6302f563d88b54cfb0442045643014ddb9d4d5be4ae0679a7f504"
)
PREDICTION_MANIFEST_SCHEMA_VERSION = "lifetwin.nasa_dynamic_gate.prediction_manifest.v2"

BASE_MODEL_IDS = (
    "target_prefix_persistence",
    "target_prefix_full_linear",
    "target_prefix_robust_recent_linear",
    "target_prefix_constrained_sqrt_linear",
)
GATED_MODEL_IDS = (
    "nested_loco_capacity_only_mean_gate",
    "nested_loco_curve_aware_mean_gate",
    "nested_loco_curve_aware_consensus_fallback_gate",
)
MODEL_IDS = (*BASE_MODEL_IDS, *GATED_MODEL_IDS)

CAPACITY_FEATURE_IDS = (
    "capacity_full_slope_pp_per_cycle",
    "capacity_recent_slope_pp_per_cycle",
    "capacity_slope_disagreement_pp_per_cycle",
    "capacity_residual_std_pp",
    "capacity_recovery_fraction",
)
CURVE_FEATURE_IDS = (
    "voltage_window_duration_relative_slope_pp_per_cycle",
    "voltage_at_1ah_slope_mv_per_cycle",
    "temperature_rise_slope_c_per_cycle",
    "discharge_cutoff_voltage_v",
)
SIMILARITY_FEATURE_IDS = (*CAPACITY_FEATURE_IDS, *CURVE_FEATURE_IDS)
SIMILARITY_SCALE_FLOORS = {
    "capacity_full_slope_pp_per_cycle": 0.005,
    "capacity_recent_slope_pp_per_cycle": 0.005,
    "capacity_slope_disagreement_pp_per_cycle": 0.005,
    "capacity_residual_std_pp": 0.1,
    "capacity_recovery_fraction": 0.02,
    "voltage_window_duration_relative_slope_pp_per_cycle": 0.01,
    "voltage_at_1ah_slope_mv_per_cycle": 0.02,
    "temperature_rise_slope_c_per_cycle": 0.005,
    "discharge_cutoff_voltage_v": 0.1,
}

REQUIRED_CYCLE_COLUMNS = (
    "dataset_id",
    "cell_id",
    "cycle_index",
    "discharge_capacity_ah",
    "discharge_cutoff_voltage_v",
    "common_window_3p8_to_3p4_duration_s",
    "voltage_at_1p0_ah_v",
    "temperature_rise_c",
)
FOLD_TABLE_COLUMNS = (
    *REQUIRED_CYCLE_COLUMNS,
    "held_out_cell_id",
    "row_role",
    "prefix_cycle",
)
PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "selected_base_model_id",
    "forecast_cycle",
    "predicted_capacity_retention_pct",
    "evidence_band_lower_pct",
    "evidence_band_upper_pct",
    "normalization_capacity_ah",
    "prefix_row_count",
    "target_prefix_sha256",
    "gate_feature_set",
    "gate_nearest_cell_ids",
    "gate_neighbor_distances_json",
    "gate_training_mae_json",
    "gate_evidence_status",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "selected_base_model_id",
    "gate_evidence_status",
    "future_observation_count",
    "trajectory_iae_pp_normalized_by_cycle_horizon",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
    "empirical_evidence_band_coverage_fraction",
    "mean_evidence_band_width_pp",
    "endpoint_inside_evidence_band",
)

_ALLOWED_CLAIMS = (
    "chronological_prefix_prediction_software_validation",
    "descriptive_nested_loco_algorithm_development_on_public_stress_data",
    "capacity_only_versus_curve_aware_gate_ablation",
)
_PROHIBITED_CLAIMS = (
    "independent_outcome_blind_confirmation",
    "lfp_chemistry_validation",
    "calendar_aging_validation",
    "fifteen_to_twenty_five_year_accuracy",
    "hithium_product_accuracy",
    "stationary_storage_field_validation",
    "inferential_significance_from_four_cells",
    "formal_uncertainty_coverage",
)


class NasaDynamicGateError(ValueError):
    """Raised when the V2 experiment contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NasaDynamicGateError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NasaDynamicGateError("Value is not canonical finite JSON") from exc


def validate_nasa_dynamic_gate_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Validate the exact preregistered V2 development protocol."""
    if not isinstance(config, Mapping):
        raise NasaDynamicGateError("NASA dynamic-gate config must be an object")
    if canonical_json_sha256(dict(config)) != CONFIG_SEMANTIC_SHA256:
        raise NasaDynamicGateError("NASA dynamic-gate frozen config changed")
    detached = json.loads(_canonical_json_text(dict(config)))
    if detached.get("schema_version") != SCHEMA_VERSION:
        raise NasaDynamicGateError("NASA dynamic-gate schema changed")
    if detached.get("experiment_id") != EXPERIMENT_ID:
        raise NasaDynamicGateError("NASA dynamic-gate experiment changed")
    return detached


def load_nasa_dynamic_gate_config(path: str | Path) -> dict[str, object]:
    """Load strict JSON and verify the frozen semantic hash."""
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NasaDynamicGateError(f"Non-finite JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NasaDynamicGateError("Cannot load NASA dynamic-gate config") from exc
    if not isinstance(value, Mapping):
        raise NasaDynamicGateError("NASA dynamic-gate config must be an object")
    return validate_nasa_dynamic_gate_config(value)


def _validated_cycles(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    parsed = validate_nasa_dynamic_gate_config(config)
    missing = sorted(set(REQUIRED_CYCLE_COLUMNS) - set(cycles.columns))
    if missing:
        raise NasaDynamicGateError(f"Missing V2 cycle columns: {missing}")
    if cycles.empty:
        raise NasaDynamicGateError("Canonical cycle table is empty")
    result = cycles.loc[:, REQUIRED_CYCLE_COLUMNS].copy()
    for column in ("dataset_id", "cell_id"):
        if result[column].isna().any():
            raise NasaDynamicGateError(f"{column} cannot contain null values")
        result[column] = result[column].astype(str)
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise NasaDynamicGateError("Canonical cycles contain an unexpected dataset")
    expected_cells = tuple(parsed["design"]["cell_ids"])
    if set(result["cell_id"]) != set(expected_cells):
        raise NasaDynamicGateError(
            "Canonical cycles do not contain the exact four cells"
        )

    numeric_columns = tuple(
        column
        for column in REQUIRED_CYCLE_COLUMNS
        if column not in {"dataset_id", "cell_id"}
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaDynamicGateError("V2 cycle values must be finite numeric values")
    raw_index = numeric["cycle_index"].to_numpy(dtype=float)
    if not np.equal(raw_index, np.floor(raw_index)).all() or (raw_index < 1).any():
        raise NasaDynamicGateError("cycle_index must contain positive integers")
    result["cycle_index"] = raw_index.astype(np.int64)
    for column in numeric_columns:
        if column != "cycle_index":
            result[column] = numeric[column].astype(float)
    positive_columns = (
        "discharge_capacity_ah",
        "common_window_3p8_to_3p4_duration_s",
    )
    if (result.loc[:, positive_columns] <= 0.0).any().any():
        raise NasaDynamicGateError("V2 positive curve features must be positive")
    if (result["temperature_rise_c"] < 0.0).any():
        raise NasaDynamicGateError("V2 temperature rise cannot be negative")
    if result.duplicated(["dataset_id", "cell_id", "cycle_index"]).any():
        raise NasaDynamicGateError("Canonical cycles contain duplicate coordinates")

    cutoffs = parsed["dataset"]["cell_discharge_cutoff_voltage_v"]
    for cell_id in expected_cells:
        cell = result.loc[result["cell_id"] == cell_id]
        if not np.allclose(
            cell["discharge_cutoff_voltage_v"].to_numpy(dtype=float),
            float(cutoffs[cell_id]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise NasaDynamicGateError(f"Cutoff voltage changed for {cell_id}")
        support = sorted(
            cell.loc[
                cell["cycle_index"] <= SCORE_END_CYCLE,
                "cycle_index",
            ].astype(int)
        )
        if support != list(range(1, SCORE_END_CYCLE + 1)):
            raise NasaDynamicGateError(
                f"{cell_id} lacks contiguous support through {SCORE_END_CYCLE}"
            )
    return result.sort_values(["cell_id", "cycle_index"], kind="stable").reset_index(
        drop=True
    )


def build_nasa_dynamic_gate_fold_table(
    cycles: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    """Build outer-fold inputs with target prefixes and training histories."""
    ordered = _validated_cycles(cycles, config)
    frames: list[pd.DataFrame] = []
    for held_out_cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            target = ordered.loc[
                (ordered["cell_id"] == held_out_cell_id)
                & (ordered["cycle_index"] <= prefix_cycle)
            ].copy()
            target["held_out_cell_id"] = held_out_cell_id
            target["row_role"] = "target_prefix"
            target["prefix_cycle"] = prefix_cycle
            training = ordered.loc[
                (ordered["cell_id"] != held_out_cell_id)
                & (ordered["cycle_index"] <= SCORE_END_CYCLE)
            ].copy()
            training["held_out_cell_id"] = held_out_cell_id
            training["row_role"] = "training_history"
            training["prefix_cycle"] = prefix_cycle
            frames.extend(
                (
                    target.loc[:, FOLD_TABLE_COLUMNS],
                    training.loc[:, FOLD_TABLE_COLUMNS],
                )
            )
    return pd.concat(frames, ignore_index=True).sort_values(
        ["held_out_cell_id", "prefix_cycle", "row_role", "cell_id", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def _validated_fold_table(
    fold_table: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    parsed = validate_nasa_dynamic_gate_config(config)
    if tuple(fold_table.columns) != FOLD_TABLE_COLUMNS:
        raise NasaDynamicGateError(
            "Prediction input must contain only the V2 fold-table columns"
        )
    if fold_table.empty:
        raise NasaDynamicGateError("Prediction fold table is empty")
    result = fold_table.copy()
    for column in ("dataset_id", "cell_id", "held_out_cell_id", "row_role"):
        if result[column].isna().any():
            raise NasaDynamicGateError(f"{column} cannot contain null values")
        result[column] = result[column].astype(str)
    numeric_columns = tuple(
        column
        for column in FOLD_TABLE_COLUMNS
        if column not in {"dataset_id", "cell_id", "held_out_cell_id", "row_role"}
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaDynamicGateError("Fold table must contain finite numeric values")
    for column in ("cycle_index", "prefix_cycle"):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise NasaDynamicGateError(f"{column} must contain integers")
        result[column] = values.astype(np.int64)
    for column in numeric_columns:
        if column not in {"cycle_index", "prefix_cycle"}:
            result[column] = numeric[column].astype(float)

    expected_cells = tuple(parsed["design"]["cell_ids"])
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise NasaDynamicGateError("Fold-table dataset identity changed")
    if set(result["cell_id"]) != set(expected_cells):
        raise NasaDynamicGateError("Fold-table source cells changed")
    if set(result["held_out_cell_id"]) != set(expected_cells):
        raise NasaDynamicGateError("Fold-table held-out cells changed")
    if set(result["row_role"]) != {"target_prefix", "training_history"}:
        raise NasaDynamicGateError("Fold-table row roles changed")
    if set(result["prefix_cycle"].astype(int)) != set(PREFIX_CYCLES):
        raise NasaDynamicGateError("Fold-table prefixes changed")
    if result.duplicated(
        ["held_out_cell_id", "prefix_cycle", "row_role", "cell_id", "cycle_index"]
    ).any():
        raise NasaDynamicGateError("Fold table contains duplicate coordinates")

    cutoffs = parsed["dataset"]["cell_discharge_cutoff_voltage_v"]
    for held_out_cell_id in expected_cells:
        training_ids = set(expected_cells) - {held_out_cell_id}
        for prefix_cycle in PREFIX_CYCLES:
            fold = result.loc[
                (result["held_out_cell_id"] == held_out_cell_id)
                & (result["prefix_cycle"] == prefix_cycle)
            ]
            target = fold.loc[fold["row_role"] == "target_prefix"]
            training = fold.loc[fold["row_role"] == "training_history"]
            if set(target["cell_id"]) != {held_out_cell_id}:
                raise NasaDynamicGateError("Target rows do not match the held-out cell")
            if target.sort_values("cycle_index")["cycle_index"].astype(
                int
            ).tolist() != list(range(1, prefix_cycle + 1)):
                raise NasaDynamicGateError(
                    f"{held_out_cell_id} prefix {prefix_cycle} is not exactly truncated"
                )
            if (target["cycle_index"] > prefix_cycle).any():
                raise NasaDynamicGateError(
                    "Prediction input contains target future rows"
                )
            if set(training["cell_id"]) != training_ids:
                raise NasaDynamicGateError(
                    "Outer-fold training-cell identities changed"
                )
            for training_cell_id in sorted(training_ids):
                support = (
                    training.loc[training["cell_id"] == training_cell_id, "cycle_index"]
                    .sort_values()
                    .astype(int)
                    .tolist()
                )
                if support != list(range(1, SCORE_END_CYCLE + 1)):
                    raise NasaDynamicGateError(
                        f"Training history is incomplete for {training_cell_id}"
                    )
            for source_cell_id, source in fold.groupby("cell_id", sort=True):
                if not np.allclose(
                    source["discharge_cutoff_voltage_v"].to_numpy(dtype=float),
                    float(cutoffs[source_cell_id]),
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise NasaDynamicGateError(
                        f"Cutoff voltage changed for {source_cell_id}"
                    )
    return result.sort_values(
        ["held_out_cell_id", "prefix_cycle", "row_role", "cell_id", "cycle_index"],
        kind="stable",
        ignore_index=True,
    )


def _normalization_capacity(cell: pd.DataFrame) -> float:
    first_five = cell.loc[
        cell["cycle_index"].between(1, 5), "discharge_capacity_ah"
    ].astype(float)
    if len(first_five) != 5:
        raise NasaDynamicGateError("Normalization requires exact cycles 1 to 5")
    value = float(median(first_five.tolist()))
    if not math.isfinite(value) or value <= 0.0:
        raise NasaDynamicGateError("Normalization capacity must be positive")
    return value


def _retention_pct(cell: pd.DataFrame, normalization: float) -> np.ndarray:
    return 100.0 * cell["discharge_capacity_ah"].to_numpy(dtype=float) / normalization


def _linear_parameters(
    cycle: Sequence[float],
    retention: Sequence[float],
    *,
    constrain_non_positive: bool,
) -> tuple[float, float]:
    x = np.asarray(cycle, dtype=float)
    y = np.asarray(retention, dtype=float)
    centered = x - float(np.mean(x))
    denominator = float(np.sum(np.square(centered)))
    if denominator <= 0.0:
        raise NasaDynamicGateError("Linear fit has no cycle support")
    slope = float(np.sum(centered * (y - float(np.mean(y)))) / denominator)
    if constrain_non_positive:
        slope = min(slope, 0.0)
    intercept = float(np.mean(y) - slope * np.mean(x))
    return intercept, slope


def _robust_recent_parameters(
    cycle: Sequence[float],
    retention: Sequence[float],
    *,
    window_cycles: int,
) -> tuple[float, float]:
    x = np.asarray(cycle, dtype=float)[-window_cycles:]
    y = np.asarray(retention, dtype=float)[-window_cycles:]
    slopes: list[float] = []
    for left in range(len(x) - 1):
        delta_x = x[left + 1 :] - x[left]
        slopes.extend(((y[left + 1 :] - y[left]) / delta_x).tolist())
    if not slopes:
        raise NasaDynamicGateError("Robust recent fit has no pairwise support")
    slope = min(float(median(slopes)), 0.0)
    intercept = float(median((y - slope * x).tolist()))
    return intercept, slope


def _constrained_sqrt_linear_parameters(
    cycle: Sequence[float],
    retention: Sequence[float],
    *,
    huber_iterations: int,
    huber_delta_mad: float,
) -> tuple[float, float]:
    x = np.asarray(cycle, dtype=float)
    loss = 100.0 - np.asarray(retention, dtype=float)
    shifted = np.maximum(x - 1.0, 0.0)
    design = np.column_stack((np.sqrt(shifted), shifted))
    if float(np.sum(np.square(design))) <= 0.0:
        raise NasaDynamicGateError("Constrained sqrt-linear fit has no support")
    weights = np.ones(len(x), dtype=float)
    coefficients = np.zeros(2, dtype=float)
    for _ in range(huber_iterations):
        root_weight = np.sqrt(weights)
        result = lsq_linear(
            design * root_weight[:, None],
            loss * root_weight,
            bounds=(0.0, np.inf),
            method="trf",
            lsmr_tol="auto",
        )
        if not result.success or not np.isfinite(result.x).all():
            raise NasaDynamicGateError("Constrained sqrt-linear fit failed")
        coefficients = result.x.astype(float)
        residual = loss - design @ coefficients
        centered = residual - float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(centered)))
        cutoff = huber_delta_mad * max(scale, 1e-9)
        magnitude = np.abs(centered)
        weights = np.ones(len(x), dtype=float)
        large = magnitude > cutoff
        weights[large] = cutoff / magnitude[large]
    return float(coefficients[0]), float(coefficients[1])


def _clip_prediction(value: float) -> float:
    return float(min(max(value, 0.0), 110.0))


def _base_model_predictions(
    prefix: pd.DataFrame,
    forecast_cycles: np.ndarray,
    config: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], float]:
    normalization = _normalization_capacity(prefix)
    cycle = prefix["cycle_index"].to_numpy(dtype=float)
    retention = _retention_pct(prefix, normalization)
    linear_intercept, linear_slope = _linear_parameters(
        cycle,
        retention,
        constrain_non_positive=True,
    )
    model_config = config["models"]
    recent_intercept, recent_slope = _robust_recent_parameters(
        cycle,
        retention,
        window_cycles=int(model_config["robust_recent_window_cycles"]),
    )
    sqrt_coefficient, linear_coefficient = _constrained_sqrt_linear_parameters(
        cycle,
        retention,
        huber_iterations=int(model_config["huber_iterations"]),
        huber_delta_mad=float(model_config["huber_delta_mad"]),
    )
    shifted_forecast = np.maximum(forecast_cycles.astype(float) - 1.0, 0.0)
    raw = {
        "target_prefix_persistence": np.full(
            len(forecast_cycles),
            float(retention[-1]),
        ),
        "target_prefix_full_linear": (
            linear_intercept + linear_slope * forecast_cycles
        ),
        "target_prefix_robust_recent_linear": (
            recent_intercept + recent_slope * forecast_cycles
        ),
        "target_prefix_constrained_sqrt_linear": (
            100.0
            - sqrt_coefficient * np.sqrt(shifted_forecast)
            - linear_coefficient * shifted_forecast
        ),
    }
    clipped = {
        model_id: np.clip(values.astype(float), 0.0, 110.0)
        for model_id, values in raw.items()
    }
    if tuple(clipped) != BASE_MODEL_IDS:
        raise NasaDynamicGateError("Base-model registry changed")
    return clipped, normalization


def _trajectory_signature(
    prefix: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, float]:
    normalization = _normalization_capacity(prefix)
    cycle = prefix["cycle_index"].to_numpy(dtype=float)
    retention = _retention_pct(prefix, normalization)
    linear_intercept, full_slope = _linear_parameters(
        cycle,
        retention,
        constrain_non_positive=False,
    )
    _, recent_slope = _robust_recent_parameters(
        cycle,
        retention,
        window_cycles=int(config["models"]["robust_recent_window_cycles"]),
    )
    residual = retention - (linear_intercept + full_slope * cycle)
    recovery_fraction = float(
        np.mean(np.diff(retention) > 0.1) if len(retention) > 1 else 0.0
    )

    window_duration = prefix["common_window_3p8_to_3p4_duration_s"].to_numpy(
        dtype=float
    )
    initial_window = float(median(window_duration[:5].tolist()))
    relative_window = 100.0 * window_duration / initial_window
    _, window_slope = _linear_parameters(
        cycle,
        relative_window,
        constrain_non_positive=False,
    )
    voltage_1ah_mv = 1_000.0 * prefix["voltage_at_1p0_ah_v"].to_numpy(dtype=float)
    _, voltage_slope = _linear_parameters(
        cycle,
        voltage_1ah_mv,
        constrain_non_positive=False,
    )
    temperature_rise = prefix["temperature_rise_c"].to_numpy(dtype=float)
    _, temperature_slope = _linear_parameters(
        cycle,
        temperature_rise,
        constrain_non_positive=False,
    )
    signature = {
        "capacity_full_slope_pp_per_cycle": float(full_slope),
        "capacity_recent_slope_pp_per_cycle": float(recent_slope),
        "capacity_slope_disagreement_pp_per_cycle": float(
            abs(full_slope - recent_slope)
        ),
        "capacity_residual_std_pp": float(np.sqrt(np.mean(np.square(residual)))),
        "capacity_recovery_fraction": recovery_fraction,
        "voltage_window_duration_relative_slope_pp_per_cycle": float(window_slope),
        "voltage_at_1ah_slope_mv_per_cycle": float(voltage_slope),
        "temperature_rise_slope_c_per_cycle": float(temperature_slope),
        "discharge_cutoff_voltage_v": float(
            prefix["discharge_cutoff_voltage_v"].iloc[0]
        ),
    }
    if tuple(signature) != SIMILARITY_FEATURE_IDS or not all(
        math.isfinite(value) for value in signature.values()
    ):
        raise NasaDynamicGateError("Trajectory signature is invalid")
    return signature


def _nearest_training_cells(
    target_signature: Mapping[str, float],
    training_signatures: Mapping[str, Mapping[str, float]],
    feature_ids: Sequence[str],
) -> tuple[list[str], dict[str, float]]:
    training_cell_ids = sorted(training_signatures)
    training_matrix = np.asarray(
        [
            [training_signatures[cell_id][feature_id] for feature_id in feature_ids]
            for cell_id in training_cell_ids
        ],
        dtype=float,
    )
    target = np.asarray(
        [target_signature[feature_id] for feature_id in feature_ids],
        dtype=float,
    )
    scale = np.std(training_matrix, axis=0, ddof=0)
    floors = np.asarray(
        [SIMILARITY_SCALE_FLOORS[feature_id] for feature_id in feature_ids],
        dtype=float,
    )
    scale = np.maximum(scale, floors)
    distances = np.sqrt(np.mean(np.square((training_matrix - target) / scale), axis=1))
    distance_map = {
        cell_id: float(distance)
        for cell_id, distance in zip(training_cell_ids, distances, strict=True)
    }
    ordered = sorted(
        training_cell_ids, key=lambda cell_id: (distance_map[cell_id], cell_id)
    )
    return ordered[:2], distance_map


def _select_lowest_mae(
    training_mae: Mapping[str, Mapping[str, float]],
    reference_cells: Sequence[str],
) -> tuple[str, dict[str, float]]:
    mean_mae = {
        model_id: float(
            np.mean([training_mae[cell_id][model_id] for cell_id in reference_cells])
        )
        for model_id in BASE_MODEL_IDS
    }
    rank = {model_id: index for index, model_id in enumerate(BASE_MODEL_IDS)}
    selected = min(
        BASE_MODEL_IDS, key=lambda model_id: (mean_mae[model_id], rank[model_id])
    )
    return selected, mean_mae


def _training_diagnostics(
    training: pd.DataFrame,
    *,
    prefix_cycle: int,
    config: Mapping[str, object],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, float]],
]:
    forecast_cycles = np.arange(prefix_cycle + 1, SCORE_END_CYCLE + 1, dtype=float)
    training_mae: dict[str, dict[str, float]] = {}
    training_errors: dict[str, dict[str, np.ndarray]] = {}
    training_signatures: dict[str, dict[str, float]] = {}
    for cell_id, cell in training.groupby("cell_id", sort=True):
        ordered = cell.sort_values("cycle_index", kind="stable")
        prefix = ordered.loc[ordered["cycle_index"] <= prefix_cycle]
        suffix = ordered.loc[
            ordered["cycle_index"].between(prefix_cycle + 1, SCORE_END_CYCLE)
        ]
        predictions, normalization = _base_model_predictions(
            prefix,
            forecast_cycles,
            config,
        )
        truth = _retention_pct(suffix, normalization)
        if len(truth) != len(forecast_cycles):
            raise NasaDynamicGateError("Training suffix support changed")
        training_errors[str(cell_id)] = {
            model_id: np.abs(values - truth) for model_id, values in predictions.items()
        }
        training_mae[str(cell_id)] = {
            model_id: float(np.mean(training_errors[str(cell_id)][model_id]))
            for model_id in BASE_MODEL_IDS
        }
        training_signatures[str(cell_id)] = _trajectory_signature(prefix, config)
    return training_mae, training_errors, training_signatures


def _band_arrays(
    center: np.ndarray,
    training_errors: Mapping[str, Mapping[str, np.ndarray]],
    reference_cells: Sequence[str],
    *,
    selected_model_id: str,
    prefix_cycle: int,
    disagreement_fallback: bool,
    config: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    error_matrix = np.vstack(
        [training_errors[cell_id][selected_model_id] for cell_id in reference_cells]
    )
    band_config = config["evidence_band"]
    half_width = np.maximum(
        np.max(error_matrix, axis=0),
        float(band_config["minimum_half_width_pp"]),
    )
    scarcity_multiplier = max(
        1.0,
        (float(band_config["scarcity_reference_prefix_cycle"]) / float(prefix_cycle))
        ** float(band_config["scarcity_exponent"]),
    )
    half_width = half_width * scarcity_multiplier
    if disagreement_fallback:
        half_width = half_width * float(band_config["neighbor_disagreement_multiplier"])
    return (
        np.clip(center - half_width, 0.0, 110.0),
        np.clip(center + half_width, 0.0, 110.0),
    )


def _prediction_rows_for_fold(
    fold: pd.DataFrame,
    *,
    held_out_cell_id: str,
    prefix_cycle: int,
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    target = fold.loc[fold["row_role"] == "target_prefix"].sort_values(
        "cycle_index", kind="stable"
    )
    training = fold.loc[fold["row_role"] == "training_history"].sort_values(
        ["cell_id", "cycle_index"], kind="stable"
    )
    training_cell_ids = sorted(training["cell_id"].astype(str).unique().tolist())
    training_cell_text = ";".join(training_cell_ids)
    forecast_cycles = np.arange(prefix_cycle + 1, SCORE_END_CYCLE + 1, dtype=float)
    target_predictions, normalization = _base_model_predictions(
        target,
        forecast_cycles,
        config,
    )
    training_mae, training_errors, training_signatures = _training_diagnostics(
        training,
        prefix_cycle=prefix_cycle,
        config=config,
    )
    target_signature = _trajectory_signature(target, config)

    capacity_nearest, capacity_distances = _nearest_training_cells(
        target_signature,
        training_signatures,
        CAPACITY_FEATURE_IDS,
    )
    curve_nearest, curve_distances = _nearest_training_cells(
        target_signature,
        training_signatures,
        SIMILARITY_FEATURE_IDS,
    )
    capacity_selected, capacity_mean_mae = _select_lowest_mae(
        training_mae,
        capacity_nearest,
    )
    curve_selected, curve_mean_mae = _select_lowest_mae(
        training_mae,
        curve_nearest,
    )
    individual_best = {
        cell_id: _select_lowest_mae(training_mae, [cell_id])[0]
        for cell_id in curve_nearest
    }
    neighbor_winners = {individual_best[cell_id] for cell_id in curve_nearest}
    if len(neighbor_winners) == 1:
        consensus_selected = next(iter(neighbor_winners))
        consensus_status = "consensus_selected"
        disagreement_fallback = False
    else:
        consensus_selected = "target_prefix_persistence"
        consensus_status = "neighbor_disagreement_fallback"
        disagreement_fallback = True

    target_prefix_for_hash = target.loc[:, REQUIRED_CYCLE_COLUMNS].reset_index(
        drop=True
    )
    target_prefix_hash = canonical_frame_sha256(
        target_prefix_for_hash,
        REQUIRED_CYCLE_COLUMNS,
    )
    gate_specs: dict[str, dict[str, object]] = {
        "nested_loco_capacity_only_mean_gate": {
            "selected": capacity_selected,
            "feature_set": "capacity_only",
            "nearest": capacity_nearest,
            "distances": capacity_distances,
            "mae": capacity_mean_mae,
            "status": "mean_selected",
            "disagreement": False,
        },
        "nested_loco_curve_aware_mean_gate": {
            "selected": curve_selected,
            "feature_set": "capacity_plus_curve",
            "nearest": curve_nearest,
            "distances": curve_distances,
            "mae": curve_mean_mae,
            "status": "mean_selected",
            "disagreement": False,
        },
        "nested_loco_curve_aware_consensus_fallback_gate": {
            "selected": consensus_selected,
            "feature_set": "capacity_plus_curve",
            "nearest": curve_nearest,
            "distances": curve_distances,
            "mae": curve_mean_mae,
            "status": consensus_status,
            "disagreement": disagreement_fallback,
        },
    }

    rows: list[dict[str, object]] = []
    for model_id in MODEL_IDS:
        if model_id in BASE_MODEL_IDS:
            selected_model_id = model_id
            center = target_predictions[model_id]
            reference_cells = training_cell_ids
            feature_set = "not_applicable"
            nearest_text = "not_applicable"
            distance_json = "{}"
            mae_json = _canonical_json_text(
                {
                    cell_id: training_mae[cell_id][model_id]
                    for cell_id in training_cell_ids
                }
            )
            evidence_status = "base_model"
            fallback = False
        else:
            spec = gate_specs[model_id]
            selected_model_id = str(spec["selected"])
            center = target_predictions[selected_model_id]
            reference_cells = list(spec["nearest"])
            feature_set = str(spec["feature_set"])
            nearest_text = ";".join(reference_cells)
            distance_json = _canonical_json_text(
                {
                    cell_id: float(spec["distances"][cell_id])
                    for cell_id in reference_cells
                }
            )
            mae_json = _canonical_json_text(spec["mae"])
            evidence_status = str(spec["status"])
            fallback = bool(spec["disagreement"])
        lower, upper = _band_arrays(
            center,
            training_errors,
            reference_cells,
            selected_model_id=selected_model_id,
            prefix_cycle=prefix_cycle,
            disagreement_fallback=fallback,
            config=config,
        )
        for index, forecast_cycle in enumerate(forecast_cycles.astype(int)):
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_id": DATASET_ID,
                    "held_out_cell_id": held_out_cell_id,
                    "training_cell_ids": training_cell_text,
                    "prefix_cycle": prefix_cycle,
                    "score_end_cycle": SCORE_END_CYCLE,
                    "model_id": model_id,
                    "selected_base_model_id": selected_model_id,
                    "forecast_cycle": int(forecast_cycle),
                    "predicted_capacity_retention_pct": _clip_prediction(
                        float(center[index])
                    ),
                    "evidence_band_lower_pct": _clip_prediction(float(lower[index])),
                    "evidence_band_upper_pct": _clip_prediction(float(upper[index])),
                    "normalization_capacity_ah": normalization,
                    "prefix_row_count": len(target),
                    "target_prefix_sha256": target_prefix_hash,
                    "gate_feature_set": feature_set,
                    "gate_nearest_cell_ids": nearest_text,
                    "gate_neighbor_distances_json": distance_json,
                    "gate_training_mae_json": mae_json,
                    "gate_evidence_status": evidence_status,
                }
            )
    return rows


def predict_nasa_dynamic_gate(
    fold_table: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict from committed outer-fold inputs without target suffix outcomes."""
    parsed = validate_nasa_dynamic_gate_config(config)
    ordered = _validated_fold_table(fold_table, parsed)
    rows: list[dict[str, object]] = []
    for held_out_cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            fold = ordered.loc[
                (ordered["held_out_cell_id"] == held_out_cell_id)
                & (ordered["prefix_cycle"] == prefix_cycle)
            ]
            rows.extend(
                _prediction_rows_for_fold(
                    fold,
                    held_out_cell_id=held_out_cell_id,
                    prefix_cycle=prefix_cycle,
                    config=parsed,
                )
            )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "config_semantic_sha256": canonical_json_sha256(parsed),
        "fold_table_sha256": canonical_frame_sha256(ordered, FOLD_TABLE_COLUMNS),
        "prediction_sha256": canonical_frame_sha256(
            predictions,
            PREDICTION_COLUMNS,
        ),
        "prediction_row_count": len(predictions),
        "held_out_cell_ids": list(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "model_ids": list(MODEL_IDS),
        "score_end_cycle": SCORE_END_CYCLE,
        "target_future_outcomes_used": False,
        "outer_fold_training_histories_used": True,
        "selection_scope": "outer_fold_training_cells_only",
        "evidence_band_scope": "descriptive_not_formal_coverage",
        "inference_scope": "descriptive_development_only_no_significance_test",
    }
    return predictions, manifest


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise NasaDynamicGateError("V2 prediction columns changed")
    result = predictions.copy()
    string_columns = tuple(
        column
        for column in PREDICTION_COLUMNS
        if column
        not in {
            "prefix_cycle",
            "score_end_cycle",
            "forecast_cycle",
            "predicted_capacity_retention_pct",
            "evidence_band_lower_pct",
            "evidence_band_upper_pct",
            "normalization_capacity_ah",
            "prefix_row_count",
        }
    )
    if result.loc[:, string_columns].isna().any().any():
        raise NasaDynamicGateError("V2 prediction strings cannot be null")
    for column in string_columns:
        result[column] = result[column].astype(str)
    numeric_columns = (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "predicted_capacity_retention_pct",
        "evidence_band_lower_pct",
        "evidence_band_upper_pct",
        "normalization_capacity_ah",
        "prefix_row_count",
    )
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaDynamicGateError("V2 predictions must be finite")
    for column in (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    ):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise NasaDynamicGateError(f"Prediction {column} must be integral")
        result[column] = values.astype(np.int64)
    for column in (
        "predicted_capacity_retention_pct",
        "evidence_band_lower_pct",
        "evidence_band_upper_pct",
        "normalization_capacity_ah",
    ):
        result[column] = numeric[column].astype(float)

    if set(result["experiment_id"]) != {EXPERIMENT_ID}:
        raise NasaDynamicGateError("Prediction experiment identity changed")
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise NasaDynamicGateError("Prediction dataset identity changed")
    if set(result["held_out_cell_id"]) != set(CELL_CUTOFFS):
        raise NasaDynamicGateError("Prediction held-out cells changed")
    if set(result["prefix_cycle"].astype(int)) != set(PREFIX_CYCLES):
        raise NasaDynamicGateError("Prediction prefixes changed")
    if set(result["model_id"]) != set(MODEL_IDS):
        raise NasaDynamicGateError("Prediction model registry changed")
    if not set(result["selected_base_model_id"]).issubset(set(BASE_MODEL_IDS)):
        raise NasaDynamicGateError("Prediction selected an unknown base model")
    expected_rows = (
        len(CELL_CUTOFFS)
        * len(MODEL_IDS)
        * sum(SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES)
    )
    if len(result) != expected_rows:
        raise NasaDynamicGateError("Prediction artifact cardinality changed")
    bounded_columns = (
        "predicted_capacity_retention_pct",
        "evidence_band_lower_pct",
        "evidence_band_upper_pct",
    )
    if not all(result[column].between(0.0, 110.0).all() for column in bounded_columns):
        raise NasaDynamicGateError("Prediction or evidence band exceeds bounds")
    if not (
        (
            result["evidence_band_lower_pct"]
            <= result["predicted_capacity_retention_pct"]
        )
        & (
            result["predicted_capacity_retention_pct"]
            <= result["evidence_band_upper_pct"]
        )
    ).all():
        raise NasaDynamicGateError("Evidence band does not contain its prediction")
    if (result["normalization_capacity_ah"] <= 0.0).any():
        raise NasaDynamicGateError("Prediction normalization must be positive")
    if result.duplicated(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"]
    ).any():
        raise NasaDynamicGateError("Prediction coordinates are duplicated")
    valid_hash = result["target_prefix_sha256"].map(
        lambda value: len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
    if not valid_hash.all():
        raise NasaDynamicGateError("Prediction target-prefix hash is invalid")
    for json_column in (
        "gate_neighbor_distances_json",
        "gate_training_mae_json",
    ):
        for value in result[json_column].unique():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise NasaDynamicGateError(
                    f"Prediction {json_column} is not valid JSON"
                ) from exc
            if _canonical_json_text(decoded) != value:
                raise NasaDynamicGateError(
                    f"Prediction {json_column} is not canonical JSON"
                )

    for held_out_cell_id in CELL_CUTOFFS:
        expected_training = ";".join(
            cell_id for cell_id in CELL_CUTOFFS if cell_id != held_out_cell_id
        )
        for prefix_cycle in PREFIX_CYCLES:
            for model_id in MODEL_IDS:
                group = result.loc[
                    (result["held_out_cell_id"] == held_out_cell_id)
                    & (result["prefix_cycle"] == prefix_cycle)
                    & (result["model_id"] == model_id)
                ].sort_values("forecast_cycle", kind="stable")
                if group["forecast_cycle"].astype(int).tolist() != list(
                    range(prefix_cycle + 1, SCORE_END_CYCLE + 1)
                ):
                    raise NasaDynamicGateError("Prediction forecast support changed")
                invariant_columns = (
                    "training_cell_ids",
                    "score_end_cycle",
                    "selected_base_model_id",
                    "normalization_capacity_ah",
                    "prefix_row_count",
                    "target_prefix_sha256",
                    "gate_feature_set",
                    "gate_nearest_cell_ids",
                    "gate_neighbor_distances_json",
                    "gate_training_mae_json",
                    "gate_evidence_status",
                )
                if any(
                    group[column].nunique(dropna=False) != 1
                    for column in invariant_columns
                ):
                    raise NasaDynamicGateError(
                        "Prediction fold metadata changes within a trajectory"
                    )
                if set(group["training_cell_ids"]) != {expected_training}:
                    raise NasaDynamicGateError("LOCO training-cell identities changed")
                if set(group["score_end_cycle"].astype(int)) != {SCORE_END_CYCLE}:
                    raise NasaDynamicGateError("Prediction endpoint changed")
                if set(group["prefix_row_count"].astype(int)) != {prefix_cycle}:
                    raise NasaDynamicGateError("Prediction prefix count changed")
                if model_id in BASE_MODEL_IDS:
                    if set(group["selected_base_model_id"]) != {model_id}:
                        raise NasaDynamicGateError(
                            "Base model selection metadata changed"
                        )
                    if set(group["gate_feature_set"]) != {"not_applicable"}:
                        raise NasaDynamicGateError("Base model claims gate features")
                else:
                    nearest = str(group["gate_nearest_cell_ids"].iloc[0]).split(";")
                    if len(nearest) != 2 or not set(nearest).issubset(
                        set(expected_training.split(";"))
                    ):
                        raise NasaDynamicGateError("Gate nearest-cell metadata changed")
    return result.sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def _validate_prediction_manifest(
    manifest: Mapping[str, object],
    *,
    config: Mapping[str, object],
    fold_table: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    expected_keys = {
        "schema_version",
        "experiment_id",
        "dataset_id",
        "evidence_role",
        "config_semantic_sha256",
        "fold_table_sha256",
        "prediction_sha256",
        "prediction_row_count",
        "held_out_cell_ids",
        "prefix_cycles",
        "model_ids",
        "score_end_cycle",
        "target_future_outcomes_used",
        "outer_fold_training_histories_used",
        "selection_scope",
        "evidence_band_scope",
        "inference_scope",
    }
    if set(manifest) != expected_keys:
        raise NasaDynamicGateError("Prediction manifest keys changed")
    expected_values: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "config_semantic_sha256": canonical_json_sha256(config),
        "fold_table_sha256": canonical_frame_sha256(
            fold_table,
            FOLD_TABLE_COLUMNS,
        ),
        "prediction_sha256": canonical_frame_sha256(
            predictions,
            PREDICTION_COLUMNS,
        ),
        "prediction_row_count": len(predictions),
        "held_out_cell_ids": list(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "model_ids": list(MODEL_IDS),
        "score_end_cycle": SCORE_END_CYCLE,
        "target_future_outcomes_used": False,
        "outer_fold_training_histories_used": True,
        "selection_scope": "outer_fold_training_cells_only",
        "evidence_band_scope": "descriptive_not_formal_coverage",
        "inference_scope": "descriptive_development_only_no_significance_test",
    }
    for key, expected in expected_values.items():
        if manifest[key] != expected:
            raise NasaDynamicGateError(f"Prediction manifest mismatch for {key}")


def score_nasa_dynamic_gate(
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score committed V2 predictions against held-out suffix outcomes."""
    parsed = validate_nasa_dynamic_gate_config(config)
    ordered_cycles = _validated_cycles(cycles, parsed)
    fold_table = build_nasa_dynamic_gate_fold_table(ordered_cycles, parsed)
    ordered_fold_table = _validated_fold_table(fold_table, parsed)
    ordered_predictions = _validated_predictions(predictions)
    _validate_prediction_manifest(
        prediction_manifest,
        config=parsed,
        fold_table=ordered_fold_table,
        predictions=ordered_predictions,
    )

    truth_frames: list[pd.DataFrame] = []
    for cell_id in CELL_CUTOFFS:
        cell = ordered_cycles.loc[
            (ordered_cycles["cell_id"] == cell_id)
            & (ordered_cycles["cycle_index"] <= SCORE_END_CYCLE)
        ].sort_values("cycle_index", kind="stable")
        normalization = _normalization_capacity(cell)
        truth_frames.append(
            pd.DataFrame(
                {
                    "held_out_cell_id": cell_id,
                    "forecast_cycle": cell["cycle_index"].astype(int),
                    "observed_capacity_retention_pct": _retention_pct(
                        cell,
                        normalization,
                    ),
                }
            )
        )
    truth_table = pd.concat(truth_frames, ignore_index=True)
    linked = ordered_predictions.merge(
        truth_table,
        on=["held_out_cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise NasaDynamicGateError("Prediction coordinates could not link to truth")

    score_rows: list[dict[str, object]] = []
    group_columns = ["held_out_cell_id", "prefix_cycle", "model_id"]
    for (cell_id, prefix_cycle, model_id), group in linked.groupby(
        group_columns,
        sort=True,
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        forecast_cycle = group["forecast_cycle"].to_numpy(dtype=float)
        observed = group["observed_capacity_retention_pct"].to_numpy(dtype=float)
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        lower = group["evidence_band_lower_pct"].to_numpy(dtype=float)
        upper = group["evidence_band_upper_pct"].to_numpy(dtype=float)
        error = predicted - observed
        absolute_error = np.abs(error)
        horizon = float(forecast_cycle[-1] - forecast_cycle[0])
        iae = (
            float(np.trapezoid(absolute_error, forecast_cycle) / horizon)
            if horizon > 0.0
            else float(absolute_error[0])
        )
        inside = (observed >= lower) & (observed <= upper)
        score_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "held_out_cell_id": str(cell_id),
                "training_cell_ids": str(group["training_cell_ids"].iloc[0]),
                "prefix_cycle": int(prefix_cycle),
                "score_end_cycle": SCORE_END_CYCLE,
                "model_id": str(model_id),
                "selected_base_model_id": str(group["selected_base_model_id"].iloc[0]),
                "gate_evidence_status": str(group["gate_evidence_status"].iloc[0]),
                "future_observation_count": len(group),
                "trajectory_iae_pp_normalized_by_cycle_horizon": iae,
                "trajectory_mae_pp": float(np.mean(absolute_error)),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(absolute_error[-1]),
                "empirical_evidence_band_coverage_fraction": float(np.mean(inside)),
                "mean_evidence_band_width_pp": float(np.mean(upper - lower)),
                "endpoint_inside_evidence_band": float(inside[-1]),
            }
        )
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    metric_columns = list(SCORE_COLUMNS[-7:])
    aggregate = (
        scores.groupby(["prefix_cycle", "model_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    aggregate_records = json.loads(
        _canonical_json_text(aggregate.to_dict(orient="records"))
    )
    selection_counts = (
        scores.loc[scores["model_id"].isin(GATED_MODEL_IDS)]
        .groupby(
            [
                "prefix_cycle",
                "model_id",
                "selected_base_model_id",
                "gate_evidence_status",
            ],
            sort=True,
        )
        .size()
        .rename("fold_count")
        .reset_index()
    )
    selection_records = json.loads(
        _canonical_json_text(selection_counts.to_dict(orient="records"))
    )
    summary: dict[str, object] = {
        "schema_version": "lifetwin.nasa_dynamic_gate.score_summary.v2",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "development_status": (
            "post_v1_development_not_independent_outcome_blind_confirmation"
        ),
        "prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "score_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "fold_count": len(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "primary_prefix_cycle": PRIMARY_PREFIX_CYCLE,
        "score_end_cycle": SCORE_END_CYCLE,
        "base_model_ids": list(BASE_MODEL_IDS),
        "gated_model_ids": list(GATED_MODEL_IDS),
        "cell_weighting": "equal_weight_per_held_out_cell",
        "evidence_band_scope": "descriptive_not_formal_coverage",
        "inference_scope": "descriptive_development_only_no_significance_test",
        "aggregate_metrics": aggregate_records,
        "gate_selection_counts": selection_records,
        "allowed_claims": list(_ALLOWED_CLAIMS),
        "prohibited_claims": list(_PROHIBITED_CLAIMS),
    }
    return scores, summary


__all__ = [
    "BASE_MODEL_IDS",
    "CAPACITY_FEATURE_IDS",
    "CONFIG_SEMANTIC_SHA256",
    "CURVE_FEATURE_IDS",
    "EXPERIMENT_ID",
    "FOLD_TABLE_COLUMNS",
    "GATED_MODEL_IDS",
    "MODEL_IDS",
    "NasaDynamicGateError",
    "PREDICTION_COLUMNS",
    "REQUIRED_CYCLE_COLUMNS",
    "SCORE_COLUMNS",
    "SIMILARITY_FEATURE_IDS",
    "build_nasa_dynamic_gate_fold_table",
    "load_nasa_dynamic_gate_config",
    "predict_nasa_dynamic_gate",
    "score_nasa_dynamic_gate",
    "validate_nasa_dynamic_gate_config",
]
