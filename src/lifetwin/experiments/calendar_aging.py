from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd

from lifetwin.data.naumann import validate_naumann_calendar_observations
from lifetwin.models.calendar import (
    CALENDAR_MODEL_NAME,
    CALENDAR_NULL_MODEL_NAME,
    EmpiricalStressSurface,
    ConditionAgnosticSqrtModel,
    estimate_empirical_bayes_ridge,
    estimate_prefix_only_k,
    estimate_target_scale,
    fit_condition_agnostic_sqrt_model,
    fit_empirical_stress_surface,
    predict_condition_agnostic_loss,
    predict_stress_surface_loss,
    select_prefix,
)


PREFIX_UPDATED_MODEL_NAME = "empirical_stress_surface_prefix_update_v1"
PREFIX_ONLY_MODEL_NAME = "target_prefix_only_sqrt_time_v1"
PERSISTENCE_MODEL_NAME = "last_prefix_observation_v1"
FULL_HISTORY_POLICY = "full_observed_history_upper_bound"
GLOBAL_LANDMARK_POLICY = "global_landmark_prefix"
ALLOWED_HISTORY_POLICIES = {FULL_HISTORY_POLICY, GLOBAL_LANDMARK_POLICY}
EXPECTED_TIME_UNIT = "hour"
EXPECTED_TIME_LAW = "fixed_square_root"
EXPECTED_ADAPTATION_METHOD = "training_only_empirical_bayes_scale_update"
MODEL_NAMES = (
    CALENDAR_NULL_MODEL_NAME,
    CALENDAR_MODEL_NAME,
    PREFIX_UPDATED_MODEL_NAME,
    PREFIX_ONLY_MODEL_NAME,
    PERSISTENCE_MODEL_NAME,
)

PREDICTION_KEY_COLUMNS = [
    "scenario",
    "training_history_policy",
    "fold_id",
    "target_condition_id",
    "prefix_checkups",
    "method",
    "target_checkup_index",
]


def _validate_backtest_configuration(
    *,
    scenarios: list[dict[str, object]],
    model_parameters: dict[str, object],
    adaptation_parameters: dict[str, object],
    gate_scenarios: list[str],
) -> dict[str, object]:
    required_model = {
        "name",
        "time_unit",
        "time_law",
        "minimum_training_conditions",
        "robust_loss_scale_pp",
        "probability_calibration_minimum_groups",
    }
    missing_model = sorted(required_model - set(model_parameters))
    unknown_model = sorted(set(model_parameters) - required_model)
    if missing_model or unknown_model:
        raise ValueError(
            "Calendar model configuration keys do not match the implemented model: "
            f"missing={missing_model}, unknown={unknown_model}"
        )
    if str(model_parameters["name"]) != CALENDAR_MODEL_NAME:
        raise ValueError(f"Unsupported calendar model name: {model_parameters['name']}")
    if str(model_parameters["time_unit"]) != EXPECTED_TIME_UNIT:
        raise ValueError(f"Unsupported calendar time unit: {model_parameters['time_unit']}")
    if str(model_parameters["time_law"]) != EXPECTED_TIME_LAW:
        raise ValueError(f"Unsupported calendar time law: {model_parameters['time_law']}")
    if int(model_parameters["minimum_training_conditions"]) < 2:
        raise ValueError("minimum_training_conditions must be at least two")
    if float(model_parameters["robust_loss_scale_pp"]) <= 0.0:
        raise ValueError("robust_loss_scale_pp must be positive")
    calibration_minimum = int(
        model_parameters["probability_calibration_minimum_groups"]
    )
    if calibration_minimum < 2:
        raise ValueError("probability_calibration_minimum_groups must be at least two")

    required_adaptation = {
        "method",
        "ridge_min",
        "ridge_max",
        "scale_min",
        "scale_max",
    }
    missing_adaptation = sorted(required_adaptation - set(adaptation_parameters))
    unknown_adaptation = sorted(set(adaptation_parameters) - required_adaptation)
    if missing_adaptation or unknown_adaptation:
        raise ValueError(
            "Calendar adaptation configuration keys do not match the implemented method: "
            f"missing={missing_adaptation}, unknown={unknown_adaptation}"
        )
    if str(adaptation_parameters["method"]) != EXPECTED_ADAPTATION_METHOD:
        raise ValueError(
            "Unsupported calendar adaptation method: "
            f"{adaptation_parameters['method']}"
        )
    ridge_min = float(adaptation_parameters["ridge_min"])
    ridge_max = float(adaptation_parameters["ridge_max"])
    scale_min = float(adaptation_parameters["scale_min"])
    scale_max = float(adaptation_parameters["scale_max"])
    if not (0.0 <= ridge_min <= ridge_max):
        raise ValueError("Adaptation ridge bounds must satisfy 0 <= min <= max")
    if not (0.0 <= scale_min < scale_max):
        raise ValueError("Adaptation scale bounds must satisfy 0 <= min < max")

    if not scenarios:
        raise ValueError("Calendar scenarios cannot be empty")
    scenario_names = [str(scenario.get("name", "")) for scenario in scenarios]
    if any(not name for name in scenario_names):
        raise ValueError("Every calendar scenario requires a non-empty name")
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("Calendar scenario names must be unique")
    scenario_policies: dict[str, str] = {}
    for scenario in scenarios:
        name = str(scenario["name"])
        policy = str(scenario.get("training_history_policy", ""))
        if policy not in ALLOWED_HISTORY_POLICIES:
            raise ValueError(
                f"Scenario {name} has unsupported training_history_policy: {policy}"
            )
        scenario_policies[name] = policy

    if not gate_scenarios:
        raise ValueError("gate_scenarios must contain at least one executed scenario")
    if any(not value for value in gate_scenarios):
        raise ValueError("Gate scenario names cannot be empty")
    if len(set(gate_scenarios)) != len(gate_scenarios):
        raise ValueError("Gate scenario names must be unique")
    unknown_gates = sorted(set(gate_scenarios) - set(scenario_names))
    if unknown_gates:
        raise ValueError(f"Gate scenarios were not configured: {unknown_gates}")
    non_landmark_gates = sorted(
        name
        for name in gate_scenarios
        if scenario_policies[name] != GLOBAL_LANDMARK_POLICY
    )
    if non_landmark_gates:
        raise ValueError(
            "Scientific gates must use the time-honest global-landmark design; "
            f"non-landmark gates={non_landmark_gates}"
        )
    return {
        "probability_calibration_minimum_groups": calibration_minimum,
        "scenario_policies": scenario_policies,
    }


def calendar_prediction_artifact_sha256(predictions: pd.DataFrame) -> str:
    """Hash the canonical label-free prediction pack."""
    missing = sorted(set(PREDICTION_KEY_COLUMNS) - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing calendar prediction hash keys: {missing}")
    forbidden = {
        "capacity_ah",
        "capacity_loss_pct",
        "capacity_retention_pct",
        "true_capacity_retention_pct",
        "prediction_error_pp",
    }
    leaked = sorted(forbidden & set(predictions.columns))
    if leaked:
        raise ValueError(f"Label-free calendar predictions contain outcomes: {leaked}")
    if predictions[PREDICTION_KEY_COLUMNS].isna().any().any():
        raise ValueError("Calendar prediction keys cannot be null")
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Calendar prediction keys must be unique")
    normalized = predictions.sort_values(PREDICTION_KEY_COLUMNS, kind="stable")
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scenario_folds(
    observations: pd.DataFrame,
    scenario: dict[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    name = str(scenario["name"])
    kind = str(scenario["kind"])
    condition_profile = (
        observations[
            ["condition_id", "temperature_c", "storage_soc_fraction"]
        ]
        .drop_duplicates()
        .sort_values("condition_id", kind="stable")
    )
    if kind == "leave_one_condition_out":
        return [
            (f"condition={condition_id}", (str(condition_id),))
            for condition_id in condition_profile["condition_id"]
        ]
    if kind == "leave_one_temperature_level_out":
        folds: list[tuple[str, tuple[str, ...]]] = []
        for temperature, rows in condition_profile.groupby("temperature_c", sort=True):
            folds.append(
                (
                    f"temperature_c={float(temperature):g}",
                    tuple(sorted(rows["condition_id"].astype(str))),
                )
            )
        return folds
    if kind == "fixed_condition_holdout":
        target_ids = tuple(sorted(str(value) for value in scenario["target_condition_ids"]))
        known = set(condition_profile["condition_id"].astype(str))
        missing = sorted(set(target_ids) - known)
        if missing:
            raise ValueError(f"Scenario {name} references unknown conditions: {missing}")
        return [(str(scenario.get("fold_id", name)), target_ids)]
    raise ValueError(f"Unsupported calendar scenario kind: {kind}")


def _prediction_state_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _support_classification(
    training: pd.DataFrame,
    target: pd.DataFrame,
) -> dict[str, object]:
    target_temperature = float(target["temperature_c"].iloc[0])
    target_soc = float(target["storage_soc_fraction"].iloc[0])
    temperatures = training["temperature_c"].drop_duplicates().to_numpy(dtype=float)
    socs = training["storage_soc_fraction"].drop_duplicates().to_numpy(dtype=float)
    temperature_in_range = bool(
        temperatures.min() <= target_temperature <= temperatures.max()
    )
    soc_in_range = bool(socs.min() <= target_soc <= socs.max())
    if temperature_in_range and soc_in_range:
        support_class = "stress_covariate_interpolation"
    elif temperature_in_range:
        support_class = "soc_range_extrapolation"
    elif soc_in_range:
        support_class = "temperature_range_extrapolation"
    else:
        support_class = "temperature_and_soc_range_extrapolation"
    return {
        "temperature_in_training_range": temperature_in_range,
        "soc_in_training_range": soc_in_range,
        "temperature_level_seen": bool(np.isclose(temperatures, target_temperature).any()),
        "soc_level_seen": bool(np.isclose(socs, target_soc).any()),
        "support_class": support_class,
    }


def build_calendar_target_prediction_state(
    *,
    scenario_name: str,
    fold_id: str,
    target_frame: pd.DataFrame,
    stress_model: EmpiricalStressSurface,
    null_model: ConditionAgnosticSqrtModel,
    ridge_state: dict[str, object],
    prefix_checkups: int,
    scale_bounds: tuple[float, float],
    training_history_policy: str = FULL_HISTORY_POLICY,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict one held-out condition while reading target outcomes only in its prefix."""
    target_ids = target_frame["condition_id"].astype(str).unique()
    if len(target_ids) != 1:
        raise ValueError("A target prediction state must contain exactly one condition")
    target_id = str(target_ids[0])
    if training_history_policy not in ALLOWED_HISTORY_POLICIES:
        raise ValueError(
            f"Unsupported training_history_policy: {training_history_policy}"
        )
    target = target_frame.sort_values("checkup_index", kind="stable").copy()
    if target["checkup_index"].duplicated().any():
        raise ValueError("Target condition contains duplicate checkup indices")
    if len(target) <= prefix_checkups:
        raise ValueError("Target condition has no future checkups after the prefix")

    prefix = select_prefix(target, prefix_checkups)
    future_coordinates = target.loc[
        pd.to_numeric(target["checkup_index"]) >= prefix_checkups,
        [
            "condition_id",
            "temperature_c",
            "storage_soc_fraction",
            "elapsed_hours",
            "elapsed_days",
            "checkup_index",
        ],
    ].copy()
    target_scale = estimate_target_scale(
        stress_model,
        prefix,
        ridge=float(ridge_state["ridge"]),
        scale_bounds=scale_bounds,
    )
    prefix_only_k = estimate_prefix_only_k(prefix)
    base_loss = predict_stress_surface_loss(stress_model, future_coordinates)
    null_loss = predict_condition_agnostic_loss(
        null_model,
        future_coordinates["elapsed_hours"].to_numpy(dtype=float),
    )
    prefix_only_loss = prefix_only_k * np.sqrt(
        future_coordinates["elapsed_hours"].to_numpy(dtype=float)
    )
    last_prefix_retention = float(
        prefix.sort_values("checkup_index", kind="stable")[
            "capacity_retention_pct"
        ].iloc[-1]
    )
    predictions_by_method = {
        CALENDAR_NULL_MODEL_NAME: 100.0 - null_loss,
        CALENDAR_MODEL_NAME: 100.0 - base_loss,
        PREFIX_UPDATED_MODEL_NAME: 100.0 - target_scale * base_loss,
        PREFIX_ONLY_MODEL_NAME: 100.0 - prefix_only_loss,
        PERSISTENCE_MODEL_NAME: np.full(len(future_coordinates), last_prefix_retention),
    }

    payload = {
        "scenario": scenario_name,
        "training_history_policy": training_history_policy,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(prefix_checkups),
        "prefix_observations": [
            {
                "checkup_index": int(row.checkup_index),
                "elapsed_hours": float(row.elapsed_hours),
                "capacity_retention_pct": float(row.capacity_retention_pct),
            }
            for row in prefix.itertuples(index=False)
        ],
        "future_coordinates": [
            {
                "checkup_index": int(row.checkup_index),
                "elapsed_hours": float(row.elapsed_hours),
            }
            for row in future_coordinates.itertuples(index=False)
        ],
        "stress_model_parameters": list(stress_model.parameters),
        "stress_model_training_conditions": list(stress_model.training_condition_ids),
        "stress_model_training_support_hours": float(
            stress_model.maximum_training_hours
        ),
        "stress_model_prediction_support_hours": float(
            stress_model.maximum_supported_hours
        ),
        "null_model_k": float(null_model.k_pct_per_sqrt_hour),
        "null_model_training_support_hours": float(null_model.maximum_training_hours),
        "null_model_prediction_support_hours": float(null_model.maximum_supported_hours),
        "ridge": float(ridge_state["ridge"]),
        "target_scale": float(target_scale),
        "prefix_only_k": float(prefix_only_k),
        "predictions": {
            method: [float(value) for value in values]
            for method, values in predictions_by_method.items()
        },
    }
    state_hash = _prediction_state_sha256(payload)
    prefix_end = prefix.sort_values("checkup_index", kind="stable").iloc[-1]
    prediction_rows: list[dict[str, object]] = []
    for method, values in predictions_by_method.items():
        for coordinate, predicted in zip(
            future_coordinates.itertuples(index=False),
            values,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "scenario": scenario_name,
                    "training_history_policy": training_history_policy,
                    "fold_id": fold_id,
                    "target_condition_id": target_id,
                    "prefix_checkups": int(prefix_checkups),
                    "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
                    "prefix_end_days": float(prefix_end["elapsed_days"]),
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "elapsed_hours": float(coordinate.elapsed_hours),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(predicted),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == target["checkup_index"].max()
                    ),
                    "prediction_state_sha256": state_hash,
                }
            )
    diagnostics = {
        "target_condition_id": target_id,
        "training_history_policy": training_history_policy,
        "prefix_checkups": int(prefix_checkups),
        "prefix_end_days": float(prefix_end["elapsed_days"]),
        "future_checkup_count": len(future_coordinates),
        "target_scale": float(target_scale),
        "prefix_only_k_pct_per_sqrt_hour": float(prefix_only_k),
        "ridge": float(ridge_state["ridge"]),
        "prediction_state_sha256": state_hash,
    }
    return pd.DataFrame(prediction_rows), diagnostics


def score_calendar_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify a label-free pack, then join future outcomes and score per condition."""
    observed_hash = calendar_prediction_artifact_sha256(predictions)
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Frozen calendar prediction hash does not match prediction content")
    outcomes = observations[
        [
            "condition_id",
            "checkup_index",
            "capacity_retention_pct",
        ]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    if outcomes.duplicated(["target_condition_id", "target_checkup_index"]).any():
        raise ValueError("Calendar scoring outcomes must be unique by condition/checkup")
    scored = predictions.merge(
        outcomes,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (scored["_merge"] != "both").any() or scored[
        "true_capacity_retention_pct"
    ].isna().any():
        raise ValueError("Every calendar prediction must match one observed outcome")
    scored = scored.drop(columns="_merge")
    scored["prediction_error_pp"] = (
        scored["predicted_capacity_retention_pct"]
        - scored["true_capacity_retention_pct"]
    )

    condition_rows: list[dict[str, object]] = []
    grouping = [
        "scenario",
        "training_history_policy",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_days",
        "method",
    ]
    for keys, rows in scored.groupby(grouping, sort=True):
        ordered = rows.sort_values("elapsed_hours", kind="stable")
        time = ordered["elapsed_hours"].to_numpy(dtype=float)
        absolute_error = np.abs(ordered["prediction_error_pp"].to_numpy(dtype=float))
        if len(time) < 2 or time[-1] <= time[0]:
            raise ValueError("At least two future checkups are required for trajectory scoring")
        trajectory_iae = float(np.trapezoid(absolute_error, time) / (time[-1] - time[0]))
        final = ordered.loc[ordered["is_final_checkup"]]
        if len(final) != 1:
            raise ValueError("Each condition prediction must contain one final checkup")
        final_row = final.iloc[0]
        condition_rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "future_checkup_count": len(ordered),
                "trajectory_iae_pp": trajectory_iae,
                "future_point_mae_pp": float(absolute_error.mean()),
                "final_true_retention_pct": float(
                    final_row["true_capacity_retention_pct"]
                ),
                "final_predicted_retention_pct": float(
                    final_row["predicted_capacity_retention_pct"]
                ),
                "final_error_pp": float(final_row["prediction_error_pp"]),
                "final_absolute_error_pp": abs(float(final_row["prediction_error_pp"])),
            }
        )
    condition_metrics = pd.DataFrame(condition_rows)

    aggregate_rows: list[dict[str, object]] = []
    for keys, rows in condition_metrics.groupby(
        ["scenario", "training_history_policy", "prefix_checkups", "method"],
        sort=True,
    ):
        final_error = rows["final_error_pp"].to_numpy(dtype=float)
        aggregate_rows.append(
            {
                "scenario": keys[0],
                "training_history_policy": keys[1],
                "prefix_checkups": int(keys[2]),
                "method": keys[3],
                "independent_condition_count": len(rows),
                "trajectory_iae_pp_mean": float(rows["trajectory_iae_pp"].mean()),
                "trajectory_iae_pp_median": float(rows["trajectory_iae_pp"].median()),
                "final_mae_pp": float(np.mean(np.abs(final_error))),
                "final_rmse_pp": float(np.sqrt(np.mean(np.square(final_error)))),
                "final_bias_pp": float(np.mean(final_error)),
                "maximum_final_absolute_error_pp": float(np.max(np.abs(final_error))),
            }
        )
    return condition_metrics, pd.DataFrame(aggregate_rows)


def _metrics_lookup(
    metrics: pd.DataFrame,
    *,
    scenario: str,
    prefix_checkups: int,
    method: str,
) -> pd.Series:
    selected = metrics.loc[
        (metrics["scenario"] == scenario)
        & (metrics["prefix_checkups"] == prefix_checkups)
        & (metrics["method"] == method)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one metric row for {scenario}/{prefix_checkups}/{method}"
        )
    return selected.iloc[0]


def _scientific_gates(
    metrics: pd.DataFrame,
    condition_metrics: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    thresholds = dict(config["validation_thresholds"])
    primary_prefix = int(config["primary_prefix_checkups"])
    gate_scenarios = [str(value) for value in config["gate_scenarios"]]
    if not gate_scenarios:
        raise ValueError("gate_scenarios must be non-empty")
    if len(set(gate_scenarios)) != len(gate_scenarios):
        raise ValueError("gate_scenarios must be unique")
    executed_scenarios = set(metrics["scenario"].astype(str))
    missing_scenarios = sorted(set(gate_scenarios) - executed_scenarios)
    if missing_scenarios:
        raise ValueError(f"Gate scenarios were not executed: {missing_scenarios}")
    minimum_improvement = float(
        thresholds["minimum_stress_surface_improvement_vs_null_fraction"]
    )
    maximum_error = float(thresholds["maximum_any_condition_final_error_pp"])
    scenario_results: list[dict[str, object]] = []
    for scenario in gate_scenarios:
        stress = _metrics_lookup(
            metrics,
            scenario=scenario,
            prefix_checkups=primary_prefix,
            method=CALENDAR_MODEL_NAME,
        )
        null = _metrics_lookup(
            metrics,
            scenario=scenario,
            prefix_checkups=primary_prefix,
            method=CALENDAR_NULL_MODEL_NAME,
        )
        null_error = float(null["trajectory_iae_pp_mean"])
        if not np.isfinite(null_error) or null_error <= 0.0:
            raise ValueError(f"Gate baseline error must be positive for {scenario}")
        improvement = 1.0 - float(stress["trajectory_iae_pp_mean"]) / null_error
        if not np.isfinite(improvement):
            raise ValueError(f"Gate improvement is not finite for scenario {scenario}")
        error_rows = condition_metrics.loc[
            (condition_metrics["scenario"] == scenario)
            & (condition_metrics["prefix_checkups"] == primary_prefix)
            & (condition_metrics["method"] == CALENDAR_MODEL_NAME)
        ]
        max_error = float(error_rows["final_absolute_error_pp"].max())
        passed = improvement >= minimum_improvement and max_error <= maximum_error
        scenario_results.append(
            {
                "scenario": scenario,
                "stress_surface_trajectory_iae_pp": float(
                    stress["trajectory_iae_pp_mean"]
                ),
                "condition_agnostic_trajectory_iae_pp": float(
                    null["trajectory_iae_pp_mean"]
                ),
                "relative_improvement_fraction": improvement,
                "minimum_required_improvement_fraction": minimum_improvement,
                "maximum_final_absolute_error_pp": max_error,
                "maximum_allowed_final_absolute_error_pp": maximum_error,
                "passed": bool(passed),
            }
        )
    return {
        "status": (
            "passed" if all(item["passed"] for item in scenario_results) else "failed"
        ),
        "primary_prefix_checkups": primary_prefix,
        "all_configured_gates_executed": True,
        "executed_gate_scenarios": gate_scenarios,
        "scenario_results": scenario_results,
    }


def run_calendar_aging_backtest(
    observations: pd.DataFrame,
    *,
    scenarios: Iterable[dict[str, object]],
    prefix_checkups: Iterable[int],
    primary_prefix_checkups: int,
    model_parameters: dict[str, object],
    adaptation_parameters: dict[str, object],
    validation_thresholds: dict[str, object],
    gate_scenarios: Iterable[str],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_naumann_calendar_observations(observations)
    raw_prefixes = [int(value) for value in prefix_checkups]
    if len(raw_prefixes) != len(set(raw_prefixes)):
        raise ValueError("Calendar backtest prefixes must be unique")
    prefixes = sorted(raw_prefixes)
    if not prefixes or prefixes[0] < 3:
        raise ValueError("Calendar backtest prefixes must contain at least three checkups")
    if primary_prefix_checkups not in prefixes:
        raise ValueError("Primary prefix must be included in prefix_checkups")
    if max(prefixes) >= int(observations.groupby("condition_id").size().min()):
        raise ValueError("Every prefix must leave future checkups for scoring")

    scenario_configs = [dict(value) for value in scenarios]
    configured_gates = [str(value) for value in gate_scenarios]
    validated_config = _validate_backtest_configuration(
        scenarios=scenario_configs,
        model_parameters=dict(model_parameters),
        adaptation_parameters=dict(adaptation_parameters),
        gate_scenarios=configured_gates,
    )

    all_conditions = set(observations["condition_id"].astype(str))
    maximum_public_hours = float(observations["elapsed_hours"].max())
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for scenario in scenario_configs:
        scenario_name = str(scenario["name"])
        history_policy = str(scenario["training_history_policy"])
        folds = _scenario_folds(observations, scenario)
        for fold_id, target_ids in folds:
            target_set = set(target_ids)
            training_ids = all_conditions - target_set
            if target_set & training_ids or target_set | training_ids != all_conditions:
                raise RuntimeError("Calendar fold does not partition complete conditions")
            training_full = observations.loc[
                observations["condition_id"].astype(str).isin(training_ids)
            ].copy()
            target_fold = observations.loc[
                observations["condition_id"].astype(str).isin(target_set)
            ].copy()
            if training_full["condition_id"].nunique() < int(
                model_parameters["minimum_training_conditions"]
            ):
                raise ValueError(f"Fold {scenario_name}/{fold_id} has too few conditions")
            split_rows.extend(
                {
                    "scenario": scenario_name,
                    "fold_id": fold_id,
                    "condition_id": condition_id,
                    "split": "test" if condition_id in target_set else "train",
                    "training_history_policy": history_policy,
                }
                for condition_id in sorted(all_conditions)
            )
            model_cache: dict[
                int | None,
                tuple[EmpiricalStressSurface, ConditionAgnosticSqrtModel, pd.DataFrame],
            ] = {}
            for prefix in prefixes:
                cache_key = prefix if history_policy == GLOBAL_LANDMARK_POLICY else None
                if cache_key not in model_cache:
                    training_fit = (
                        select_prefix(training_full, prefix)
                        if history_policy == GLOBAL_LANDMARK_POLICY
                        else training_full.copy()
                    )
                    prediction_limit = (
                        maximum_public_hours
                        if history_policy == GLOBAL_LANDMARK_POLICY
                        else None
                    )
                    stress_model = fit_empirical_stress_surface(
                        training_fit,
                        minimum_conditions=int(
                            model_parameters["minimum_training_conditions"]
                        ),
                        robust_loss_scale_pp=float(
                            model_parameters["robust_loss_scale_pp"]
                        ),
                        maximum_prediction_hours=prediction_limit,
                    )
                    null_model = fit_condition_agnostic_sqrt_model(
                        training_fit,
                        maximum_prediction_hours=prediction_limit,
                    )
                    model_cache[cache_key] = (stress_model, null_model, training_fit)
                stress_model, null_model, training_fit = model_cache[cache_key]
                training_support_hours = float(training_fit["elapsed_hours"].max())
                training_max_checkup_index = int(
                    pd.to_numeric(training_fit["checkup_index"]).max()
                )
                training_landmark_checkups = (
                    int(prefix) if history_policy == GLOBAL_LANDMARK_POLICY else None
                )
                if not np.isclose(
                    training_support_hours,
                    stress_model.maximum_training_hours,
                ):
                    raise RuntimeError("Calendar model training support was not preserved")
                ridge_state = estimate_empirical_bayes_ridge(
                    stress_model,
                    training_fit,
                    prefix_checkups=prefix,
                    ridge_bounds=(
                        float(adaptation_parameters["ridge_min"]),
                        float(adaptation_parameters["ridge_max"]),
                    ),
                )
                for parameter_name, parameter_value in stress_model.parameter_map().items():
                    parameter_rows.append(
                        {
                            "scenario": scenario_name,
                            "fold_id": fold_id,
                            "prefix_checkups": int(prefix),
                            "training_history_policy": history_policy,
                            "parameter": parameter_name,
                            "value": parameter_value,
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_fit),
                            "training_landmark_checkups": training_landmark_checkups,
                            "training_max_checkup_index": training_max_checkup_index,
                            "training_support_hours": training_support_hours,
                            "training_support_days": training_support_hours / 24.0,
                            "optimizer_cost": stress_model.optimizer_cost,
                            "optimizer_evaluations": stress_model.optimizer_evaluations,
                        }
                    )
                for target_id in sorted(target_set):
                    target = target_fold.loc[
                        target_fold["condition_id"].astype(str) == target_id
                    ].copy()
                    support = _support_classification(training_fit, target)
                    validation_horizon_hours = float(target["elapsed_hours"].max())
                    if training_support_hours <= 0.0:
                        raise ValueError("Calendar training support must extend beyond time zero")
                    extrapolation_ratio = validation_horizon_hours / training_support_hours
                    predictions, diagnostics = build_calendar_target_prediction_state(
                        scenario_name=scenario_name,
                        fold_id=fold_id,
                        target_frame=target,
                        stress_model=stress_model,
                        null_model=null_model,
                        ridge_state=ridge_state,
                        prefix_checkups=prefix,
                        scale_bounds=(
                            float(adaptation_parameters["scale_min"]),
                            float(adaptation_parameters["scale_max"]),
                        ),
                        training_history_policy=history_policy,
                    )
                    predictions["training_support_days"] = training_support_hours / 24.0
                    predictions["validation_horizon_days"] = (
                        validation_horizon_hours / 24.0
                    )
                    predictions["time_extrapolation_ratio"] = extrapolation_ratio
                    prediction_frames.append(predictions)
                    diagnostic_rows.append(
                        {
                            "scenario": scenario_name,
                            "fold_id": fold_id,
                            **diagnostics,
                            **support,
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_fit),
                            "training_landmark_checkups": training_landmark_checkups,
                            "training_max_checkup_index": training_max_checkup_index,
                            "target_condition_count_in_fold": len(target_set),
                            "training_history_policy": history_policy,
                            "training_support_hours": training_support_hours,
                            "training_support_days": training_support_hours / 24.0,
                            "validation_horizon_hours": validation_horizon_hours,
                            "validation_horizon_days": validation_horizon_hours / 24.0,
                            "forecast_beyond_training_hours": (
                                validation_horizon_hours - training_support_hours
                            ),
                            "forecast_beyond_training_days": (
                                validation_horizon_hours - training_support_hours
                            )
                            / 24.0,
                            "time_extrapolation_ratio": extrapolation_ratio,
                        }
                    )

    label_free_predictions = pd.concat(prediction_frames, ignore_index=True)
    label_free_predictions = label_free_predictions.sort_values(
        PREDICTION_KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)
    prediction_hash = calendar_prediction_artifact_sha256(label_free_predictions)
    condition_metrics, aggregate_metrics = score_calendar_predictions(
        label_free_predictions,
        observations,
        frozen_prediction_sha256=prediction_hash,
    )
    gate_config = {
        "primary_prefix_checkups": int(primary_prefix_checkups),
        "validation_thresholds": dict(validation_thresholds),
        "gate_scenarios": configured_gates,
    }
    performance_gate = _scientific_gates(
        aggregate_metrics,
        condition_metrics,
        gate_config,
    )
    horizons = observations.groupby("condition_id")["elapsed_days"].max()
    result: dict[str, object] = {
        "status": (
            "public_method_gate_passed"
            if performance_gate["status"] == "passed"
            else "public_method_gate_failed"
        ),
        "model_name": CALENDAR_MODEL_NAME,
        "model_scope": (
            "Empirical temperature/SOC stress surface with fixed square-root time. "
            "This is not the full Naumann model or a mechanistic degradation model."
        ),
        "dataset": {
            "independent_condition_count": int(
                observations["condition_id"].nunique()
            ),
            "observation_count": len(observations),
            "statistical_unit": "temperature_soc_condition_mean_trajectory",
            "replicates_per_condition": 3,
            "minimum_horizon_days": float(horizons.min()),
            "maximum_horizon_days": float(horizons.max()),
        },
        "design": {
            "scenarios": scenario_configs,
            "prefix_checkups": prefixes,
            "primary_prefix_checkups": int(primary_prefix_checkups),
            "scientific_gate_design": GLOBAL_LANDMARK_POLICY,
            "full_history_design_role": "condition_transfer_upper_bound_only",
            "split_unit": "condition_id",
            "condition_balanced_metrics": True,
            "target_future_outcomes_used_for_fit": False,
            "target_prefix_outcomes_used_for_state_update": True,
            "global_landmark_training_future_outcomes_used_for_fit": False,
            "projection_beyond_observed_horizon": False,
            "projection_beyond_training_landmark_reported": True,
            "method_roles": {
                CALENDAR_MODEL_NAME: "static_stress_surface",
                PREFIX_UPDATED_MODEL_NAME: "target_prefix_dynamic_scale_update",
                CALENDAR_NULL_MODEL_NAME: "condition_agnostic_training_baseline",
                PREFIX_ONLY_MODEL_NAME: "target_prefix_only_baseline",
                PERSISTENCE_MODEL_NAME: "last_prefix_observation_baseline",
            },
        },
        "metrics": aggregate_metrics.to_dict(orient="records"),
        "performance_gate": performance_gate,
        "probability_calibration_gate": {
            "status": (
                "blocked_insufficient_independent_conditions"
                if int(observations["condition_id"].nunique())
                < int(validated_config["probability_calibration_minimum_groups"])
                else "not_evaluated_point_predictions_only"
            ),
            "independent_condition_count": int(
                observations["condition_id"].nunique()
            ),
            "minimum_required_groups": int(
                validated_config["probability_calibration_minimum_groups"]
            ),
            "reason": (
                "The configured independent-group threshold is not met, or no "
                "probabilistic predictions were evaluated. This release reports "
                "point errors only."
            ),
        },
        "long_horizon_gate": {
            "status": "blocked",
            "maximum_observed_days": float(horizons.max()),
            "reason": "No prediction beyond the 885-day public support is permitted.",
        },
        "prediction_firewall": {
            "label_free_prediction_sha256": prediction_hash,
            "future_outcome_columns_in_prediction_pack": [],
            "score_after_hash_verification": True,
        },
        "claim_boundary": (
            "Evidence is limited to retrospective public 3 Ah LFP condition-mean "
            "trajectories. It does not validate Hithium cells, utility-scale storage, "
            "or 15-25 year forecasts."
        ),
    }
    return (
        result,
        label_free_predictions,
        condition_metrics.sort_values(
            ["scenario", "target_condition_id", "prefix_checkups", "method"],
            kind="stable",
        ).reset_index(drop=True),
        pd.DataFrame(diagnostic_rows).sort_values(
            ["scenario", "target_condition_id", "prefix_checkups"],
            kind="stable",
        ).reset_index(drop=True),
        pd.DataFrame(parameter_rows).sort_values(
            ["scenario", "fold_id", "prefix_checkups", "parameter"],
            kind="stable",
        ).reset_index(drop=True),
        pd.DataFrame(split_rows).sort_values(
            ["scenario", "fold_id", "condition_id"],
            kind="stable",
        ).reset_index(drop=True),
    )
