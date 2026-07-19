from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.experiments.calendar_v2_development import (
    EXPECTED_DATASET_SNAPSHOT_ID,
    EXPECTED_LABEL_VERSION,
    GLOBAL_LANDMARK_POLICY,
    HIERARCHICAL_POWER_METHOD,
    SOC_SCENARIO,
    SOC_TARGET_CONDITIONS,
    TEMPERATURE_SCENARIO,
    calendar_v2_prediction_sha256,
)
from lifetwin.models.calendar_v2 import (
    fit_hierarchical_power_prior,
    fit_power_law,
    fit_sqrt_linear_coefficients,
    predict_power_loss,
    predict_sqrt_linear_loss,
    update_hierarchical_power_law,
)
from lifetwin.models.calendar_v2_uncertainty import (
    INTERVAL_METHODS,
    LAPLACE_INTERVAL_METHOD,
    PREFIX_CONFORMAL_INTERVAL_METHOD,
    finite_sample_higher_quantile,
    gaussian_central_multiplier,
    interval_score,
    physical_interval,
    power_law_predictive_sd,
)


EXPERIMENT_ID = "naumann_calendar_v2_uncertainty_development_v1"
BASE_EXPERIMENT_ID = "naumann_calendar_v2_development_bakeoff_v1"
DESIGN_STATUS = "isolated_post_hoc_uncertainty_development_diagnostic"
EVIDENCE_ROLE = "reused_naumann_uncertainty_calibration_diagnostic_only"
RUNNER_SCOPE = "naumann_reuse_uncertainty_development_only"
CONFIRMATION_STATUS = "blocked_pending_independent_dataset"
CALIBRATION_HISTORY_POLICY = "nested_training_prefix_before_outer_landmark"
EXPECTED_PREFIXES = (5, 8, 10, 14)
PRIMARY_PREFIX = 10
INNER_PREFIX_BY_OUTER = {5: 3, 8: 4, 10: 5, 14: 7}
INTERVAL_LEVELS = (0.8, 0.9, 0.95)
EXPECTED_PROHIBITED_CLAIMS = (
    "confirmatory_uncertainty_calibration_on_naumann_reuse",
    "finite_sample_guarantee_beyond_outer_landmark",
    "independent_external_validation",
    "individual_cell_variability",
    "hithium_product_uncertainty",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

PREDICTION_KEY_COLUMNS = [
    "scenario",
    "fold_id",
    "target_condition_id",
    "prefix_checkups",
    "interval_method",
    "requested_coverage",
    "target_checkup_index",
]
PREDICTION_COLUMNS = [
    *PREDICTION_KEY_COLUMNS,
    "training_history_policy",
    "calibration_history_policy",
    "inner_prefix_checkups",
    "prefix_end_checkup_index",
    "prefix_end_days",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_days",
    "predicted_capacity_retention_pct",
    "predictive_sd_pp",
    "interval_multiplier",
    "multiplier_status",
    "lower_capacity_retention_pct",
    "upper_capacity_retention_pct",
    "is_final_checkup",
    "calibration_condition_count",
    "calibration_max_checkup_index",
    "training_support_days",
    "validation_horizon_days",
    "time_extrapolation_ratio",
    "training_state_sha256",
    "calibration_state_sha256",
    "prediction_state_sha256",
]

CALIBRATION_SCORE_COLUMNS = [
    "scenario",
    "fold_id",
    "outer_prefix_checkups",
    "inner_prefix_checkups",
    "calibration_condition_id",
    "calibration_validation_point_count",
    "calibration_max_checkup_index",
    "maximum_standardized_error",
    "maximum_absolute_error_pp",
    "predictive_sd_min_pp",
    "predictive_sd_max_pp",
    "pseudo_training_condition_count",
    "pseudo_training_observation_count",
]

TOP_LEVEL_CONFIG_KEYS = {
    "experiment_id",
    "base_experiment_id",
    "design_status",
    "dataset_id",
    "dataset_snapshot_id",
    "label_version",
    "evidence_role",
    "runner_scope",
    "scenarios",
    "prefix_checkups",
    "primary_prefix_checkups",
    "training_history_policy",
    "primary_point_model",
    "model",
    "uncertainty",
    "calibration",
    "development_gate",
    "prohibited_claims",
}


def _require_exact_keys(
    value: Mapping[str, object],
    *,
    expected: set[str],
    context: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys must be exact: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return dict(value)


def _number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    return parsed


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame, *, sort_by: list[str]) -> str:
    ordered = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    encoded = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_calendar_v2_uncertainty_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    parsed = dict(config)
    _require_exact_keys(
        parsed,
        expected=TOP_LEVEL_CONFIG_KEYS,
        context="Calendar V2 uncertainty config",
    )
    expected_scalars = {
        "experiment_id": EXPERIMENT_ID,
        "base_experiment_id": BASE_EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "evidence_role": EVIDENCE_ROLE,
        "runner_scope": RUNNER_SCOPE,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "primary_point_model": HIERARCHICAL_POWER_METHOD,
    }
    for key, expected in expected_scalars.items():
        if parsed[key] != expected:
            raise ValueError(f"Calendar V2 uncertainty {key} must remain {expected}")
    if tuple(int(value) for value in parsed["prefix_checkups"]) != EXPECTED_PREFIXES:
        raise ValueError("Calendar V2 uncertainty prefixes must remain frozen")
    if int(parsed["primary_prefix_checkups"]) != PRIMARY_PREFIX:
        raise ValueError("Calendar V2 uncertainty primary prefix must remain 10")

    scenarios = list(parsed["scenarios"])
    if len(scenarios) != 2 or not all(isinstance(item, Mapping) for item in scenarios):
        raise ValueError("Calendar V2 uncertainty scenarios must contain two objects")
    scenario_by_name = {str(item["name"]): dict(item) for item in scenarios}
    if set(scenario_by_name) != {TEMPERATURE_SCENARIO, SOC_SCENARIO}:
        raise ValueError("Calendar V2 uncertainty scenarios must remain exact")
    temperature = scenario_by_name[TEMPERATURE_SCENARIO]
    _require_exact_keys(
        temperature,
        expected={"name", "kind", "training_history_policy"},
        context="Unseen-temperature scenario",
    )
    if temperature != {
        "name": TEMPERATURE_SCENARIO,
        "kind": "leave_one_temperature_level_out",
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
    }:
        raise ValueError("Unseen-temperature scenario changed")
    soc = scenario_by_name[SOC_SCENARIO]
    _require_exact_keys(
        soc,
        expected={
            "name",
            "kind",
            "training_history_policy",
            "fold_id",
            "target_condition_ids",
        },
        context="SOC scenario",
    )
    if (
        soc["kind"] != "fixed_condition_holdout"
        or soc["training_history_policy"] != GLOBAL_LANDMARK_POLICY
        or soc["fold_id"] != "40c_intermediate_soc_v2_uncertainty_development"
        or tuple(soc["target_condition_ids"]) != SOC_TARGET_CONDITIONS
    ):
        raise ValueError("SOC scenario changed")

    model = _mapping(parsed["model"], context="model")
    _require_exact_keys(
        model,
        expected={
            "time_unit",
            "minimum_training_conditions",
            "robust_loss_scale_pp",
            "power_exponent_bounds",
            "stress_surface_ridge",
            "parameter_scale_floors",
            "observation_scale_floor_pp",
        },
        context="model",
    )
    if model["time_unit"] != "day":
        raise ValueError("Calendar V2 uncertainty time unit must remain day")
    numeric_model = {
        "minimum_training_conditions": int(model["minimum_training_conditions"]),
        "robust_loss_scale_pp": _number(
            model["robust_loss_scale_pp"], context="robust_loss_scale_pp"
        ),
        "stress_surface_ridge": _number(
            model["stress_surface_ridge"], context="stress_surface_ridge"
        ),
        "observation_scale_floor_pp": _number(
            model["observation_scale_floor_pp"],
            context="observation_scale_floor_pp",
        ),
    }
    exponent_bounds = tuple(float(value) for value in model["power_exponent_bounds"])
    parameter_floors = tuple(float(value) for value in model["parameter_scale_floors"])
    if (
        numeric_model["minimum_training_conditions"] != 6
        or numeric_model["robust_loss_scale_pp"] != 0.25
        or exponent_bounds != (0.05, 1.5)
        or numeric_model["stress_surface_ridge"] != 1.0
        or parameter_floors != (0.1, 0.05)
        or numeric_model["observation_scale_floor_pp"] != 0.1
    ):
        raise ValueError("Calendar V2 uncertainty base point model changed")

    uncertainty = _mapping(parsed["uncertainty"], context="uncertainty")
    _require_exact_keys(
        uncertainty,
        expected={
            "interval_levels",
            "interval_methods",
            "predictive_scale",
            "include_observation_scale",
            "base_scale_floor_pp",
            "physical_bounds_pct",
        },
        context="uncertainty",
    )
    levels = tuple(float(value) for value in uncertainty["interval_levels"])
    methods = tuple(str(value) for value in uncertainty["interval_methods"])
    bounds = tuple(float(value) for value in uncertainty["physical_bounds_pct"])
    scale_floor = _number(
        uncertainty["base_scale_floor_pp"], context="base_scale_floor_pp"
    )
    if (
        levels != INTERVAL_LEVELS
        or methods != INTERVAL_METHODS
        or uncertainty["predictive_scale"]
        != "posterior_delta_method_plus_training_prefix_observation_scale"
        or uncertainty["include_observation_scale"] is not True
        or scale_floor != 0.1
        or bounds != (0.0, 100.0)
    ):
        raise ValueError("Calendar V2 uncertainty interval design changed")

    calibration = _mapping(parsed["calibration"], context="calibration")
    _require_exact_keys(
        calibration,
        expected={
            "calibration_unit",
            "score",
            "quantile_rule",
            "inner_prefix_checkups_by_outer",
            "validation_window",
            "target_future_outcomes_used",
            "training_rows_at_or_after_outer_prefix_used",
            "formal_coverage_claim_allowed",
        },
        context="calibration",
    )
    inner = {
        int(key): int(value)
        for key, value in _mapping(
            calibration["inner_prefix_checkups_by_outer"],
            context="inner_prefix_checkups_by_outer",
        ).items()
    }
    if (
        calibration["calibration_unit"] != "condition_mean_trajectory"
        or calibration["score"] != "maximum_standardized_error_within_inner_to_outer_window"
        or calibration["quantile_rule"] != "ceil((n+1)*coverage)_higher_or_infinity"
        or inner != INNER_PREFIX_BY_OUTER
        or calibration["validation_window"] != "inner_prefix_inclusive_outer_prefix_exclusive"
        or calibration["target_future_outcomes_used"] is not False
        or calibration["training_rows_at_or_after_outer_prefix_used"] is not False
        or calibration["formal_coverage_claim_allowed"] is not False
    ):
        raise ValueError("Calendar V2 uncertainty calibration design changed")

    gate = _mapping(parsed["development_gate"], context="development_gate")
    _require_exact_keys(
        gate,
        expected={
            "confirmation_status",
            "current_dataset_relationship",
            "decision_role",
        },
        context="development_gate",
    )
    if (
        gate["confirmation_status"] != CONFIRMATION_STATUS
        or gate["current_dataset_relationship"]
        != "reused_and_outcomes_already_inspected"
        or gate["decision_role"] != "descriptive_coverage_diagnostic_only"
    ):
        raise ValueError("Calendar V2 uncertainty gate must remain blocked")
    if tuple(parsed["prohibited_claims"]) != EXPECTED_PROHIBITED_CLAIMS:
        raise ValueError("Calendar V2 uncertainty prohibited claims must remain exact")

    return {
        **parsed,
        "scenarios": scenarios,
        "prefix_checkups": list(EXPECTED_PREFIXES),
        "model": {
            **model,
            **numeric_model,
            "power_exponent_bounds": list(exponent_bounds),
            "parameter_scale_floors": list(parameter_floors),
        },
        "uncertainty": {
            **uncertainty,
            "interval_levels": list(levels),
            "interval_methods": list(methods),
            "physical_bounds_pct": list(bounds),
            "base_scale_floor_pp": scale_floor,
        },
        "calibration": {
            **calibration,
            "inner_prefix_checkups_by_outer": {
                str(key): value for key, value in sorted(inner.items())
            },
        },
    }


def _scenario_folds(
    observations: pd.DataFrame,
    scenario: Mapping[str, object],
) -> list[tuple[str, tuple[str, ...]]]:
    profile = (
        observations[["condition_id", "temperature_c", "storage_soc_fraction"]]
        .drop_duplicates()
        .sort_values("condition_id", kind="stable")
    )
    if scenario["kind"] == "leave_one_temperature_level_out":
        return [
            (
                f"temperature_c={float(temperature):g}",
                tuple(sorted(group["condition_id"].astype(str))),
            )
            for temperature, group in profile.groupby("temperature_c", sort=True)
        ]
    targets = tuple(str(value) for value in scenario["target_condition_ids"])
    missing = sorted(set(targets) - set(profile["condition_id"].astype(str)))
    if missing:
        raise ValueError(f"Uncertainty scenario references unknown conditions: {missing}")
    return [(str(scenario["fold_id"]), targets)]


def _select_prefix(frame: pd.DataFrame, prefix_checkups: int) -> pd.DataFrame:
    selected = frame.loc[pd.to_numeric(frame["checkup_index"]) < prefix_checkups].copy()
    counts = selected.groupby("condition_id", sort=True).size()
    if selected.empty or (counts != prefix_checkups).any():
        raise ValueError("Every condition must contain the complete requested prefix")
    return selected.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)


def _model_kwargs(model: Mapping[str, object]) -> dict[str, object]:
    return {
        "minimum_conditions": int(model["minimum_training_conditions"]),
        "exponent_bounds": tuple(float(value) for value in model["power_exponent_bounds"]),
        "robust_loss_scale_pp": float(model["robust_loss_scale_pp"]),
        "stress_surface_ridge": float(model["stress_surface_ridge"]),
        "parameter_scale_floors": tuple(
            float(value) for value in model["parameter_scale_floors"]
        ),
        "observation_scale_floor_pp": float(model["observation_scale_floor_pp"]),
    }


def _fit_prior(training: pd.DataFrame, model: Mapping[str, object]):
    return fit_hierarchical_power_prior(training, **_model_kwargs(model))


def _calibration_scores(
    *,
    scenario: str,
    fold_id: str,
    training_outer_prefix: pd.DataFrame,
    outer_prefix: int,
    inner_prefix: int,
    model: Mapping[str, object],
    scale_floor_pp: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    condition_ids = tuple(
        sorted(training_outer_prefix["condition_id"].astype(str).unique())
    )
    for calibration_id in condition_ids:
        pseudo_target = training_outer_prefix.loc[
            training_outer_prefix["condition_id"].astype(str) == calibration_id
        ].copy()
        pseudo_target_prefix = _select_prefix(pseudo_target, inner_prefix)
        pseudo_training = training_outer_prefix.loc[
            training_outer_prefix["condition_id"].astype(str) != calibration_id
        ].copy()
        pseudo_training_prefix = _select_prefix(pseudo_training, inner_prefix)
        prior = _fit_prior(pseudo_training_prefix, model)
        posterior = update_hierarchical_power_law(
            prior,
            pseudo_target_prefix,
            exponent_bounds=tuple(
                float(value) for value in model["power_exponent_bounds"]
            ),
        )
        validation = pseudo_target.loc[
            (pd.to_numeric(pseudo_target["checkup_index"]) >= inner_prefix)
            & (pd.to_numeric(pseudo_target["checkup_index"]) < outer_prefix)
        ].sort_values("checkup_index", kind="stable")
        if len(validation) != outer_prefix - inner_prefix:
            raise ValueError("Calibration window must be complete for every condition")
        elapsed = validation["elapsed_days"].to_numpy(dtype=float)
        predicted_loss = predict_power_loss(posterior, elapsed)
        scale = power_law_predictive_sd(
            posterior,
            elapsed,
            observation_scale_pp=prior.observation_scale_pp,
            scale_floor_pp=scale_floor_pp,
        )
        absolute_error = np.abs(
            predicted_loss - validation["capacity_loss_pct"].to_numpy(dtype=float)
        )
        standardized = absolute_error / scale
        rows.append(
            {
                "scenario": scenario,
                "fold_id": fold_id,
                "outer_prefix_checkups": int(outer_prefix),
                "inner_prefix_checkups": int(inner_prefix),
                "calibration_condition_id": calibration_id,
                "calibration_validation_point_count": len(validation),
                "calibration_max_checkup_index": int(
                    validation["checkup_index"].max()
                ),
                "maximum_standardized_error": float(standardized.max()),
                "maximum_absolute_error_pp": float(absolute_error.max()),
                "predictive_sd_min_pp": float(scale.min()),
                "predictive_sd_max_pp": float(scale.max()),
                "pseudo_training_condition_count": int(
                    pseudo_training_prefix["condition_id"].nunique()
                ),
                "pseudo_training_observation_count": len(pseudo_training_prefix),
            }
        )
    return pd.DataFrame(rows)[CALIBRATION_SCORE_COLUMNS].sort_values(
        "calibration_condition_id", kind="stable"
    ).reset_index(drop=True)


def _training_state_sha256(
    *,
    scenario: str,
    fold_id: str,
    outer_prefix: int,
    inner_prefix: int,
    training_outer_prefix: pd.DataFrame,
    prior: object,
    calibration_scores: pd.DataFrame,
    model: Mapping[str, object],
) -> str:
    training_rows = [
        {
            "condition_id": str(row.condition_id),
            "checkup_index": int(row.checkup_index),
            "temperature_c": float(row.temperature_c),
            "storage_soc_fraction": float(row.storage_soc_fraction),
            "elapsed_days": float(row.elapsed_days),
            "capacity_loss_pct": float(row.capacity_loss_pct),
        }
        for row in training_outer_prefix.sort_values(
            ["condition_id", "checkup_index"], kind="stable"
        ).itertuples(index=False)
    ]
    payload = {
        "scenario": scenario,
        "fold_id": fold_id,
        "outer_prefix_checkups": int(outer_prefix),
        "inner_prefix_checkups": int(inner_prefix),
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "calibration_history_policy": CALIBRATION_HISTORY_POLICY,
        "training_rows": training_rows,
        "surface_coefficients": [list(values) for values in prior.surface_coefficients],
        "parameter_scales": list(prior.parameter_scales),
        "observation_scale_pp": float(prior.observation_scale_pp),
        "condition_parameters": [list(values) for values in prior.condition_parameters],
        "calibration_scores_sha256": _canonical_frame_sha256(
            calibration_scores,
            sort_by=["calibration_condition_id"],
        ),
        "model": dict(model),
    }
    return _canonical_json_sha256(payload)


def uncertainty_prediction_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V2 uncertainty prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty or predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Calendar V2 uncertainty prediction keys must be non-empty and unique")
    if set(predictions["interval_method"].astype(str)) != set(INTERVAL_METHODS):
        raise ValueError("Every uncertainty interval method must be present")
    if set(predictions["requested_coverage"].astype(float)) != set(INTERVAL_LEVELS):
        raise ValueError("Every uncertainty interval level must be present")
    if (
        predictions[
            [
                "predicted_capacity_retention_pct",
                "predictive_sd_pp",
                "lower_capacity_retention_pct",
                "upper_capacity_retention_pct",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Core uncertainty predictions cannot be null")
    if (
        predictions["lower_capacity_retention_pct"]
        > predictions["upper_capacity_retention_pct"]
    ).any():
        raise ValueError("Uncertainty interval bounds are reversed")
    unavailable = predictions["multiplier_status"].eq(
        "unavailable_finite_sample_full_physical_range"
    )
    if not predictions.loc[unavailable, "interval_multiplier"].isna().all():
        raise ValueError("Unavailable finite-sample multipliers must be null")
    if predictions.loc[~unavailable, "interval_multiplier"].isna().any():
        raise ValueError("Available interval multipliers cannot be null")
    for column in (
        "training_state_sha256",
        "calibration_state_sha256",
        "prediction_state_sha256",
    ):
        if not predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid uncertainty state hash: {column}")
    return _canonical_frame_sha256(
        predictions[PREDICTION_COLUMNS], sort_by=PREDICTION_KEY_COLUMNS
    )


def _target_predictions(
    *,
    scenario: str,
    fold_id: str,
    target: pd.DataFrame,
    outer_prefix: int,
    inner_prefix: int,
    prior: object,
    training_state_sha256: str,
    calibration_state_sha256: str,
    quantiles: list[dict[str, object]],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = target.sort_values("checkup_index", kind="stable").reset_index(drop=True)
    prefix = _select_prefix(ordered, outer_prefix)
    future = ordered.loc[
        pd.to_numeric(ordered["checkup_index"]) >= outer_prefix,
        [
            "condition_id",
            "checkup_index",
            "temperature_c",
            "storage_soc_fraction",
            "elapsed_days",
        ],
    ].sort_values("checkup_index", kind="stable")
    model = config["model"]
    uncertainty = config["uncertainty"]
    posterior = update_hierarchical_power_law(
        prior,
        prefix,
        exponent_bounds=tuple(float(value) for value in model["power_exponent_bounds"]),
    )
    elapsed = future["elapsed_days"].to_numpy(dtype=float)
    predicted_retention = 100.0 - predict_power_loss(posterior, elapsed)
    predictive_sd = power_law_predictive_sd(
        posterior,
        elapsed,
        observation_scale_pp=prior.observation_scale_pp,
        scale_floor_pp=float(uncertainty["base_scale_floor_pp"]),
    )
    intervals: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    target_id = str(ordered["condition_id"].iloc[0])
    prefix_end = prefix.sort_values("checkup_index", kind="stable").iloc[-1]
    calibration_count = int(quantiles[0]["calibration_condition_count"])
    for quantile in quantiles:
        coverage = float(quantile["requested_coverage"])
        candidates = [
            (
                LAPLACE_INTERVAL_METHOD,
                gaussian_central_multiplier(coverage),
                "gaussian_asymptotic_pointwise",
            ),
            (
                PREFIX_CONFORMAL_INTERVAL_METHOD,
                quantile["multiplier"],
                str(quantile["status"]),
            ),
        ]
        for method, multiplier, status in candidates:
            lower, upper = physical_interval(
                predicted_retention,
                predictive_sd,
                multiplier=multiplier,
                physical_bounds_pct=tuple(
                    float(value) for value in uncertainty["physical_bounds_pct"]
                ),
            )
            intervals.append(
                {
                    "method": method,
                    "coverage": coverage,
                    "multiplier": multiplier,
                    "status": status,
                    "lower": [float(value) for value in lower],
                    "upper": [float(value) for value in upper],
                }
            )
            for coordinate, point, scale, low, high in zip(
                future.itertuples(index=False),
                predicted_retention,
                predictive_sd,
                lower,
                upper,
                strict=True,
            ):
                rows.append(
                    {
                        "scenario": scenario,
                        "fold_id": fold_id,
                        "target_condition_id": target_id,
                        "prefix_checkups": int(outer_prefix),
                        "interval_method": method,
                        "requested_coverage": coverage,
                        "target_checkup_index": int(coordinate.checkup_index),
                        "training_history_policy": GLOBAL_LANDMARK_POLICY,
                        "calibration_history_policy": CALIBRATION_HISTORY_POLICY,
                        "inner_prefix_checkups": int(inner_prefix),
                        "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
                        "prefix_end_days": float(prefix_end["elapsed_days"]),
                        "temperature_c": float(coordinate.temperature_c),
                        "storage_soc_fraction": float(
                            coordinate.storage_soc_fraction
                        ),
                        "elapsed_days": float(coordinate.elapsed_days),
                        "predicted_capacity_retention_pct": float(point),
                        "predictive_sd_pp": float(scale),
                        "interval_multiplier": (
                            float(multiplier) if multiplier is not None else None
                        ),
                        "multiplier_status": status,
                        "lower_capacity_retention_pct": float(low),
                        "upper_capacity_retention_pct": float(high),
                        "is_final_checkup": bool(
                            coordinate.checkup_index == ordered["checkup_index"].max()
                        ),
                        "calibration_condition_count": calibration_count,
                        "calibration_max_checkup_index": int(outer_prefix - 1),
                        "training_support_days": float(prior.maximum_training_days),
                        "validation_horizon_days": float(ordered["elapsed_days"].max()),
                        "time_extrapolation_ratio": float(
                            ordered["elapsed_days"].max() / prior.maximum_training_days
                        ),
                        "training_state_sha256": training_state_sha256,
                        "calibration_state_sha256": calibration_state_sha256,
                        "prediction_state_sha256": "",
                    }
                )
    covariance = np.asarray(posterior.parameter_covariance, dtype=float)
    state = {
        "scenario": scenario,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "outer_prefix_checkups": int(outer_prefix),
        "inner_prefix_checkups": int(inner_prefix),
        "training_state_sha256": training_state_sha256,
        "calibration_state_sha256": calibration_state_sha256,
        "prefix_observations": [
            {
                "checkup_index": int(row.checkup_index),
                "elapsed_days": float(row.elapsed_days),
                "capacity_loss_pct": float(row.capacity_loss_pct),
            }
            for row in prefix.itertuples(index=False)
        ],
        "future_coordinates": [
            {
                "checkup_index": int(row.checkup_index),
                "elapsed_days": float(row.elapsed_days),
            }
            for row in future.itertuples(index=False)
        ],
        "posterior_parameters": posterior.parameter_map(),
        "posterior_covariance": covariance.tolist(),
        "observation_scale_pp": float(prior.observation_scale_pp),
        "predicted_capacity_retention_pct": [
            float(value) for value in predicted_retention
        ],
        "predictive_sd_pp": [float(value) for value in predictive_sd],
        "intervals": intervals,
    }
    prediction_hash = _canonical_json_sha256(state)
    for row in rows:
        row["prediction_state_sha256"] = prediction_hash
    prior_mean = prior.prior_mean(prefix)
    diagnostic = {
        "scenario": scenario,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(outer_prefix),
        "inner_prefix_checkups": int(inner_prefix),
        "prefix_end_days": float(prefix_end["elapsed_days"]),
        "future_checkup_count": len(future),
        "hierarchical_prior_log_amplitude": float(prior_mean[0]),
        "hierarchical_prior_time_exponent": float(prior_mean[1]),
        "hierarchical_posterior_log_amplitude": posterior.log_amplitude,
        "hierarchical_posterior_time_exponent": posterior.time_exponent,
        "hierarchical_posterior_log_amplitude_sd": float(
            math.sqrt(max(covariance[0, 0], 0.0))
        ),
        "hierarchical_posterior_time_exponent_sd": float(
            math.sqrt(max(covariance[1, 1], 0.0))
        ),
        "observation_scale_pp": float(prior.observation_scale_pp),
        "predictive_sd_min_pp": float(predictive_sd.min()),
        "predictive_sd_max_pp": float(predictive_sd.max()),
        "calibration_condition_count": calibration_count,
        "training_condition_count": len(prior.training_condition_ids),
        "training_observation_count": int(prior.training_observation_count),
        "training_max_checkup_index": int(outer_prefix - 1),
        "training_state_sha256": training_state_sha256,
        "calibration_state_sha256": calibration_state_sha256,
        "prediction_state_sha256": prediction_hash,
    }
    return pd.DataFrame(rows)[PREDICTION_COLUMNS], diagnostic


def score_uncertainty_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    observed_hash = uncertainty_prediction_sha256(predictions)
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Calendar V2 uncertainty prediction hash mismatch")
    lookup = observations[
        ["condition_id", "checkup_index", "capacity_retention_pct"]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "capacity_retention_pct": "observed_capacity_retention_pct",
        }
    )
    scored = predictions.merge(
        lookup,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
    )
    if scored["observed_capacity_retention_pct"].isna().any():
        raise ValueError("Uncertainty scoring could not resolve every outcome")
    truth = scored["observed_capacity_retention_pct"].to_numpy(dtype=float)
    lower = scored["lower_capacity_retention_pct"].to_numpy(dtype=float)
    upper = scored["upper_capacity_retention_pct"].to_numpy(dtype=float)
    scored["covered"] = (truth >= lower) & (truth <= upper)
    scored["interval_width_pp"] = upper - lower
    scored["absolute_point_error_pp"] = np.abs(
        truth - scored["predicted_capacity_retention_pct"].to_numpy(dtype=float)
    )
    scored["interval_score_pp"] = np.nan
    for coverage in INTERVAL_LEVELS:
        mask = np.isclose(scored["requested_coverage"].to_numpy(dtype=float), coverage)
        scored.loc[mask, "interval_score_pp"] = interval_score(
            scored.loc[mask, "observed_capacity_retention_pct"].to_numpy(dtype=float),
            scored.loc[mask, "lower_capacity_retention_pct"].to_numpy(dtype=float),
            scored.loc[mask, "upper_capacity_retention_pct"].to_numpy(dtype=float),
            coverage=coverage,
        )
    scored = scored.sort_values(PREDICTION_KEY_COLUMNS, kind="stable").reset_index(
        drop=True
    )

    group_keys = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "interval_method",
        "requested_coverage",
    ]
    condition_rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(group_keys, sort=True):
        condition_rows.append(
            {
                **dict(zip(group_keys, keys, strict=True)),
                "future_point_count": len(group),
                "pointwise_coverage_fraction": float(group["covered"].mean()),
                "simultaneous_trajectory_covered": bool(group["covered"].all()),
                "mean_interval_width_pp": float(group["interval_width_pp"].mean()),
                "mean_interval_score_pp": float(group["interval_score_pp"].mean()),
                "mean_absolute_point_error_pp": float(
                    group["absolute_point_error_pp"].mean()
                ),
                "final_checkup_covered": bool(
                    group.loc[group["is_final_checkup"], "covered"].iloc[0]
                ),
                "calibration_condition_count": int(
                    group["calibration_condition_count"].iloc[0]
                ),
                "multiplier_status": str(group["multiplier_status"].iloc[0]),
            }
        )
    condition_metrics = pd.DataFrame(condition_rows).sort_values(
        group_keys, kind="stable"
    ).reset_index(drop=True)

    summary_keys = [
        "scenario",
        "prefix_checkups",
        "interval_method",
        "requested_coverage",
    ]
    summary_rows: list[dict[str, object]] = []
    for keys, group in condition_metrics.groupby(summary_keys, sort=True):
        point_group = scored.merge(
            group[group_keys], on=group_keys, how="inner", validate="many_to_one"
        )
        statuses = set(group["multiplier_status"].astype(str))
        summary_rows.append(
            {
                **dict(zip(summary_keys, keys, strict=True)),
                "independent_condition_count": len(group),
                "shared_fold_count": int(group["fold_id"].nunique()),
                "future_point_count": len(point_group),
                "empirical_pointwise_coverage_fraction": float(
                    point_group["covered"].mean()
                ),
                "empirical_simultaneous_trajectory_coverage_fraction": float(
                    group["simultaneous_trajectory_covered"].mean()
                ),
                "final_checkup_coverage_fraction": float(
                    group["final_checkup_covered"].mean()
                ),
                "mean_interval_width_pp": float(point_group["interval_width_pp"].mean()),
                "mean_interval_score_pp": float(point_group["interval_score_pp"].mean()),
                "calibration_condition_count_min": int(
                    group["calibration_condition_count"].min()
                ),
                "calibration_condition_count_max": int(
                    group["calibration_condition_count"].max()
                ),
                "finite_multiplier_available_for_all_targets": (
                    "unavailable_finite_sample_full_physical_range" not in statuses
                ),
                "full_physical_range_target_fraction": float(
                    group["multiplier_status"]
                    .eq("unavailable_finite_sample_full_physical_range")
                    .mean()
                ),
                "inference_role": "descriptive_on_reused_outcomes_only",
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        summary_keys, kind="stable"
    ).reset_index(drop=True)
    return scored, condition_metrics, summary


def run_calendar_v2_uncertainty_development(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    validate_naumann_calendar_observations(observations)
    parsed = validate_calendar_v2_uncertainty_config(config)
    observations = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    all_ids = set(observations["condition_id"].astype(str))
    prediction_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    quantile_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    for scenario_raw in parsed["scenarios"]:
        scenario = dict(scenario_raw)
        scenario_name = str(scenario["name"])
        for fold_id, target_ids in _scenario_folds(observations, scenario):
            target_set = set(target_ids)
            training_ids = sorted(all_ids - target_set)
            for condition_id in sorted(all_ids):
                split_rows.append(
                    {
                        "scenario": scenario_name,
                        "fold_id": fold_id,
                        "condition_id": condition_id,
                        "role": "target" if condition_id in target_set else "training",
                    }
                )
            for outer_prefix in EXPECTED_PREFIXES:
                inner_prefix = INNER_PREFIX_BY_OUTER[outer_prefix]
                training_outer = _select_prefix(
                    observations.loc[
                        observations["condition_id"].astype(str).isin(training_ids)
                    ],
                    outer_prefix,
                )
                prior = _fit_prior(training_outer, parsed["model"])
                calibration = _calibration_scores(
                    scenario=scenario_name,
                    fold_id=fold_id,
                    training_outer_prefix=training_outer,
                    outer_prefix=outer_prefix,
                    inner_prefix=inner_prefix,
                    model=parsed["model"],
                    scale_floor_pp=float(
                        parsed["uncertainty"]["base_scale_floor_pp"]
                    ),
                )
                calibration_frames.append(calibration)
                calibration_hash = _canonical_frame_sha256(
                    calibration,
                    sort_by=["calibration_condition_id"],
                )
                quantiles: list[dict[str, object]] = []
                scores = calibration["maximum_standardized_error"].to_numpy(
                    dtype=float
                )
                for coverage in INTERVAL_LEVELS:
                    quantile = finite_sample_higher_quantile(
                        scores, coverage=coverage
                    )
                    row = {
                        "scenario": scenario_name,
                        "fold_id": fold_id,
                        "outer_prefix_checkups": int(outer_prefix),
                        "inner_prefix_checkups": int(inner_prefix),
                        "requested_coverage": float(coverage),
                        "calibration_condition_count": quantile.calibration_count,
                        "order_statistic_rank": quantile.order_statistic_rank,
                        "multiplier": quantile.multiplier,
                        "status": quantile.status,
                        "calibration_state_sha256": calibration_hash,
                    }
                    quantiles.append(row)
                    quantile_rows.append(row)
                training_hash = _training_state_sha256(
                    scenario=scenario_name,
                    fold_id=fold_id,
                    outer_prefix=outer_prefix,
                    inner_prefix=inner_prefix,
                    training_outer_prefix=training_outer,
                    prior=prior,
                    calibration_scores=calibration,
                    model=parsed["model"],
                )
                for target_id in sorted(target_set):
                    target = observations.loc[
                        observations["condition_id"].astype(str) == target_id
                    ].copy()
                    predictions, diagnostic = _target_predictions(
                        scenario=scenario_name,
                        fold_id=fold_id,
                        target=target,
                        outer_prefix=outer_prefix,
                        inner_prefix=inner_prefix,
                        prior=prior,
                        training_state_sha256=training_hash,
                        calibration_state_sha256=calibration_hash,
                        quantiles=quantiles,
                        config=parsed,
                    )
                    prediction_frames.append(predictions)
                    diagnostic_rows.append(diagnostic)

    predictions = pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS]
    predictions = predictions.sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    prediction_hash = uncertainty_prediction_sha256(predictions)
    point_scores, condition_metrics, summary = score_uncertainty_predictions(
        predictions,
        observations,
        frozen_prediction_sha256=prediction_hash,
    )
    calibration_scores = pd.concat(calibration_frames, ignore_index=True).sort_values(
        [
            "scenario",
            "fold_id",
            "outer_prefix_checkups",
            "calibration_condition_id",
        ],
        kind="stable",
    ).reset_index(drop=True)
    calibration_quantiles = pd.DataFrame(quantile_rows).sort_values(
        ["scenario", "fold_id", "outer_prefix_checkups", "requested_coverage"],
        kind="stable",
    ).reset_index(drop=True)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["scenario", "fold_id", "target_condition_id", "prefix_checkups"],
        kind="stable",
    ).reset_index(drop=True)
    splits = pd.DataFrame(split_rows).drop_duplicates().sort_values(
        ["scenario", "fold_id", "condition_id"], kind="stable"
    ).reset_index(drop=True)
    primary = summary.loc[summary["prefix_checkups"] == PRIMARY_PREFIX]
    result: dict[str, object] = {
        "status": "uncertainty_development_diagnostic_complete_confirmation_blocked",
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "base_experiment_id": BASE_EXPERIMENT_ID,
        "config_sha256": _canonical_json_sha256(parsed),
        "dataset": {
            "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
            "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
            "label_version": EXPECTED_LABEL_VERSION,
            "statistical_unit": NAUMANN_STATISTICAL_UNIT,
            "independent_condition_count": int(
                observations["condition_id"].nunique()
            ),
            "observation_count": len(observations),
            "maximum_observed_days": float(observations["elapsed_days"].max()),
        },
        "design": {
            "design_status": DESIGN_STATUS,
            "evidence_role": EVIDENCE_ROLE,
            "training_history_policy": GLOBAL_LANDMARK_POLICY,
            "calibration_history_policy": CALIBRATION_HISTORY_POLICY,
            "outer_prefix_checkups": list(EXPECTED_PREFIXES),
            "inner_prefix_checkups_by_outer": {
                str(key): value for key, value in INNER_PREFIX_BY_OUTER.items()
            },
            "interval_levels": list(INTERVAL_LEVELS),
            "interval_methods": list(INTERVAL_METHODS),
            "target_future_outcomes_used_for_prediction": False,
            "training_rows_at_or_after_outer_prefix_used": False,
            "condition_is_calibration_unit": True,
            "calibration_validation_horizon_matches_target_horizon": False,
        },
        "primary_prefix_summary": primary.to_dict(orient="records"),
        "development_gate": {
            "confirmation_status": CONFIRMATION_STATUS,
            "coverage_status": "descriptive_on_reused_development_data",
            "current_dataset_relationship": "reused_and_outcomes_already_inspected",
            "formal_coverage_claim_allowed": False,
            "reason": (
                "Calibration uses only labels before each outer landmark, but its "
                "inner-to-outer validation horizon is shorter than the scored target "
                "future and the Naumann outcomes were already inspected."
            ),
        },
        "finite_sample_limits": {
            "minimum_calibration_count_for_finite_80pct_quantile": 4,
            "minimum_calibration_count_for_finite_90pct_quantile": 9,
            "minimum_calibration_count_for_finite_95pct_quantile": 19,
            "maximum_available_calibration_condition_count": int(
                calibration_quantiles["calibration_condition_count"].max()
            ),
            "all_95pct_conformal_bands_fall_back_to_full_physical_range": bool(
                calibration_quantiles.loc[
                    np.isclose(
                        calibration_quantiles["requested_coverage"].to_numpy(
                            dtype=float
                        ),
                        0.95,
                    ),
                    "status",
                ].eq("unavailable_finite_sample_full_physical_range").all()
            ),
        },
        "future_label_firewall": {
            "label_free_prediction_sha256": prediction_hash,
            "calibration_scores_sha256": _canonical_frame_sha256(
                calibration_scores,
                sort_by=[
                    "scenario",
                    "fold_id",
                    "outer_prefix_checkups",
                    "calibration_condition_id",
                ],
            ),
            "score_after_prediction_hash_verification": True,
            "future_outcome_columns_in_prediction_pack": [],
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
        "claim_boundary": (
            "This layer diagnoses interval behavior on 17 reused condition-mean "
            "trajectories through 885 days. It does not establish calibrated "
            "individual-cell, Hithium-product, plant, or 15-25 year uncertainty."
        ),
    }
    return (
        result,
        predictions,
        calibration_scores,
        calibration_quantiles,
        point_scores,
        condition_metrics,
        summary,
        diagnostics,
        splits,
    )


def build_t40_soc12_5_failure_audit(
    observations: pd.DataFrame,
    *,
    phase6_predictions: pd.DataFrame,
    phase6_diagnostics: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_naumann_calendar_observations(observations)
    calendar_v2_prediction_sha256(phase6_predictions)
    target_id = "NAUMANN_CAL_T40_SOC12.5"
    prefix = 10
    target = observations.loc[
        observations["condition_id"].astype(str) == target_id
    ].sort_values("checkup_index", kind="stable")
    future_observed = target.loc[target["checkup_index"] >= prefix]
    selected = phase6_predictions.loc[
        (phase6_predictions["scenario"] == SOC_SCENARIO)
        & (phase6_predictions["target_condition_id"] == target_id)
        & (phase6_predictions["prefix_checkups"] == prefix)
    ].copy()
    if len(selected) != 4 * len(future_observed):
        raise ValueError("Frozen Phase 6 T40/SOC12.5 prediction slice is incomplete")
    observed_lookup = future_observed[
        ["checkup_index", "elapsed_days", "capacity_retention_pct", "capacity_loss_pct"]
    ].rename(
        columns={
            "checkup_index": "target_checkup_index",
            "elapsed_days": "observed_elapsed_days",
        }
    )
    residuals = selected[
        [
            "method",
            "target_checkup_index",
            "elapsed_days",
            "predicted_capacity_retention_pct",
        ]
    ].merge(
        observed_lookup,
        on="target_checkup_index",
        validate="many_to_one",
    )
    if len(residuals) != len(selected) or not np.allclose(
        residuals["elapsed_days"].to_numpy(dtype=float),
        residuals["observed_elapsed_days"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-9,
    ):
        raise ValueError("Frozen Phase 6 T40/SOC12.5 coordinates do not align")
    residuals = residuals.drop(columns="observed_elapsed_days")
    residuals["prediction_source"] = "phase6_prefix10_label_free_prediction"
    residuals["residual_retention_pp"] = (
        residuals["predicted_capacity_retention_pct"]
        - residuals["capacity_retention_pct"]
    )
    residuals["absolute_residual_pp"] = residuals["residual_retention_pp"].abs()

    full_power = fit_power_law(target)
    full_sqrt_linear = fit_sqrt_linear_coefficients(target)
    elapsed = future_observed["elapsed_days"].to_numpy(dtype=float)
    oracle_predictions = {
        "oracle_full_history_power_law_outcome_seen": 100.0
        - predict_power_loss(full_power, elapsed),
        "oracle_full_history_sqrt_plus_linear_outcome_seen": 100.0
        - predict_sqrt_linear_loss(full_sqrt_linear, elapsed),
    }
    oracle_rows: list[dict[str, object]] = []
    for method, predicted in oracle_predictions.items():
        for coordinate, value in zip(
            future_observed.itertuples(index=False), predicted, strict=True
        ):
            error = float(value - coordinate.capacity_retention_pct)
            oracle_rows.append(
                {
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(value),
                    "capacity_retention_pct": float(
                        coordinate.capacity_retention_pct
                    ),
                    "capacity_loss_pct": float(coordinate.capacity_loss_pct),
                    "prediction_source": "outcome_seen_oracle_diagnostic_not_candidate",
                    "residual_retention_pp": error,
                    "absolute_residual_pp": abs(error),
                }
            )
    residuals = pd.concat(
        [residuals, pd.DataFrame(oracle_rows)], ignore_index=True
    ).sort_values(["method", "target_checkup_index"], kind="stable").reset_index(
        drop=True
    )

    diagnostic_path = phase6_diagnostics.loc[
        (phase6_diagnostics["scenario"] == SOC_SCENARIO)
        & (phase6_diagnostics["target_condition_id"] == target_id)
        & (phase6_diagnostics["prefix_checkups"].isin(EXPECTED_PREFIXES))
    ].sort_values("prefix_checkups", kind="stable")
    if len(diagnostic_path) != len(EXPECTED_PREFIXES):
        raise ValueError("Frozen Phase 6 T40/SOC12.5 diagnostics are incomplete")
    parameter_rows = [
        {
            "fit_scope": f"prefix_{int(row.prefix_checkups)}",
            "prefix_checkups": int(row.prefix_checkups),
            "prefix_end_days": float(row.prefix_end_days),
            "target_power_log_amplitude": float(row.target_power_log_amplitude),
            "target_power_time_exponent": float(row.target_power_time_exponent),
            "hierarchical_prior_log_amplitude": float(
                row.hierarchical_prior_log_amplitude
            ),
            "hierarchical_prior_time_exponent": float(
                row.hierarchical_prior_time_exponent
            ),
            "hierarchical_posterior_log_amplitude": float(
                row.hierarchical_posterior_log_amplitude
            ),
            "hierarchical_posterior_time_exponent": float(
                row.hierarchical_posterior_time_exponent
            ),
            "outcome_seen": False,
        }
        for row in diagnostic_path.itertuples(index=False)
    ]
    parameter_rows.append(
        {
            "fit_scope": "full_history_oracle",
            "prefix_checkups": len(target),
            "prefix_end_days": float(target["elapsed_days"].max()),
            "target_power_log_amplitude": full_power.log_amplitude,
            "target_power_time_exponent": full_power.time_exponent,
            "hierarchical_prior_log_amplitude": np.nan,
            "hierarchical_prior_time_exponent": np.nan,
            "hierarchical_posterior_log_amplitude": np.nan,
            "hierarchical_posterior_time_exponent": np.nan,
            "outcome_seen": True,
        }
    )
    parameter_path = pd.DataFrame(parameter_rows)

    neighbor_rows: list[dict[str, object]] = []
    for condition_id in (
        "NAUMANN_CAL_T40_SOC0",
        "NAUMANN_CAL_T40_SOC12.5",
        "NAUMANN_CAL_T40_SOC25",
    ):
        condition = observations.loc[
            observations["condition_id"].astype(str) == condition_id
        ].sort_values("checkup_index", kind="stable")
        prefix_fit = fit_power_law(condition.loc[condition["checkup_index"] < prefix])
        full_fit = fit_power_law(condition)
        neighbor_rows.append(
            {
                "condition_id": condition_id,
                "storage_soc_fraction": float(
                    condition["storage_soc_fraction"].iloc[0]
                ),
                "checkup1_capacity_loss_pct": float(
                    condition.loc[condition["checkup_index"] == 1, "capacity_loss_pct"].iloc[0]
                ),
                "checkup2_capacity_loss_pct": float(
                    condition.loc[condition["checkup_index"] == 2, "capacity_loss_pct"].iloc[0]
                ),
                "prefix10_time_exponent": prefix_fit.time_exponent,
                "full_history_time_exponent": full_fit.time_exponent,
                "prefix_minus_full_time_exponent": (
                    prefix_fit.time_exponent - full_fit.time_exponent
                ),
            }
        )
    neighbors = pd.DataFrame(neighbor_rows)

    method_summary: list[dict[str, object]] = []
    for method, group in residuals.groupby("method", sort=True):
        ordered = group.sort_values("elapsed_days", kind="stable")
        horizon = float(ordered["elapsed_days"].iloc[-1] - ordered["elapsed_days"].iloc[0])
        iae = float(
            np.trapezoid(
                ordered["absolute_residual_pp"].to_numpy(dtype=float),
                ordered["elapsed_days"].to_numpy(dtype=float),
            )
            / horizon
        )
        method_summary.append(
            {
                "method": method,
                "trajectory_iae_pp": iae,
                "future_point_mae_pp": float(ordered["absolute_residual_pp"].mean()),
                "final_residual_retention_pp": float(
                    ordered["residual_retention_pp"].iloc[-1]
                ),
            }
        )
    p10_diag = diagnostic_path.loc[
        diagnostic_path["prefix_checkups"] == prefix
    ].iloc[0]
    target_neighbor = neighbors.loc[
        neighbors["condition_id"] == target_id
    ].iloc[0]
    audit = {
        "status": "outcome_seen_residual_diagnostic_complete",
        "target_condition_id": target_id,
        "prefix_checkups": prefix,
        "method_summary": method_summary,
        "classification": {
            "primary_driver": (
                "low_soc_early_capacity_rebound_and_prefix_exponent_overestimate"
            ),
            "stress_prior_is_primary_driver": False,
            "condition_specific_shape_mismatch_supported": True,
            "evidence": {
                "checkup1_capacity_loss_pct": float(
                    target_neighbor["checkup1_capacity_loss_pct"]
                ),
                "checkup2_capacity_loss_pct": float(
                    target_neighbor["checkup2_capacity_loss_pct"]
                ),
                "prefix10_target_time_exponent": float(
                    p10_diag["target_power_time_exponent"]
                ),
                "prefix10_hierarchical_time_exponent": float(
                    p10_diag["hierarchical_posterior_time_exponent"]
                ),
                "prefix10_stress_prior_time_exponent": float(
                    p10_diag["hierarchical_prior_time_exponent"]
                ),
                "full_history_oracle_time_exponent": full_power.time_exponent,
                "prefix10_target_minus_oracle_exponent": float(
                    p10_diag["target_power_time_exponent"]
                    - full_power.time_exponent
                ),
                "prefix10_prior_minus_oracle_exponent": float(
                    p10_diag["hierarchical_prior_time_exponent"]
                    - full_power.time_exponent
                ),
            },
            "interpretation": (
                "The first two positive-time checks show apparent capacity gain. A "
                "zero-intercept power law then fits an overly steep p=10 exponent. "
                "The stress prior pulls the exponent toward the full-history oracle "
                "but cannot overcome the prefix likelihood. The same rebound appears "
                "at 40 C / 0% SOC and is absent at 25% SOC."
            ),
        },
        "oracle_warning": (
            "Full-history fits use the scored future outcomes and are diagnostic "
            "oracles only; they are not eligible models, baselines, or evidence gates."
        ),
    }
    return audit, residuals, parameter_path, neighbors
