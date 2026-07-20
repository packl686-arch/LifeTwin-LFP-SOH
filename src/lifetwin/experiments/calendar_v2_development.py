from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.models.calendar_v2 import (
    HIERARCHICAL_POWER_METHOD,
    METHOD_NAMES,
    POWER_PARAMETER_NAMES,
    STRESS_FEATURE_NAMES,
    TARGET_POWER_METHOD,
    TARGET_SQRT_LINEAR_METHOD,
    TARGET_SQRT_METHOD,
    fit_hierarchical_power_prior,
    fit_power_law,
    fit_sqrt_linear_coefficients,
    fit_sqrt_rate,
    predict_power_loss,
    predict_sqrt_linear_loss,
    predict_sqrt_loss,
    update_hierarchical_power_law,
)


EXPERIMENT_ID = "naumann_calendar_v2_development_bakeoff_v1"
DESIGN_STATUS = "isolated_post_hoc_development_diagnostic"
EVIDENCE_ROLE = "reused_naumann_development_model_family_bakeoff_only"
RUNNER_SCOPE = "naumann_reuse_development_only"
CONFIRMATION_STATUS = "blocked_pending_independent_dataset"
GLOBAL_LANDMARK_POLICY = "global_landmark_prefix"
EXPECTED_DATASET_SNAPSHOT_ID = (
    "celljar_5c9601a027751c84ee8346f7c0ab9c6851330202_"
    "cycle_summary_6d4fd8d102ce9605c2c9bff86662b03201ae85411af583e23f3315b52bf81bc5"
)
EXPECTED_LABEL_VERSION = "published_condition_mean_capacity_retention_v1"
EXPECTED_PREFIXES = (5, 8, 10, 14)
EXPECTED_CHECKUP_INDICES = tuple(range(35))
PRIMARY_PREFIX = 10
TEMPERATURE_SCENARIO = "v2_unseen_temperature_level"
SOC_SCENARIO = "v2_soc_interpolation_at_40c"
SOC_FOLD_ID = "40c_intermediate_soc_v2_development"
GATE_SCENARIOS = (TEMPERATURE_SCENARIO, SOC_SCENARIO)
SOC_TARGET_CONDITIONS = (
    "NAUMANN_CAL_T40_SOC12.5",
    "NAUMANN_CAL_T40_SOC37.5",
    "NAUMANN_CAL_T40_SOC62.5",
    "NAUMANN_CAL_T40_SOC87.5",
)
EXPECTED_PROHIBITED_CLAIMS = (
    "confirmatory_superiority_on_naumann_reuse",
    "independent_external_validation",
    "individual_cell_variability",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
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
PREDICTION_COLUMNS = [
    *PREDICTION_KEY_COLUMNS,
    "prefix_end_checkup_index",
    "prefix_end_days",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_days",
    "predicted_capacity_retention_pct",
    "is_final_checkup",
    "training_support_days",
    "validation_horizon_days",
    "time_extrapolation_ratio",
    "training_state_sha256",
    "prediction_state_sha256",
]

TOP_LEVEL_CONFIG_KEYS = {
    "experiment_id",
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
    "methods",
    "primary_candidate",
    "baseline_method",
    "model",
    "diagnostics",
    "development_gate",
    "prohibited_claims",
}


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys do not match the implementation: "
            f"missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return dict(value)


def _require_number(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    return converted


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_frame_sha256(frame: pd.DataFrame, *, sort_by: list[str]) -> str:
    normalized = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_calendar_v2_config(config: Mapping[str, object]) -> dict[str, object]:
    parsed = dict(config)
    _require_exact_keys(parsed, TOP_LEVEL_CONFIG_KEYS, context="Calendar V2 config")
    exact_scalars = {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "evidence_role": EVIDENCE_ROLE,
        "runner_scope": RUNNER_SCOPE,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "primary_candidate": HIERARCHICAL_POWER_METHOD,
        "baseline_method": TARGET_SQRT_METHOD,
    }
    for key, expected in exact_scalars.items():
        if parsed[key] != expected:
            raise ValueError(f"{key} must remain {expected!r}")

    prefixes = tuple(int(value) for value in parsed["prefix_checkups"])
    if prefixes != EXPECTED_PREFIXES:
        raise ValueError(f"prefix_checkups must remain {list(EXPECTED_PREFIXES)}")
    if int(parsed["primary_prefix_checkups"]) != PRIMARY_PREFIX:
        raise ValueError(f"Primary prefix must remain {PRIMARY_PREFIX}")
    methods = tuple(str(value) for value in parsed["methods"])
    if methods != METHOD_NAMES:
        raise ValueError(f"Methods must remain {list(METHOD_NAMES)}")

    raw_scenarios = parsed["scenarios"]
    if not isinstance(raw_scenarios, list) or len(raw_scenarios) != 2:
        raise ValueError("Exactly two Calendar V2 scenarios are required")
    scenarios = [_require_mapping(value, context="scenario") for value in raw_scenarios]
    expected_scenarios = [
        {
            "name": TEMPERATURE_SCENARIO,
            "kind": "leave_one_temperature_level_out",
            "training_history_policy": GLOBAL_LANDMARK_POLICY,
            "comparison_family": "unseen_temperature_level",
        },
        {
            "name": SOC_SCENARIO,
            "kind": "fixed_condition_holdout",
            "training_history_policy": GLOBAL_LANDMARK_POLICY,
            "comparison_family": "soc_interpolation_at_40c",
            "fold_id": "40c_intermediate_soc_v2_development",
            "target_condition_ids": list(SOC_TARGET_CONDITIONS),
        },
    ]
    if scenarios != expected_scenarios:
        raise ValueError("Calendar V2 scenarios do not match the locked development design")

    model = _require_mapping(parsed["model"], context="model")
    _require_exact_keys(
        model,
        {
            "time_unit",
            "minimum_training_conditions",
            "robust_loss_scale_pp",
            "power_exponent_bounds",
            "stress_surface_ridge",
            "parameter_scale_floors",
            "observation_scale_floor_pp",
            "sqrt_linear_time_scale_days",
        },
        context="model",
    )
    if model["time_unit"] != "day":
        raise ValueError("Calendar V2 time unit must remain day")
    if int(model["minimum_training_conditions"]) < 6:
        raise ValueError("At least six training conditions are required")
    robust_scale = _require_number(
        model["robust_loss_scale_pp"],
        context="robust_loss_scale_pp",
    )
    exponent_bounds = tuple(
        _require_number(value, context="power_exponent_bounds")
        for value in model["power_exponent_bounds"]
    )
    parameter_scale_floors = tuple(
        _require_number(value, context="parameter_scale_floors")
        for value in model["parameter_scale_floors"]
    )
    if len(exponent_bounds) != 2 or not 0.0 < exponent_bounds[0] < exponent_bounds[1]:
        raise ValueError("Power exponent bounds must be positive and ordered")
    if len(parameter_scale_floors) != 2 or min(parameter_scale_floors) <= 0.0:
        raise ValueError("Two positive parameter scale floors are required")
    numeric_model = {
        "stress_surface_ridge": _require_number(
            model["stress_surface_ridge"], context="stress_surface_ridge"
        ),
        "observation_scale_floor_pp": _require_number(
            model["observation_scale_floor_pp"],
            context="observation_scale_floor_pp",
        ),
        "sqrt_linear_time_scale_days": _require_number(
            model["sqrt_linear_time_scale_days"],
            context="sqrt_linear_time_scale_days",
        ),
    }
    if robust_scale <= 0.0 or min(numeric_model.values()) <= 0.0:
        raise ValueError("Calendar V2 model scales and ridge must be positive")

    diagnostics = _require_mapping(parsed["diagnostics"], context="diagnostics")
    _require_exact_keys(
        diagnostics,
        {
            "bootstrap_unit",
            "bootstrap_resamples",
            "random_seed",
            "confidence_level",
            "inference_role",
            "shared_fold_dependence_adjusted",
        },
        context="diagnostics",
    )
    if diagnostics["bootstrap_unit"] != "condition":
        raise ValueError("Bootstrap unit must remain condition")
    if int(diagnostics["bootstrap_resamples"]) < 1000:
        raise ValueError("At least 1,000 bootstrap resamples are required")
    confidence = _require_number(
        diagnostics["confidence_level"], context="confidence_level"
    )
    if not 0.0 < confidence < 1.0:
        raise ValueError("Confidence level must lie in (0, 1)")
    if diagnostics["inference_role"] != "descriptive_resampling_only":
        raise ValueError("Bootstrap inference must remain descriptive")
    if diagnostics["shared_fold_dependence_adjusted"] is not False:
        raise ValueError("Shared-fold dependence is not adjusted in this experiment")

    gate = _require_mapping(parsed["development_gate"], context="development_gate")
    expected_gate = {
        "decision_rule": (
            "primary_candidate_bootstrap_upper_ci_below_zero_in_both_scenarios"
        ),
        "confirmation_status": CONFIRMATION_STATUS,
        "current_dataset_relationship": "reused_and_outcomes_already_inspected",
    }
    if gate != expected_gate:
        raise ValueError("Development gate must remain blocked and diagnostic")
    if tuple(parsed["prohibited_claims"]) != EXPECTED_PROHIBITED_CLAIMS:
        raise ValueError("Calendar V2 prohibited claims must remain exact")

    return {
        **parsed,
        "scenarios": scenarios,
        "prefix_checkups": list(prefixes),
        "methods": list(methods),
        "model": {
            **model,
            "robust_loss_scale_pp": robust_scale,
            "power_exponent_bounds": list(exponent_bounds),
            "parameter_scale_floors": list(parameter_scale_floors),
            **numeric_model,
        },
        "diagnostics": {
            **diagnostics,
            "confidence_level": confidence,
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
    kind = str(scenario["kind"])
    if kind == "leave_one_temperature_level_out":
        return [
            (
                f"temperature_c={float(temperature):g}",
                tuple(sorted(rows["condition_id"].astype(str))),
            )
            for temperature, rows in profile.groupby("temperature_c", sort=True)
        ]
    if kind == "fixed_condition_holdout":
        targets = tuple(sorted(str(value) for value in scenario["target_condition_ids"]))
        missing = sorted(set(targets) - set(profile["condition_id"].astype(str)))
        if missing:
            raise ValueError(f"Scenario references unknown conditions: {missing}")
        return [(str(scenario["fold_id"]), targets)]
    raise ValueError(f"Unsupported Calendar V2 scenario kind: {kind}")


def _select_prefix(frame: pd.DataFrame, prefix_checkups: int) -> pd.DataFrame:
    selected = frame.loc[pd.to_numeric(frame["checkup_index"]) < prefix_checkups].copy()
    counts = selected.groupby("condition_id", sort=True).size()
    if selected.empty or (counts != prefix_checkups).any():
        raise ValueError("Every condition must contain the complete requested prefix")
    return selected.sort_values(
        ["condition_id", "checkup_index"],
        kind="stable",
    ).reset_index(drop=True)


def calendar_v2_prediction_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V2 prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty:
        raise ValueError("Calendar V2 prediction pack cannot be empty")
    if predictions[PREDICTION_KEY_COLUMNS].isna().any().any():
        raise ValueError("Calendar V2 prediction keys cannot be null")
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Calendar V2 prediction keys must be unique")
    if set(predictions["method"].astype(str)) != set(METHOD_NAMES):
        raise ValueError("Calendar V2 prediction pack must contain every method")
    support_columns = [
        column for column in PREDICTION_KEY_COLUMNS if column != "method"
    ]
    methods_by_coordinate = predictions.groupby(
        support_columns,
        sort=False,
        dropna=False,
    )["method"].agg(lambda values: frozenset(values.astype(str)))
    expected_methods = frozenset(METHOD_NAMES)
    if not methods_by_coordinate.map(
        lambda methods: methods == expected_methods
    ).all():
        raise ValueError(
            "Calendar V2 prediction support must contain every method at "
            "every future coordinate"
        )
    for column in ("training_state_sha256", "prediction_state_sha256"):
        if not predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid Calendar V2 state hash column: {column}")
    return _canonical_frame_sha256(
        predictions[PREDICTION_COLUMNS],
        sort_by=PREDICTION_KEY_COLUMNS,
    )


def _training_state(
    *,
    scenario: str,
    fold_id: str,
    prefix_checkups: int,
    training_frame: pd.DataFrame,
    prior: object,
    model_config: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    training_rows = [
        {
            "condition_id": str(row.condition_id),
            "checkup_index": int(row.checkup_index),
            "temperature_c": float(row.temperature_c),
            "storage_soc_fraction": float(row.storage_soc_fraction),
            "elapsed_days": float(row.elapsed_days),
            "capacity_loss_pct": float(row.capacity_loss_pct),
        }
        for row in training_frame.sort_values(
            ["condition_id", "checkup_index"], kind="stable"
        ).itertuples(index=False)
    ]
    payload = {
        "scenario": scenario,
        "fold_id": fold_id,
        "prefix_checkups": int(prefix_checkups),
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "training_rows": training_rows,
        "surface_coefficients": [
            list(values) for values in prior.surface_coefficients
        ],
        "parameter_scales": list(prior.parameter_scales),
        "observation_scale_pp": float(prior.observation_scale_pp),
        "condition_parameters": [list(values) for values in prior.condition_parameters],
        "model_config": dict(model_config),
    }
    return _canonical_json_sha256(payload), payload


def _target_prediction_state(
    *,
    scenario: str,
    fold_id: str,
    target: pd.DataFrame,
    prefix_checkups: int,
    training_state_sha256: str,
    prior: object,
    model_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    ordered = target.sort_values("checkup_index", kind="stable").reset_index(drop=True)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("Each Calendar V2 target state requires one condition")
    if len(ordered) <= prefix_checkups:
        raise ValueError("Target condition has no future checkups")
    prefix = _select_prefix(ordered, prefix_checkups)
    future = ordered.loc[
        pd.to_numeric(ordered["checkup_index"]) >= prefix_checkups,
        [
            "condition_id",
            "checkup_index",
            "temperature_c",
            "storage_soc_fraction",
            "elapsed_days",
        ],
    ].sort_values("checkup_index", kind="stable")
    exponent_bounds = tuple(float(value) for value in model_config["power_exponent_bounds"])
    robust_scale = float(model_config["robust_loss_scale_pp"])
    sqrt_rate = fit_sqrt_rate(prefix)
    target_power = fit_power_law(
        prefix,
        exponent_bounds=exponent_bounds,
        robust_loss_scale_pp=robust_scale,
    )
    sqrt_linear = fit_sqrt_linear_coefficients(
        prefix,
        time_scale_days=float(model_config["sqrt_linear_time_scale_days"]),
        robust_loss_scale_pp=robust_scale,
    )
    hierarchical_power = update_hierarchical_power_law(
        prior,
        prefix,
        exponent_bounds=exponent_bounds,
    )
    elapsed = future["elapsed_days"].to_numpy(dtype=float)
    predicted_loss = {
        TARGET_SQRT_METHOD: predict_sqrt_loss(sqrt_rate, elapsed),
        TARGET_POWER_METHOD: predict_power_loss(target_power, elapsed),
        TARGET_SQRT_LINEAR_METHOD: predict_sqrt_linear_loss(
            sqrt_linear,
            elapsed,
            time_scale_days=float(model_config["sqrt_linear_time_scale_days"]),
        ),
        HIERARCHICAL_POWER_METHOD: predict_power_loss(hierarchical_power, elapsed),
    }
    target_id = str(ordered["condition_id"].iloc[0])
    prior_mean = prior.prior_mean(prefix)
    covariance = hierarchical_power.parameter_covariance
    state_payload = {
        "scenario": scenario,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(prefix_checkups),
        "training_state_sha256": training_state_sha256,
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
        "parameters": {
            TARGET_SQRT_METHOD: {"rate": float(sqrt_rate)},
            TARGET_POWER_METHOD: target_power.parameter_map(),
            TARGET_SQRT_LINEAR_METHOD: {
                "sqrt_coefficient": float(sqrt_linear[0]),
                "linear_coefficient": float(sqrt_linear[1]),
            },
            HIERARCHICAL_POWER_METHOD: hierarchical_power.parameter_map(),
        },
        "hierarchical_prior_mean": {
            "log_amplitude": float(prior_mean[0]),
            "time_exponent": float(prior_mean[1]),
        },
        "hierarchical_posterior_covariance": (
            [list(row) for row in covariance] if covariance is not None else None
        ),
        "predictions": {
            method: [float(value) for value in values]
            for method, values in predicted_loss.items()
        },
    }
    prediction_state_sha256 = _canonical_json_sha256(state_payload)
    prefix_end = prefix.sort_values("checkup_index", kind="stable").iloc[-1]
    rows: list[dict[str, object]] = []
    for method in METHOD_NAMES:
        for coordinate, loss in zip(
            future.itertuples(index=False),
            predicted_loss[method],
            strict=True,
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "training_history_policy": GLOBAL_LANDMARK_POLICY,
                    "fold_id": fold_id,
                    "target_condition_id": target_id,
                    "prefix_checkups": int(prefix_checkups),
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
                    "prefix_end_days": float(prefix_end["elapsed_days"]),
                    "temperature_c": float(coordinate.temperature_c),
                    "storage_soc_fraction": float(coordinate.storage_soc_fraction),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": 100.0 - float(loss),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == ordered["checkup_index"].max()
                    ),
                    "training_support_days": float(prior.maximum_training_days),
                    "validation_horizon_days": float(ordered["elapsed_days"].max()),
                    "time_extrapolation_ratio": float(
                        ordered["elapsed_days"].max() / prior.maximum_training_days
                    ),
                    "training_state_sha256": training_state_sha256,
                    "prediction_state_sha256": prediction_state_sha256,
                }
            )
    posterior_sd = (
        np.sqrt(np.maximum(np.diag(np.asarray(covariance)), 0.0))
        if covariance is not None
        else np.full(2, np.nan)
    )
    diagnostics = {
        "scenario": scenario,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(prefix_checkups),
        "prefix_end_days": float(prefix_end["elapsed_days"]),
        "future_checkup_count": len(future),
        "sqrt_rate": float(sqrt_rate),
        "target_power_log_amplitude": target_power.log_amplitude,
        "target_power_time_exponent": target_power.time_exponent,
        "sqrt_linear_sqrt_coefficient": float(sqrt_linear[0]),
        "sqrt_linear_linear_coefficient": float(sqrt_linear[1]),
        "hierarchical_prior_log_amplitude": float(prior_mean[0]),
        "hierarchical_prior_time_exponent": float(prior_mean[1]),
        "hierarchical_posterior_log_amplitude": hierarchical_power.log_amplitude,
        "hierarchical_posterior_time_exponent": hierarchical_power.time_exponent,
        "hierarchical_posterior_log_amplitude_sd": float(posterior_sd[0]),
        "hierarchical_posterior_time_exponent_sd": float(posterior_sd[1]),
        "training_state_sha256": training_state_sha256,
        "prediction_state_sha256": prediction_state_sha256,
    }
    return pd.DataFrame(rows), diagnostics


def _validated_scoring_frame(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    grouping_columns: list[str],
) -> pd.DataFrame:
    _validate_frozen_protocol_target_coverage(predictions, observations)
    required_scoring_columns = set(grouping_columns) | {
        "target_checkup_index",
        "elapsed_days",
        "predicted_capacity_retention_pct",
        "is_final_checkup",
    }
    if predictions[list(required_scoring_columns)].isna().any().any():
        raise ValueError("Calendar V2 scoring fields must be non-null")
    numeric_columns = [
        "prefix_checkups",
        "target_checkup_index",
        "prefix_end_checkup_index",
        "prefix_end_days",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "predicted_capacity_retention_pct",
        "training_support_days",
        "validation_horizon_days",
        "time_extrapolation_ratio",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Calendar V2 prediction column must be finite: {column}")
    for column in (
        "prefix_checkups",
        "target_checkup_index",
        "prefix_end_checkup_index",
    ):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Calendar V2 prediction column must be integral: {column}")
    if not pd.api.types.is_bool_dtype(predictions["is_final_checkup"]):
        raise ValueError("Calendar V2 final-checkup flags must be boolean")

    truth = observations[
        [
            "condition_id",
            "checkup_index",
            "elapsed_days",
            "temperature_c",
            "storage_soc_fraction",
            "capacity_retention_pct",
        ]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "elapsed_days": "truth_elapsed_days",
            "temperature_c": "truth_temperature_c",
            "storage_soc_fraction": "truth_storage_soc_fraction",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    if truth["target_condition_id"].isna().any():
        raise ValueError("Calendar V2 truth condition ids must be non-null")
    if truth.duplicated(["target_condition_id", "target_checkup_index"]).any():
        raise ValueError("Calendar V2 truth coordinates must be unique")
    truth_numeric = truth[
        [
            "target_checkup_index",
            "truth_elapsed_days",
            "truth_temperature_c",
            "truth_storage_soc_fraction",
            "true_capacity_retention_pct",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(truth_numeric.to_numpy(dtype=float)).all():
        raise ValueError("Calendar V2 truth coordinates and outcomes must be finite")
    if not np.equal(
        truth_numeric["target_checkup_index"],
        np.floor(truth_numeric["target_checkup_index"]),
    ).all():
        raise ValueError("Calendar V2 truth checkup indices must be integral")
    truth[truth_numeric.columns] = truth_numeric

    target_ids = set(predictions["target_condition_id"].astype(str))
    relevant_truth = truth.loc[
        truth["target_condition_id"].astype(str).isin(target_ids)
    ].copy()
    if set(relevant_truth["target_condition_id"].astype(str)) != target_ids:
        raise ValueError("Every Calendar V2 target condition must exist in truth")
    for _, condition in relevant_truth.groupby("target_condition_id", sort=True):
        indices = sorted(
            pd.to_numeric(condition["target_checkup_index"]).astype(int).tolist()
        )
        if indices != list(EXPECTED_CHECKUP_INDICES):
            raise ValueError(
                "Calendar V2 truth must contain checkup indices range(0, 35)"
            )

    scored = predictions.merge(
        truth,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
        indicator="_future_truth_merge",
    )
    if (scored["_future_truth_merge"] != "both").any() or scored[
        "true_capacity_retention_pct"
    ].isna().any():
        raise ValueError("Every Calendar V2 prediction must match one future outcome")
    scored = scored.drop(columns="_future_truth_merge")

    for prediction_column, truth_column in (
        ("elapsed_days", "truth_elapsed_days"),
        ("temperature_c", "truth_temperature_c"),
        ("storage_soc_fraction", "truth_storage_soc_fraction"),
    ):
        matches = np.isclose(
            pd.to_numeric(scored[prediction_column]).to_numpy(dtype=float),
            pd.to_numeric(scored[truth_column]).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        if not matches.all():
            raise ValueError(
                f"Calendar V2 prediction coordinate disagrees with truth: "
                f"{prediction_column}"
            )

    prefixes = pd.to_numeric(scored["prefix_checkups"]).astype(int)
    target_indices = pd.to_numeric(scored["target_checkup_index"]).astype(int)
    prefix_end_indices = pd.to_numeric(
        scored["prefix_end_checkup_index"]
    ).astype(int)
    if not prefixes.isin(EXPECTED_PREFIXES).all():
        raise ValueError("Calendar V2 prediction uses an unsupported prefix")
    if (target_indices < prefixes).any():
        raise ValueError("Calendar V2 prediction target precedes its prefix")
    if not (prefix_end_indices == prefixes - 1).all():
        raise ValueError("Calendar V2 prefix-end index disagrees with prefix")

    prefix_truth = truth[
        ["target_condition_id", "target_checkup_index", "truth_elapsed_days"]
    ].rename(
        columns={
            "target_checkup_index": "prefix_end_checkup_index",
            "truth_elapsed_days": "truth_prefix_end_days",
        }
    )
    scored = scored.merge(
        prefix_truth,
        on=["target_condition_id", "prefix_end_checkup_index"],
        how="left",
        validate="many_to_one",
    )
    if not np.isclose(
        pd.to_numeric(scored["prefix_end_days"]).to_numpy(dtype=float),
        pd.to_numeric(scored["truth_prefix_end_days"]).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError("Calendar V2 prefix-end day disagrees with truth")

    truth_horizons = relevant_truth.groupby("target_condition_id", sort=True)[
        "truth_elapsed_days"
    ].max()
    expected_horizons = scored["target_condition_id"].map(truth_horizons)
    if not np.isclose(
        pd.to_numeric(scored["validation_horizon_days"]).to_numpy(dtype=float),
        pd.to_numeric(expected_horizons).to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError("Calendar V2 validation horizon disagrees with truth")
    expected_ratios = (
        pd.to_numeric(scored["validation_horizon_days"])
        / pd.to_numeric(scored["training_support_days"])
    )
    if not np.isclose(
        pd.to_numeric(scored["time_extrapolation_ratio"]).to_numpy(dtype=float),
        expected_ratios.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ).all():
        raise ValueError("Calendar V2 extrapolation ratio is inconsistent")

    for _, group in scored.groupby(grouping_columns, sort=True, dropna=False):
        prefix_values = pd.to_numeric(group["prefix_checkups"]).astype(int).unique()
        if len(prefix_values) != 1:
            raise ValueError("Calendar V2 trajectory must use one prefix")
        expected_future = list(range(int(prefix_values[0]), 35))
        actual_future = sorted(
            pd.to_numeric(group["target_checkup_index"]).astype(int).tolist()
        )
        if actual_future != expected_future:
            raise ValueError(
                "Calendar V2 trajectory must contain every future checkup in "
                "range(prefix, 35)"
            )

    truth_final_indices = relevant_truth.groupby("target_condition_id", sort=True)[
        "target_checkup_index"
    ].max()
    scored_target_indices = pd.to_numeric(scored["target_checkup_index"]).astype(int)
    expected_final = scored_target_indices == scored["target_condition_id"].map(
        truth_final_indices
    ).astype(int)
    if not (scored["is_final_checkup"].to_numpy(dtype=bool) == expected_final).all():
        raise ValueError("Calendar V2 final-checkup flag disagrees with truth")
    scored["truth_is_final_checkup"] = expected_final
    return scored


def _validate_frozen_protocol_target_coverage(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
) -> None:
    observed_scenarios = set(predictions["scenario"].astype(str))
    frozen_scenarios = set(GATE_SCENARIOS)
    profile = observations[
        ["condition_id", "temperature_c", "storage_soc_fraction"]
    ].drop_duplicates()
    observation_ids = set(profile["condition_id"].astype(str))
    soc_targets = frozenset(SOC_TARGET_CONDITIONS)
    if not soc_targets.issubset(observation_ids):
        if not observed_scenarios.intersection(frozen_scenarios):
            return
        raise ValueError(
            "Calendar V2 frozen SOC targets are absent from scoring truth"
        )

    expected_groups: dict[tuple[str, str, int], frozenset[str]] = {}
    for temperature, rows in profile.groupby("temperature_c", sort=True):
        targets = frozenset(rows["condition_id"].astype(str))
        fold_id = f"temperature_c={float(temperature):g}"
        for prefix in EXPECTED_PREFIXES:
            expected_groups[(TEMPERATURE_SCENARIO, fold_id, prefix)] = targets
    for prefix in EXPECTED_PREFIXES:
        expected_groups[(SOC_SCENARIO, SOC_FOLD_ID, prefix)] = soc_targets

    target_rows = predictions[
        ["scenario", "fold_id", "prefix_checkups", "target_condition_id"]
    ].copy()
    target_rows["scenario"] = target_rows["scenario"].astype(str)
    target_rows["fold_id"] = target_rows["fold_id"].astype(str)
    target_rows["prefix_checkups"] = pd.to_numeric(
        target_rows["prefix_checkups"], errors="coerce"
    )
    target_rows["target_condition_id"] = target_rows[
        "target_condition_id"
    ].astype(str)
    actual_groups = {
        (str(scenario), str(fold_id), int(prefix)): frozenset(
            group["target_condition_id"]
        )
        for (scenario, fold_id, prefix), group in target_rows.groupby(
            ["scenario", "fold_id", "prefix_checkups"],
            sort=False,
            dropna=False,
        )
        if np.isfinite(prefix) and float(prefix).is_integer()
    }
    if actual_groups != expected_groups:
        raise ValueError(
            "Calendar V2 prediction target coverage does not match the frozen "
            "scenario/fold/prefix protocol"
        )


def score_calendar_v2_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    if calendar_v2_prediction_sha256(predictions) != frozen_prediction_sha256:
        raise ValueError("Frozen Calendar V2 prediction hash does not match content")
    grouping = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_days",
        "method",
        "training_state_sha256",
        "prediction_state_sha256",
    ]
    scored = _validated_scoring_frame(
        predictions,
        observations,
        grouping_columns=grouping,
    )
    scored["prediction_error_pp"] = (
        scored["predicted_capacity_retention_pct"]
        - scored["true_capacity_retention_pct"]
    )
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(grouping, sort=True):
        ordered = group.sort_values("truth_elapsed_days", kind="stable")
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        error = ordered["prediction_error_pp"].to_numpy(dtype=float)
        absolute_error = np.abs(error)
        if len(elapsed) < 2 or elapsed[-1] <= elapsed[0]:
            raise ValueError("Trajectory IAE requires at least two future checkups")
        final = ordered.loc[ordered["truth_is_final_checkup"]]
        if len(final) != 1:
            raise ValueError("Each Calendar V2 trajectory requires one final point")
        final_row = final.iloc[0]
        rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "future_checkup_count": len(ordered),
                "trajectory_iae_pp": float(
                    np.trapezoid(absolute_error, elapsed) / (elapsed[-1] - elapsed[0])
                ),
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
    return pd.DataFrame(rows).sort_values(
        ["scenario", "target_condition_id", "prefix_checkups", "method"],
        kind="stable",
    ).reset_index(drop=True)


def _paired_metrics(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    key = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_days",
    ]
    baseline = condition_metrics.loc[
        condition_metrics["method"] == TARGET_SQRT_METHOD,
        [*key, "trajectory_iae_pp", "final_absolute_error_pp"],
    ].rename(
        columns={
            "trajectory_iae_pp": "baseline_trajectory_iae_pp",
            "final_absolute_error_pp": "baseline_final_absolute_error_pp",
        }
    )
    candidates = condition_metrics.loc[
        condition_metrics["method"] != TARGET_SQRT_METHOD,
        [*key, "method", "trajectory_iae_pp", "final_absolute_error_pp"],
    ].rename(
        columns={
            "method": "candidate_method",
            "trajectory_iae_pp": "candidate_trajectory_iae_pp",
            "final_absolute_error_pp": "candidate_final_absolute_error_pp",
        }
    )
    paired = candidates.merge(baseline, on=key, how="left", validate="many_to_one")
    if paired[["baseline_trajectory_iae_pp", "baseline_final_absolute_error_pp"]].isna().any().any():
        raise RuntimeError("Every Calendar V2 candidate must have a paired baseline")
    paired["paired_delta_iae_pp"] = (
        paired["candidate_trajectory_iae_pp"]
        - paired["baseline_trajectory_iae_pp"]
    )
    paired["paired_delta_final_absolute_error_pp"] = (
        paired["candidate_final_absolute_error_pp"]
        - paired["baseline_final_absolute_error_pp"]
    )
    return paired.sort_values(
        ["scenario", "prefix_checkups", "candidate_method", "target_condition_id"],
        kind="stable",
    ).reset_index(drop=True)


def _bootstrap_seed(base_seed: int, scenario: str, prefix: int, method: str) -> int:
    payload = f"{base_seed}|{scenario}|{prefix}|{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _comparison_summary(
    paired: pd.DataFrame,
    *,
    diagnostics_config: Mapping[str, object],
) -> pd.DataFrame:
    resamples = int(diagnostics_config["bootstrap_resamples"])
    confidence = float(diagnostics_config["confidence_level"])
    base_seed = int(diagnostics_config["random_seed"])
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, object]] = []
    grouping = ["scenario", "prefix_checkups", "candidate_method"]
    for keys, group in paired.groupby(grouping, sort=True):
        scenario, prefix, method = keys
        ordered = group.sort_values("target_condition_id", kind="stable")
        delta = ordered["paired_delta_iae_pp"].to_numpy(dtype=float)
        rng = np.random.default_rng(
            _bootstrap_seed(base_seed, str(scenario), int(prefix), str(method))
        )
        indices = rng.integers(0, len(delta), size=(resamples, len(delta)))
        bootstrapped = delta[indices].mean(axis=1)
        worst = ordered.iloc[int(np.argmax(delta))]
        rows.append(
            {
                "scenario": str(scenario),
                "prefix_checkups": int(prefix),
                "candidate_method": str(method),
                "independent_condition_count": len(ordered),
                "shared_training_fold_count": int(ordered["fold_id"].nunique()),
                "candidate_trajectory_iae_pp_mean": float(
                    ordered["candidate_trajectory_iae_pp"].mean()
                ),
                "baseline_trajectory_iae_pp_mean": float(
                    ordered["baseline_trajectory_iae_pp"].mean()
                ),
                "mean_paired_delta_iae_pp": float(delta.mean()),
                "relative_iae_improvement_fraction": float(
                    1.0
                    - ordered["candidate_trajectory_iae_pp"].mean()
                    / ordered["baseline_trajectory_iae_pp"].mean()
                ),
                "bootstrap_mean_delta_ci_lower_pp": float(
                    np.quantile(bootstrapped, alpha)
                ),
                "bootstrap_mean_delta_ci_upper_pp": float(
                    np.quantile(bootstrapped, 1.0 - alpha)
                ),
                "candidate_better_condition_count": int(np.sum(delta < 0.0)),
                "candidate_worse_condition_count": int(np.sum(delta > 0.0)),
                "worst_condition_id": str(worst["target_condition_id"]),
                "worst_condition_delta_iae_pp": float(worst["paired_delta_iae_pp"]),
                "descriptive_superiority_criterion_met": bool(
                    np.quantile(bootstrapped, 1.0 - alpha) < 0.0
                ),
                "bootstrap_inference_role": "descriptive_resampling_only",
                "shared_fold_dependence_adjusted": False,
            }
        )
    return pd.DataFrame(rows).sort_values(grouping, kind="stable").reset_index(drop=True)


def run_calendar_v2_development(
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
    parsed = validate_calendar_v2_config(config)
    validate_naumann_calendar_observations(observations)
    all_conditions = set(observations["condition_id"].astype(str))
    model_config = parsed["model"]
    prediction_frames: list[pd.DataFrame] = []
    diagnostic_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for scenario in parsed["scenarios"]:
        scenario_name = str(scenario["name"])
        for fold_id, target_ids in _scenario_folds(observations, scenario):
            target_set = set(target_ids)
            training_ids = all_conditions - target_set
            if target_set & training_ids or target_set | training_ids != all_conditions:
                raise RuntimeError("Calendar V2 fold does not partition complete conditions")
            training_full = observations.loc[
                observations["condition_id"].astype(str).isin(training_ids)
            ].copy()
            target_fold = observations.loc[
                observations["condition_id"].astype(str).isin(target_set)
            ].copy()
            split_rows.extend(
                {
                    "scenario": scenario_name,
                    "fold_id": fold_id,
                    "condition_id": condition_id,
                    "split": "test" if condition_id in target_set else "train",
                    "training_history_policy": GLOBAL_LANDMARK_POLICY,
                }
                for condition_id in sorted(all_conditions)
            )
            for prefix in parsed["prefix_checkups"]:
                training_prefix = _select_prefix(training_full, int(prefix))
                prior = fit_hierarchical_power_prior(
                    training_prefix,
                    minimum_conditions=int(model_config["minimum_training_conditions"]),
                    exponent_bounds=tuple(
                        float(value) for value in model_config["power_exponent_bounds"]
                    ),
                    robust_loss_scale_pp=float(model_config["robust_loss_scale_pp"]),
                    stress_surface_ridge=float(model_config["stress_surface_ridge"]),
                    parameter_scale_floors=tuple(
                        float(value) for value in model_config["parameter_scale_floors"]
                    ),
                    observation_scale_floor_pp=float(
                        model_config["observation_scale_floor_pp"]
                    ),
                )
                training_hash, _ = _training_state(
                    scenario=scenario_name,
                    fold_id=fold_id,
                    prefix_checkups=int(prefix),
                    training_frame=training_prefix,
                    prior=prior,
                    model_config=model_config,
                )
                for parameter_index, parameter_name in enumerate(POWER_PARAMETER_NAMES):
                    for feature_index, feature_name in enumerate(STRESS_FEATURE_NAMES):
                        parameter_rows.append(
                            {
                                "scenario": scenario_name,
                                "fold_id": fold_id,
                                "prefix_checkups": int(prefix),
                                "parameter_group": "stress_surface",
                                "parameter": f"{parameter_name}:{feature_name}",
                                "value": float(
                                    prior.surface_coefficients[parameter_index][feature_index]
                                ),
                                "training_condition_count": len(training_ids),
                                "training_observation_count": len(training_prefix),
                                "training_max_checkup_index": int(prefix) - 1,
                                "training_support_days": prior.maximum_training_days,
                                "training_state_sha256": training_hash,
                            }
                        )
                    parameter_rows.append(
                        {
                            "scenario": scenario_name,
                            "fold_id": fold_id,
                            "prefix_checkups": int(prefix),
                            "parameter_group": "hierarchical_scale",
                            "parameter": f"{parameter_name}:scale",
                            "value": float(prior.parameter_scales[parameter_index]),
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_prefix),
                            "training_max_checkup_index": int(prefix) - 1,
                            "training_support_days": prior.maximum_training_days,
                            "training_state_sha256": training_hash,
                        }
                    )
                parameter_rows.append(
                    {
                        "scenario": scenario_name,
                        "fold_id": fold_id,
                        "prefix_checkups": int(prefix),
                        "parameter_group": "observation_scale",
                        "parameter": "capacity_loss_residual_scale_pp",
                        "value": float(prior.observation_scale_pp),
                        "training_condition_count": len(training_ids),
                        "training_observation_count": len(training_prefix),
                        "training_max_checkup_index": int(prefix) - 1,
                        "training_support_days": prior.maximum_training_days,
                        "training_state_sha256": training_hash,
                    }
                )
                for target_id in sorted(target_set):
                    target = target_fold.loc[
                        target_fold["condition_id"].astype(str) == target_id
                    ].copy()
                    predictions, diagnostics = _target_prediction_state(
                        scenario=scenario_name,
                        fold_id=fold_id,
                        target=target,
                        prefix_checkups=int(prefix),
                        training_state_sha256=training_hash,
                        prior=prior,
                        model_config=model_config,
                    )
                    prediction_frames.append(predictions)
                    diagnostic_rows.append(
                        {
                            **diagnostics,
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_prefix),
                            "training_max_checkup_index": int(prefix) - 1,
                            "training_support_days": prior.maximum_training_days,
                            "validation_horizon_days": float(target["elapsed_days"].max()),
                            "time_extrapolation_ratio": float(
                                target["elapsed_days"].max() / prior.maximum_training_days
                            ),
                        }
                    )

    predictions = pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS]
    predictions = predictions.sort_values(PREDICTION_KEY_COLUMNS, kind="stable").reset_index(
        drop=True
    )
    prediction_hash = calendar_v2_prediction_sha256(predictions)
    condition_metrics = score_calendar_v2_predictions(
        predictions,
        observations,
        frozen_prediction_sha256=prediction_hash,
    )
    paired = _paired_metrics(condition_metrics)
    comparisons = _comparison_summary(
        paired,
        diagnostics_config=parsed["diagnostics"],
    )
    primary_rows = comparisons.loc[
        (comparisons["prefix_checkups"] == PRIMARY_PREFIX)
        & (comparisons["candidate_method"] == HIERARCHICAL_POWER_METHOD)
        & (comparisons["scenario"].isin(GATE_SCENARIOS))
    ]
    all_scenarios_present = set(primary_rows["scenario"].astype(str)) == set(
        GATE_SCENARIOS
    )
    descriptive_signal = bool(
        all_scenarios_present
        and primary_rows["descriptive_superiority_criterion_met"].all()
    )
    diagnostics_frame = pd.DataFrame(diagnostic_rows).sort_values(
        ["scenario", "target_condition_id", "prefix_checkups"],
        kind="stable",
    ).reset_index(drop=True)
    parameters_frame = pd.DataFrame(parameter_rows).sort_values(
        ["scenario", "fold_id", "prefix_checkups", "parameter_group", "parameter"],
        kind="stable",
    ).reset_index(drop=True)
    splits_frame = pd.DataFrame(split_rows).drop_duplicates().sort_values(
        ["scenario", "fold_id", "condition_id"],
        kind="stable",
    ).reset_index(drop=True)

    result: dict[str, object] = {
        "status": "development_diagnostic_complete_confirmation_blocked",
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
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
            "prefix_checkups": list(EXPECTED_PREFIXES),
            "primary_prefix_checkups": PRIMARY_PREFIX,
            "methods": list(METHOD_NAMES),
            "baseline_method": TARGET_SQRT_METHOD,
            "primary_candidate": HIERARCHICAL_POWER_METHOD,
            "target_future_outcomes_used_for_fit": False,
            "training_future_outcomes_used_for_fit": False,
            "condition_bootstrap_is_descriptive_only": True,
            "shared_fold_dependence_adjusted": False,
        },
        "comparison_summary": comparisons.to_dict(orient="records"),
        "development_gate": {
            "confirmation_status": CONFIRMATION_STATUS,
            "descriptive_signal_status": (
                "passed_on_reused_development_data"
                if descriptive_signal
                else "failed_on_reused_development_data"
            ),
            "all_primary_scenarios_present": all_scenarios_present,
            "all_primary_descriptive_criteria_met": descriptive_signal,
            "current_dataset_relationship": "reused_and_outcomes_already_inspected",
            "reason": (
                "A development signal cannot become confirmation on reused Naumann "
                "outcomes. Independent data require a new dataset-specific protocol."
            ),
        },
        "future_label_firewall": {
            "label_free_prediction_sha256": prediction_hash,
            "diagnostics_sha256": _canonical_frame_sha256(
                diagnostics_frame,
                sort_by=["scenario", "target_condition_id", "prefix_checkups"],
            ),
            "parameters_sha256": _canonical_frame_sha256(
                parameters_frame,
                sort_by=[
                    "scenario",
                    "fold_id",
                    "prefix_checkups",
                    "parameter_group",
                    "parameter",
                ],
            ),
            "score_after_prediction_hash_verification": True,
            "future_outcome_columns_in_prediction_pack": [],
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
        "claim_boundary": (
            "This is a post-hoc development bakeoff on 17 reused Naumann condition-"
            "mean trajectories. It does not validate individual cells, Hithium "
            "products, utility-scale storage, or 15-25 year forecasts."
        ),
    }
    return (
        result,
        predictions,
        condition_metrics,
        paired,
        comparisons,
        diagnostics_frame,
        parameters_frame,
        splits_frame,
    )
