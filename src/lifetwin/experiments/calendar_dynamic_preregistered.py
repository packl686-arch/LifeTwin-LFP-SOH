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
from lifetwin.models.calendar import (
    CALENDAR_MODEL_NAME,
    EmpiricalStressSurface,
    estimate_empirical_bayes_ridge,
    estimate_prefix_only_k,
    estimate_target_scale,
    fit_empirical_stress_surface,
    predict_stress_surface_loss,
    select_prefix,
)


EXPERIMENT_ID = "naumann_calendar_dynamic_update_preregistered_v1"
DESIGN_STATUS = "frozen_future_replication_design_with_naumann_diagnostic"
EVIDENCE_ROLE = "reused_phase41_development_dataset_diagnostic_only"
RUNNER_SCOPE = "naumann_reuse_development_diagnostic_only"
INDEPENDENT_RUNNER_STATUS = (
    "not_implemented_requires_new_dataset_specific_preregistration"
)
GLOBAL_LANDMARK_POLICY = "global_landmark_prefix"
DYNAMIC_METHOD_NAME = "empirical_stress_surface_prefix_update_v1"
COMPARATOR_METHOD_NAME = "target_prefix_only_sqrt_time_v1"
ADAPTATION_METHOD = "training_only_empirical_bayes_scale_update"
CONFIRMATION_BLOCK_STATUS = "blocked_pending_independent_dataset"
EXPECTED_DATASET_SNAPSHOT_ID = (
    "celljar_5c9601a027751c84ee8346f7c0ab9c6851330202_"
    "cycle_summary_6d4fd8d102ce9605c2c9bff86662b03201ae85411af583e23f3315b52bf81bc5"
)
EXPECTED_LABEL_VERSION = "published_condition_mean_capacity_retention_v1"
EXPECTED_PREFIXES = [5, 8, 10, 14]
PRIMARY_PREFIX = 10
TEMPERATURE_SCENARIO = "dynamic_preregistered_unseen_temperature_level"
SOC_SCENARIO = "dynamic_preregistered_soc_interpolation_at_40c"
GATE_SCENARIOS = [TEMPERATURE_SCENARIO, SOC_SCENARIO]
SOC_TARGET_CONDITIONS = (
    "NAUMANN_CAL_T40_SOC12.5",
    "NAUMANN_CAL_T40_SOC37.5",
    "NAUMANN_CAL_T40_SOC62.5",
    "NAUMANN_CAL_T40_SOC87.5",
)
EXPECTED_PROHIBITED_CLAIMS = [
    "validated_dynamic_update",
    "confirmatory_superiority_on_naumann_reuse",
    "individual_cell_variability",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
    "this_naumann_runner_can_confirm_on_an_independent_dataset",
]
METHOD_NAMES = (DYNAMIC_METHOD_NAME, COMPARATOR_METHOD_NAME)
SIGN_ZERO_TOLERANCE_PP = 1e-12

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
    "elapsed_hours",
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
    "independent_confirmation_runner_status",
    "scenarios",
    "prefix_checkups",
    "primary_prefix_checkups",
    "training_history_policy",
    "model",
    "dynamic_method",
    "comparator",
    "estimand",
    "diagnostics",
    "gate_scenarios",
    "confirmation_gate",
    "effective_independent_condition_count",
    "maximum_supported_horizon_days",
    "allow_projection_beyond_observed_horizon",
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
            f"{context} configuration keys do not match the preregistration: "
            f"missing={missing}, unknown={unknown}"
        )


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return dict(value)


def _require_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _require_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
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
    if frame.empty:
        raise ValueError("Cannot hash an empty preregistered result table")
    if sorted(set(sort_by) - set(frame.columns)):
        raise ValueError("Canonical result hash is missing sort columns")
    normalized = frame.sort_values(sort_by, kind="stable").reset_index(drop=True)
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_dynamic_preregistered_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    """Fail closed on any method, estimand, diagnostic, or Gate drift."""
    normalized = _require_mapping(config, context="Dynamic calendar preregistration")
    _require_exact_keys(
        normalized,
        TOP_LEVEL_CONFIG_KEYS,
        context="Dynamic calendar preregistration",
    )
    frozen_strings = {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "evidence_role": EVIDENCE_ROLE,
        "runner_scope": RUNNER_SCOPE,
        "independent_confirmation_runner_status": INDEPENDENT_RUNNER_STATUS,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
    }
    for key, expected in frozen_strings.items():
        if normalized[key] != expected:
            raise ValueError(f"Unsupported preregistered {key}: {normalized[key]}")
    if normalized["dataset_snapshot_id"] != EXPECTED_DATASET_SNAPSHOT_ID:
        raise ValueError("dataset_snapshot_id must remain the frozen Naumann snapshot")
    if normalized["label_version"] != EXPECTED_LABEL_VERSION:
        raise ValueError("label_version must remain the frozen published definition")

    raw_prefixes = normalized["prefix_checkups"]
    if not isinstance(raw_prefixes, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_prefixes
    ):
        raise ValueError("prefix_checkups must be an integer list")
    if raw_prefixes != EXPECTED_PREFIXES:
        raise ValueError(
            f"Preregistered diagnostic prefixes must remain {EXPECTED_PREFIXES}"
        )
    primary_prefix = _require_int(
        normalized["primary_prefix_checkups"],
        context="primary_prefix_checkups",
    )
    if primary_prefix != PRIMARY_PREFIX:
        raise ValueError(f"Primary prefix must remain fixed at {PRIMARY_PREFIX}")

    raw_scenarios = normalized["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty list")
    scenarios = [
        _require_mapping(value, context=f"scenarios[{index}]")
        for index, value in enumerate(raw_scenarios)
    ]
    scenario_names = [str(value.get("name", "")) for value in scenarios]
    if scenario_names != GATE_SCENARIOS:
        raise ValueError(
            "Preregistered scenarios and order must remain "
            f"{GATE_SCENARIOS}"
        )
    for scenario in scenarios:
        name = str(scenario["name"])
        if name == TEMPERATURE_SCENARIO:
            expected_keys = {
                "name",
                "kind",
                "training_history_policy",
                "comparison_family",
                "evidence_role",
            }
            _require_exact_keys(scenario, expected_keys, context=f"Scenario {name}")
            expected_values = {
                "kind": "leave_one_temperature_level_out",
                "training_history_policy": GLOBAL_LANDMARK_POLICY,
                "comparison_family": "unseen_temperature_level",
                "evidence_role": "primary_time_honest_temperature_generalization",
            }
        else:
            expected_keys = {
                "name",
                "kind",
                "training_history_policy",
                "comparison_family",
                "fold_id",
                "target_condition_ids",
                "evidence_role",
            }
            _require_exact_keys(scenario, expected_keys, context=f"Scenario {name}")
            expected_values = {
                "kind": "fixed_condition_holdout",
                "training_history_policy": GLOBAL_LANDMARK_POLICY,
                "comparison_family": "soc_interpolation_at_40c",
                "fold_id": "40c_intermediate_soc_dynamic_preregistered",
                "evidence_role": "primary_time_honest_soc_interpolation",
            }
            target_ids = scenario["target_condition_ids"]
            if not isinstance(target_ids, list) or tuple(target_ids) != SOC_TARGET_CONDITIONS:
                raise ValueError(
                    "The preregistered 40 C target conditions cannot be changed"
                )
        for key, expected in expected_values.items():
            if scenario[key] != expected:
                raise ValueError(f"Scenario {name} has unsupported {key}: {scenario[key]}")

    model = _require_mapping(normalized["model"], context="model")
    _require_exact_keys(
        model,
        {
            "name",
            "time_unit",
            "time_law",
            "minimum_training_conditions",
            "robust_loss_scale_pp",
        },
        context="model",
    )
    if model["name"] != CALENDAR_MODEL_NAME:
        raise ValueError(f"Unsupported model name: {model['name']}")
    if model["time_unit"] != "hour":
        raise ValueError(f"Unsupported model time unit: {model['time_unit']}")
    if model["time_law"] != "fixed_square_root":
        raise ValueError(f"Unsupported model time law: {model['time_law']}")
    minimum_conditions = _require_int(
        model["minimum_training_conditions"],
        context="minimum_training_conditions",
    )
    if minimum_conditions != 6:
        raise ValueError("minimum_training_conditions must remain frozen at six")
    robust_scale = _require_float(
        model["robust_loss_scale_pp"],
        context="robust_loss_scale_pp",
    )
    if robust_scale != 0.25:
        raise ValueError("robust_loss_scale_pp must remain frozen at 0.25")

    dynamic = _require_mapping(normalized["dynamic_method"], context="dynamic_method")
    _require_exact_keys(
        dynamic,
        {
            "name",
            "adaptation_method",
            "ridge_training_scope",
            "target_update_scope",
            "ridge_min",
            "ridge_max",
            "scale_min",
            "scale_max",
        },
        context="dynamic_method",
    )
    expected_dynamic = {
        "name": DYNAMIC_METHOD_NAME,
        "adaptation_method": ADAPTATION_METHOD,
        "ridge_training_scope": "training_conditions_prefix_only",
        "target_update_scope": "target_condition_prefix_only",
    }
    for key, expected in expected_dynamic.items():
        if dynamic[key] != expected:
            raise ValueError(f"Unsupported dynamic method {key}: {dynamic[key]}")
    ridge_min = _require_float(dynamic["ridge_min"], context="ridge_min")
    ridge_max = _require_float(dynamic["ridge_max"], context="ridge_max")
    scale_min = _require_float(dynamic["scale_min"], context="scale_min")
    scale_max = _require_float(dynamic["scale_max"], context="scale_max")
    if (ridge_min, ridge_max, scale_min, scale_max) != (0.01, 100.0, 0.0, 4.0):
        raise ValueError("Dynamic ridge and scale bounds are frozen by preregistration")

    comparator = _require_mapping(normalized["comparator"], context="comparator")
    _require_exact_keys(
        comparator,
        {"name", "fit_scope", "time_unit", "time_law", "nonnegative_rate"},
        context="comparator",
    )
    expected_comparator = {
        "name": COMPARATOR_METHOD_NAME,
        "fit_scope": "target_condition_prefix_only",
        "time_unit": "hour",
        "time_law": "fixed_square_root",
        "nonnegative_rate": True,
    }
    if comparator != expected_comparator:
        raise ValueError("Comparator must remain the target-prefix-only sqrt(time) model")

    estimand = _require_mapping(normalized["estimand"], context="estimand")
    _require_exact_keys(
        estimand,
        {
            "statistical_unit",
            "paired_delta_definition",
            "primary_summary",
            "lower_is_better",
        },
        context="estimand",
    )
    expected_estimand = {
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "paired_delta_definition": (
            "dynamic_trajectory_iae_pp_minus_target_prefix_only_trajectory_iae_pp"
        ),
        "primary_summary": "mean_paired_condition_delta_iae_pp",
        "lower_is_better": True,
    }
    if estimand != expected_estimand:
        raise ValueError("The paired condition-level estimand cannot be changed")

    diagnostics = _require_mapping(normalized["diagnostics"], context="diagnostics")
    _require_exact_keys(
        diagnostics,
        {
            "bootstrap_unit",
            "bootstrap_resamples",
            "random_seed",
            "confidence_level",
            "exact_sign_test",
            "worst_condition_rule",
            "inference_role",
            "shared_fold_dependence_adjusted",
        },
        context="diagnostics",
    )
    expected_diagnostics = {
        "bootstrap_unit": "condition",
        "bootstrap_resamples": 10_000,
        "random_seed": 20_260_719,
        "confidence_level": 0.95,
        "exact_sign_test": True,
        "worst_condition_rule": "maximum_paired_delta_iae_pp",
        "inference_role": "descriptive_resampling_only",
        "shared_fold_dependence_adjusted": False,
    }
    if diagnostics != expected_diagnostics:
        raise ValueError(
            "Diagnostic parameters and their descriptive-only role are frozen"
        )

    gate_scenarios = normalized["gate_scenarios"]
    if not isinstance(gate_scenarios, list) or not gate_scenarios:
        raise ValueError("gate_scenarios must contain at least one scenario")
    if any(not isinstance(value, str) or not value for value in gate_scenarios):
        raise ValueError("Gate scenario names must be non-empty strings")
    if len(gate_scenarios) != len(set(gate_scenarios)):
        raise ValueError("gate_scenarios must be unique")
    if gate_scenarios != GATE_SCENARIOS:
        raise ValueError(f"Preregistered Gate scenarios must remain {GATE_SCENARIOS}")

    gate = _require_mapping(normalized["confirmation_gate"], context="confirmation_gate")
    _require_exact_keys(
        gate,
        {
            "decision_rule",
            "superiority_margin_delta_iae_pp",
            "required_dataset_role",
            "current_dataset_relationship",
            "status_if_current_dataset",
        },
        context="confirmation_gate",
    )
    expected_gate_strings = {
        "decision_rule": "all_scenarios_bootstrap_upper_ci_below_margin",
        "required_dataset_role": "independent_external_replication",
        "current_dataset_relationship": (
            "reused_phase41_naumann_17_condition_development_dataset"
        ),
        "status_if_current_dataset": CONFIRMATION_BLOCK_STATUS,
    }
    for key, expected in expected_gate_strings.items():
        if gate[key] != expected:
            raise ValueError(f"Unsupported confirmation Gate {key}: {gate[key]}")
    if _require_float(
        gate["superiority_margin_delta_iae_pp"],
        context="superiority_margin_delta_iae_pp",
    ) != 0.0:
        raise ValueError("The superiority margin must remain zero pp")

    condition_count = _require_int(
        normalized["effective_independent_condition_count"],
        context="effective_independent_condition_count",
    )
    if condition_count != 17:
        raise ValueError("Naumann effective independent condition count must remain 17")
    horizon = _require_float(
        normalized["maximum_supported_horizon_days"],
        context="maximum_supported_horizon_days",
    )
    if not np.isclose(horizon, 885.0416666666666, rtol=0.0, atol=1e-12):
        raise ValueError("The public maximum supported horizon cannot be changed")
    if normalized["allow_projection_beyond_observed_horizon"] is not False:
        raise ValueError("Projection beyond observed horizon must remain prohibited")
    if normalized["prohibited_claims"] != EXPECTED_PROHIBITED_CLAIMS:
        raise ValueError("The preregistered prohibited claims cannot be changed")

    return normalized


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
        target_ids = tuple(str(value) for value in scenario["target_condition_ids"])
        known = set(profile["condition_id"].astype(str))
        missing = sorted(set(target_ids) - known)
        if missing:
            raise ValueError(f"Scenario references unknown conditions: {missing}")
        return [(str(scenario["fold_id"]), target_ids)]
    raise ValueError(f"Unsupported preregistered scenario kind: {kind}")


def dynamic_prediction_artifact_sha256(predictions: pd.DataFrame) -> str:
    """Hash the exact, canonical, outcome-free dynamic prediction pack."""
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Dynamic prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty:
        raise ValueError("Dynamic prediction pack cannot be empty")
    if predictions[PREDICTION_KEY_COLUMNS].isna().any().any():
        raise ValueError("Dynamic prediction keys cannot be null")
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Dynamic prediction keys must be unique")
    if set(predictions["method"].astype(str)) != set(METHOD_NAMES):
        raise ValueError("Dynamic prediction pack must contain both frozen methods")
    for column in ("training_state_sha256", "prediction_state_sha256"):
        valid = predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}")
        if not valid.all():
            raise ValueError(f"{column} must contain canonical SHA-256 values")
    return _canonical_frame_sha256(predictions[PREDICTION_COLUMNS], sort_by=PREDICTION_KEY_COLUMNS)


def _training_state(
    *,
    scenario_name: str,
    fold_id: str,
    prefix: int,
    training_prefix: pd.DataFrame,
    model: EmpiricalStressSurface,
    ridge_state: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    input_columns = [
        "condition_id",
        "checkup_index",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_hours",
        "capacity_loss_pct",
    ]
    ordered = training_prefix.sort_values(
        ["condition_id", "checkup_index"],
        kind="stable",
    )
    payload: dict[str, object] = {
        "experiment_id": config["experiment_id"],
        "scenario": scenario_name,
        "fold_id": fold_id,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "prefix_checkups": int(prefix),
        "training_prefix_rows": [
            {
                column: (
                    int(getattr(row, column))
                    if column == "checkup_index"
                    else str(getattr(row, column))
                    if column == "condition_id"
                    else float(getattr(row, column))
                )
                for column in input_columns
            }
            for row in ordered[input_columns].itertuples(index=False)
        ],
        "model_parameters": [float(value) for value in model.parameters],
        "training_condition_ids": list(model.training_condition_ids),
        "maximum_training_hours": float(model.maximum_training_hours),
        "maximum_supported_hours": float(model.maximum_supported_hours),
        "ridge_state": {
            "ridge": float(ridge_state["ridge"]),
            "raw_ridge": float(ridge_state["raw_ridge"]),
            "residual_variance_pp2": float(ridge_state["residual_variance_pp2"]),
            "between_condition_scale_variance": float(
                ridge_state["between_condition_scale_variance"]
            ),
            "training_scale_count": int(ridge_state["training_scale_count"]),
            "training_scales": [
                float(value) for value in ridge_state["training_scales"]
            ],
        },
        "model_config": config["model"],
        "dynamic_method_config": config["dynamic_method"],
    }
    return _canonical_json_sha256(payload), payload


def _target_predictions(
    *,
    scenario_name: str,
    fold_id: str,
    target_frame: pd.DataFrame,
    model: EmpiricalStressSurface,
    ridge_state: Mapping[str, object],
    training_state_sha256: str,
    prefix: int,
    config: Mapping[str, object],
    training_support_days: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    target_ids = target_frame["condition_id"].astype(str).unique()
    if len(target_ids) != 1:
        raise ValueError("Each target prediction state must contain one condition")
    target_id = str(target_ids[0])
    target = target_frame.sort_values("checkup_index", kind="stable").reset_index(drop=True)
    if target["checkup_index"].duplicated().any():
        raise ValueError("Target condition contains duplicate checkup indices")
    prefix_frame = select_prefix(target, prefix).sort_values(
        "checkup_index",
        kind="stable",
    )
    future = target.loc[
        pd.to_numeric(target["checkup_index"]) >= prefix,
        [
            "condition_id",
            "temperature_c",
            "storage_soc_fraction",
            "elapsed_hours",
            "elapsed_days",
            "checkup_index",
        ],
    ].sort_values("checkup_index", kind="stable")
    if len(future) < 2:
        raise ValueError("Every prefix must leave at least two future checkups")

    dynamic = dict(config["dynamic_method"])
    target_scale = estimate_target_scale(
        model,
        prefix_frame,
        ridge=float(ridge_state["ridge"]),
        scale_bounds=(float(dynamic["scale_min"]), float(dynamic["scale_max"])),
    )
    prefix_only_k = estimate_prefix_only_k(prefix_frame)
    base_loss = predict_stress_surface_loss(model, future)
    elapsed_hours = future["elapsed_hours"].to_numpy(dtype=float)
    predictions_by_method = {
        DYNAMIC_METHOD_NAME: 100.0 - target_scale * base_loss,
        COMPARATOR_METHOD_NAME: 100.0 - prefix_only_k * np.sqrt(elapsed_hours),
    }
    prefix_end = prefix_frame.iloc[-1]
    validation_horizon_days = float(future["elapsed_days"].max())
    target_prefix_columns = [
        "checkup_index",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_hours",
        "capacity_loss_pct",
    ]
    state_payload = {
        "experiment_id": config["experiment_id"],
        "scenario": scenario_name,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "prefix_checkups": int(prefix),
        "training_state_sha256": training_state_sha256,
        "target_prefix_rows": [
            {
                column: (
                    int(getattr(row, column))
                    if column == "checkup_index"
                    else float(getattr(row, column))
                )
                for column in target_prefix_columns
            }
            for row in prefix_frame[target_prefix_columns].itertuples(index=False)
        ],
        "future_coordinates": [
            {
                "checkup_index": int(row.checkup_index),
                "elapsed_hours": float(row.elapsed_hours),
                "temperature_c": float(row.temperature_c),
                "storage_soc_fraction": float(row.storage_soc_fraction),
            }
            for row in future.itertuples(index=False)
        ],
        "ridge": float(ridge_state["ridge"]),
        "target_scale": float(target_scale),
        "prefix_only_k_pct_per_sqrt_hour": float(prefix_only_k),
        "predictions": {
            method: [float(value) for value in predictions_by_method[method]]
            for method in METHOD_NAMES
        },
    }
    state_hash = _canonical_json_sha256(state_payload)
    rows: list[dict[str, object]] = []
    final_checkup_index = int(target["checkup_index"].max())
    for method in METHOD_NAMES:
        for coordinate, predicted in zip(
            future.itertuples(index=False),
            predictions_by_method[method],
            strict=True,
        ):
            rows.append(
                {
                    "scenario": scenario_name,
                    "training_history_policy": GLOBAL_LANDMARK_POLICY,
                    "fold_id": fold_id,
                    "target_condition_id": target_id,
                    "prefix_checkups": int(prefix),
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
                    "prefix_end_days": float(prefix_end["elapsed_days"]),
                    "elapsed_hours": float(coordinate.elapsed_hours),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(predicted),
                    "is_final_checkup": bool(
                        int(coordinate.checkup_index) == final_checkup_index
                    ),
                    "training_support_days": float(training_support_days),
                    "validation_horizon_days": validation_horizon_days,
                    "time_extrapolation_ratio": (
                        validation_horizon_days / float(training_support_days)
                    ),
                    "training_state_sha256": training_state_sha256,
                    "prediction_state_sha256": state_hash,
                }
            )
    diagnostics = {
        "scenario": scenario_name,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(prefix),
        "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
        "prefix_end_days": float(prefix_end["elapsed_days"]),
        "future_checkup_count": len(future),
        "target_scale": float(target_scale),
        "prefix_only_k_pct_per_sqrt_hour": float(prefix_only_k),
        "ridge": float(ridge_state["ridge"]),
        "training_support_days": float(training_support_days),
        "validation_horizon_days": validation_horizon_days,
        "time_extrapolation_ratio": validation_horizon_days / training_support_days,
        "training_state_sha256": training_state_sha256,
        "prediction_state_sha256": state_hash,
    }
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS), diagnostics


def score_dynamic_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    """Verify the outcome-free pack before joining any future labels."""
    observed_hash = dynamic_prediction_artifact_sha256(predictions)
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Frozen dynamic prediction hash does not match prediction content")
    outcomes = observations[
        ["condition_id", "checkup_index", "capacity_retention_pct"]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    if outcomes.duplicated(["target_condition_id", "target_checkup_index"]).any():
        raise ValueError("Scoring outcomes must be unique by condition/checkup")
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
        raise ValueError("Every dynamic prediction must match one future outcome")
    scored = scored.drop(columns="_merge")
    scored["prediction_error_pp"] = (
        scored["predicted_capacity_retention_pct"]
        - scored["true_capacity_retention_pct"]
    )

    grouping = [
        "scenario",
        "training_history_policy",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_days",
        "method",
        "training_state_sha256",
        "prediction_state_sha256",
    ]
    rows: list[dict[str, object]] = []
    for keys, condition in scored.groupby(grouping, sort=True):
        ordered = condition.sort_values("elapsed_hours", kind="stable")
        elapsed = ordered["elapsed_hours"].to_numpy(dtype=float)
        absolute_error = np.abs(ordered["prediction_error_pp"].to_numpy(dtype=float))
        if len(elapsed) < 2 or elapsed[-1] <= elapsed[0]:
            raise ValueError("Trajectory IAE requires at least two future checkups")
        final = ordered.loc[ordered["is_final_checkup"]]
        if len(final) != 1:
            raise ValueError("Each condition/method must contain one final checkup")
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
    condition_metrics = pd.DataFrame(rows)
    return condition_metrics.sort_values(
        ["scenario", "target_condition_id", "prefix_checkups", "method"],
        kind="stable",
    ).reset_index(drop=True)


def _paired_condition_metrics(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scenario",
        "training_history_policy",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_days",
    ]
    dynamic = condition_metrics.loc[
        condition_metrics["method"] == DYNAMIC_METHOD_NAME,
        [
            *keys,
            "trajectory_iae_pp",
            "final_absolute_error_pp",
            "training_state_sha256",
            "prediction_state_sha256",
        ],
    ].rename(
        columns={
            "trajectory_iae_pp": "dynamic_trajectory_iae_pp",
            "final_absolute_error_pp": "dynamic_final_absolute_error_pp",
            "training_state_sha256": "dynamic_training_state_sha256",
            "prediction_state_sha256": "dynamic_prediction_state_sha256",
        }
    )
    comparator = condition_metrics.loc[
        condition_metrics["method"] == COMPARATOR_METHOD_NAME,
        [
            *keys,
            "trajectory_iae_pp",
            "final_absolute_error_pp",
            "training_state_sha256",
            "prediction_state_sha256",
        ],
    ].rename(
        columns={
            "trajectory_iae_pp": "comparator_trajectory_iae_pp",
            "final_absolute_error_pp": "comparator_final_absolute_error_pp",
            "training_state_sha256": "comparator_training_state_sha256",
            "prediction_state_sha256": "comparator_prediction_state_sha256",
        }
    )
    paired = dynamic.merge(comparator, on=keys, how="outer", validate="one_to_one", indicator=True)
    if (paired["_merge"] != "both").any():
        raise ValueError("Every condition must have both dynamic and comparator metrics")
    paired = paired.drop(columns="_merge")
    if not (
        paired["dynamic_training_state_sha256"]
        == paired["comparator_training_state_sha256"]
    ).all():
        raise RuntimeError("Paired methods did not share a training state")
    if not (
        paired["dynamic_prediction_state_sha256"]
        == paired["comparator_prediction_state_sha256"]
    ).all():
        raise RuntimeError("Paired methods did not share a target prediction state")
    paired["paired_delta_iae_pp"] = (
        paired["dynamic_trajectory_iae_pp"]
        - paired["comparator_trajectory_iae_pp"]
    )
    denominator = paired["comparator_trajectory_iae_pp"].to_numpy(dtype=float)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise ValueError("Comparator condition IAE must be finite and positive")
    paired["dynamic_relative_improvement_fraction"] = (
        1.0
        - paired["dynamic_trajectory_iae_pp"].to_numpy(dtype=float) / denominator
    )
    return paired.sort_values(
        ["scenario", "prefix_checkups", "target_condition_id"],
        kind="stable",
    ).reset_index(drop=True)


def _group_bootstrap_seed(base_seed: int, scenario: str, prefix: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{scenario}:{prefix}".encode()).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _binomial_upper_tail(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    numerator = sum(math.comb(trials, value) for value in range(successes, trials + 1))
    return float(numerator / (2**trials))


def _exact_sign_diagnostics(delta: np.ndarray) -> dict[str, object]:
    better = int(np.sum(delta < -SIGN_ZERO_TOLERANCE_PP))
    worse = int(np.sum(delta > SIGN_ZERO_TOLERANCE_PP))
    ties = int(len(delta) - better - worse)
    trials = better + worse
    superiority_p = _binomial_upper_tail(better, trials)
    if trials == 0:
        two_sided_p = 1.0
    else:
        opposite_tail = _binomial_upper_tail(worse, trials)
        two_sided_p = min(1.0, 2.0 * min(superiority_p, opposite_tail))
    return {
        "dynamic_better_condition_count": better,
        "dynamic_worse_condition_count": worse,
        "exact_tie_condition_count": ties,
        "exact_sign_effective_condition_count": trials,
        "exact_sign_superiority_p_value": superiority_p,
        "exact_sign_two_sided_p_value": two_sided_p,
        "sign_zero_tolerance_pp": SIGN_ZERO_TOLERANCE_PP,
    }


def _comparison_summary(
    paired: pd.DataFrame,
    *,
    diagnostics_config: Mapping[str, object],
    superiority_margin: float,
) -> pd.DataFrame:
    resamples = int(diagnostics_config["bootstrap_resamples"])
    confidence = float(diagnostics_config["confidence_level"])
    base_seed = int(diagnostics_config["random_seed"])
    alpha = 1.0 - confidence
    rows: list[dict[str, object]] = []
    for (scenario, prefix), group in paired.groupby(
        ["scenario", "prefix_checkups"],
        sort=True,
    ):
        ordered = group.sort_values("target_condition_id", kind="stable")
        if ordered["target_condition_id"].duplicated().any():
            raise ValueError("Condition bootstrap cannot include repeated condition units")
        delta = ordered["paired_delta_iae_pp"].to_numpy(dtype=float)
        if len(delta) == 0 or not np.isfinite(delta).all():
            raise ValueError("Paired condition deltas must be finite and non-empty")
        bootstrap_seed = _group_bootstrap_seed(base_seed, str(scenario), int(prefix))
        rng = np.random.default_rng(bootstrap_seed)
        sampled_means = rng.choice(
            delta,
            size=(resamples, len(delta)),
            replace=True,
        ).mean(axis=1)
        lower, upper = np.quantile(
            sampled_means,
            [alpha / 2.0, 1.0 - alpha / 2.0],
            method="linear",
        )
        worst = ordered.sort_values(
            ["paired_delta_iae_pp", "target_condition_id"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        dynamic_mean = float(ordered["dynamic_trajectory_iae_pp"].mean())
        comparator_mean = float(ordered["comparator_trajectory_iae_pp"].mean())
        rows.append(
            {
                "scenario": str(scenario),
                "prefix_checkups": int(prefix),
                "independent_condition_count": len(ordered),
                "shared_training_fold_count": int(ordered["fold_id"].nunique()),
                "dynamic_trajectory_iae_pp_mean": dynamic_mean,
                "comparator_trajectory_iae_pp_mean": comparator_mean,
                "mean_paired_delta_iae_pp": float(delta.mean()),
                "median_paired_delta_iae_pp": float(np.median(delta)),
                "dynamic_relative_improvement_fraction": (
                    1.0 - dynamic_mean / comparator_mean
                ),
                "bootstrap_unit": "condition",
                "bootstrap_resamples": resamples,
                "bootstrap_seed": bootstrap_seed,
                "bootstrap_confidence_level": confidence,
                "bootstrap_inference_role": diagnostics_config["inference_role"],
                "shared_fold_dependence_adjusted": diagnostics_config[
                    "shared_fold_dependence_adjusted"
                ],
                "bootstrap_mean_delta_ci_lower_pp": float(lower),
                "bootstrap_mean_delta_ci_upper_pp": float(upper),
                "worst_condition_id": str(worst["target_condition_id"]),
                "worst_condition_paired_delta_iae_pp": float(
                    worst["paired_delta_iae_pp"]
                ),
                "worst_condition_dynamic_iae_pp": float(
                    worst["dynamic_trajectory_iae_pp"]
                ),
                "worst_condition_comparator_iae_pp": float(
                    worst["comparator_trajectory_iae_pp"]
                ),
                **_exact_sign_diagnostics(delta),
                "diagnostic_superiority_criterion_met": bool(
                    float(upper) < superiority_margin
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "prefix_checkups"],
        kind="stable",
    ).reset_index(drop=True)


def _confirmation_gate(
    comparison_summary: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> dict[str, object]:
    gate_scenarios = [str(value) for value in config["gate_scenarios"]]
    if not gate_scenarios:
        raise ValueError("gate_scenarios must be non-empty")
    primary_prefix = int(config["primary_prefix_checkups"])
    primary = comparison_summary.loc[
        (comparison_summary["prefix_checkups"] == primary_prefix)
        & comparison_summary["scenario"].isin(gate_scenarios)
    ].copy()
    counts = primary.groupby("scenario", sort=True).size().to_dict()
    missing_or_duplicate = {
        scenario: int(counts.get(scenario, 0))
        for scenario in gate_scenarios
        if int(counts.get(scenario, 0)) != 1
    }
    if missing_or_duplicate:
        raise ValueError(
            "Every configured Gate scenario must execute exactly once at the primary "
            f"prefix: {missing_or_duplicate}"
        )
    primary = primary.set_index("scenario").loc[gate_scenarios].reset_index()
    scenario_diagnostics = [
        {
            "scenario": str(row.scenario),
            "independent_condition_count": int(row.independent_condition_count),
            "mean_paired_delta_iae_pp": float(row.mean_paired_delta_iae_pp),
            "bootstrap_mean_delta_ci_lower_pp": float(
                row.bootstrap_mean_delta_ci_lower_pp
            ),
            "bootstrap_mean_delta_ci_upper_pp": float(
                row.bootstrap_mean_delta_ci_upper_pp
            ),
            "worst_condition_id": str(row.worst_condition_id),
            "worst_condition_paired_delta_iae_pp": float(
                row.worst_condition_paired_delta_iae_pp
            ),
            "exact_sign_superiority_p_value": float(
                row.exact_sign_superiority_p_value
            ),
            "diagnostic_superiority_criterion_met": bool(
                row.diagnostic_superiority_criterion_met
            ),
        }
        for row in primary.itertuples(index=False)
    ]
    return {
        "status": CONFIRMATION_BLOCK_STATUS,
        "primary_prefix_checkups": primary_prefix,
        "gate_scenarios": gate_scenarios,
        "all_configured_gates_executed": True,
        "decision_rule": config["confirmation_gate"]["decision_rule"],
        "superiority_margin_delta_iae_pp": float(
            config["confirmation_gate"]["superiority_margin_delta_iae_pp"]
        ),
        "all_diagnostic_superiority_criteria_met": all(
            row["diagnostic_superiority_criterion_met"]
            for row in scenario_diagnostics
        ),
        "scenario_diagnostics": scenario_diagnostics,
        "required_dataset_role": config["confirmation_gate"][
            "required_dataset_role"
        ],
        "current_dataset_relationship": config["confirmation_gate"][
            "current_dataset_relationship"
        ],
        "reason": (
            "These 17 Naumann condition-mean trajectories were already used during "
            "Phase 4.1 method development. Their reuse is diagnostic only; an "
            "independent dataset and a separately frozen dataset-specific runner are "
            "required before a confirmatory decision."
        ),
    }


def run_calendar_dynamic_preregistered(
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
]:
    """Run the separate dynamic global-landmark preregistered comparison."""
    validated = validate_dynamic_preregistered_config(config)
    validate_naumann_calendar_observations(observations)
    if set(observations["dataset_id"].astype(str)) != {validated["dataset_id"]}:
        raise ValueError("Observed dataset id does not match the preregistration")
    condition_count = int(observations["condition_id"].nunique())
    if condition_count != int(validated["effective_independent_condition_count"]):
        raise ValueError("Observed independent condition count does not match config")
    if set(observations["statistical_unit"].astype(str)) != {
        validated["estimand"]["statistical_unit"]
    }:
        raise ValueError("Observed statistical unit does not match the estimand")
    maximum_days = float(observations["elapsed_days"].max())
    if not np.isclose(
        maximum_days,
        float(validated["maximum_supported_horizon_days"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Observed horizon does not match the frozen public support")

    ordered_observations = observations.sort_values(
        ["condition_id", "checkup_index"],
        kind="stable",
    ).reset_index(drop=True)
    all_conditions = set(ordered_observations["condition_id"].astype(str))
    maximum_public_hours = float(ordered_observations["elapsed_hours"].max())
    model_config = dict(validated["model"])
    dynamic_config = dict(validated["dynamic_method"])
    prediction_frames: list[pd.DataFrame] = []
    diagnostics_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []

    for scenario in validated["scenarios"]:
        scenario_name = str(scenario["name"])
        seen_targets: set[str] = set()
        for fold_id, target_ids in _scenario_folds(ordered_observations, scenario):
            target_set = set(target_ids)
            if not target_set or seen_targets & target_set:
                raise ValueError("Scenario folds must contain disjoint non-empty targets")
            seen_targets.update(target_set)
            training_ids = all_conditions - target_set
            if target_set & training_ids or target_set | training_ids != all_conditions:
                raise RuntimeError("Dynamic calendar fold does not partition conditions")
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
            training_full = ordered_observations.loc[
                ordered_observations["condition_id"].astype(str).isin(training_ids)
            ].copy()
            target_fold = ordered_observations.loc[
                ordered_observations["condition_id"].astype(str).isin(target_set)
            ].copy()
            if training_full["condition_id"].nunique() < int(
                model_config["minimum_training_conditions"]
            ):
                raise ValueError(f"Fold {scenario_name}/{fold_id} has too few conditions")

            for prefix in validated["prefix_checkups"]:
                training_prefix = select_prefix(training_full, int(prefix)).sort_values(
                    ["condition_id", "checkup_index"],
                    kind="stable",
                ).reset_index(drop=True)
                if int(training_prefix["checkup_index"].max()) >= int(prefix):
                    raise RuntimeError("Training future labels crossed the global landmark")
                model = fit_empirical_stress_surface(
                    training_prefix,
                    minimum_conditions=int(model_config["minimum_training_conditions"]),
                    robust_loss_scale_pp=float(model_config["robust_loss_scale_pp"]),
                    maximum_prediction_hours=maximum_public_hours,
                )
                ridge_state = estimate_empirical_bayes_ridge(
                    model,
                    training_prefix,
                    prefix_checkups=int(prefix),
                    ridge_bounds=(
                        float(dynamic_config["ridge_min"]),
                        float(dynamic_config["ridge_max"]),
                    ),
                )
                training_support_days = float(training_prefix["elapsed_days"].max())
                training_state_sha256, _ = _training_state(
                    scenario_name=scenario_name,
                    fold_id=fold_id,
                    prefix=int(prefix),
                    training_prefix=training_prefix,
                    model=model,
                    ridge_state=ridge_state,
                    config=validated,
                )
                for parameter_name, parameter_value in model.parameter_map().items():
                    parameter_rows.append(
                        {
                            "scenario": scenario_name,
                            "fold_id": fold_id,
                            "prefix_checkups": int(prefix),
                            "training_history_policy": GLOBAL_LANDMARK_POLICY,
                            "parameter": parameter_name,
                            "value": float(parameter_value),
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_prefix),
                            "training_max_checkup_index": int(
                                training_prefix["checkup_index"].max()
                            ),
                            "training_support_days": training_support_days,
                            "training_state_sha256": training_state_sha256,
                        }
                    )
                for parameter_name in (
                    "ridge",
                    "raw_ridge",
                    "residual_variance_pp2",
                    "between_condition_scale_variance",
                ):
                    parameter_rows.append(
                        {
                            "scenario": scenario_name,
                            "fold_id": fold_id,
                            "prefix_checkups": int(prefix),
                            "training_history_policy": GLOBAL_LANDMARK_POLICY,
                            "parameter": f"empirical_bayes_{parameter_name}",
                            "value": float(ridge_state[parameter_name]),
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_prefix),
                            "training_max_checkup_index": int(
                                training_prefix["checkup_index"].max()
                            ),
                            "training_support_days": training_support_days,
                            "training_state_sha256": training_state_sha256,
                        }
                    )

                for target_id in sorted(target_set):
                    target = target_fold.loc[
                        target_fold["condition_id"].astype(str) == target_id
                    ].copy()
                    predictions, target_diagnostics = _target_predictions(
                        scenario_name=scenario_name,
                        fold_id=fold_id,
                        target_frame=target,
                        model=model,
                        ridge_state=ridge_state,
                        training_state_sha256=training_state_sha256,
                        prefix=int(prefix),
                        config=validated,
                        training_support_days=training_support_days,
                    )
                    prediction_frames.append(predictions)
                    diagnostics_rows.append(
                        {
                            **target_diagnostics,
                            "training_history_policy": GLOBAL_LANDMARK_POLICY,
                            "training_condition_count": len(training_ids),
                            "training_observation_count": len(training_prefix),
                            "training_max_checkup_index": int(
                                training_prefix["checkup_index"].max()
                            ),
                            "target_condition_count_in_fold": len(target_set),
                        }
                    )
        if seen_targets != (
            all_conditions if scenario_name == TEMPERATURE_SCENARIO else set(SOC_TARGET_CONDITIONS)
        ):
            raise RuntimeError(f"Scenario {scenario_name} did not execute all frozen targets")

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        PREDICTION_KEY_COLUMNS,
        kind="stable",
    ).reset_index(drop=True)
    prediction_hash = dynamic_prediction_artifact_sha256(predictions)
    condition_metrics = score_dynamic_predictions(
        predictions,
        ordered_observations,
        frozen_prediction_sha256=prediction_hash,
    )
    paired_metrics = _paired_condition_metrics(condition_metrics)
    comparison_summary = _comparison_summary(
        paired_metrics,
        diagnostics_config=validated["diagnostics"],
        superiority_margin=float(
            validated["confirmation_gate"]["superiority_margin_delta_iae_pp"]
        ),
    )
    confirmation_gate = _confirmation_gate(comparison_summary, config=validated)
    diagnostics = pd.DataFrame(diagnostics_rows).sort_values(
        ["scenario", "target_condition_id", "prefix_checkups"],
        kind="stable",
    ).reset_index(drop=True)
    parameters = pd.DataFrame(parameter_rows).sort_values(
        ["scenario", "fold_id", "prefix_checkups", "parameter"],
        kind="stable",
    ).reset_index(drop=True)
    splits = pd.DataFrame(split_rows).sort_values(
        ["scenario", "fold_id", "condition_id"],
        kind="stable",
    ).reset_index(drop=True)
    parameter_hash = _canonical_frame_sha256(
        parameters,
        sort_by=["scenario", "fold_id", "prefix_checkups", "parameter"],
    )
    training_state_set_hash = _canonical_json_sha256(
        sorted(parameters["training_state_sha256"].astype(str).unique())
    )
    prediction_state_set_hash = _canonical_json_sha256(
        sorted(predictions["prediction_state_sha256"].astype(str).unique())
    )

    result: dict[str, object] = {
        "status": CONFIRMATION_BLOCK_STATUS,
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _canonical_json_sha256(validated),
        "dataset": {
            "dataset_id": validated["dataset_id"],
            "dataset_snapshot_id": validated["dataset_snapshot_id"],
            "label_version": validated["label_version"],
            "independent_condition_count": condition_count,
            "observation_count": len(ordered_observations),
            "statistical_unit": NAUMANN_STATISTICAL_UNIT,
            "maximum_horizon_days": maximum_days,
            "relationship_to_phase41": validated["confirmation_gate"][
                "current_dataset_relationship"
            ],
        },
        "design": {
            "design_status": DESIGN_STATUS,
            "runner_scope": RUNNER_SCOPE,
            "independent_confirmation_runner_status": INDEPENDENT_RUNNER_STATUS,
            "training_history_policy": GLOBAL_LANDMARK_POLICY,
            "scenarios": validated["scenarios"],
            "prefix_checkups": validated["prefix_checkups"],
            "primary_prefix_checkups": validated["primary_prefix_checkups"],
            "dynamic_method": validated["dynamic_method"],
            "comparator": validated["comparator"],
            "estimand": validated["estimand"],
            "training_future_outcomes_used_for_fit_or_update": False,
            "target_future_outcomes_used_for_fit_or_update": False,
            "target_prefix_outcomes_used_for_update": True,
            "condition_balanced_paired_analysis": True,
            "condition_bootstrap_is_descriptive_only": True,
            "shared_fold_dependence_adjusted": False,
            "projection_beyond_observed_horizon": False,
        },
        "comparison_summary": comparison_summary.to_dict(orient="records"),
        "confirmation_gate": confirmation_gate,
        "future_label_firewall": {
            "label_free_prediction_sha256": prediction_hash,
            "parameter_artifact_sha256": parameter_hash,
            "training_state_set_sha256": training_state_set_hash,
            "prediction_state_set_sha256": prediction_state_set_hash,
            "training_and_target_rows_at_or_after_each_prefix_used_only_for_scoring": True,
            "score_after_prediction_hash_verification": True,
        },
        "claim_boundary": (
            "This run reuses the 17 Naumann condition-mean trajectories already seen "
            "in Phase 4.1. It is a deterministic development diagnostic and cannot "
            "confirm dynamic-update superiority, product accuracy, individual-cell "
            "risk, or long-horizon storage performance. This Naumann-specific runner "
            "cannot issue a confirmation pass; an independent dataset requires a new "
            "dataset-specific preregistration and runner."
        ),
        "prohibited_claims": validated["prohibited_claims"],
    }
    return (
        result,
        predictions,
        condition_metrics,
        paired_metrics,
        diagnostics,
        parameters,
        splits,
    )
