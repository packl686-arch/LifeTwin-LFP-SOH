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
from lifetwin.experiments.calendar_v2_development import (
    EXPECTED_DATASET_SNAPSHOT_ID,
    EXPECTED_LABEL_VERSION,
    GLOBAL_LANDMARK_POLICY,
    SOC_TARGET_CONDITIONS,
)
from lifetwin.models.calendar_v2 import (
    HIERARCHICAL_POWER_METHOD,
    POWER_PARAMETER_NAMES,
    STRESS_FEATURE_NAMES,
    TARGET_SQRT_METHOD,
    fit_hierarchical_power_prior,
    fit_sqrt_rate,
    predict_power_loss,
    predict_sqrt_loss,
    update_hierarchical_power_law,
)
from lifetwin.models.calendar_v3_activation import (
    ACTIVATION_PARAMETER_NAMES,
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    GATED_TARGET_ACTIVATION_METHOD,
    HIERARCHICAL_ACTIVATION_METHOD,
    TARGET_ACTIVATION_METHOD,
    activation_mechanism_gate,
    fit_activation_offset_power_law,
    fit_hierarchical_activation_offset_prior,
    predict_activation_offset_loss,
    update_hierarchical_activation_offset,
)


EXPERIMENT_ID = "naumann_calendar_v3_activation_offset_development_v1"
BASE_EXPERIMENT_ID = "naumann_calendar_v2_development_bakeoff_v1"
DESIGN_STATUS = "isolated_post_hoc_mechanism_development_diagnostic"
EVIDENCE_ROLE = "reused_naumann_low_soc_mechanism_development_only"
RUNNER_SCOPE = "naumann_reuse_activation_offset_development_only"
CONFIRMATION_STATUS = "blocked_pending_independent_dataset"
EXPECTED_PREFIXES = (5, 8, 10, 14)
EXPECTED_CHECKUP_INDICES = tuple(range(35))
PRIMARY_PREFIX = 10
TEMPERATURE_SCENARIO = "v3_unseen_temperature_level"
SOC_SCENARIO = "v3_soc_interpolation_at_40c"
SOC_FOLD_ID = "40c_intermediate_soc_v3_activation_development"
GATE_SCENARIOS = (TEMPERATURE_SCENARIO, SOC_SCENARIO)
METHOD_NAMES = (
    TARGET_SQRT_METHOD,
    HIERARCHICAL_POWER_METHOD,
    TARGET_ACTIVATION_METHOD,
    HIERARCHICAL_ACTIVATION_METHOD,
    GATED_TARGET_ACTIVATION_METHOD,
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
)
PRIMARY_CANDIDATE = GATED_TARGET_ACTIVATION_METHOD
PRIMARY_COMPARATOR = HIERARCHICAL_POWER_METHOD
TAU_SENSITIVITY_DAYS = (3.0, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0)
CORE_TAU_DAYS = (3.0, 5.0, 7.0, 10.0, 14.0)
EXPECTED_PROHIBITED_CLAIMS = (
    "confirmatory_superiority_on_naumann_reuse",
    "preregistered_tau_selection",
    "general_low_soc_mechanism_validation",
    "independent_external_validation",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

PREDICTION_KEY_COLUMNS = [
    "scenario",
    "fold_id",
    "target_condition_id",
    "prefix_checkups",
    "method",
    "target_checkup_index",
]
PREDICTION_COLUMNS = [
    *PREDICTION_KEY_COLUMNS,
    "training_history_policy",
    "prefix_end_checkup_index",
    "prefix_end_days",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_days",
    "predicted_capacity_retention_pct",
    "is_final_checkup",
    "activation_gate_ready",
    "negative_loss_evidence",
    "positive_time_observation_count",
    "minimum_prefix_capacity_loss_pct",
    "activation_component_selected",
    "training_support_days",
    "validation_horizon_days",
    "time_extrapolation_ratio",
    "training_state_sha256",
    "prediction_state_sha256",
]

SENSITIVITY_KEY_COLUMNS = [
    "scenario",
    "fold_id",
    "target_condition_id",
    "prefix_checkups",
    "activation_timescale_days",
    "target_checkup_index",
]
SENSITIVITY_COLUMNS = [
    *SENSITIVITY_KEY_COLUMNS,
    "training_history_policy",
    "elapsed_days",
    "predicted_capacity_retention_pct",
    "is_final_checkup",
    "activation_gate_ready",
    "activation_component_selected",
    "training_state_sha256",
    "prediction_state_sha256",
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
    "methods",
    "primary_candidate",
    "primary_comparator",
    "model",
    "mechanism_gate",
    "timescale_sensitivity",
    "diagnostics",
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
        index=False, lineterminator="\n", float_format="%.17g"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_calendar_v3_activation_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    parsed = dict(config)
    _require_exact_keys(
        parsed,
        expected=TOP_LEVEL_CONFIG_KEYS,
        context="Calendar V3 activation config",
    )
    exact_scalars = {
        "experiment_id": EXPERIMENT_ID,
        "base_experiment_id": BASE_EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "evidence_role": EVIDENCE_ROLE,
        "runner_scope": RUNNER_SCOPE,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_comparator": PRIMARY_COMPARATOR,
    }
    for key, expected in exact_scalars.items():
        if parsed[key] != expected:
            raise ValueError(f"Calendar V3 activation {key} must remain {expected}")
    if parsed["prefix_checkups"] != list(EXPECTED_PREFIXES) or not all(
        type(value) is int for value in parsed["prefix_checkups"]
    ):
        raise ValueError("Calendar V3 activation prefixes must remain exact")
    if (
        type(parsed["primary_prefix_checkups"]) is not int
        or parsed["primary_prefix_checkups"] != PRIMARY_PREFIX
    ):
        raise ValueError("Calendar V3 activation primary prefix must remain 10")
    if tuple(str(value) for value in parsed["methods"]) != METHOD_NAMES:
        raise ValueError("Calendar V3 activation methods must remain exact")

    scenarios = list(parsed["scenarios"])
    if len(scenarios) != 2 or not all(isinstance(item, Mapping) for item in scenarios):
        raise ValueError("Calendar V3 activation requires two scenarios")
    by_name = {str(item["name"]): dict(item) for item in scenarios}
    if set(by_name) != set(GATE_SCENARIOS):
        raise ValueError("Calendar V3 activation scenarios changed")
    temperature = by_name[TEMPERATURE_SCENARIO]
    _require_exact_keys(
        temperature,
        expected={"name", "kind", "training_history_policy"},
        context="V3 temperature scenario",
    )
    if temperature != {
        "name": TEMPERATURE_SCENARIO,
        "kind": "leave_one_temperature_level_out",
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
    }:
        raise ValueError("Calendar V3 temperature scenario changed")
    soc = by_name[SOC_SCENARIO]
    _require_exact_keys(
        soc,
        expected={
            "name",
            "kind",
            "training_history_policy",
            "fold_id",
            "target_condition_ids",
        },
        context="V3 SOC scenario",
    )
    if (
        soc["kind"] != "fixed_condition_holdout"
        or soc["training_history_policy"] != GLOBAL_LANDMARK_POLICY
        or soc["fold_id"] != "40c_intermediate_soc_v3_activation_development"
        or tuple(soc["target_condition_ids"]) != SOC_TARGET_CONDITIONS
    ):
        raise ValueError("Calendar V3 SOC scenario changed")

    model = _mapping(parsed["model"], context="model")
    _require_exact_keys(
        model,
        expected={
            "time_unit",
            "minimum_training_conditions",
            "robust_loss_scale_pp",
            "power_exponent_bounds",
            "stress_surface_ridge",
            "base_parameter_scale_floors",
            "activation_parameter_scale_floors",
            "observation_scale_floor_pp",
            "activation_timescale_days",
            "activation_offset_bounds_pp",
            "activation_formula",
        },
        context="model",
    )
    expected_model = {
        "time_unit": "day",
        "minimum_training_conditions": 6,
        "robust_loss_scale_pp": 0.25,
        "power_exponent_bounds": [0.05, 1.5],
        "stress_surface_ridge": 1.0,
        "base_parameter_scale_floors": [0.1, 0.05],
        "activation_parameter_scale_floors": [0.1, 0.05, 0.1],
        "observation_scale_floor_pp": 0.1,
        "activation_timescale_days": 7.0,
        "activation_offset_bounds_pp": [0.0, 10.0],
        "activation_formula": (
            "exp(log_amplitude)*t^time_exponent-activation_offset*"
            "(1-exp(-t/tau))"
        ),
    }
    if _canonical_json_sha256(model) != _canonical_json_sha256(expected_model):
        raise ValueError("Calendar V3 activation model specification changed")

    gate = _mapping(parsed["mechanism_gate"], context="mechanism_gate")
    expected_gate = {
        "minimum_positive_time_observations": 7,
        "negative_loss_threshold_pp": 0.0,
        "requires_both_readiness_and_negative_loss": True,
        "fallback_method": HIERARCHICAL_POWER_METHOD,
        "target_future_outcomes_used": False,
    }
    if _canonical_json_sha256(gate) != _canonical_json_sha256(expected_gate):
        raise ValueError("Calendar V3 activation mechanism gate changed")

    sensitivity = _mapping(
        parsed["timescale_sensitivity"], context="timescale_sensitivity"
    )
    expected_sensitivity = {
        "prefix_checkups": PRIMARY_PREFIX,
        "timescale_days": list(TAU_SENSITIVITY_DAYS),
        "core_robustness_timescale_days": list(CORE_TAU_DAYS),
        "selection_status": "post_hoc_fixed_after_phase7_failure_audit",
        "primary_timescale_days": 7.0,
        "future_outcomes_used_for_timescale_selection": True,
        "formal_hyperparameter_selection_claim_allowed": False,
    }
    if _canonical_json_sha256(sensitivity) != _canonical_json_sha256(
        expected_sensitivity
    ):
        raise ValueError("Calendar V3 activation timescale sensitivity changed")

    diagnostics = _mapping(parsed["diagnostics"], context="diagnostics")
    _require_exact_keys(
        diagnostics,
        expected={
            "bootstrap_unit",
            "bootstrap_resamples",
            "random_seed",
            "confidence_level",
            "inference_role",
            "shared_fold_dependence_adjusted",
        },
        context="diagnostics",
    )
    expected_diagnostics = {
        "bootstrap_unit": "condition",
        "bootstrap_resamples": 10000,
        "random_seed": 20260719,
        "confidence_level": 0.95,
        "inference_role": "descriptive_resampling_only",
        "shared_fold_dependence_adjusted": False,
    }
    if _canonical_json_sha256(diagnostics) != _canonical_json_sha256(
        expected_diagnostics
    ):
        raise ValueError("Calendar V3 activation diagnostics changed")
    development_gate = _mapping(
        parsed["development_gate"], context="development_gate"
    )
    if development_gate != {
        "confirmation_status": CONFIRMATION_STATUS,
        "current_dataset_relationship": "reused_and_outcomes_already_inspected",
        "decision_role": "mechanism_development_diagnostic_only",
    }:
        raise ValueError("Calendar V3 activation gate must remain blocked")
    if tuple(parsed["prohibited_claims"]) != EXPECTED_PROHIBITED_CLAIMS:
        raise ValueError("Calendar V3 activation prohibited claims changed")
    return parsed


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
        raise ValueError(f"Calendar V3 scenario has unknown conditions: {missing}")
    return [(str(scenario["fold_id"]), targets)]


def _select_prefix(frame: pd.DataFrame, prefix_checkups: int) -> pd.DataFrame:
    selected = frame.loc[pd.to_numeric(frame["checkup_index"]) < prefix_checkups].copy()
    counts = selected.groupby("condition_id", sort=True).size()
    if selected.empty or (counts != prefix_checkups).any():
        raise ValueError("Every condition must contain the complete requested prefix")
    return selected.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)


def _base_prior(training: pd.DataFrame, model: Mapping[str, object]):
    return fit_hierarchical_power_prior(
        training,
        minimum_conditions=int(model["minimum_training_conditions"]),
        exponent_bounds=tuple(float(v) for v in model["power_exponent_bounds"]),
        robust_loss_scale_pp=float(model["robust_loss_scale_pp"]),
        stress_surface_ridge=float(model["stress_surface_ridge"]),
        parameter_scale_floors=tuple(
            float(v) for v in model["base_parameter_scale_floors"]
        ),
        observation_scale_floor_pp=float(model["observation_scale_floor_pp"]),
    )


def _activation_prior(training: pd.DataFrame, model: Mapping[str, object]):
    return fit_hierarchical_activation_offset_prior(
        training,
        activation_timescale_days=float(model["activation_timescale_days"]),
        minimum_conditions=int(model["minimum_training_conditions"]),
        exponent_bounds=tuple(float(v) for v in model["power_exponent_bounds"]),
        activation_offset_bounds_pp=tuple(
            float(v) for v in model["activation_offset_bounds_pp"]
        ),
        robust_loss_scale_pp=float(model["robust_loss_scale_pp"]),
        stress_surface_ridge=float(model["stress_surface_ridge"]),
        parameter_scale_floors=tuple(
            float(v) for v in model["activation_parameter_scale_floors"]
        ),
        observation_scale_floor_pp=float(model["observation_scale_floor_pp"]),
    )


def _training_state(
    *,
    scenario: str,
    fold_id: str,
    prefix_checkups: int,
    training: pd.DataFrame,
    base_prior: object,
    activation_prior: object,
    model: Mapping[str, object],
) -> str:
    payload = {
        "scenario": scenario,
        "fold_id": fold_id,
        "prefix_checkups": int(prefix_checkups),
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "training_rows": [
            {
                "condition_id": str(row.condition_id),
                "checkup_index": int(row.checkup_index),
                "temperature_c": float(row.temperature_c),
                "storage_soc_fraction": float(row.storage_soc_fraction),
                "elapsed_days": float(row.elapsed_days),
                "capacity_loss_pct": float(row.capacity_loss_pct),
            }
            for row in training.sort_values(
                ["condition_id", "checkup_index"], kind="stable"
            ).itertuples(index=False)
        ],
        "base_prior": {
            "surface_coefficients": [
                list(values) for values in base_prior.surface_coefficients
            ],
            "parameter_scales": list(base_prior.parameter_scales),
            "observation_scale_pp": float(base_prior.observation_scale_pp),
            "condition_parameters": [
                list(values) for values in base_prior.condition_parameters
            ],
        },
        "activation_prior": {
            "surface_coefficients": [
                list(values) for values in activation_prior.surface_coefficients
            ],
            "parameter_scales": list(activation_prior.parameter_scales),
            "observation_scale_pp": float(activation_prior.observation_scale_pp),
            "activation_timescale_days": float(
                activation_prior.activation_timescale_days
            ),
            "condition_parameters": [
                list(values) for values in activation_prior.condition_parameters
            ],
        },
        "model": dict(model),
    }
    return _canonical_json_sha256(payload)


def _prior_parameter_rows(
    *,
    scenario: str,
    fold_id: str,
    prefix_checkups: int,
    training_state_sha256: str,
    base_prior: object,
    activation_prior: object,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, prior, names in (
        (HIERARCHICAL_POWER_METHOD, base_prior, POWER_PARAMETER_NAMES),
        (HIERARCHICAL_ACTIVATION_METHOD, activation_prior, ACTIVATION_PARAMETER_NAMES),
    ):
        for parameter, coefficients in zip(
            names, prior.surface_coefficients, strict=True
        ):
            for feature, value in zip(
                STRESS_FEATURE_NAMES, coefficients, strict=True
            ):
                rows.append(
                    {
                        "scenario": scenario,
                        "fold_id": fold_id,
                        "prefix_checkups": int(prefix_checkups),
                        "model": model_name,
                        "parameter_group": "stress_surface",
                        "parameter": parameter,
                        "feature": feature,
                        "value": float(value),
                        "training_state_sha256": training_state_sha256,
                    }
                )
        for parameter, value in zip(names, prior.parameter_scales, strict=True):
            rows.append(
                {
                    "scenario": scenario,
                    "fold_id": fold_id,
                    "prefix_checkups": int(prefix_checkups),
                    "model": model_name,
                    "parameter_group": "parameter_scale",
                    "parameter": parameter,
                    "feature": "none",
                    "value": float(value),
                    "training_state_sha256": training_state_sha256,
                }
            )
        rows.append(
            {
                "scenario": scenario,
                "fold_id": fold_id,
                "prefix_checkups": int(prefix_checkups),
                "model": model_name,
                "parameter_group": "observation_scale",
                "parameter": "observation_scale_pp",
                "feature": "none",
                "value": float(prior.observation_scale_pp),
                "training_state_sha256": training_state_sha256,
            }
        )
    rows.append(
        {
            "scenario": scenario,
            "fold_id": fold_id,
            "prefix_checkups": int(prefix_checkups),
            "model": HIERARCHICAL_ACTIVATION_METHOD,
            "parameter_group": "fixed_timescale",
            "parameter": "activation_timescale_days",
            "feature": "none",
            "value": float(activation_prior.activation_timescale_days),
            "training_state_sha256": training_state_sha256,
        }
    )
    return rows


def _target_state(
    *,
    scenario: str,
    fold_id: str,
    target: pd.DataFrame,
    prefix_checkups: int,
    training_state_sha256: str,
    base_prior: object,
    activation_prior: object,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object], list[dict[str, object]]]:
    ordered = target.sort_values("checkup_index", kind="stable").reset_index(drop=True)
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
    model = config["model"]
    gate_config = config["mechanism_gate"]
    exponent_bounds = tuple(float(v) for v in model["power_exponent_bounds"])
    offset_bounds = tuple(float(v) for v in model["activation_offset_bounds_pp"])
    gate = activation_mechanism_gate(
        prefix,
        minimum_positive_time_observations=int(
            gate_config["minimum_positive_time_observations"]
        ),
        negative_loss_threshold_pp=float(
            gate_config["negative_loss_threshold_pp"]
        ),
    )
    sqrt_rate = fit_sqrt_rate(prefix)
    base_fit = update_hierarchical_power_law(
        base_prior, prefix, exponent_bounds=exponent_bounds
    )
    elapsed = future["elapsed_days"].to_numpy(dtype=float)
    base_loss = predict_power_loss(base_fit, elapsed)

    target_activation = None
    target_activation_loss = base_loss
    target_activation_error: str | None = None
    try:
        target_activation = fit_activation_offset_power_law(
            prefix,
            activation_timescale_days=float(model["activation_timescale_days"]),
            exponent_bounds=exponent_bounds,
            activation_offset_bounds_pp=offset_bounds,
            robust_loss_scale_pp=float(model["robust_loss_scale_pp"]),
        )
        target_activation_loss = predict_activation_offset_loss(
            target_activation, elapsed
        )
        if not np.isfinite(target_activation_loss).all():
            raise RuntimeError("Target activation specialist returned non-finite values")
    except Exception as exc:  # Numerical specialist failure must not block V2.
        target_activation = None
        target_activation_loss = base_loss
        target_activation_error = f"{type(exc).__name__}: {exc}"

    hierarchical_activation = None
    hierarchical_activation_loss = base_loss
    hierarchical_activation_error: str | None = None
    try:
        hierarchical_activation = update_hierarchical_activation_offset(
            activation_prior,
            prefix,
            exponent_bounds=exponent_bounds,
            activation_offset_bounds_pp=offset_bounds,
        )
        hierarchical_activation_loss = predict_activation_offset_loss(
            hierarchical_activation, elapsed
        )
        if not np.isfinite(hierarchical_activation_loss).all():
            raise RuntimeError(
                "Hierarchical activation specialist returned non-finite values"
            )
    except Exception as exc:  # Keep the primary gated prediction operational.
        hierarchical_activation = None
        hierarchical_activation_loss = base_loss
        hierarchical_activation_error = f"{type(exc).__name__}: {exc}"
    target_specialist_selected = bool(gate.ready and target_activation is not None)
    hierarchical_specialist_selected = bool(
        gate.ready and hierarchical_activation is not None
    )
    target_fallback_reason = (
        "none"
        if target_specialist_selected
        else (
            "gate_not_ready"
            if not gate.ready
            else f"specialist_fit_failed:{target_activation_error}"
        )
    )
    hierarchical_fallback_reason = (
        "none"
        if hierarchical_specialist_selected
        else (
            "gate_not_ready"
            if not gate.ready
            else f"specialist_fit_failed:{hierarchical_activation_error}"
        )
    )
    predictions = {
        TARGET_SQRT_METHOD: predict_sqrt_loss(sqrt_rate, elapsed),
        HIERARCHICAL_POWER_METHOD: base_loss,
        TARGET_ACTIVATION_METHOD: target_activation_loss,
        HIERARCHICAL_ACTIVATION_METHOD: hierarchical_activation_loss,
        GATED_TARGET_ACTIVATION_METHOD: (
            target_activation_loss if target_specialist_selected else base_loss
        ),
        GATED_HIERARCHICAL_ACTIVATION_METHOD: (
            hierarchical_activation_loss
            if hierarchical_specialist_selected
            else base_loss
        ),
    }
    target_id = str(ordered["condition_id"].iloc[0])
    state = {
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
        "gate": gate.__dict__,
        "parameters": {
            TARGET_SQRT_METHOD: {"sqrt_rate": float(sqrt_rate)},
            HIERARCHICAL_POWER_METHOD: base_fit.parameter_map(),
            TARGET_ACTIVATION_METHOD: (
                target_activation.parameter_map()
                if target_activation is not None
                else {
                    "fit_status": "failed",
                    "fallback_method": HIERARCHICAL_POWER_METHOD,
                    "error": target_activation_error,
                }
            ),
            HIERARCHICAL_ACTIVATION_METHOD: (
                hierarchical_activation.parameter_map()
                if hierarchical_activation is not None
                else {
                    "fit_status": "failed",
                    "fallback_method": HIERARCHICAL_POWER_METHOD,
                    "error": hierarchical_activation_error,
                }
            ),
        },
        "predictions": {
            method: [float(value) for value in values]
            for method, values in predictions.items()
        },
    }
    prediction_hash = _canonical_json_sha256(state)
    prefix_end = prefix.sort_values("checkup_index", kind="stable").iloc[-1]
    rows: list[dict[str, object]] = []
    for method in METHOD_NAMES:
        selected = {
            TARGET_ACTIVATION_METHOD: target_activation is not None,
            HIERARCHICAL_ACTIVATION_METHOD: hierarchical_activation is not None,
            GATED_TARGET_ACTIVATION_METHOD: target_specialist_selected,
            GATED_HIERARCHICAL_ACTIVATION_METHOD: (
                hierarchical_specialist_selected
            ),
        }.get(method, False)
        for coordinate, loss in zip(
            future.itertuples(index=False), predictions[method], strict=True
        ):
            rows.append(
                {
                    "scenario": scenario,
                    "fold_id": fold_id,
                    "target_condition_id": target_id,
                    "prefix_checkups": int(prefix_checkups),
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "training_history_policy": GLOBAL_LANDMARK_POLICY,
                    "prefix_end_checkup_index": int(prefix_end["checkup_index"]),
                    "prefix_end_days": float(prefix_end["elapsed_days"]),
                    "temperature_c": float(coordinate.temperature_c),
                    "storage_soc_fraction": float(coordinate.storage_soc_fraction),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": 100.0 - float(loss),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == ordered["checkup_index"].max()
                    ),
                    "activation_gate_ready": gate.ready,
                    "negative_loss_evidence": gate.negative_loss_evidence,
                    "positive_time_observation_count": (
                        gate.positive_time_observation_count
                    ),
                    "minimum_prefix_capacity_loss_pct": (
                        gate.minimum_capacity_loss_pct
                    ),
                    "activation_component_selected": bool(selected),
                    "training_support_days": float(base_prior.maximum_training_days),
                    "validation_horizon_days": float(ordered["elapsed_days"].max()),
                    "time_extrapolation_ratio": float(
                        ordered["elapsed_days"].max()
                        / base_prior.maximum_training_days
                    ),
                    "training_state_sha256": training_state_sha256,
                    "prediction_state_sha256": prediction_hash,
                }
            )
    base_prior_mean = base_prior.prior_mean(prefix)
    activation_prior_mean = activation_prior.prior_mean(prefix)
    diagnostic = {
        "scenario": scenario,
        "fold_id": fold_id,
        "target_condition_id": target_id,
        "prefix_checkups": int(prefix_checkups),
        "prefix_end_days": float(prefix_end["elapsed_days"]),
        "future_checkup_count": len(future),
        "activation_gate_ready": gate.ready,
        "negative_loss_evidence": gate.negative_loss_evidence,
        "positive_time_observation_count": gate.positive_time_observation_count,
        "minimum_prefix_capacity_loss_pct": gate.minimum_capacity_loss_pct,
        "sqrt_rate": float(sqrt_rate),
        "base_prior_log_amplitude": float(base_prior_mean[0]),
        "base_prior_time_exponent": float(base_prior_mean[1]),
        "base_posterior_log_amplitude": base_fit.log_amplitude,
        "base_posterior_time_exponent": base_fit.time_exponent,
        "activation_target_log_amplitude": (
            target_activation.log_amplitude
            if target_activation is not None
            else None
        ),
        "activation_target_time_exponent": (
            target_activation.time_exponent
            if target_activation is not None
            else None
        ),
        "activation_target_offset_pp": (
            target_activation.activation_offset_pp
            if target_activation is not None
            else None
        ),
        "activation_prior_log_amplitude": float(activation_prior_mean[0]),
        "activation_prior_time_exponent": float(activation_prior_mean[1]),
        "activation_prior_offset_pp": float(activation_prior_mean[2]),
        "activation_posterior_log_amplitude": (
            hierarchical_activation.log_amplitude
            if hierarchical_activation is not None
            else None
        ),
        "activation_posterior_time_exponent": (
            hierarchical_activation.time_exponent
            if hierarchical_activation is not None
            else None
        ),
        "activation_posterior_offset_pp": (
            hierarchical_activation.activation_offset_pp
            if hierarchical_activation is not None
            else None
        ),
        "activation_timescale_days": float(model["activation_timescale_days"]),
        "target_activation_fit_status": (
            "fitted" if target_activation is not None else "failed"
        ),
        "target_activation_fit_error": target_activation_error or "none",
        "hierarchical_activation_fit_status": (
            "fitted" if hierarchical_activation is not None else "failed"
        ),
        "hierarchical_activation_fit_error": (
            hierarchical_activation_error or "none"
        ),
        "fallback_reason": target_fallback_reason,
        "hierarchical_fallback_reason": hierarchical_fallback_reason,
        "primary_gated_specialist_selected": target_specialist_selected,
        "hierarchical_gated_specialist_selected": (
            hierarchical_specialist_selected
        ),
        "training_condition_count": len(base_prior.training_condition_ids),
        "training_observation_count": int(base_prior.training_observation_count),
        "training_max_checkup_index": int(prefix_checkups - 1),
        "training_state_sha256": training_state_sha256,
        "prediction_state_sha256": prediction_hash,
    }
    sensitivity_rows: list[dict[str, object]] = []
    sensitivity_fit_errors: list[str] = []
    if prefix_checkups == PRIMARY_PREFIX:
        for tau in TAU_SENSITIVITY_DAYS:
            sensitivity_fit = None
            sensitivity_loss = base_loss
            if gate.ready:
                try:
                    sensitivity_fit = fit_activation_offset_power_law(
                        prefix,
                        activation_timescale_days=tau,
                        exponent_bounds=exponent_bounds,
                        activation_offset_bounds_pp=offset_bounds,
                        robust_loss_scale_pp=float(model["robust_loss_scale_pp"]),
                    )
                    sensitivity_loss = predict_activation_offset_loss(
                        sensitivity_fit, elapsed
                    )
                    if not np.isfinite(sensitivity_loss).all():
                        raise RuntimeError(
                            "Sensitivity specialist returned non-finite values"
                        )
                except Exception as exc:  # Sensitivity failure is diagnostic only.
                    sensitivity_fit = None
                    sensitivity_loss = base_loss
                    sensitivity_fit_errors.append(
                        f"tau={tau:g}:{type(exc).__name__}: {exc}"
                    )
            sensitivity_state = {
                "scenario": scenario,
                "fold_id": fold_id,
                "target_condition_id": target_id,
                "prefix_checkups": PRIMARY_PREFIX,
                "activation_timescale_days": tau,
                "training_state_sha256": training_state_sha256,
                "gate": gate.__dict__,
                "parameters": (
                    sensitivity_fit.parameter_map()
                    if sensitivity_fit is not None
                    else base_fit.parameter_map()
                ),
                "selected_activation": sensitivity_fit is not None,
                "future_coordinates": [
                    {
                        "checkup_index": int(row.checkup_index),
                        "elapsed_days": float(row.elapsed_days),
                    }
                    for row in future.itertuples(index=False)
                ],
                "predictions": [float(value) for value in sensitivity_loss],
            }
            sensitivity_hash = _canonical_json_sha256(sensitivity_state)
            for coordinate, loss in zip(
                future.itertuples(index=False), sensitivity_loss, strict=True
            ):
                sensitivity_rows.append(
                    {
                        "scenario": scenario,
                        "fold_id": fold_id,
                        "target_condition_id": target_id,
                        "prefix_checkups": PRIMARY_PREFIX,
                        "activation_timescale_days": tau,
                        "target_checkup_index": int(coordinate.checkup_index),
                        "training_history_policy": GLOBAL_LANDMARK_POLICY,
                        "elapsed_days": float(coordinate.elapsed_days),
                        "predicted_capacity_retention_pct": 100.0 - float(loss),
                        "is_final_checkup": bool(
                            coordinate.checkup_index
                            == ordered["checkup_index"].max()
                        ),
                        "activation_gate_ready": gate.ready,
                        "activation_component_selected": (
                            gate.ready and sensitivity_fit is not None
                        ),
                        "training_state_sha256": training_state_sha256,
                        "prediction_state_sha256": sensitivity_hash,
                    }
                )
    diagnostic["sensitivity_fit_failure_count"] = len(sensitivity_fit_errors)
    diagnostic["sensitivity_fit_errors"] = (
        ";".join(sensitivity_fit_errors) if sensitivity_fit_errors else "none"
    )
    return pd.DataFrame(rows)[PREDICTION_COLUMNS], diagnostic, sensitivity_rows


def calendar_v3_prediction_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V3 prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty or predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Calendar V3 prediction keys must be non-empty and unique")
    if predictions[PREDICTION_KEY_COLUMNS].isna().any().any():
        raise ValueError("Calendar V3 prediction keys cannot be null")
    if set(predictions["method"].astype(str)) != set(METHOD_NAMES):
        raise ValueError("Calendar V3 prediction pack must contain every method")
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
            "Calendar V3 prediction support must contain every method at "
            "every future coordinate"
        )
    for column in ("training_state_sha256", "prediction_state_sha256"):
        if not predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid Calendar V3 state hash column: {column}")
    return _canonical_frame_sha256(
        predictions[PREDICTION_COLUMNS], sort_by=PREDICTION_KEY_COLUMNS
    )


def calendar_v3_sensitivity_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(SENSITIVITY_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(SENSITIVITY_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V3 sensitivity schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty or predictions.duplicated(SENSITIVITY_KEY_COLUMNS).any():
        raise ValueError("Calendar V3 sensitivity keys must be non-empty and unique")
    if predictions[SENSITIVITY_KEY_COLUMNS].isna().any().any():
        raise ValueError("Calendar V3 sensitivity keys cannot be null")
    if set(predictions["activation_timescale_days"].astype(float)) != set(
        TAU_SENSITIVITY_DAYS
    ):
        raise ValueError("Calendar V3 sensitivity must contain the frozen tau grid")
    support_columns = [
        column
        for column in SENSITIVITY_KEY_COLUMNS
        if column != "activation_timescale_days"
    ]
    timescales_by_coordinate = predictions.groupby(
        support_columns,
        sort=False,
        dropna=False,
    )["activation_timescale_days"].agg(
        lambda values: frozenset(float(value) for value in values)
    )
    expected_timescales = frozenset(TAU_SENSITIVITY_DAYS)
    if not timescales_by_coordinate.map(
        lambda timescales: timescales == expected_timescales
    ).all():
        raise ValueError(
            "Calendar V3 sensitivity support must contain every frozen tau at "
            "every future coordinate"
        )
    return _canonical_frame_sha256(
        predictions[SENSITIVITY_COLUMNS], sort_by=SENSITIVITY_KEY_COLUMNS
    )


def _validated_scoring_frame(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    grouping_columns: list[str],
) -> pd.DataFrame:
    is_sensitivity = "activation_timescale_days" in predictions.columns
    _validate_frozen_protocol_target_coverage(
        predictions,
        observations,
        expected_prefixes=(PRIMARY_PREFIX,) if is_sensitivity else EXPECTED_PREFIXES,
        context=(
            "Calendar V3 sensitivity" if is_sensitivity else "Calendar V3 prediction"
        ),
    )
    required_scoring_columns = set(grouping_columns) | {
        "target_checkup_index",
        "elapsed_days",
        "predicted_capacity_retention_pct",
        "is_final_checkup",
    }
    if predictions[list(required_scoring_columns)].isna().any().any():
        raise ValueError("Calendar V3 scoring fields must be non-null")
    numeric_columns = [
        "prefix_checkups",
        "target_checkup_index",
        "prefix_end_checkup_index",
        "prefix_end_days",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "predicted_capacity_retention_pct",
        "positive_time_observation_count",
        "minimum_prefix_capacity_loss_pct",
        "training_support_days",
        "validation_horizon_days",
        "time_extrapolation_ratio",
        "activation_timescale_days",
    ]
    for column in numeric_columns:
        if column not in predictions:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Calendar V3 prediction column must be finite: {column}")

    for column in (
        "prefix_checkups",
        "target_checkup_index",
        "prefix_end_checkup_index",
    ):
        if column not in predictions:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"Calendar V3 prediction column must be integral: {column}")
    if not pd.api.types.is_bool_dtype(predictions["is_final_checkup"]):
        raise ValueError("Calendar V3 final-checkup flags must be boolean")

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
        raise ValueError("Calendar V3 truth condition ids must be non-null")
    if truth.duplicated(["target_condition_id", "target_checkup_index"]).any():
        raise ValueError("Calendar V3 truth coordinates must be unique")
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
        raise ValueError("Calendar V3 truth coordinates and outcomes must be finite")
    if not np.equal(
        truth_numeric["target_checkup_index"],
        np.floor(truth_numeric["target_checkup_index"]),
    ).all():
        raise ValueError("Calendar V3 truth checkup indices must be integral")
    truth[truth_numeric.columns] = truth_numeric

    target_ids = set(predictions["target_condition_id"].astype(str))
    relevant_truth = truth.loc[
        truth["target_condition_id"].astype(str).isin(target_ids)
    ].copy()
    if set(relevant_truth["target_condition_id"].astype(str)) != target_ids:
        raise ValueError("Every Calendar V3 target condition must exist in truth")
    for _, condition in relevant_truth.groupby("target_condition_id", sort=True):
        indices = sorted(
            pd.to_numeric(condition["target_checkup_index"]).astype(int).tolist()
        )
        if indices != list(EXPECTED_CHECKUP_INDICES):
            raise ValueError(
                "Calendar V3 truth must contain checkup indices range(0, 35)"
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
        raise ValueError("Every Calendar V3 prediction must match a future outcome")
    scored = scored.drop(columns="_future_truth_merge")

    for prediction_column, truth_column in (
        ("elapsed_days", "truth_elapsed_days"),
        ("temperature_c", "truth_temperature_c"),
        ("storage_soc_fraction", "truth_storage_soc_fraction"),
    ):
        if prediction_column not in scored:
            continue
        matches = np.isclose(
            pd.to_numeric(scored[prediction_column]).to_numpy(dtype=float),
            pd.to_numeric(scored[truth_column]).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        )
        if not matches.all():
            raise ValueError(
                f"Calendar V3 prediction coordinate disagrees with truth: "
                f"{prediction_column}"
            )

    prefixes = pd.to_numeric(scored["prefix_checkups"]).astype(int)
    target_indices = pd.to_numeric(scored["target_checkup_index"]).astype(int)
    if not prefixes.isin(EXPECTED_PREFIXES).all():
        raise ValueError("Calendar V3 prediction uses an unsupported prefix")
    if (target_indices < prefixes).any():
        raise ValueError("Calendar V3 prediction target precedes its prefix")

    if "prefix_end_checkup_index" in scored:
        prefix_end_indices = pd.to_numeric(
            scored["prefix_end_checkup_index"]
        ).astype(int)
        if not (prefix_end_indices == prefixes - 1).all():
            raise ValueError("Calendar V3 prefix-end index disagrees with prefix")
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
            raise ValueError("Calendar V3 prefix-end day disagrees with truth")

    truth_horizons = relevant_truth.groupby("target_condition_id", sort=True)[
        "truth_elapsed_days"
    ].max()
    if "validation_horizon_days" in scored:
        expected_horizons = scored["target_condition_id"].map(truth_horizons)
        if not np.isclose(
            pd.to_numeric(scored["validation_horizon_days"]).to_numpy(dtype=float),
            pd.to_numeric(expected_horizons).to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-12,
        ).all():
            raise ValueError("Calendar V3 validation horizon disagrees with truth")
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
            raise ValueError("Calendar V3 extrapolation ratio is inconsistent")

    for _, group in scored.groupby(grouping_columns, sort=True, dropna=False):
        prefix_values = pd.to_numeric(group["prefix_checkups"]).astype(int).unique()
        if len(prefix_values) != 1:
            raise ValueError("Calendar V3 trajectory must use one prefix")
        expected_future = list(range(int(prefix_values[0]), 35))
        actual_future = sorted(
            pd.to_numeric(group["target_checkup_index"]).astype(int).tolist()
        )
        if actual_future != expected_future:
            raise ValueError(
                "Calendar V3 trajectory must contain every future checkup in "
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
        raise ValueError("Calendar V3 final-checkup flag disagrees with truth")
    scored["truth_is_final_checkup"] = expected_final
    return scored


def _validate_frozen_protocol_target_coverage(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    expected_prefixes: tuple[int, ...],
    context: str,
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
        raise ValueError(f"{context} frozen SOC targets are absent from scoring truth")

    expected_groups: dict[tuple[str, str, int], frozenset[str]] = {}
    for temperature, rows in profile.groupby("temperature_c", sort=True):
        targets = frozenset(rows["condition_id"].astype(str))
        fold_id = f"temperature_c={float(temperature):g}"
        for prefix in expected_prefixes:
            expected_groups[(TEMPERATURE_SCENARIO, fold_id, prefix)] = targets
    for prefix in expected_prefixes:
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
            f"{context} target coverage does not match the frozen "
            "scenario/fold/prefix protocol"
        )


def _score_prediction_pack(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    key_columns: list[str],
    grouping_columns: list[str],
) -> pd.DataFrame:
    scored = _validated_scoring_frame(
        predictions,
        observations,
        grouping_columns=grouping_columns,
    )
    scored["prediction_error_pp"] = (
        scored["predicted_capacity_retention_pct"]
        - scored["true_capacity_retention_pct"]
    )
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(grouping_columns, sort=True):
        ordered = group.sort_values("truth_elapsed_days", kind="stable")
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        error = ordered["prediction_error_pp"].to_numpy(dtype=float)
        absolute = np.abs(error)
        final = ordered.loc[ordered["truth_is_final_checkup"]]
        if len(elapsed) < 2 or elapsed[-1] <= elapsed[0] or len(final) != 1:
            raise ValueError("Calendar V3 trajectory scoring support is incomplete")
        rows.append(
            {
                **dict(zip(grouping_columns, keys, strict=True)),
                "future_checkup_count": len(ordered),
                "trajectory_iae_pp": float(
                    np.trapezoid(absolute, elapsed) / (elapsed[-1] - elapsed[0])
                ),
                "future_point_mae_pp": float(absolute.mean()),
                "final_true_retention_pct": float(
                    final["true_capacity_retention_pct"].iloc[0]
                ),
                "final_predicted_retention_pct": float(
                    final["predicted_capacity_retention_pct"].iloc[0]
                ),
                "final_error_pp": float(final["prediction_error_pp"].iloc[0]),
                "activation_gate_ready": bool(
                    ordered["activation_gate_ready"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(key_columns, kind="stable").reset_index(
        drop=True
    )


def score_calendar_v3_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    if calendar_v3_prediction_sha256(predictions) != frozen_prediction_sha256:
        raise ValueError("Calendar V3 frozen prediction hash mismatch")
    return _score_prediction_pack(
        predictions,
        observations,
        key_columns=["scenario", "target_condition_id", "prefix_checkups", "method"],
        grouping_columns=[
            "scenario",
            "fold_id",
            "target_condition_id",
            "prefix_checkups",
            "method",
            "training_state_sha256",
            "prediction_state_sha256",
        ],
    )


def score_calendar_v3_sensitivity(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    if calendar_v3_sensitivity_sha256(predictions) != frozen_prediction_sha256:
        raise ValueError("Calendar V3 sensitivity prediction hash mismatch")
    return _score_prediction_pack(
        predictions,
        observations,
        key_columns=[
            "scenario",
            "target_condition_id",
            "activation_timescale_days",
        ],
        grouping_columns=[
            "scenario",
            "fold_id",
            "target_condition_id",
            "prefix_checkups",
            "activation_timescale_days",
            "training_state_sha256",
            "prediction_state_sha256",
        ],
    )


def _paired_metrics(condition_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = condition_metrics.loc[
        condition_metrics["method"] == PRIMARY_COMPARATOR,
        [
            "scenario",
            "target_condition_id",
            "prefix_checkups",
            "trajectory_iae_pp",
        ],
    ].rename(columns={"trajectory_iae_pp": "comparator_trajectory_iae_pp"})
    candidates = condition_metrics.loc[
        condition_metrics["method"] != PRIMARY_COMPARATOR
    ].rename(
        columns={
            "method": "candidate_method",
            "trajectory_iae_pp": "candidate_trajectory_iae_pp",
        }
    )
    paired = candidates.merge(
        baseline,
        on=["scenario", "target_condition_id", "prefix_checkups"],
        validate="many_to_one",
    )
    paired["paired_delta_iae_pp"] = (
        paired["candidate_trajectory_iae_pp"]
        - paired["comparator_trajectory_iae_pp"]
    )
    return paired.sort_values(
        ["scenario", "target_condition_id", "prefix_checkups", "candidate_method"],
        kind="stable",
    ).reset_index(drop=True)


def _bootstrap_seed(base_seed: int, scenario: str, prefix: int, method: str) -> int:
    payload = f"{base_seed}|{scenario}|{prefix}|{method}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _comparison_summary(
    paired: pd.DataFrame,
    *,
    diagnostics: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouping = ["scenario", "prefix_checkups", "candidate_method"]
    confidence = float(diagnostics["confidence_level"])
    alpha = 1.0 - confidence
    for keys, group in paired.groupby(grouping, sort=True):
        scenario, prefix, method = keys
        deltas = group["paired_delta_iae_pp"].to_numpy(dtype=float)
        rng = np.random.default_rng(
            _bootstrap_seed(
                int(diagnostics["random_seed"]),
                str(scenario),
                int(prefix),
                str(method),
            )
        )
        samples = rng.choice(
            deltas,
            size=(int(diagnostics["bootstrap_resamples"]), len(deltas)),
            replace=True,
        ).mean(axis=1)
        lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
        candidate_mean = float(group["candidate_trajectory_iae_pp"].mean())
        comparator_mean = float(group["comparator_trajectory_iae_pp"].mean())
        tolerance = 1e-12
        rows.append(
            {
                "scenario": scenario,
                "prefix_checkups": int(prefix),
                "candidate_method": method,
                "independent_condition_count": len(group),
                "shared_training_fold_count": int(group["fold_id"].nunique()),
                "candidate_trajectory_iae_pp_mean": candidate_mean,
                "comparator_trajectory_iae_pp_mean": comparator_mean,
                "mean_paired_delta_iae_pp": float(deltas.mean()),
                "relative_iae_improvement_fraction": float(
                    (comparator_mean - candidate_mean) / comparator_mean
                ),
                "bootstrap_mean_delta_ci_lower_pp": float(lower),
                "bootstrap_mean_delta_ci_upper_pp": float(upper),
                "candidate_better_condition_count": int(
                    np.sum(deltas < -tolerance)
                ),
                "candidate_worse_condition_count": int(
                    np.sum(deltas > tolerance)
                ),
                "candidate_equal_condition_count": int(
                    np.sum(np.abs(deltas) <= tolerance)
                ),
                "mechanism_gate_ready_condition_count": int(
                    group["activation_gate_ready"].sum()
                ),
                "descriptive_strict_superiority_criterion_met": bool(upper < 0.0),
                "bootstrap_inference_role": diagnostics["inference_role"],
                "shared_fold_dependence_adjusted": diagnostics[
                    "shared_fold_dependence_adjusted"
                ],
            }
        )
    return pd.DataFrame(rows).sort_values(grouping, kind="stable").reset_index(
        drop=True
    )


def _sensitivity_summary(
    metrics: pd.DataFrame,
    condition_metrics: pd.DataFrame,
) -> pd.DataFrame:
    baseline = condition_metrics.loc[
        (condition_metrics["method"] == PRIMARY_COMPARATOR)
        & (condition_metrics["prefix_checkups"] == PRIMARY_PREFIX),
        ["scenario", "target_condition_id", "trajectory_iae_pp"],
    ].rename(columns={"trajectory_iae_pp": "comparator_trajectory_iae_pp"})
    paired = metrics.merge(
        baseline,
        on=["scenario", "target_condition_id"],
        validate="many_to_one",
    )
    paired["paired_delta_iae_pp"] = (
        paired["trajectory_iae_pp"] - paired["comparator_trajectory_iae_pp"]
    )
    rows: list[dict[str, object]] = []
    for (scenario, tau), group in paired.groupby(
        ["scenario", "activation_timescale_days"], sort=True
    ):
        candidate = float(group["trajectory_iae_pp"].mean())
        comparator = float(group["comparator_trajectory_iae_pp"].mean())
        delta = group["paired_delta_iae_pp"].to_numpy(dtype=float)
        rows.append(
            {
                "scenario": scenario,
                "prefix_checkups": PRIMARY_PREFIX,
                "activation_timescale_days": float(tau),
                "independent_condition_count": len(group),
                "mechanism_gate_ready_condition_count": int(
                    group["activation_gate_ready"].sum()
                ),
                "candidate_trajectory_iae_pp_mean": candidate,
                "comparator_trajectory_iae_pp_mean": comparator,
                "mean_paired_delta_iae_pp": float(delta.mean()),
                "relative_iae_improvement_fraction": float(
                    (comparator - candidate) / comparator
                ),
                "candidate_better_condition_count": int(np.sum(delta < -1e-12)),
                "candidate_worse_condition_count": int(np.sum(delta > 1e-12)),
                "candidate_equal_condition_count": int(
                    np.sum(np.abs(delta) <= 1e-12)
                ),
                "is_core_robustness_timescale": float(tau) in CORE_TAU_DAYS,
                "mean_improvement_observed": bool(candidate < comparator),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario", "activation_timescale_days"], kind="stable"
    ).reset_index(drop=True)


def run_calendar_v3_activation_development(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> tuple:
    validate_naumann_calendar_observations(observations)
    parsed = validate_calendar_v3_activation_config(config)
    observations = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    all_ids = set(observations["condition_id"].astype(str))
    prediction_frames: list[pd.DataFrame] = []
    sensitivity_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []
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
            for prefix in EXPECTED_PREFIXES:
                training = _select_prefix(
                    observations.loc[
                        observations["condition_id"].astype(str).isin(training_ids)
                    ],
                    prefix,
                )
                base_prior = _base_prior(training, parsed["model"])
                activation_prior = _activation_prior(training, parsed["model"])
                training_hash = _training_state(
                    scenario=scenario_name,
                    fold_id=fold_id,
                    prefix_checkups=prefix,
                    training=training,
                    base_prior=base_prior,
                    activation_prior=activation_prior,
                    model=parsed["model"],
                )
                parameter_rows.extend(
                    _prior_parameter_rows(
                        scenario=scenario_name,
                        fold_id=fold_id,
                        prefix_checkups=prefix,
                        training_state_sha256=training_hash,
                        base_prior=base_prior,
                        activation_prior=activation_prior,
                    )
                )
                for target_id in sorted(target_set):
                    target = observations.loc[
                        observations["condition_id"].astype(str) == target_id
                    ].copy()
                    predictions, diagnostic, sensitivity = _target_state(
                        scenario=scenario_name,
                        fold_id=fold_id,
                        target=target,
                        prefix_checkups=prefix,
                        training_state_sha256=training_hash,
                        base_prior=base_prior,
                        activation_prior=activation_prior,
                        config=parsed,
                    )
                    prediction_frames.append(predictions)
                    diagnostic_rows.append(diagnostic)
                    sensitivity_rows.extend(sensitivity)
    predictions = pd.concat(prediction_frames, ignore_index=True)[PREDICTION_COLUMNS]
    predictions = predictions.sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    prediction_hash = calendar_v3_prediction_sha256(predictions)
    condition_metrics = score_calendar_v3_predictions(
        predictions, observations, frozen_prediction_sha256=prediction_hash
    )
    paired = _paired_metrics(condition_metrics)
    comparisons = _comparison_summary(paired, diagnostics=parsed["diagnostics"])

    sensitivity_predictions = pd.DataFrame(sensitivity_rows)[SENSITIVITY_COLUMNS]
    sensitivity_predictions = sensitivity_predictions.sort_values(
        SENSITIVITY_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    sensitivity_hash = calendar_v3_sensitivity_sha256(sensitivity_predictions)
    sensitivity_metrics = score_calendar_v3_sensitivity(
        sensitivity_predictions,
        observations,
        frozen_prediction_sha256=sensitivity_hash,
    )
    sensitivity_summary = _sensitivity_summary(
        sensitivity_metrics, condition_metrics
    )
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["scenario", "target_condition_id", "prefix_checkups"], kind="stable"
    ).reset_index(drop=True)
    parameters = pd.DataFrame(parameter_rows).sort_values(
        [
            "scenario",
            "fold_id",
            "prefix_checkups",
            "model",
            "parameter_group",
            "parameter",
            "feature",
        ],
        kind="stable",
    ).reset_index(drop=True)
    splits = pd.DataFrame(split_rows).drop_duplicates().sort_values(
        ["scenario", "fold_id", "condition_id"], kind="stable"
    ).reset_index(drop=True)
    primary = comparisons.loc[
        (comparisons["prefix_checkups"] == PRIMARY_PREFIX)
        & (comparisons["candidate_method"] == PRIMARY_CANDIDATE)
    ]
    primary_scenarios_present = set(primary["scenario"].astype(str)) == set(
        GATE_SCENARIOS
    )
    mean_improvement_all = bool(
        primary_scenarios_present
        and (primary["mean_paired_delta_iae_pp"] < 0.0).all()
    )
    strict_bootstrap_all = bool(
        primary_scenarios_present
        and primary["descriptive_strict_superiority_criterion_met"].all()
    )
    core_sensitivity = sensitivity_summary.loc[
        sensitivity_summary["is_core_robustness_timescale"]
    ]
    core_tau_improvement_all = bool(
        len(core_sensitivity) == len(GATE_SCENARIOS) * len(CORE_TAU_DAYS)
        and core_sensitivity["mean_improvement_observed"].all()
    )
    gate_ready = diagnostics.loc[
        (diagnostics["prefix_checkups"] == PRIMARY_PREFIX)
        & diagnostics["activation_gate_ready"]
    ]
    result: dict[str, object] = {
        "status": "mechanism_development_diagnostic_complete_confirmation_blocked",
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
            "prefix_checkups": list(EXPECTED_PREFIXES),
            "primary_prefix_checkups": PRIMARY_PREFIX,
            "methods": list(METHOD_NAMES),
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_comparator": PRIMARY_COMPARATOR,
            "mechanism_gate": dict(parsed["mechanism_gate"]),
            "activation_timescale_days": float(
                parsed["model"]["activation_timescale_days"]
            ),
            "timescale_selection_status": parsed["timescale_sensitivity"][
                "selection_status"
            ],
            "target_future_outcomes_used_for_prediction": False,
            "training_future_outcomes_used_for_fit": False,
        },
        "primary_comparison_summary": primary.to_dict(orient="records"),
        "mechanism_support": {
            "unique_gate_ready_condition_ids_at_primary_prefix": sorted(
                gate_ready["target_condition_id"].astype(str).unique()
            ),
            "unique_gate_ready_condition_count_at_primary_prefix": int(
                gate_ready["target_condition_id"].nunique()
            ),
            "gate_ready_scenario_condition_rows": len(gate_ready),
            "effect_is_sparse_and_low_soc_specific": True,
        },
        "timescale_sensitivity": {
            "timescale_days": list(TAU_SENSITIVITY_DAYS),
            "core_robustness_timescale_days": list(CORE_TAU_DAYS),
            "all_core_timescales_improve_mean_iae_in_both_scenarios": (
                core_tau_improvement_all
            ),
            "selection_used_inspected_outcomes": True,
            "formal_hyperparameter_selection_claim_allowed": False,
        },
        "development_gate": {
            "confirmation_status": CONFIRMATION_STATUS,
            "mean_improvement_in_both_primary_scenarios": mean_improvement_all,
            "strict_bootstrap_superiority_in_both_primary_scenarios": (
                strict_bootstrap_all
            ),
            "core_timescale_robustness_passed": core_tau_improvement_all,
            "descriptive_signal_status": (
                "mechanistic_mean_signal_with_sparse_condition_support"
                if mean_improvement_all and core_tau_improvement_all
                else "mechanistic_signal_failed"
            ),
            "current_dataset_relationship": "reused_and_outcomes_already_inspected",
            "reason": (
                "The mechanism and seven-day timescale were introduced after "
                "inspecting the Phase 7 failure. Only three unique conditions trigger "
                "the gate, so mean improvement cannot establish general low-SOC "
                "validity or independent superiority."
            ),
        },
        "future_label_firewall": {
            "label_free_prediction_sha256": prediction_hash,
            "tau_sensitivity_label_free_prediction_sha256": sensitivity_hash,
            "score_after_prediction_hash_verification": True,
            "future_outcome_columns_in_prediction_packs": [],
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
        "claim_boundary": (
            "This is a post-hoc mechanism-development experiment on 17 reused "
            "Naumann condition means. It does not validate activation physics, "
            "individual cells, Hithium products, plants, or 15-25 year forecasts."
        ),
    }
    return (
        result,
        predictions,
        condition_metrics,
        paired,
        comparisons,
        diagnostics,
        parameters,
        splits,
        sensitivity_predictions,
        sensitivity_metrics,
        sensitivity_summary,
    )
