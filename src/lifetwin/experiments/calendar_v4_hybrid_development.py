from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.experiments.calendar_v2_development import (
    EXPECTED_DATASET_SNAPSHOT_ID,
    EXPECTED_LABEL_VERSION,
)
from lifetwin.models.calendar_v2 import (
    fit_hierarchical_power_prior,
    predict_power_loss,
    stress_features_for_condition,
    update_hierarchical_power_law,
)
from lifetwin.models.calendar_v2_uncertainty import (
    finite_sample_higher_quantile,
    interval_score,
    power_law_predictive_sd,
)
from lifetwin.models.calendar_v3_activation import (
    activation_mechanism_gate,
    fit_hierarchical_activation_offset_prior,
    predict_activation_offset_loss,
    update_hierarchical_activation_offset,
)
from lifetwin.models.calendar_v4_hybrid import (
    MeanPredictionRoute,
    ResidualSupportError,
    activation_offset_predictive_sd,
    conservative_issuance_decision,
    fit_bounded_residual_correction,
    predict_bounded_residual_correction,
)


EXPERIMENT_ID = "naumann_calendar_v4_hybrid_development_v1"
DESIGN_STATUS = "retrospective_locked_after_outcome_access"
EVIDENCE_ROLE = "reused_naumann_hybrid_uncertainty_development_only"
SPLIT_ID = "naumann_v4_fixed_7_train_6_calibration_4_test_v1"
SELECTION_STATUS = "post_hoc_locked_after_outcome_access"
CONFIRMATION_STATUS = "not_confirmed"
PREFIX_CHECKUPS = 10
PREFIX_END_INDEX = 9
FORECAST_START_INDEX = 10
FORECAST_END_INDEX = 34
FORECAST_INDICES = tuple(range(FORECAST_START_INDEX, FORECAST_END_INDEX + 1))
REQUESTED_COVERAGES = (0.8, 0.9, 0.95)
PRIMARY_COVERAGE = 0.8
SPECIALIST_ROUTE = MeanPredictionRoute.SPECIALIST.value
FALLBACK_ROUTE = MeanPredictionRoute.FALLBACK.value
ROUTES = (SPECIALIST_ROUTE, FALLBACK_ROUTE)
DIAGNOSTIC_AVAILABLE = "available"
DIAGNOSTIC_UNAVAILABLE = "unavailable"
OPERATIONAL_ABSTAINED = "abstained"
NONE_REASON = "none"
OPERATIONAL_BASE_REASONS = (
    "calibration_evidence_not_independent",
    "independent_long_term_evidence_missing",
)

TRAINING_CONDITION_IDS = (
    "NAUMANN_CAL_T0_SOC50",
    "NAUMANN_CAL_T10_SOC50",
    "NAUMANN_CAL_T25_SOC0",
    "NAUMANN_CAL_T25_SOC100",
    "NAUMANN_CAL_T60_SOC0",
    "NAUMANN_CAL_T60_SOC50",
    "NAUMANN_CAL_T60_SOC100",
)
CALIBRATION_CONDITION_IDS = (
    "NAUMANN_CAL_T25_SOC50",
    "NAUMANN_CAL_T40_SOC0",
    "NAUMANN_CAL_T40_SOC25",
    "NAUMANN_CAL_T40_SOC50",
    "NAUMANN_CAL_T40_SOC75",
    "NAUMANN_CAL_T40_SOC100",
)
TEST_CONDITION_IDS = (
    "NAUMANN_CAL_T40_SOC12.5",
    "NAUMANN_CAL_T40_SOC37.5",
    "NAUMANN_CAL_T40_SOC62.5",
    "NAUMANN_CAL_T40_SOC87.5",
)
EXPECTED_PROHIBITED_CLAIMS = (
    "confirmatory_performance_on_naumann_reuse",
    "formal_finite_sample_coverage_on_naumann_reuse",
    "independent_external_validation",
    "individual_cell_uncertainty",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

PREDICTION_KEY_COLUMNS = [
    "target_condition_id",
    "prefix_checkups",
    "requested_coverage",
    "target_checkup_index",
]
PREDICTION_COLUMNS = [
    "experiment_id",
    "split_id",
    "dataset_id",
    *PREDICTION_KEY_COLUMNS,
    "prefix_end_checkup_index",
    "prefix_end_days",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_days",
    "forecast_horizon_days",
    "is_final_checkup",
    "mean_route",
    "mean_fallback_reasons",
    "activation_gate_ready",
    "domain_supported",
    "residual_support_ok",
    "residual_cap_hit",
    "residual_correction_pp",
    "predicted_capacity_retention_pct",
    "predictive_sd_pp",
    "calibration_condition_count",
    "calibration_order_statistic_rank",
    "calibration_multiplier",
    "calibration_horizon_matched",
    "diagnostic_interval_status",
    "diagnostic_abstention_reasons",
    "diagnostic_lower_pct",
    "diagnostic_upper_pct",
    "diagnostic_width_pp",
    "operational_issuance_status",
    "operational_abstention_reasons",
    "operational_lower_pct",
    "operational_upper_pct",
    "mechanistic_training_support_days",
    "residual_support_horizon_days",
    "training_state_sha256",
    "calibration_state_sha256",
    "prediction_state_sha256",
]

RESIDUAL_COLUMNS = [
    "source_condition_id",
    "source_condition_role",
    "pseudo_training_condition_ids",
    "pseudo_training_condition_count",
    "prefix_checkups",
    "target_checkup_index",
    "prefix_end_days",
    "elapsed_days",
    "forecast_horizon_days",
    "true_capacity_retention_pct",
    "raw_activation_predicted_retention_pct",
    "retention_residual_pp",
    "source_model_state_sha256",
    "source_residual_state_sha256",
]

CALIBRATION_SCORE_COLUMNS = [
    "calibration_condition_id",
    "mean_route",
    "mean_fallback_reasons",
    "activation_gate_ready",
    "domain_supported",
    "residual_support_ok",
    "residual_cap_hit",
    "calibration_point_count",
    "calibration_start_checkup_index",
    "calibration_end_checkup_index",
    "calibration_start_horizon_days",
    "calibration_end_horizon_days",
    "maximum_standardized_error",
    "maximum_absolute_error_pp",
    "predictive_sd_min_pp",
    "predictive_sd_max_pp",
    "training_state_sha256",
    "calibration_prediction_state_sha256",
]

CALIBRATION_QUANTILE_COLUMNS = [
    "mean_route",
    "requested_coverage",
    "calibration_condition_count",
    "order_statistic_rank",
    "multiplier",
    "status",
]

CONDITION_METRIC_COLUMNS = [
    "target_condition_id",
    "mean_route",
    "requested_coverage",
    "future_point_count",
    "trajectory_iae_pp",
    "point_mae_pp",
    "final_error_pp",
    "final_absolute_error_pp",
    "diagnostic_interval_status",
    "diagnostic_simultaneous_covered",
    "diagnostic_pointwise_coverage_fraction",
    "diagnostic_mean_width_pp",
    "diagnostic_interval_score_mean",
    "operational_issuance_status",
]


@dataclass(frozen=True)
class _ConditionPrediction:
    condition_id: str
    prefix_end_days: float
    future: pd.DataFrame
    predicted_retention_pct: np.ndarray
    predictive_sd_pp: np.ndarray
    residual_correction_pp: np.ndarray
    mean_route: str
    mean_fallback_reasons: tuple[str, ...]
    activation_gate_ready: bool
    domain_supported: bool
    residual_support_ok: bool
    residual_cap_hit: bool
    prediction_state_sha256: str


def default_calendar_v4_hybrid_config() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "evidence_role": EVIDENCE_ROLE,
        "split": {
            "split_id": SPLIT_ID,
            "selection_status": SELECTION_STATUS,
            "training_condition_ids": list(TRAINING_CONDITION_IDS),
            "calibration_condition_ids": list(CALIBRATION_CONDITION_IDS),
            "test_condition_ids": list(TEST_CONDITION_IDS),
        },
        "landmark": {
            "prefix_checkups": PREFIX_CHECKUPS,
            "prefix_end_checkup_index": PREFIX_END_INDEX,
            "forecast_start_checkup_index": FORECAST_START_INDEX,
            "forecast_end_checkup_index": FORECAST_END_INDEX,
            "history_policy": (
                "target_prefix_with_historical_training_residual_library"
            ),
        },
        "model": {
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
            "mechanism_gate_minimum_positive_time_observations": 7,
            "mechanism_gate_negative_loss_threshold_pp": 0.0,
        },
        "residual": {
            "method": "landmark_anchored_bounded_ridge_v1",
            "source_prediction_policy": (
                "training_condition_leave_one_out_activation_specialist"
            ),
            "minimum_source_conditions": 7,
            "ridge_penalty": 1.0,
            "correction_cap_pp": 2.0,
            "support_end_checkup_index": FORECAST_END_INDEX,
            "apply_to_route": SPECIALIST_ROUTE,
        },
        "uncertainty": {
            "requested_coverages": list(REQUESTED_COVERAGES),
            "primary_coverage": PRIMARY_COVERAGE,
            "calibration_unit": "condition_mean_trajectory",
            "calibration_partition": "selected_mean_route",
            "score": "maximum_standardized_error_over_checkups_10_to_34",
            "quantile_rule": (
                "ceil((n+1)*coverage)_higher_or_unavailable"
            ),
            "predictive_scale_floor_pp": 0.1,
            "physical_bounds_pct": [0.0, 100.0],
            "max_interval_width_pp": None,
            "formal_coverage_claim_allowed": False,
        },
        "issuance": {
            "diagnostic_interval_role": (
                "reused_data_horizon_matched_diagnostic_only"
            ),
            "operational_issuance_enabled": False,
            "operational_abstention_reasons": list(OPERATIONAL_BASE_REASONS),
        },
        "confirmation_status": CONFIRMATION_STATUS,
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
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


def validate_calendar_v4_hybrid_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("Calendar V4 config must be an object")
    parsed = json.loads(json.dumps(dict(config), allow_nan=False))
    expected = default_calendar_v4_hybrid_config()
    if _canonical_json_sha256(parsed) != _canonical_json_sha256(expected):
        raise ValueError(
            "Calendar V4 config differs from the locked retrospective protocol"
        )
    return parsed


def _select_prefix(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        pd.to_numeric(frame["checkup_index"]) < PREFIX_CHECKUPS
    ].copy()
    counts = selected.groupby("condition_id", sort=True).size()
    if selected.empty or (counts != PREFIX_CHECKUPS).any():
        raise ValueError("Every selected condition must have the complete p=10 prefix")
    return selected.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)


def _future_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition_id",
        "checkup_index",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
    ]
    future = frame.loc[
        pd.to_numeric(frame["checkup_index"]).between(
            FORECAST_START_INDEX, FORECAST_END_INDEX
        ),
        columns,
    ].copy()
    future = future.sort_values("checkup_index", kind="stable").reset_index(drop=True)
    indices = pd.to_numeric(future["checkup_index"]).astype(int).tolist()
    if indices != list(FORECAST_INDICES):
        raise ValueError("Every V4 condition must have checkups 10 through 34")
    return future


def _condition_frame(observations: pd.DataFrame, condition_id: str) -> pd.DataFrame:
    selected = observations.loc[
        observations["condition_id"].astype(str) == condition_id
    ].copy()
    if selected.empty:
        raise ValueError(f"Missing locked V4 condition: {condition_id}")
    return selected.sort_values("checkup_index", kind="stable").reset_index(drop=True)


def _model_kwargs(config: Mapping[str, object]) -> dict[str, object]:
    model = dict(config["model"])
    return {
        "minimum_conditions": int(model["minimum_training_conditions"]),
        "exponent_bounds": tuple(
            float(value) for value in model["power_exponent_bounds"]
        ),
        "robust_loss_scale_pp": float(model["robust_loss_scale_pp"]),
        "stress_surface_ridge": float(model["stress_surface_ridge"]),
        "observation_scale_floor_pp": float(
            model["observation_scale_floor_pp"]
        ),
    }


def _fit_power_prior(training_prefix: pd.DataFrame, config: Mapping[str, object]):
    model = dict(config["model"])
    kwargs = _model_kwargs(config)
    return fit_hierarchical_power_prior(
        training_prefix,
        parameter_scale_floors=tuple(
            float(value) for value in model["base_parameter_scale_floors"]
        ),
        **kwargs,
    )


def _fit_activation_prior(
    training_prefix: pd.DataFrame,
    config: Mapping[str, object],
):
    model = dict(config["model"])
    kwargs = _model_kwargs(config)
    return fit_hierarchical_activation_offset_prior(
        training_prefix,
        activation_timescale_days=float(model["activation_timescale_days"]),
        activation_offset_bounds_pp=tuple(
            float(value) for value in model["activation_offset_bounds_pp"]
        ),
        parameter_scale_floors=tuple(
            float(value)
            for value in model["activation_parameter_scale_floors"]
        ),
        **kwargs,
    )


def _prior_payload(prior: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "surface_coefficients": [
            list(values) for values in prior.surface_coefficients
        ],
        "parameter_scales": list(prior.parameter_scales),
        "observation_scale_pp": float(prior.observation_scale_pp),
        "training_condition_ids": list(prior.training_condition_ids),
        "training_observation_count": int(prior.training_observation_count),
        "maximum_training_days": float(prior.maximum_training_days),
        "condition_parameters": [
            list(values) for values in prior.condition_parameters
        ],
    }
    if hasattr(prior, "activation_timescale_days"):
        payload["activation_timescale_days"] = float(
            prior.activation_timescale_days
        )
    return payload


def _stress_design(
    observations: pd.DataFrame,
    condition_ids: Sequence[str],
) -> np.ndarray:
    rows = [
        stress_features_for_condition(
            _select_prefix(_condition_frame(observations, condition_id))
        )
        for condition_id in condition_ids
    ]
    return np.vstack(rows)


def validate_calendar_v4_split_and_rank(observations: pd.DataFrame) -> pd.DataFrame:
    expected = set(TRAINING_CONDITION_IDS) | set(CALIBRATION_CONDITION_IDS) | set(
        TEST_CONDITION_IDS
    )
    observed = set(observations["condition_id"].astype(str))
    if expected != observed or len(expected) != 17:
        raise ValueError("The locked V4 split must partition all 17 conditions")
    roles = [
        *(dict(condition_id=value, role="training") for value in TRAINING_CONDITION_IDS),
        *(
            dict(condition_id=value, role="calibration")
            for value in CALIBRATION_CONDITION_IDS
        ),
        *(dict(condition_id=value, role="test") for value in TEST_CONDITION_IDS),
    ]
    splits = pd.DataFrame(roles)
    if splits["condition_id"].duplicated().any():
        raise ValueError("A condition cannot occupy multiple V4 roles")

    full_design = _stress_design(observations, TRAINING_CONDITION_IDS)
    if full_design.shape != (7, 5) or np.linalg.matrix_rank(full_design) != 5:
        raise ValueError("The seven-condition V4 stress design must have rank five")
    for source_id in TRAINING_CONDITION_IDS:
        remaining = tuple(value for value in TRAINING_CONDITION_IDS if value != source_id)
        design = _stress_design(observations, remaining)
        if design.shape != (6, 5) or np.linalg.matrix_rank(design) != 5:
            raise ValueError(
                f"LOCO stress design must retain rank five for {source_id}"
            )
    return splits.sort_values("condition_id", kind="stable").reset_index(drop=True)


def _domain_hull(observations: pd.DataFrame) -> ConvexHull:
    points: list[tuple[float, float]] = []
    for condition_id in TRAINING_CONDITION_IDS:
        condition = _condition_frame(observations, condition_id)
        points.append(
            (
                float(condition["temperature_c"].iloc[0]),
                float(condition["storage_soc_fraction"].iloc[0]),
            )
        )
    return ConvexHull(np.asarray(points, dtype=float))


def _condition_in_training_hull(condition: pd.DataFrame, hull: ConvexHull) -> bool:
    point = np.asarray(
        [
            float(condition["temperature_c"].iloc[0]),
            float(condition["storage_soc_fraction"].iloc[0]),
        ],
        dtype=float,
    )
    values = hull.equations[:, :-1] @ point + hull.equations[:, -1]
    return bool(np.all(values <= 1e-10))


def _fit_training_residual(
    observations: pd.DataFrame,
    config: Mapping[str, object],
    *,
    prefix_end_days: float,
    support_horizon_days: float,
):
    rows: list[dict[str, object]] = []
    model = dict(config["model"])
    exponent_bounds = tuple(float(v) for v in model["power_exponent_bounds"])
    offset_bounds = tuple(float(v) for v in model["activation_offset_bounds_pp"])
    for source_id in TRAINING_CONDITION_IDS:
        pseudo_training_ids = tuple(
            value for value in TRAINING_CONDITION_IDS if value != source_id
        )
        pseudo_training = observations.loc[
            observations["condition_id"].astype(str).isin(pseudo_training_ids)
        ]
        pseudo_training_prefix = _select_prefix(pseudo_training)
        prior = _fit_activation_prior(pseudo_training_prefix, config)
        source = _condition_frame(observations, source_id)
        source_prefix = _select_prefix(source)
        fitted = update_hierarchical_activation_offset(
            prior,
            source_prefix,
            exponent_bounds=exponent_bounds,
            activation_offset_bounds_pp=offset_bounds,
        )
        future_coordinates = _future_coordinates(source)
        elapsed = future_coordinates["elapsed_days"].to_numpy(dtype=float)
        predicted = 100.0 - predict_activation_offset_loss(fitted, elapsed)
        truth = source.loc[
            source["checkup_index"].isin(FORECAST_INDICES),
            ["checkup_index", "capacity_retention_pct"],
        ].sort_values("checkup_index", kind="stable")
        if truth["checkup_index"].astype(int).tolist() != list(FORECAST_INDICES):
            raise ValueError("Residual source future must be complete")
        observed = truth["capacity_retention_pct"].to_numpy(dtype=float)
        residual = observed - predicted
        source_model_state = {
            "source_condition_id": source_id,
            "pseudo_training_condition_ids": list(pseudo_training_ids),
            "pseudo_training_prefix": _prefix_payload(pseudo_training_prefix),
            "source_prefix": _prefix_payload(source_prefix),
            "activation_prior": _prior_payload(prior),
            "posterior_parameters": fitted.parameter_map(),
            "future_coordinates": _coordinate_payload(future_coordinates),
            "predicted_retention_pct": [float(value) for value in predicted],
        }
        model_hash = _canonical_json_sha256(source_model_state)
        residual_state = {
            "source_model_state_sha256": model_hash,
            "observed_retention_pct": [float(value) for value in observed],
            "retention_residual_pp": [float(value) for value in residual],
        }
        residual_hash = _canonical_json_sha256(residual_state)
        for coordinate, observed_value, predicted_value, residual_value in zip(
            future_coordinates.itertuples(index=False),
            observed,
            predicted,
            residual,
            strict=True,
        ):
            rows.append(
                {
                    "source_condition_id": source_id,
                    "source_condition_role": "training",
                    "pseudo_training_condition_ids": ";".join(
                        pseudo_training_ids
                    ),
                    "pseudo_training_condition_count": len(pseudo_training_ids),
                    "prefix_checkups": PREFIX_CHECKUPS,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "prefix_end_days": prefix_end_days,
                    "elapsed_days": float(coordinate.elapsed_days),
                    "forecast_horizon_days": (
                        float(coordinate.elapsed_days) - prefix_end_days
                    ),
                    "true_capacity_retention_pct": float(observed_value),
                    "raw_activation_predicted_retention_pct": float(
                        predicted_value
                    ),
                    "retention_residual_pp": float(residual_value),
                    "source_model_state_sha256": model_hash,
                    "source_residual_state_sha256": residual_hash,
                }
            )
    residual_frame = pd.DataFrame(rows, columns=RESIDUAL_COLUMNS).sort_values(
        ["source_condition_id", "target_checkup_index"], kind="stable"
    ).reset_index(drop=True)
    expected_rows = len(TRAINING_CONDITION_IDS) * len(FORECAST_INDICES)
    if len(residual_frame) != expected_rows:
        raise ValueError("V4 residual cross-fit must contain exactly 175 rows")
    residual_config = dict(config["residual"])
    residual_input_state_sha256 = _canonical_json_sha256(
        {
            "split_id": SPLIT_ID,
            "prefix_checkups": PREFIX_CHECKUPS,
            "source_condition_ids": list(TRAINING_CONDITION_IDS),
            "residual_crossfit_sha256": _canonical_frame_sha256(
                residual_frame,
                sort_by=["source_condition_id", "target_checkup_index"],
            ),
        }
    )
    fitted = fit_bounded_residual_correction(
        residual_frame["forecast_horizon_days"].to_numpy(dtype=float),
        residual_frame["retention_residual_pp"].to_numpy(dtype=float),
        support_horizon_days=support_horizon_days,
        correction_cap_pp=float(residual_config["correction_cap_pp"]),
        ridge_penalty=float(residual_config["ridge_penalty"]),
        training_condition_ids=residual_frame["source_condition_id"].astype(
            str
        ),
        landmark_days=prefix_end_days,
        upstream_training_state_sha256=residual_input_state_sha256,
    )
    return fitted, residual_frame


def _prefix_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "condition_id": str(row.condition_id),
            "checkup_index": int(row.checkup_index),
            "temperature_c": float(row.temperature_c),
            "storage_soc_fraction": float(row.storage_soc_fraction),
            "elapsed_days": float(row.elapsed_days),
            "capacity_loss_pct": float(row.capacity_loss_pct),
        }
        for row in frame.sort_values(
            ["condition_id", "checkup_index"], kind="stable"
        ).itertuples(index=False)
    ]


def _coordinate_payload(frame: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {
            "condition_id": str(row.condition_id),
            "checkup_index": int(row.checkup_index),
            "temperature_c": float(row.temperature_c),
            "storage_soc_fraction": float(row.storage_soc_fraction),
            "elapsed_days": float(row.elapsed_days),
        }
        for row in frame.sort_values("checkup_index", kind="stable").itertuples(
            index=False
        )
    ]


def _predict_condition(
    condition: pd.DataFrame,
    *,
    power_prior: object,
    activation_prior: object,
    residual_fit: object,
    training_state_sha256: str,
    config: Mapping[str, object],
    domain_hull: ConvexHull,
) -> _ConditionPrediction:
    ordered = condition.sort_values("checkup_index", kind="stable").reset_index(
        drop=True
    )
    condition_id = str(ordered["condition_id"].iloc[0])
    prefix = _select_prefix(ordered)
    prefix_end_days = float(
        prefix.loc[prefix["checkup_index"] == PREFIX_END_INDEX, "elapsed_days"].iloc[
            0
        ]
    )
    future = _future_coordinates(ordered)
    elapsed = future["elapsed_days"].to_numpy(dtype=float)
    horizons = elapsed - prefix_end_days
    model = dict(config["model"])
    exponent_bounds = tuple(float(v) for v in model["power_exponent_bounds"])
    offset_bounds = tuple(float(v) for v in model["activation_offset_bounds_pp"])
    gate = activation_mechanism_gate(
        prefix,
        minimum_positive_time_observations=int(
            model["mechanism_gate_minimum_positive_time_observations"]
        ),
        negative_loss_threshold_pp=float(
            model["mechanism_gate_negative_loss_threshold_pp"]
        ),
    )

    fallback_fit = update_hierarchical_power_law(
        power_prior,
        prefix,
        exponent_bounds=exponent_bounds,
    )
    fallback_retention = 100.0 - predict_power_loss(fallback_fit, elapsed)
    fallback_sd = power_law_predictive_sd(
        fallback_fit,
        elapsed,
        observation_scale_pp=float(power_prior.observation_scale_pp),
        scale_floor_pp=float(
            dict(config["uncertainty"])["predictive_scale_floor_pp"]
        ),
    )

    route = FALLBACK_ROUTE
    fallback_reasons: list[str] = []
    predicted = fallback_retention
    predictive_sd = fallback_sd
    correction = np.zeros(len(future), dtype=float)
    residual_support_ok = True
    residual_cap_hit = False
    specialist_fit_succeeded = False
    specialist_parameters: dict[str, object] = {"fit_status": "not_attempted"}
    if not gate.ready:
        fallback_reasons.append("specialist_gate_not_ready")
    else:
        try:
            activation_fit = update_hierarchical_activation_offset(
                activation_prior,
                prefix,
                exponent_bounds=exponent_bounds,
                activation_offset_bounds_pp=offset_bounds,
            )
            specialist_fit_succeeded = True
            specialist_parameters = activation_fit.parameter_map()
            raw_specialist = 100.0 - predict_activation_offset_loss(
                activation_fit, elapsed
            )
            residual_prediction = predict_bounded_residual_correction(
                residual_fit, horizons
            )
            residual_cap_hit = residual_prediction.any_cap_hit
            correction = np.asarray(
                residual_prediction.correction_pp, dtype=float
            )
            if residual_cap_hit:
                fallback_reasons.append("residual_cap_hit")
            else:
                route = SPECIALIST_ROUTE
                predicted = raw_specialist + correction
                predictive_sd = activation_offset_predictive_sd(
                    activation_fit,
                    elapsed,
                    observation_scale_pp=float(
                        activation_prior.observation_scale_pp
                    ),
                    scale_floor_pp=float(
                        dict(config["uncertainty"])[
                            "predictive_scale_floor_pp"
                        ]
                    ),
                )
        except ResidualSupportError:
            residual_support_ok = False
            fallback_reasons.append("residual_outside_support")
        except Exception as exc:
            fallback_reasons.append("specialist_fit_failed")
            specialist_parameters = {
                "fit_status": "failed",
                "error_type": type(exc).__name__,
            }

    if not np.isfinite(predicted).all() or not np.isfinite(predictive_sd).all():
        raise RuntimeError("Calendar V4 mean and scale must remain finite")
    if np.any(predictive_sd <= 0.0):
        raise RuntimeError("Calendar V4 predictive scales must be positive")
    if np.any((predicted < 0.0) | (predicted > 100.0)):
        raise RuntimeError("Calendar V4 mean must remain within physical bounds")
    domain_supported = _condition_in_training_hull(ordered, domain_hull)
    state = {
        "condition_id": condition_id,
        "prefix_checkups": PREFIX_CHECKUPS,
        "training_state_sha256": training_state_sha256,
        "prefix": _prefix_payload(prefix),
        "future_coordinates": _coordinate_payload(future),
        "gate": gate.__dict__,
        "domain_supported": domain_supported,
        "mean_route": route,
        "mean_fallback_reasons": fallback_reasons,
        "specialist_fit_succeeded": specialist_fit_succeeded,
        "specialist_parameters": specialist_parameters,
        "fallback_parameters": fallback_fit.parameter_map(),
        "residual_support_ok": residual_support_ok,
        "residual_cap_hit": residual_cap_hit,
        "residual_correction_pp": [float(value) for value in correction],
        "predicted_retention_pct": [float(value) for value in predicted],
        "predictive_sd_pp": [float(value) for value in predictive_sd],
    }
    return _ConditionPrediction(
        condition_id=condition_id,
        prefix_end_days=prefix_end_days,
        future=future,
        predicted_retention_pct=np.asarray(predicted, dtype=float),
        predictive_sd_pp=np.asarray(predictive_sd, dtype=float),
        residual_correction_pp=np.asarray(correction, dtype=float),
        mean_route=route,
        mean_fallback_reasons=tuple(fallback_reasons),
        activation_gate_ready=gate.ready,
        domain_supported=domain_supported,
        residual_support_ok=residual_support_ok,
        residual_cap_hit=residual_cap_hit,
        prediction_state_sha256=_canonical_json_sha256(state),
    )


def _build_training_state(
    observations: pd.DataFrame,
    config: Mapping[str, object],
):
    training = observations.loc[
        observations["condition_id"].astype(str).isin(TRAINING_CONDITION_IDS)
    ].copy()
    training_prefix = _select_prefix(training)
    prefix_end_days = float(
        training_prefix.loc[
            training_prefix["checkup_index"] == PREFIX_END_INDEX, "elapsed_days"
        ].iloc[0]
    )
    support_end_days = float(
        training.loc[
            training["checkup_index"] == FORECAST_END_INDEX, "elapsed_days"
        ].iloc[0]
    )
    support_horizon_days = support_end_days - prefix_end_days
    if support_horizon_days <= 0.0:
        raise ValueError("Residual support horizon must be positive")
    power_prior = _fit_power_prior(training_prefix, config)
    activation_prior = _fit_activation_prior(training_prefix, config)
    residual_fit, residual_frame = _fit_training_residual(
        observations,
        config,
        prefix_end_days=prefix_end_days,
        support_horizon_days=support_horizon_days,
    )
    residual_hash = _canonical_frame_sha256(
        residual_frame,
        sort_by=["source_condition_id", "target_checkup_index"],
    )
    state = {
        "split_id": SPLIT_ID,
        "training_condition_ids": list(TRAINING_CONDITION_IDS),
        "training_prefix": _prefix_payload(training_prefix),
        "power_prior": _prior_payload(power_prior),
        "activation_prior": _prior_payload(activation_prior),
        "residual_fit": {
            "coefficients": list(residual_fit.coefficients),
            "ridge_penalty": float(residual_fit.ridge_penalty),
            "support_horizon_days": float(residual_fit.support_horizon_days),
            "correction_cap_pp": float(residual_fit.correction_cap_pp),
            "training_observation_count": int(
                residual_fit.training_observation_count
            ),
            "training_condition_ids": list(
                residual_fit.training_condition_ids
            ),
            "landmark_days": float(residual_fit.landmark_days),
            "observed_max_horizon_days": float(
                residual_fit.observed_max_horizon_days
            ),
            "upstream_training_state_sha256": str(
                residual_fit.upstream_training_state_sha256
            ),
            "residual_training_state_sha256": str(
                residual_fit.residual_training_state_sha256
            ),
            "basis_name": str(residual_fit.basis_name),
        },
        "residual_crossfit_sha256": residual_hash,
        "config_sha256": _canonical_json_sha256(config),
    }
    training_hash = _canonical_json_sha256(state)
    return (
        power_prior,
        activation_prior,
        residual_fit,
        residual_frame,
        training_hash,
        prefix_end_days,
        support_horizon_days,
    )


def _build_calibration(
    observations: pd.DataFrame,
    *,
    power_prior: object,
    activation_prior: object,
    residual_fit: object,
    training_state_sha256: str,
    config: Mapping[str, object],
    domain_hull: ConvexHull,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows: list[dict[str, object]] = []
    for condition_id in CALIBRATION_CONDITION_IDS:
        condition = _condition_frame(observations, condition_id)
        predicted = _predict_condition(
            condition,
            power_prior=power_prior,
            activation_prior=activation_prior,
            residual_fit=residual_fit,
            training_state_sha256=training_state_sha256,
            config=config,
            domain_hull=domain_hull,
        )
        truth = condition.loc[
            condition["checkup_index"].isin(FORECAST_INDICES),
            ["checkup_index", "capacity_retention_pct"],
        ].sort_values("checkup_index", kind="stable")
        if truth["checkup_index"].astype(int).tolist() != list(FORECAST_INDICES):
            raise ValueError("Calibration truth must contain the complete horizon")
        observed = truth["capacity_retention_pct"].to_numpy(dtype=float)
        absolute = np.abs(observed - predicted.predicted_retention_pct)
        standardized = absolute / predicted.predictive_sd_pp
        horizons = (
            predicted.future["elapsed_days"].to_numpy(dtype=float)
            - predicted.prefix_end_days
        )
        rows.append(
            {
                "calibration_condition_id": condition_id,
                "mean_route": predicted.mean_route,
                "mean_fallback_reasons": _reason_string(
                    predicted.mean_fallback_reasons
                ),
                "activation_gate_ready": predicted.activation_gate_ready,
                "domain_supported": predicted.domain_supported,
                "residual_support_ok": predicted.residual_support_ok,
                "residual_cap_hit": predicted.residual_cap_hit,
                "calibration_point_count": len(FORECAST_INDICES),
                "calibration_start_checkup_index": FORECAST_START_INDEX,
                "calibration_end_checkup_index": FORECAST_END_INDEX,
                "calibration_start_horizon_days": float(horizons[0]),
                "calibration_end_horizon_days": float(horizons[-1]),
                "maximum_standardized_error": float(standardized.max()),
                "maximum_absolute_error_pp": float(absolute.max()),
                "predictive_sd_min_pp": float(
                    predicted.predictive_sd_pp.min()
                ),
                "predictive_sd_max_pp": float(
                    predicted.predictive_sd_pp.max()
                ),
                "training_state_sha256": training_state_sha256,
                "calibration_prediction_state_sha256": (
                    predicted.prediction_state_sha256
                ),
            }
        )
    scores = pd.DataFrame(rows, columns=CALIBRATION_SCORE_COLUMNS).sort_values(
        "calibration_condition_id", kind="stable"
    ).reset_index(drop=True)
    quantile_rows: list[dict[str, object]] = []
    for route in ROUTES:
        route_scores = scores.loc[
            scores["mean_route"] == route, "maximum_standardized_error"
        ].to_numpy(dtype=float)
        if route_scores.size == 0:
            raise ValueError(f"No calibration conditions are available for {route}")
        for coverage in REQUESTED_COVERAGES:
            quantile = finite_sample_higher_quantile(
                route_scores, coverage=coverage
            )
            quantile_rows.append(
                {
                    "mean_route": route,
                    "requested_coverage": coverage,
                    "calibration_condition_count": quantile.calibration_count,
                    "order_statistic_rank": quantile.order_statistic_rank,
                    "multiplier": quantile.multiplier,
                    "status": quantile.status,
                }
            )
    quantiles = pd.DataFrame(
        quantile_rows, columns=CALIBRATION_QUANTILE_COLUMNS
    ).sort_values(["mean_route", "requested_coverage"], kind="stable").reset_index(
        drop=True
    )
    calibration_state = {
        "calibration_condition_ids": list(CALIBRATION_CONDITION_IDS),
        "training_state_sha256": training_state_sha256,
        "calibration_scores_sha256": _canonical_frame_sha256(
            scores, sort_by=["calibration_condition_id"]
        ),
        "quantiles": [
            {
                "mean_route": str(row.mean_route),
                "requested_coverage": float(row.requested_coverage),
                "calibration_condition_count": int(
                    row.calibration_condition_count
                ),
                "order_statistic_rank": int(row.order_statistic_rank),
                "multiplier": (
                    None if pd.isna(row.multiplier) else float(row.multiplier)
                ),
                "status": str(row.status),
            }
            for row in quantiles.itertuples(index=False)
        ],
        "formal_coverage_claim_allowed": False,
    }
    calibration_hash = _canonical_json_sha256(calibration_state)
    return scores, quantiles, calibration_hash


def _reason_string(reasons: Sequence[str]) -> str:
    return ";".join(reasons) if reasons else NONE_REASON


def _quantile_lookup(
    quantiles: pd.DataFrame,
    *,
    route: str,
    coverage: float,
) -> pd.Series:
    selected = quantiles.loc[
        (quantiles["mean_route"].astype(str) == route)
        & np.isclose(
            pd.to_numeric(quantiles["requested_coverage"]).to_numpy(dtype=float),
            coverage,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(selected) != 1:
        raise ValueError("Every V4 route/coverage requires one calibration quantile")
    return selected.iloc[0]


def _prediction_rows(
    observations: pd.DataFrame,
    *,
    power_prior: object,
    activation_prior: object,
    residual_fit: object,
    training_state_sha256: str,
    calibration_state_sha256: str,
    quantiles: pd.DataFrame,
    config: Mapping[str, object],
    domain_hull: ConvexHull,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    uncertainty = dict(config["uncertainty"])
    lower_physical, upper_physical = (
        float(value) for value in uncertainty["physical_bounds_pct"]
    )
    for condition_id in TEST_CONDITION_IDS:
        condition = _condition_frame(observations, condition_id)
        predicted = _predict_condition(
            condition,
            power_prior=power_prior,
            activation_prior=activation_prior,
            residual_fit=residual_fit,
            training_state_sha256=training_state_sha256,
            config=config,
            domain_hull=domain_hull,
        )
        horizons = (
            predicted.future["elapsed_days"].to_numpy(dtype=float)
            - predicted.prefix_end_days
        )
        horizon_matched = bool(
            np.all(horizons >= -1e-12)
            and np.all(horizons <= float(residual_fit.support_horizon_days) + 1e-12)
            and predicted.future["checkup_index"].astype(int).tolist()
            == list(FORECAST_INDICES)
        )
        target_state = {
            "target_condition_id": condition_id,
            "training_state_sha256": training_state_sha256,
            "calibration_state_sha256": calibration_state_sha256,
            "condition_prediction_state_sha256": (
                predicted.prediction_state_sha256
            ),
            "requested_coverages": list(REQUESTED_COVERAGES),
        }
        target_state_hash = _canonical_json_sha256(target_state)
        for coverage in REQUESTED_COVERAGES:
            quantile = _quantile_lookup(
                quantiles, route=predicted.mean_route, coverage=coverage
            )
            multiplier_value = quantile["multiplier"]
            multiplier = (
                None if pd.isna(multiplier_value) else float(multiplier_value)
            )
            diagnostic_reasons: list[str] = []
            if multiplier is None:
                diagnostic_reasons.append("insufficient_same_route_calibration")
            if not horizon_matched:
                diagnostic_reasons.append("horizon_mismatch")
            if not predicted.domain_supported:
                diagnostic_reasons.append("domain_unsupported")
            if not predicted.residual_support_ok:
                diagnostic_reasons.append("residual_outside_support")
            if predicted.residual_cap_hit:
                diagnostic_reasons.append("residual_cap_hit")
            available = not diagnostic_reasons
            if available:
                radius = multiplier * predicted.predictive_sd_pp
                diagnostic_lower = np.clip(
                    predicted.predicted_retention_pct - radius,
                    lower_physical,
                    upper_physical,
                )
                diagnostic_upper = np.clip(
                    predicted.predicted_retention_pct + radius,
                    lower_physical,
                    upper_physical,
                )
                diagnostic_width = diagnostic_upper - diagnostic_lower
            else:
                diagnostic_lower = np.full(len(FORECAST_INDICES), np.nan)
                diagnostic_upper = np.full(len(FORECAST_INDICES), np.nan)
                diagnostic_width = np.full(len(FORECAST_INDICES), np.nan)
            operational_decision = conservative_issuance_decision(
                specialist_gate_ready=predicted.activation_gate_ready,
                specialist_fit_succeeded=(
                    "specialist_fit_failed"
                    not in predicted.mean_fallback_reasons
                ),
                fallback_fit_succeeded=True,
                residual_support_ok=predicted.residual_support_ok,
                residual_cap_hit=predicted.residual_cap_hit,
                calibration_multiplier=multiplier,
                calibration_evidence_independent=False,
                sufficient_same_route_calibration=multiplier is not None,
                calibration_horizon_matched=horizon_matched,
                domain_supported=predicted.domain_supported,
                independent_long_term_evidence_required=True,
                independent_long_term_evidence_available=False,
                interval_width_pp=(
                    float(np.max(diagnostic_width)) if available else None
                ),
                max_interval_width_pp=uncertainty["max_interval_width_pp"],
            )
            if operational_decision.mean_route.value != predicted.mean_route:
                raise RuntimeError("V4 primitive and experiment mean routes disagree")
            decision_fallback_reasons = tuple(
                reason.value
                for reason in operational_decision.mean_fallback_reasons
            )
            if decision_fallback_reasons != predicted.mean_fallback_reasons:
                raise RuntimeError(
                    "V4 primitive and experiment fallback reasons disagree"
                )
            operational_reasons = tuple(
                reason.value for reason in operational_decision.abstention_reasons
            )
            if not set(OPERATIONAL_BASE_REASONS).issubset(operational_reasons):
                raise RuntimeError("Naumann evidence abstention reasons are required")
            for index, coordinate in enumerate(
                predicted.future.itertuples(index=False)
            ):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "split_id": SPLIT_ID,
                        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
                        "target_condition_id": condition_id,
                        "prefix_checkups": PREFIX_CHECKUPS,
                        "requested_coverage": coverage,
                        "target_checkup_index": int(coordinate.checkup_index),
                        "prefix_end_checkup_index": PREFIX_END_INDEX,
                        "prefix_end_days": predicted.prefix_end_days,
                        "temperature_c": float(coordinate.temperature_c),
                        "storage_soc_fraction": float(
                            coordinate.storage_soc_fraction
                        ),
                        "elapsed_days": float(coordinate.elapsed_days),
                        "forecast_horizon_days": float(horizons[index]),
                        "is_final_checkup": bool(
                            coordinate.checkup_index == FORECAST_END_INDEX
                        ),
                        "mean_route": predicted.mean_route,
                        "mean_fallback_reasons": _reason_string(
                            predicted.mean_fallback_reasons
                        ),
                        "activation_gate_ready": predicted.activation_gate_ready,
                        "domain_supported": predicted.domain_supported,
                        "residual_support_ok": predicted.residual_support_ok,
                        "residual_cap_hit": predicted.residual_cap_hit,
                        "residual_correction_pp": float(
                            predicted.residual_correction_pp[index]
                        ),
                        "predicted_capacity_retention_pct": float(
                            predicted.predicted_retention_pct[index]
                        ),
                        "predictive_sd_pp": float(
                            predicted.predictive_sd_pp[index]
                        ),
                        "calibration_condition_count": int(
                            quantile["calibration_condition_count"]
                        ),
                        "calibration_order_statistic_rank": int(
                            quantile["order_statistic_rank"]
                        ),
                        "calibration_multiplier": multiplier,
                        "calibration_horizon_matched": horizon_matched,
                        "diagnostic_interval_status": (
                            DIAGNOSTIC_AVAILABLE
                            if available
                            else DIAGNOSTIC_UNAVAILABLE
                        ),
                        "diagnostic_abstention_reasons": _reason_string(
                            diagnostic_reasons
                        ),
                        "diagnostic_lower_pct": (
                            float(diagnostic_lower[index]) if available else None
                        ),
                        "diagnostic_upper_pct": (
                            float(diagnostic_upper[index]) if available else None
                        ),
                        "diagnostic_width_pp": (
                            float(diagnostic_width[index]) if available else None
                        ),
                        "operational_issuance_status": (
                            operational_decision.issuance_status.value
                        ),
                        "operational_abstention_reasons": _reason_string(
                            operational_reasons
                        ),
                        "operational_lower_pct": None,
                        "operational_upper_pct": None,
                        "mechanistic_training_support_days": float(
                            power_prior.maximum_training_days
                        ),
                        "residual_support_horizon_days": float(
                            residual_fit.support_horizon_days
                        ),
                        "training_state_sha256": training_state_sha256,
                        "calibration_state_sha256": calibration_state_sha256,
                        "prediction_state_sha256": target_state_hash,
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    return predictions


def build_calendar_v4_label_free_predictions(
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a pack that never reads any test-condition future outcome."""
    validate_naumann_calendar_observations(observations)
    parsed = validate_calendar_v4_hybrid_config(config)
    ordered = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    splits = validate_calendar_v4_split_and_rank(ordered)
    (
        power_prior,
        activation_prior,
        residual_fit,
        residual_frame,
        training_hash,
        _,
        _,
    ) = _build_training_state(ordered, parsed)
    hull = _domain_hull(ordered)
    calibration_scores, calibration_quantiles, calibration_hash = (
        _build_calibration(
            ordered,
            power_prior=power_prior,
            activation_prior=activation_prior,
            residual_fit=residual_fit,
            training_state_sha256=training_hash,
            config=parsed,
            domain_hull=hull,
        )
    )
    predictions = _prediction_rows(
        ordered,
        power_prior=power_prior,
        activation_prior=activation_prior,
        residual_fit=residual_fit,
        training_state_sha256=training_hash,
        calibration_state_sha256=calibration_hash,
        quantiles=calibration_quantiles,
        config=parsed,
        domain_hull=hull,
    )
    calendar_v4_prediction_sha256(predictions)
    return (
        predictions,
        residual_frame,
        calibration_scores,
        calibration_quantiles,
        splits,
    )


def calendar_v4_prediction_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V4 prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if predictions.empty or predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("Calendar V4 prediction keys must be non-empty and unique")
    if predictions[PREDICTION_KEY_COLUMNS].isna().any().any():
        raise ValueError("Calendar V4 prediction keys cannot be null")
    for column in (
        "training_state_sha256",
        "calibration_state_sha256",
        "prediction_state_sha256",
    ):
        if not predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"Invalid Calendar V4 state hash: {column}")
    return _canonical_frame_sha256(
        predictions[PREDICTION_COLUMNS], sort_by=PREDICTION_KEY_COLUMNS
    )


def _validate_prediction_pack_against_authority(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
) -> None:
    expected_keys = {
        (condition_id, PREFIX_CHECKUPS, coverage, index)
        for condition_id in TEST_CONDITION_IDS
        for coverage in REQUESTED_COVERAGES
        for index in FORECAST_INDICES
    }
    observed_keys = {
        (
            str(row.target_condition_id),
            int(row.prefix_checkups),
            float(row.requested_coverage),
            int(row.target_checkup_index),
        )
        for row in predictions.itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise ValueError("Calendar V4 prediction support differs from the lock")
    truth = observations[
        [
            "condition_id",
            "checkup_index",
            "temperature_c",
            "storage_soc_fraction",
            "elapsed_days",
        ]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "temperature_c": "truth_temperature_c",
            "storage_soc_fraction": "truth_storage_soc_fraction",
            "elapsed_days": "truth_elapsed_days",
        }
    )
    merged = predictions.merge(
        truth,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Every V4 prediction must match an authoritative coordinate")
    for prediction_column, truth_column in (
        ("temperature_c", "truth_temperature_c"),
        ("storage_soc_fraction", "truth_storage_soc_fraction"),
        ("elapsed_days", "truth_elapsed_days"),
    ):
        if not np.allclose(
            pd.to_numeric(merged[prediction_column]),
            pd.to_numeric(merged[truth_column]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"Calendar V4 coordinate disagrees with truth: {prediction_column}"
            )
    prefix_end = observations.loc[
        observations["checkup_index"] == PREFIX_END_INDEX,
        "elapsed_days",
    ].to_numpy(dtype=float)
    if not np.allclose(prefix_end, prefix_end[0], rtol=0.0, atol=1e-12):
        raise ValueError("Calendar V4 prefix-end time must be common")
    if not (
        pd.to_numeric(predictions["prefix_end_checkup_index"]) == PREFIX_END_INDEX
    ).all() or not np.allclose(
        pd.to_numeric(predictions["prefix_end_days"]),
        prefix_end[0],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Calendar V4 prefix boundary was altered")
    expected_horizon = (
        pd.to_numeric(predictions["elapsed_days"]).to_numpy(dtype=float)
        - prefix_end[0]
    )
    if not np.allclose(
        pd.to_numeric(predictions["forecast_horizon_days"]),
        expected_horizon,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Calendar V4 forecast horizon arithmetic changed")
    expected_final = (
        pd.to_numeric(predictions["target_checkup_index"]).astype(int)
        == FORECAST_END_INDEX
    )
    if not pd.api.types.is_bool_dtype(predictions["is_final_checkup"]) or not (
        predictions["is_final_checkup"].to_numpy(dtype=bool)
        == expected_final.to_numpy(dtype=bool)
    ).all():
        raise ValueError("Calendar V4 final-checkup flags changed")

    numeric_required = [
        "predicted_capacity_retention_pct",
        "predictive_sd_pp",
        "calibration_condition_count",
        "calibration_order_statistic_rank",
        "mechanistic_training_support_days",
        "residual_support_horizon_days",
    ]
    converted = predictions[numeric_required].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any() or not np.isfinite(converted.to_numpy()).all():
        raise ValueError("Calendar V4 required numeric fields must be finite")
    if (converted["predictive_sd_pp"] <= 0.0).any():
        raise ValueError("Calendar V4 predictive scales must be positive")
    if not converted["predicted_capacity_retention_pct"].between(0.0, 100.0).all():
        raise ValueError("Calendar V4 mean predictions must be physically bounded")
    if set(predictions["operational_issuance_status"].astype(str)) != {
        OPERATIONAL_ABSTAINED
    }:
        raise ValueError("Operational issuance must remain abstained on Naumann")
    if predictions[["operational_lower_pct", "operational_upper_pct"]].notna().any().any():
        raise ValueError("Abstained operational intervals must have null bounds")
    if not predictions["operational_abstention_reasons"].astype(str).str.contains(
        "calibration_evidence_not_independent", regex=False
    ).all() or not predictions["operational_abstention_reasons"].astype(str).str.contains(
        "independent_long_term_evidence_missing", regex=False
    ).all():
        raise ValueError("Operational abstention evidence reasons were removed")

    for row in predictions.itertuples(index=False):
        available = row.diagnostic_interval_status == DIAGNOSTIC_AVAILABLE
        bounds = (
            row.diagnostic_lower_pct,
            row.diagnostic_upper_pct,
            row.diagnostic_width_pp,
        )
        if available:
            if row.calibration_multiplier is None or any(
                value is None or not np.isfinite(float(value)) for value in bounds
            ):
                raise ValueError("Available diagnostic intervals must be finite")
            radius = float(row.calibration_multiplier) * float(row.predictive_sd_pp)
            expected_lower = float(
                np.clip(
                    float(row.predicted_capacity_retention_pct) - radius,
                    0.0,
                    100.0,
                )
            )
            expected_upper = float(
                np.clip(
                    float(row.predicted_capacity_retention_pct) + radius,
                    0.0,
                    100.0,
                )
            )
            if not np.allclose(
                [row.diagnostic_lower_pct, row.diagnostic_upper_pct],
                [expected_lower, expected_upper],
                rtol=0.0,
                atol=1e-12,
            ) or not np.isclose(
                float(row.diagnostic_width_pp),
                expected_upper - expected_lower,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("Calendar V4 diagnostic interval arithmetic changed")
        elif any(value is not None and not pd.isna(value) for value in bounds):
            raise ValueError("Unavailable diagnostic intervals must have null bounds")


def _assert_regenerated_pack(
    predictions: pd.DataFrame,
    regenerated: pd.DataFrame,
) -> None:
    submitted = predictions[PREDICTION_COLUMNS].sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    expected = regenerated[PREDICTION_COLUMNS].sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            submitted,
            expected,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(
            "Calendar V4 pack disagrees with independent deterministic regeneration"
        ) from exc


def score_calendar_v4_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    config: Mapping[str, object],
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    validate_naumann_calendar_observations(observations)
    parsed = validate_calendar_v4_hybrid_config(config)
    observed_hash = calendar_v4_prediction_sha256(predictions)
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Calendar V4 prediction pack changed after freezing")
    _validate_prediction_pack_against_authority(predictions, observations)
    regenerated, _, _, _, _ = build_calendar_v4_label_free_predictions(
        observations, config=parsed
    )
    _assert_regenerated_pack(predictions, regenerated)

    truth = observations[
        ["condition_id", "checkup_index", "elapsed_days", "capacity_retention_pct"]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "elapsed_days": "truth_elapsed_days",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    scored = predictions.merge(
        truth,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
    )
    if scored[["truth_elapsed_days", "true_capacity_retention_pct"]].isna().any().any():
        raise ValueError("Calendar V4 scoring truth is incomplete")
    scored["error_pp"] = (
        pd.to_numeric(scored["predicted_capacity_retention_pct"])
        - pd.to_numeric(scored["true_capacity_retention_pct"])
    )
    rows: list[dict[str, object]] = []
    grouping = ["target_condition_id", "mean_route", "requested_coverage"]
    for keys, group in scored.groupby(grouping, sort=True):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        if ordered["target_checkup_index"].astype(int).tolist() != list(
            FORECAST_INDICES
        ):
            raise ValueError("Scored V4 trajectories must retain complete support")
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        error = ordered["error_pp"].to_numpy(dtype=float)
        absolute = np.abs(error)
        available = (
            ordered["diagnostic_interval_status"].iloc[0]
            == DIAGNOSTIC_AVAILABLE
        )
        if not ordered["diagnostic_interval_status"].eq(
            ordered["diagnostic_interval_status"].iloc[0]
        ).all():
            raise ValueError("Diagnostic availability must be trajectory-level")
        if available:
            lower = ordered["diagnostic_lower_pct"].to_numpy(dtype=float)
            upper = ordered["diagnostic_upper_pct"].to_numpy(dtype=float)
            truth_values = ordered["true_capacity_retention_pct"].to_numpy(
                dtype=float
            )
            covered = (truth_values >= lower) & (truth_values <= upper)
            widths = upper - lower
            interval_scores = interval_score(
                truth_values,
                lower,
                upper,
                coverage=float(keys[2]),
            )
            simultaneous_covered: bool | None = bool(covered.all())
            pointwise_coverage: float | None = float(covered.mean())
            mean_width: float | None = float(widths.mean())
            score_mean: float | None = float(interval_scores.mean())
        else:
            simultaneous_covered = None
            pointwise_coverage = None
            mean_width = None
            score_mean = None
        rows.append(
            {
                "target_condition_id": str(keys[0]),
                "mean_route": str(keys[1]),
                "requested_coverage": float(keys[2]),
                "future_point_count": len(ordered),
                "trajectory_iae_pp": float(
                    np.trapezoid(absolute, elapsed) / (elapsed[-1] - elapsed[0])
                ),
                "point_mae_pp": float(absolute.mean()),
                "final_error_pp": float(error[-1]),
                "final_absolute_error_pp": float(absolute[-1]),
                "diagnostic_interval_status": (
                    DIAGNOSTIC_AVAILABLE if available else DIAGNOSTIC_UNAVAILABLE
                ),
                "diagnostic_simultaneous_covered": simultaneous_covered,
                "diagnostic_pointwise_coverage_fraction": pointwise_coverage,
                "diagnostic_mean_width_pp": mean_width,
                "diagnostic_interval_score_mean": score_mean,
                "operational_issuance_status": OPERATIONAL_ABSTAINED,
            }
        )
    return pd.DataFrame(rows, columns=CONDITION_METRIC_COLUMNS).sort_values(
        ["target_condition_id", "requested_coverage"], kind="stable"
    ).reset_index(drop=True)


def run_calendar_v4_hybrid_development(
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
    parsed = validate_calendar_v4_hybrid_config(config)
    (
        predictions,
        residual_crossfit,
        calibration_scores,
        calibration_quantiles,
        splits,
    ) = build_calendar_v4_label_free_predictions(observations, config=parsed)
    prediction_hash = calendar_v4_prediction_sha256(predictions)
    condition_metrics = score_calendar_v4_predictions(
        predictions,
        observations,
        config=parsed,
        frozen_prediction_sha256=prediction_hash,
    )
    primary = condition_metrics.loc[
        np.isclose(
            condition_metrics["requested_coverage"].to_numpy(dtype=float),
            PRIMARY_COVERAGE,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    diagnostic_available = primary.loc[
        primary["diagnostic_interval_status"] == DIAGNOSTIC_AVAILABLE
    ]
    result: dict[str, object] = {
        "status": "retrospective_hybrid_diagnostic_complete_not_confirmed",
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _canonical_json_sha256(parsed),
        "prediction_pack_sha256": prediction_hash,
        "design": {
            "design_status": DESIGN_STATUS,
            "selection_status": SELECTION_STATUS,
            "evidence_role": EVIDENCE_ROLE,
            "split_id": SPLIT_ID,
            "training_condition_count": len(TRAINING_CONDITION_IDS),
            "calibration_condition_count": len(CALIBRATION_CONDITION_IDS),
            "test_condition_count": len(TEST_CONDITION_IDS),
            "prefix_checkups": PREFIX_CHECKUPS,
            "forecast_checkup_indices": list(FORECAST_INDICES),
            "target_future_outcomes_used_for_prediction": False,
            "calibration_outcomes_used_for_mean_or_residual_fit": False,
            "test_outcomes_used_before_prediction_freeze": False,
            "residual_source_unit": "training_condition_trajectory",
            "calibration_unit": "condition_mean_trajectory",
        },
        "calibration": {
            "partition": "selected_mean_route",
            "horizon_matched": True,
            "formal_coverage_claim_allowed": False,
            "primary_coverage": PRIMARY_COVERAGE,
            "diagnostic_available_test_trajectory_count": len(
                diagnostic_available
            ),
            "diagnostic_unavailable_test_trajectory_count": int(
                len(primary) - len(diagnostic_available)
            ),
            "operational_issued_trajectory_count": 0,
        },
        "confirmation": {
            "status": CONFIRMATION_STATUS,
            "current_dataset_relationship": (
                "reused_and_outcomes_already_inspected"
            ),
            "independent_long_term_dataset_available": False,
            "15_to_25_year_claim_allowed": False,
        },
        "future_label_firewall": {
            "label_free_prediction_pack_frozen_before_test_scoring": True,
            "scorer_independently_regenerates_prediction_pack": True,
            "future_outcome_columns_in_prediction_pack": [],
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
        "claim_boundary": (
            "This is a post-hoc diagnostic on 17 reused Naumann condition-mean "
            "trajectories through 885 days. It does not validate individual-cell, "
            "Hithium-product, plant, or 15-25 year predictions or intervals."
        ),
    }
    return (
        result,
        predictions,
        residual_crossfit,
        calibration_scores,
        calibration_quantiles,
        condition_metrics,
        splits,
    )
