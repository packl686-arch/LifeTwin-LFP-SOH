from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import numpy as np
import pandas as pd

from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_DATASET_ID,
    GEISBAUER_CALENDAR_EVIDENCE_ROLE,
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
    GEISBAUER_CALENDAR_STATISTICAL_UNIT,
    geisbauer_calendar_observations_sha256,
    validate_geisbauer_calendar_observations,
)
from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)
from lifetwin.models.calendar_v2 import (
    HIERARCHICAL_POWER_METHOD,
    TARGET_SQRT_METHOD,
    fit_hierarchical_power_prior,
    fit_sqrt_rate,
    predict_power_loss,
    predict_sqrt_loss,
    update_hierarchical_power_law,
)
from lifetwin.models.calendar_v3_activation import (
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    activation_mechanism_gate,
)


EXPERIMENT_ID = "geisbauer_lfp_calendar_external_stress_screen_v1"
DESIGN_STATUS = "retrospective_protocol_frozen_after_public_outcome_access"
SOURCE_HISTORY_POLICY = "all_available_source_history"
PERSISTENCE_METHOD = "target_prefix_persistence_day59_v1"
TARGET_PREFIX_DAYS = (0, 39, 59)
TARGET_SCORING_DAYS = (84, 120)
METHOD_NAMES = (
    PERSISTENCE_METHOD,
    TARGET_SQRT_METHOD,
    HIERARCHICAL_POWER_METHOD,
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
)
PRIMARY_CANDIDATE = GATED_HIERARCHICAL_ACTIVATION_METHOD
PRIMARY_COMPARATOR = TARGET_SQRT_METHOD
LONG_TERM_CONFIRMATION_STATUS = "blocked_target_horizon_120_days"
EXPECTED_PROHIBITED_CLAIMS = (
    "outcome_blind_external_validation",
    "independent_long_term_validation",
    "activation_mechanism_confirmation",
    "formal_uncertainty_calibration",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

PREDICTION_KEY_COLUMNS = [
    "cell_id",
    "method",
    "target_checkup_index",
]
PREDICTION_COLUMNS = [
    "experiment_id",
    "design_status",
    "target_dataset_id",
    "cell_id",
    "target_condition_id",
    "source_cell_number",
    "temperature_c",
    "storage_soc_fraction",
    "prefix_observation_count",
    "prefix_end_days",
    "method",
    "target_checkup_index",
    "target_elapsed_days",
    "predicted_capacity_retention_pct",
    "mechanism_gate_ready",
    "negative_loss_evidence",
    "positive_time_observation_count",
    "activation_component_selected",
    "mean_route",
    "fallback_reason",
    "temperature_within_source_range",
    "soc_within_source_range",
    "time_within_source_range",
    "cross_dataset_domain_confirmed",
    "source_training_state_sha256",
    "target_prefix_state_sha256",
    "prediction_state_sha256",
]

CELL_METRIC_COLUMNS = [
    "cell_id",
    "target_condition_id",
    "source_cell_number",
    "storage_soc_fraction",
    "method",
    "mechanism_gate_ready",
    "activation_component_selected",
    "mean_route",
    "fallback_reason",
    "future_point_count",
    "trajectory_iae_pp",
    "trajectory_signed_bias_pp",
    "point_mae_pp",
    "final_true_retention_pct",
    "final_predicted_retention_pct",
    "final_error_pp",
    "final_absolute_error_pp",
    "prediction_state_sha256",
]


def default_geisbauer_external_stress_protocol() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
        "source_dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "target_dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
        "source_training_history_policy": SOURCE_HISTORY_POLICY,
        "target_prefix_days": list(TARGET_PREFIX_DAYS),
        "target_scoring_days": list(TARGET_SCORING_DAYS),
        "methods": list(METHOD_NAMES),
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_comparator": PRIMARY_COMPARATOR,
        "model": {
            "minimum_training_conditions": 6,
            "robust_loss_scale_pp": 0.25,
            "power_exponent_bounds": [0.05, 1.5],
            "stress_surface_ridge": 1.0,
            "base_parameter_scale_floors": [0.1, 0.05],
            "activation_parameter_scale_floors": [0.1, 0.05, 0.1],
            "observation_scale_floor_pp": 0.1,
            "activation_timescale_days": 7.0,
            "activation_offset_bounds_pp": [0.0, 10.0],
        },
        "mechanism_gate": {
            "minimum_positive_time_observations": 7,
            "negative_loss_threshold_pp": 0.0,
            "fallback_method": HIERARCHICAL_POWER_METHOD,
        },
        "support_policy": {
            "temperature_policy": "within_source_axis_aligned_range",
            "soc_policy": "within_source_axis_aligned_range",
            "time_policy": (
                "target_scoring_day_not_beyond_source_maximum_day"
            ),
            "cross_dataset_domain_confirmation": False,
            "formal_uncertainty_calibration_allowed": False,
        },
        "decision_policy": {
            "pass_fail_threshold_defined": False,
            "long_term_confirmation_status": LONG_TERM_CONFIRMATION_STATUS,
            "independent_long_term_validation_claim_allowed": False,
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_geisbauer_external_stress_protocol(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(protocol, Mapping):
        raise ValueError("Geisbauer external stress protocol must be an object")
    parsed = dict(protocol)
    expected = default_geisbauer_external_stress_protocol()
    if _canonical_json_sha256(parsed) != _canonical_json_sha256(expected):
        raise ValueError("Geisbauer external stress protocol changed")
    return deepcopy(parsed)


def _canonical_prediction_frame_sha256(predictions: pd.DataFrame) -> str:
    _validate_prediction_pack(predictions)
    ordered = predictions.sort_values(PREDICTION_KEY_COLUMNS, kind="stable")
    payload = ordered.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def geisbauer_external_prediction_sha256(predictions: pd.DataFrame) -> str:
    return _canonical_prediction_frame_sha256(predictions)


def _source_prior(
    source: pd.DataFrame,
    model: Mapping[str, object],
):
    return fit_hierarchical_power_prior(
        source,
        minimum_conditions=int(model["minimum_training_conditions"]),
        exponent_bounds=tuple(float(v) for v in model["power_exponent_bounds"]),
        robust_loss_scale_pp=float(model["robust_loss_scale_pp"]),
        stress_surface_ridge=float(model["stress_surface_ridge"]),
        parameter_scale_floors=tuple(
            float(v) for v in model["base_parameter_scale_floors"]
        ),
        observation_scale_floor_pp=float(model["observation_scale_floor_pp"]),
    )


def _source_training_state(
    source: pd.DataFrame,
    prior: object,
    protocol: Mapping[str, object],
) -> str:
    outcome_sha256 = canonical_naumann_outcome_sha256(source)
    if outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Naumann source outcome snapshot mismatch")
    return _canonical_json_sha256(
        {
            "source_dataset_id": NAUMANN_CALENDAR_DATASET_ID,
            "source_outcome_sha256": outcome_sha256,
            "source_training_history_policy": SOURCE_HISTORY_POLICY,
            "model": protocol["model"],
            "prior": asdict(prior),
        }
    )


def _model_cell_frame(cell: pd.DataFrame) -> pd.DataFrame:
    modeled = cell.copy()
    modeled["condition_id"] = modeled["cell_id"].astype(str)
    return modeled


def _target_prefix_state(prefix: pd.DataFrame) -> str:
    ordered = prefix.sort_values("elapsed_days", kind="stable")
    rows = [
        {
            "cell_id": str(row.cell_id),
            "source_cell_number": int(row.source_cell_number),
            "temperature_c_hex": float(row.temperature_c).hex(),
            "storage_soc_fraction_hex": float(row.storage_soc_fraction).hex(),
            "elapsed_days_hex": float(row.elapsed_days).hex(),
            "capacity_loss_pct_hex": float(row.capacity_loss_pct).hex(),
        }
        for row in ordered.itertuples(index=False)
    ]
    return _canonical_json_sha256(
        {
            "target_dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
            "prefix_days": list(TARGET_PREFIX_DAYS),
            "rows": rows,
        }
    )


def _prediction_state(
    *,
    cell_id: str,
    method: str,
    mean_route: str,
    fallback_reason: str,
    source_training_state_sha256: str,
    target_prefix_state_sha256: str,
    model_state: object,
    gate_state: object,
    predicted_retention: np.ndarray,
) -> str:
    return _canonical_json_sha256(
        {
            "cell_id": cell_id,
            "method": method,
            "mean_route": mean_route,
            "fallback_reason": fallback_reason,
            "source_training_state_sha256": source_training_state_sha256,
            "target_prefix_state_sha256": target_prefix_state_sha256,
            "model_state": model_state,
            "gate_state": gate_state,
            "target_scoring_days": list(TARGET_SCORING_DAYS),
            "predicted_retention_hex": [
                float(value).hex() for value in predicted_retention
            ],
        }
    )


def generate_geisbauer_external_predictions(
    source_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    """Generate target predictions using only source history and target prefixes."""
    validate_naumann_calendar_observations(source_observations)
    validate_geisbauer_calendar_observations(target_observations)
    parsed = validate_geisbauer_external_stress_protocol(protocol)
    source = source_observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    target = target_observations.sort_values(
        ["cell_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    prior = _source_prior(source, parsed["model"])
    source_state = _source_training_state(source, prior, parsed)
    source_temperature = source["temperature_c"].to_numpy(dtype=float)
    source_soc = source["storage_soc_fraction"].to_numpy(dtype=float)
    source_maximum_days = float(source["elapsed_days"].max())
    scoring_days = np.asarray(TARGET_SCORING_DAYS, dtype=float)
    rows: list[dict[str, object]] = []

    for cell_id, raw_cell in target.groupby("cell_id", sort=True):
        cell = _model_cell_frame(raw_cell)
        prefix = cell.loc[cell["elapsed_days"].isin(TARGET_PREFIX_DAYS)].copy()
        observed_prefix_days = prefix["elapsed_days"].astype(int).tolist()
        if observed_prefix_days != list(TARGET_PREFIX_DAYS):
            raise ValueError(f"External target prefix changed for {cell_id}")
        prefix_state = _target_prefix_state(prefix)
        gate = activation_mechanism_gate(
            prefix,
            minimum_positive_time_observations=int(
                parsed["mechanism_gate"]["minimum_positive_time_observations"]
            ),
            negative_loss_threshold_pp=float(
                parsed["mechanism_gate"]["negative_loss_threshold_pp"]
            ),
        )
        if gate.ready:
            raise ValueError(
                "The locked three-checkup target prefix cannot activate the specialist"
            )
        power_fit = update_hierarchical_power_law(
            prior,
            prefix,
            exponent_bounds=tuple(
                float(v) for v in parsed["model"]["power_exponent_bounds"]
            ),
        )
        sqrt_rate = fit_sqrt_rate(prefix)
        power_retention = 100.0 - predict_power_loss(power_fit, scoring_days)
        prediction_specs = {
            PERSISTENCE_METHOD: {
                "values": np.repeat(
                    float(prefix.iloc[-1]["capacity_retention_pct"]),
                    len(scoring_days),
                ),
                "route": "target_prefix_persistence",
                "fallback": "not_applicable",
                "state": {
                    "retention_at_prefix_end": float(
                        prefix.iloc[-1]["capacity_retention_pct"]
                    )
                },
            },
            TARGET_SQRT_METHOD: {
                "values": 100.0 - predict_sqrt_loss(sqrt_rate, scoring_days),
                "route": "target_prefix_sqrt",
                "fallback": "not_applicable",
                "state": {"sqrt_rate": float(sqrt_rate)},
            },
            HIERARCHICAL_POWER_METHOD: {
                "values": power_retention,
                "route": "hierarchical_power",
                "fallback": "not_applicable",
                "state": asdict(power_fit),
            },
            GATED_HIERARCHICAL_ACTIVATION_METHOD: {
                "values": power_retention,
                "route": "hierarchical_power_fallback",
                "fallback": "specialist_gate_not_ready",
                "state": asdict(power_fit),
            },
        }
        first = prefix.iloc[0]
        temperature_supported = bool(
            source_temperature.min()
            <= float(first["temperature_c"])
            <= source_temperature.max()
        )
        soc_supported = bool(
            source_soc.min()
            <= float(first["storage_soc_fraction"])
            <= source_soc.max()
        )
        time_supported = bool(scoring_days.max() <= source_maximum_days)
        for method in METHOD_NAMES:
            spec = prediction_specs[method]
            values = np.asarray(spec["values"], dtype=float)
            prediction_state = _prediction_state(
                cell_id=str(cell_id),
                method=method,
                mean_route=str(spec["route"]),
                fallback_reason=str(spec["fallback"]),
                source_training_state_sha256=source_state,
                target_prefix_state_sha256=prefix_state,
                model_state=spec["state"],
                gate_state=asdict(gate),
                predicted_retention=values,
            )
            for elapsed_days, predicted in zip(
                TARGET_SCORING_DAYS,
                values,
                strict=True,
            ):
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "design_status": DESIGN_STATUS,
                        "target_dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
                        "cell_id": str(cell_id),
                        "target_condition_id": str(
                            first["target_condition_id"]
                            if "target_condition_id" in first
                            else raw_cell.iloc[0]["condition_id"]
                        ),
                        "source_cell_number": int(first["source_cell_number"]),
                        "temperature_c": float(first["temperature_c"]),
                        "storage_soc_fraction": float(
                            first["storage_soc_fraction"]
                        ),
                        "prefix_observation_count": len(prefix),
                        "prefix_end_days": float(TARGET_PREFIX_DAYS[-1]),
                        "method": method,
                        "target_checkup_index": (
                            list(TARGET_PREFIX_DAYS + TARGET_SCORING_DAYS).index(
                                elapsed_days
                            )
                        ),
                        "target_elapsed_days": float(elapsed_days),
                        "predicted_capacity_retention_pct": float(predicted),
                        "mechanism_gate_ready": gate.ready,
                        "negative_loss_evidence": gate.negative_loss_evidence,
                        "positive_time_observation_count": (
                            gate.positive_time_observation_count
                        ),
                        "activation_component_selected": False,
                        "mean_route": spec["route"],
                        "fallback_reason": spec["fallback"],
                        "temperature_within_source_range": temperature_supported,
                        "soc_within_source_range": soc_supported,
                        "time_within_source_range": time_supported,
                        "cross_dataset_domain_confirmed": False,
                        "source_training_state_sha256": source_state,
                        "target_prefix_state_sha256": prefix_state,
                        "prediction_state_sha256": prediction_state,
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    _validate_prediction_pack(predictions)
    return predictions


def _strict_boolean_column(frame: pd.DataFrame, column: str) -> None:
    if not frame[column].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError(f"External prediction {column} must contain booleans")


def _validate_prediction_pack(predictions: pd.DataFrame) -> None:
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError("External stress prediction schema or column order changed")
    if len(predictions) != 15 * len(METHOD_NAMES) * len(TARGET_SCORING_DAYS):
        raise ValueError("External stress prediction row count changed")
    if predictions.isna().any().any():
        raise ValueError("External stress predictions must be non-null")
    if predictions.duplicated(PREDICTION_KEY_COLUMNS).any():
        raise ValueError("External stress prediction keys must be unique")
    numeric_columns = [
        "source_cell_number",
        "temperature_c",
        "storage_soc_fraction",
        "prefix_observation_count",
        "prefix_end_days",
        "target_checkup_index",
        "target_elapsed_days",
        "predicted_capacity_retention_pct",
        "positive_time_observation_count",
    ]
    numeric = predictions[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("External stress prediction numeric values must be finite")
    for column in (
        "mechanism_gate_ready",
        "negative_loss_evidence",
        "activation_component_selected",
        "temperature_within_source_range",
        "soc_within_source_range",
        "time_within_source_range",
        "cross_dataset_domain_confirmed",
    ):
        _strict_boolean_column(predictions, column)
    exact_columns = {
        "experiment_id": {EXPERIMENT_ID},
        "design_status": {DESIGN_STATUS},
        "target_dataset_id": {GEISBAUER_CALENDAR_DATASET_ID},
        "method": set(METHOD_NAMES),
    }
    for column, expected in exact_columns.items():
        if set(predictions[column].astype(str)) != expected:
            raise ValueError(f"External stress prediction {column} changed")
    for column in (
        "source_training_state_sha256",
        "target_prefix_state_sha256",
        "prediction_state_sha256",
    ):
        if not predictions[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"External stress prediction {column} is invalid")
    if not (numeric["prefix_observation_count"] == len(TARGET_PREFIX_DAYS)).all():
        raise ValueError("External target prefix count changed")
    if not (numeric["prefix_end_days"] == TARGET_PREFIX_DAYS[-1]).all():
        raise ValueError("External target prefix endpoint changed")
    if predictions["mechanism_gate_ready"].any():
        raise ValueError("External three-checkup prefix cannot pass the mechanism gate")
    if predictions["activation_component_selected"].any():
        raise ValueError("External stress screen cannot select the activation specialist")
    if not predictions[
        [
            "temperature_within_source_range",
            "soc_within_source_range",
            "time_within_source_range",
        ]
    ].all().all():
        raise ValueError("External stress coordinates left the declared source range")
    if predictions["cross_dataset_domain_confirmed"].any():
        raise ValueError("Cross-dataset domain confirmation must remain false")

    expected_days = list(TARGET_SCORING_DAYS)
    expected_indices = [3, 4]
    for (cell_id, method), group in predictions.groupby(
        ["cell_id", "method"], sort=True
    ):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        if ordered["target_elapsed_days"].astype(int).tolist() != expected_days:
            raise ValueError(f"External scoring days changed for {cell_id}/{method}")
        if ordered["target_checkup_index"].astype(int).tolist() != expected_indices:
            raise ValueError(f"External scoring indices changed for {cell_id}/{method}")
        for state_column in (
            "source_training_state_sha256",
            "target_prefix_state_sha256",
            "prediction_state_sha256",
        ):
            if ordered[state_column].nunique() != 1:
                raise ValueError(
                    f"External {state_column} changes within {cell_id}/{method}"
                )

    gated = predictions.loc[
        predictions["method"] == GATED_HIERARCHICAL_ACTIVATION_METHOD
    ].sort_values(["cell_id", "target_checkup_index"], kind="stable")
    power = predictions.loc[
        predictions["method"] == HIERARCHICAL_POWER_METHOD
    ].sort_values(["cell_id", "target_checkup_index"], kind="stable")
    if not np.array_equal(
        gated["predicted_capacity_retention_pct"].to_numpy(dtype=float),
        power["predicted_capacity_retention_pct"].to_numpy(dtype=float),
    ):
        raise ValueError("The gated external method must exactly reuse its fallback")
    if set(gated["mean_route"].astype(str)) != {"hierarchical_power_fallback"}:
        raise ValueError("External gated mean route changed")
    if set(gated["fallback_reason"].astype(str)) != {
        "specialist_gate_not_ready"
    }:
        raise ValueError("External gated fallback reason changed")


def _require_expected_prediction_pack(
    predictions: pd.DataFrame,
    source_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
    protocol: Mapping[str, object],
) -> None:
    observed_sha256 = geisbauer_external_prediction_sha256(predictions)
    if observed_sha256 != frozen_prediction_sha256:
        raise ValueError("External frozen prediction hash mismatch")
    expected = generate_geisbauer_external_predictions(
        source_observations,
        target_observations,
        protocol=protocol,
    )
    expected_sha256 = geisbauer_external_prediction_sha256(expected)
    if frozen_prediction_sha256 != expected_sha256:
        raise ValueError("External prediction pack differs from independent replay")


def score_geisbauer_external_predictions(
    predictions: pd.DataFrame,
    source_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    validate_naumann_calendar_observations(source_observations)
    validate_geisbauer_calendar_observations(target_observations)
    if canonical_naumann_outcome_sha256(source_observations) != (
        EXPECTED_CANONICAL_OUTCOME_SHA256
    ):
        raise ValueError("Naumann source outcome snapshot mismatch")
    target_sha256 = geisbauer_calendar_observations_sha256(target_observations)
    if target_sha256 != GEISBAUER_CALENDAR_OBSERVATIONS_SHA256:
        raise ValueError("Geisbauer target outcome snapshot mismatch")
    parsed = validate_geisbauer_external_stress_protocol(protocol)
    _require_expected_prediction_pack(
        predictions,
        source_observations,
        target_observations,
        frozen_prediction_sha256=frozen_prediction_sha256,
        protocol=parsed,
    )
    truth = target_observations[
        ["cell_id", "checkup_index", "elapsed_days", "capacity_retention_pct"]
    ].rename(
        columns={
            "checkup_index": "target_checkup_index",
            "elapsed_days": "truth_elapsed_days",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    scored = predictions.merge(
        truth,
        on=["cell_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (scored["_merge"] != "both").any():
        raise ValueError("Every external prediction must match authoritative truth")
    if not np.array_equal(
        scored["target_elapsed_days"].to_numpy(dtype=float),
        scored["truth_elapsed_days"].to_numpy(dtype=float),
    ):
        raise ValueError("External prediction time differs from authoritative truth")
    scored["prediction_error_pp"] = (
        scored["predicted_capacity_retention_pct"]
        - scored["true_capacity_retention_pct"]
    )
    rows: list[dict[str, object]] = []
    grouping = [
        "cell_id",
        "target_condition_id",
        "source_cell_number",
        "storage_soc_fraction",
        "method",
        "mechanism_gate_ready",
        "activation_component_selected",
        "mean_route",
        "fallback_reason",
        "prediction_state_sha256",
    ]
    for keys, group in scored.groupby(grouping, sort=True):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        error = ordered["prediction_error_pp"].to_numpy(dtype=float)
        absolute = np.abs(error)
        duration = float(elapsed[-1] - elapsed[0])
        if duration <= 0.0 or np.any(np.diff(elapsed) <= 0.0):
            raise ValueError("External truth time must increase")
        final = ordered.iloc[-1]
        rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "future_point_count": len(ordered),
                "trajectory_iae_pp": float(
                    np.trapezoid(absolute, elapsed) / duration
                ),
                "trajectory_signed_bias_pp": float(
                    np.trapezoid(error, elapsed) / duration
                ),
                "point_mae_pp": float(absolute.mean()),
                "final_true_retention_pct": float(
                    final["true_capacity_retention_pct"]
                ),
                "final_predicted_retention_pct": float(
                    final["predicted_capacity_retention_pct"]
                ),
                "final_error_pp": float(final["prediction_error_pp"]),
                "final_absolute_error_pp": abs(
                    float(final["prediction_error_pp"])
                ),
            }
        )
    return pd.DataFrame(rows, columns=CELL_METRIC_COLUMNS).sort_values(
        ["cell_id", "method"], kind="stable"
    ).reset_index(drop=True)


def _condition_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (condition_id, soc, method), group in cell_metrics.groupby(
        ["target_condition_id", "storage_soc_fraction", "method"], sort=True
    ):
        rows.append(
            {
                "target_condition_id": condition_id,
                "storage_soc_fraction": float(soc),
                "method": method,
                "physical_cell_count": len(group),
                "trajectory_iae_pp_mean": float(group["trajectory_iae_pp"].mean()),
                "trajectory_iae_pp_median": float(
                    group["trajectory_iae_pp"].median()
                ),
                "trajectory_iae_pp_std": float(group["trajectory_iae_pp"].std()),
                "trajectory_signed_bias_pp_mean": float(
                    group["trajectory_signed_bias_pp"].mean()
                ),
                "final_absolute_error_pp_mean": float(
                    group["final_absolute_error_pp"].mean()
                ),
                "mechanism_gate_ready_cell_count": int(
                    group["mechanism_gate_ready"].sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["storage_soc_fraction", "method"], kind="stable"
    ).reset_index(drop=True)


def _comparison_summary(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    comparator = cell_metrics.loc[
        cell_metrics["method"] == PRIMARY_COMPARATOR,
        ["cell_id", "target_condition_id", "trajectory_iae_pp"],
    ].rename(columns={"trajectory_iae_pp": "comparator_trajectory_iae_pp"})
    candidate = cell_metrics.loc[
        cell_metrics["method"] == PRIMARY_CANDIDATE,
        ["cell_id", "target_condition_id", "trajectory_iae_pp"],
    ].rename(columns={"trajectory_iae_pp": "candidate_trajectory_iae_pp"})
    paired = candidate.merge(
        comparator,
        on=["cell_id", "target_condition_id"],
        validate="one_to_one",
    )
    paired["paired_delta_iae_pp"] = (
        paired["candidate_trajectory_iae_pp"]
        - paired["comparator_trajectory_iae_pp"]
    )
    rows: list[dict[str, object]] = []
    groups = [("all_cells", paired)] + [
        (str(condition_id), group)
        for condition_id, group in paired.groupby("target_condition_id", sort=True)
    ]
    for scope, group in groups:
        delta = group["paired_delta_iae_pp"].to_numpy(dtype=float)
        rows.append(
            {
                "scope": scope,
                "physical_cell_count": len(group),
                "candidate_method": PRIMARY_CANDIDATE,
                "comparator_method": PRIMARY_COMPARATOR,
                "candidate_trajectory_iae_pp_mean": float(
                    group["candidate_trajectory_iae_pp"].mean()
                ),
                "comparator_trajectory_iae_pp_mean": float(
                    group["comparator_trajectory_iae_pp"].mean()
                ),
                "mean_paired_delta_iae_pp": float(delta.mean()),
                "candidate_better_cell_count": int(np.sum(delta < -1e-12)),
                "candidate_worse_cell_count": int(np.sum(delta > 1e-12)),
                "candidate_equal_cell_count": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def run_geisbauer_external_stress(
    source_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    protocol: Mapping[str, object],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    parsed = validate_geisbauer_external_stress_protocol(protocol)
    predictions = generate_geisbauer_external_predictions(
        source_observations,
        target_observations,
        protocol=parsed,
    )
    prediction_sha256 = geisbauer_external_prediction_sha256(predictions)
    cell_metrics = score_geisbauer_external_predictions(
        predictions,
        source_observations,
        target_observations,
        frozen_prediction_sha256=prediction_sha256,
        protocol=parsed,
    )
    condition_summary = _condition_summary(cell_metrics)
    comparison_summary = _comparison_summary(cell_metrics)
    overall = comparison_summary.loc[
        comparison_summary["scope"] == "all_cells"
    ].iloc[0]
    mean_delta = float(overall["mean_paired_delta_iae_pp"])
    result: dict[str, object] = {
        "status": (
            "accelerated_external_stress_screen_complete_"
            "long_term_confirmation_blocked"
        ),
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "model_validation_status": "not_confirmed",
        "config_sha256": _canonical_json_sha256(parsed),
        "source_dataset": {
            "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
            "statistical_unit": NAUMANN_STATISTICAL_UNIT,
            "condition_count": int(source_observations["condition_id"].nunique()),
            "maximum_observed_days": float(
                source_observations["elapsed_days"].max()
            ),
            "outcome_sha256": canonical_naumann_outcome_sha256(
                source_observations
            ),
        },
        "target_dataset": {
            "dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
            "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
            "statistical_unit": GEISBAUER_CALENDAR_STATISTICAL_UNIT,
            "physical_cell_count": int(target_observations["cell_id"].nunique()),
            "condition_count": int(
                target_observations["condition_id"].nunique()
            ),
            "maximum_observed_days": float(
                target_observations["elapsed_days"].max()
            ),
            "outcome_sha256": geisbauer_calendar_observations_sha256(
                target_observations
            ),
        },
        "protocol": {
            "target_prefix_days": list(TARGET_PREFIX_DAYS),
            "target_scoring_days": list(TARGET_SCORING_DAYS),
            "methods": list(METHOD_NAMES),
            "primary_candidate": PRIMARY_CANDIDATE,
            "primary_comparator": PRIMARY_COMPARATOR,
            "pass_fail_threshold_defined": False,
            "target_future_outcomes_used_for_prediction": False,
            "outcomes_were_public_before_protocol_freeze": True,
        },
        "mechanism_gate": {
            "gate_ready_physical_cell_count": int(
                cell_metrics.groupby("cell_id")["mechanism_gate_ready"].first().sum()
            ),
            "fallback_physical_cell_count": int(
                cell_metrics.loc[
                    cell_metrics["method"] == PRIMARY_CANDIDATE,
                    "fallback_reason",
                ].eq("specialist_gate_not_ready").sum()
            ),
            "activation_mechanism_tested": False,
            "reason": (
                "The locked day-0/39/59 prefix has only two positive-time points, "
                "below the seven-point specialist gate."
            ),
        },
        "primary_comparison": overall.to_dict(),
        "descriptive_signal_status": (
            "primary_candidate_mean_improvement_observed"
            if mean_delta < 0.0
            else "primary_candidate_did_not_outperform_comparator"
        ),
        "decision": {
            "long_term_confirmation_status": LONG_TERM_CONFIRMATION_STATUS,
            "independent_long_term_validation_claim_allowed": False,
            "formal_uncertainty_calibration_allowed": False,
            "reason": (
                "The target cohort is independent and cell-level, but all cells are "
                "stored at 60 C for only 120 days across three SOC conditions. It is "
                "an accelerated transfer stress screen, not long-term confirmation."
            ),
        },
        "future_label_firewall": {
            "label_free_prediction_sha256": prediction_sha256,
            "prediction_pack_independently_replayed_before_scoring": True,
            "future_target_outcomes_in_prediction_state": False,
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }
    return (
        result,
        predictions,
        cell_metrics,
        condition_summary,
        comparison_summary,
    )
