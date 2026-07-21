from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

from lifetwin import __version__
from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
)
from lifetwin.experiments.calendar_v4_hybrid_development import (
    CALIBRATION_CONDITION_IDS,
    CalendarV4ReferenceState,
    DIAGNOSTIC_AVAILABLE,
    DIAGNOSTIC_UNAVAILABLE,
    EXPERIMENT_ID,
    FORECAST_END_INDEX,
    FORECAST_INDICES,
    FORECAST_START_INDEX,
    OPERATIONAL_BASE_REASONS,
    PREFIX_CHECKUPS,
    PREFIX_END_INDEX,
    REQUESTED_COVERAGES,
    TRAINING_CONDITION_IDS,
    fit_calendar_v4_reference_state,
    predict_calendar_v4_condition,
)
from lifetwin.models.calendar_v4_hybrid import (
    RESIDUAL_SUPPORT_BOUNDARY_ATOL_DAYS,
    conservative_issuance_decision,
)


REQUEST_SCHEMA_VERSION = "lifetwin.calendar_prefix_request.v1"
CHEMISTRY = "LFP/graphite"
LOCKED_CALENDAR_ELAPSED_DAYS = (
    0.0,
    6.680416666666667,
    11.540833333333335,
    17.04125,
    29.504583333333333,
    42.045833333333334,
    53.56666666666667,
    81.04166666666667,
    107.46666666666667,
    136.94583333333333,
    161.3625,
    186.79166666666666,
    213.275,
    239.7125,
    266.17083333333335,
    292.77916666666664,
    319.30833333333334,
    345.8375,
    376.31666666666666,
    402.7416666666667,
    429.25,
    455.7916666666667,
    482.2916666666667,
    508.7916666666667,
    535.1666666666666,
    561.7083333333334,
    588.2916666666666,
    612.2916666666666,
    631.1666666666666,
    658.7916666666666,
    708.375,
    759.9583333333334,
    799.5833333333334,
    840.125,
    885.0416666666666,
)
ROOT_FIELDS = {
    "schema_version",
    "request_id",
    "model_id",
    "chemistry",
    "statistical_unit",
    "temperature_c",
    "storage_soc_fraction",
    "requested_coverage",
    "prefix",
    "forecast",
}
PREFIX_FIELDS = {
    "observation_index",
    "elapsed_days",
    "capacity_retention_pct",
}
FORECAST_FIELDS = {"forecast_index", "elapsed_days"}
FORECAST_COLUMNS = [
    "request_id",
    "model_id",
    "forecast_index",
    "elapsed_days",
    "forecast_horizon_days",
    "predicted_capacity_retention_pct",
    "predictive_sd_pp",
    "residual_correction_pp",
    "mean_route",
    "mean_fallback_reasons",
    "domain_supported",
    "calibration_horizon_matched",
    "diagnostic_interval_status",
    "diagnostic_abstention_reasons",
    "diagnostic_lower_pct",
    "diagnostic_upper_pct",
    "operational_issuance_status",
]
CLAIM_BOUNDARY = [
    "public_condition_mean_research_demo_not_individual_cell_validation",
    "retrospective_naumann_reference_not_independent_confirmation",
    "no_hithium_internal_data_or_product_accuracy_claim",
    "no_utility_scale_storage_plant_validation",
    "no_15_to_25_year_extrapolation_or_accuracy_claim",
    "schema_valid_prefix_does_not_imply_model_support",
]


class CalendarPrefixRequestError(ValueError):
    """Raised when a target request crosses the locked inference contract."""


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    encoded = frame.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalendarPrefixRequestError(f"{field} must be a finite number")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise CalendarPrefixRequestError(f"{field} must be a finite number")
    return parsed


def _strict_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalendarPrefixRequestError(f"{field} must be an integer")
    parsed = float(value)
    if not np.isfinite(parsed) or not parsed.is_integer():
        raise CalendarPrefixRequestError(f"{field} must be an integer")
    return int(parsed)


def _exact_fields(
    payload: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing or unknown:
        raise CalendarPrefixRequestError(
            f"{context} fields differ from the locked contract: "
            f"missing={missing}, unknown={unknown}"
        )


def validate_calendar_prefix_request(
    request: Mapping[str, object],
    *,
    schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Validate a p=10 request whose future contains coordinates, never labels."""

    if not isinstance(request, Mapping):
        raise CalendarPrefixRequestError("Calendar prefix request must be an object")
    try:
        parsed = json.loads(
            json.dumps(dict(request), ensure_ascii=False, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise CalendarPrefixRequestError(
            "Calendar prefix request must be finite JSON"
        ) from exc
    if schema is not None:
        errors = sorted(
            Draft202012Validator(dict(schema)).iter_errors(parsed),
            key=lambda item: tuple(str(value) for value in item.absolute_path),
        )
        if errors:
            first = errors[0]
            location = ".".join(str(value) for value in first.absolute_path)
            raise CalendarPrefixRequestError(
                f"Request schema validation failed at {location or '<root>'}: "
                f"{first.message}"
            )

    _exact_fields(parsed, ROOT_FIELDS, context="request")
    fixed_values = {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "model_id": EXPERIMENT_ID,
        "chemistry": CHEMISTRY,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
    }
    for field, expected in fixed_values.items():
        if parsed[field] != expected:
            raise CalendarPrefixRequestError(
                f"{field} must equal {expected!r}"
            )
    request_id = parsed["request_id"]
    if (
        not isinstance(request_id, str)
        or not request_id
        or len(request_id) > 128
        or request_id[0]
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in request_id
        )
    ):
        raise CalendarPrefixRequestError(
            "request_id must start with an ASCII letter or digit and then use only "
            "letters, digits, dot, dash, or underscore (1-128 characters)"
        )

    temperature = _finite_number(parsed["temperature_c"], field="temperature_c")
    if temperature < -80.0 or temperature > 120.0:
        raise CalendarPrefixRequestError("temperature_c is outside physical input limits")
    soc = _finite_number(
        parsed["storage_soc_fraction"], field="storage_soc_fraction"
    )
    if soc < 0.0 or soc > 1.0:
        raise CalendarPrefixRequestError("storage_soc_fraction must be in [0, 1]")
    requested = _finite_number(
        parsed["requested_coverage"], field="requested_coverage"
    )
    matched = [value for value in REQUESTED_COVERAGES if requested == value]
    if len(matched) != 1:
        raise CalendarPrefixRequestError(
            f"requested_coverage must be one of {REQUESTED_COVERAGES}"
        )

    prefix = parsed["prefix"]
    forecast = parsed["forecast"]
    if not isinstance(prefix, list) or len(prefix) != PREFIX_CHECKUPS:
        raise CalendarPrefixRequestError("prefix must contain exactly 10 observations")
    if not isinstance(forecast, list) or len(forecast) != len(FORECAST_INDICES):
        raise CalendarPrefixRequestError("forecast must contain exactly 25 coordinates")

    prefix_times: list[float] = []
    retention: list[float] = []
    prefix_indices: list[int] = []
    for position, row in enumerate(prefix):
        if not isinstance(row, Mapping):
            raise CalendarPrefixRequestError(f"prefix[{position}] must be an object")
        _exact_fields(row, PREFIX_FIELDS, context=f"prefix[{position}]")
        prefix_indices.append(
            _strict_integer(
                row["observation_index"],
                field=f"prefix[{position}].observation_index",
            )
        )
        prefix_times.append(
            _finite_number(
                row["elapsed_days"], field=f"prefix[{position}].elapsed_days"
            )
        )
        retention.append(
            _finite_number(
                row["capacity_retention_pct"],
                field=f"prefix[{position}].capacity_retention_pct",
            )
        )

    forecast_times: list[float] = []
    forecast_indices: list[int] = []
    for position, row in enumerate(forecast):
        if not isinstance(row, Mapping):
            raise CalendarPrefixRequestError(f"forecast[{position}] must be an object")
        _exact_fields(row, FORECAST_FIELDS, context=f"forecast[{position}]")
        forecast_indices.append(
            _strict_integer(
                row["forecast_index"],
                field=f"forecast[{position}].forecast_index",
            )
        )
        forecast_times.append(
            _finite_number(
                row["elapsed_days"], field=f"forecast[{position}].elapsed_days"
            )
        )

    if prefix_indices != list(range(PREFIX_CHECKUPS)):
        raise CalendarPrefixRequestError("prefix indices must be ordered 0 through 9")
    if forecast_indices != list(FORECAST_INDICES):
        raise CalendarPrefixRequestError("forecast indices must be ordered 10 through 34")
    all_times = np.asarray([*prefix_times, *forecast_times], dtype=float)
    if not np.all(np.diff(all_times) > 0.0):
        raise CalendarPrefixRequestError("elapsed_days must be strictly increasing")
    if prefix_times[0] != 0.0:
        raise CalendarPrefixRequestError("the first prefix observation must be at day 0")
    if retention[0] != 100.0:
        raise CalendarPrefixRequestError("the first capacity retention must equal 100%")
    if any(value < 0.0 or value > 110.0 for value in retention):
        raise CalendarPrefixRequestError("prefix capacity retention must be in [0, 110]")

    locked_days = np.asarray(LOCKED_CALENDAR_ELAPSED_DAYS, dtype=float)
    if not np.array_equal(all_times, locked_days):
        raise CalendarPrefixRequestError(
            "v1 accepts only the locked Naumann p=10 checkup grid; "
            "other horizons require fresh calibration"
        )
    for index, row in enumerate(parsed["prefix"]):
        row["observation_index"] = index
        row["elapsed_days"] = float(locked_days[index])
    for offset, row in enumerate(parsed["forecast"], start=FORECAST_START_INDEX):
        row["forecast_index"] = offset
        row["elapsed_days"] = float(locked_days[offset])
    parsed["temperature_c"] = temperature
    parsed["storage_soc_fraction"] = soc
    parsed["requested_coverage"] = matched[0]
    return parsed


def _target_condition(
    request: Mapping[str, object],
) -> pd.DataFrame:
    request_id = str(request["request_id"])
    temperature = float(request["temperature_c"])
    soc = float(request["storage_soc_fraction"])
    rows: list[dict[str, object]] = []
    for row in request["prefix"]:
        assert isinstance(row, Mapping)
        retention = float(row["capacity_retention_pct"])
        rows.append(
            {
                "condition_id": request_id,
                "checkup_index": int(row["observation_index"]),
                "temperature_c": temperature,
                "storage_soc_fraction": soc,
                "elapsed_days": float(row["elapsed_days"]),
                "capacity_retention_pct": retention,
                "capacity_loss_pct": 100.0 - retention,
            }
        )
    for row in request["forecast"]:
        assert isinstance(row, Mapping)
        rows.append(
            {
                "condition_id": request_id,
                "checkup_index": int(row["forecast_index"]),
                "temperature_c": temperature,
                "storage_soc_fraction": soc,
                "elapsed_days": float(row["elapsed_days"]),
                "capacity_retention_pct": np.nan,
                "capacity_loss_pct": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _calibration_quantile(
    quantiles: pd.DataFrame,
    *,
    route: str,
    coverage: float,
) -> pd.Series:
    selected = quantiles.loc[
        quantiles["mean_route"].astype(str).eq(route)
        & np.isclose(
            pd.to_numeric(quantiles["requested_coverage"]).to_numpy(dtype=float),
            coverage,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(selected) != 1:
        raise RuntimeError("Locked route and coverage need one calibration quantile")
    return selected.iloc[0]


def _request_in_reference_domain(
    request: Mapping[str, object],
    reference_state: CalendarV4ReferenceState,
) -> bool:
    point = np.asarray(
        [
            float(request["temperature_c"]),
            float(request["storage_soc_fraction"]),
        ],
        dtype=float,
    )
    hull = reference_state.domain_hull
    values = hull.equations[:, :-1] @ point + hull.equations[:, -1]
    return bool(np.all(values <= 1e-10))


def _prefix_in_reference_support(
    request: Mapping[str, object],
    *,
    reference_observations: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[bool, dict[str, object]]:
    """Check a joint prefix against a local stress-slice reference trajectory."""

    reference_ids = set(TRAINING_CONDITION_IDS) | set(CALIBRATION_CONDITION_IDS)
    prefix = reference_observations.loc[
        reference_observations["condition_id"].astype(str).isin(reference_ids)
        & pd.to_numeric(reference_observations["checkup_index"]).lt(PREFIX_CHECKUPS),
        [
            "condition_id",
            "checkup_index",
            "temperature_c",
            "storage_soc_fraction",
            "capacity_retention_pct",
        ],
    ].copy()
    matrix = prefix.pivot(
        index="condition_id",
        columns="checkup_index",
        values="capacity_retention_pct",
    ).sort_index(axis=1)
    expected_shape = (len(reference_ids), PREFIX_CHECKUPS)
    if matrix.shape != expected_shape or matrix.isna().any().any():
        raise RuntimeError("Reference prefix support matrix is incomplete")
    if [int(value) for value in matrix.columns] != list(range(PREFIX_CHECKUPS)):
        raise RuntimeError("Reference prefix support indices are incomplete")

    metadata = (
        prefix.sort_values(["condition_id", "checkup_index"], kind="stable")
        .groupby("condition_id", sort=True)
        .first()[["temperature_c", "storage_soc_fraction"]]
        .loc[matrix.index]
    )
    temperatures = metadata["temperature_c"].to_numpy(dtype=float)
    socs = metadata["storage_soc_fraction"].to_numpy(dtype=float)
    reference_values = matrix.to_numpy(dtype=float)
    target_temperature = float(request["temperature_c"])
    target_soc = float(request["storage_soc_fraction"])
    tolerance = 1e-12

    exact = np.flatnonzero(
        np.isclose(temperatures, target_temperature, rtol=0.0, atol=tolerance)
        & np.isclose(socs, target_soc, rtol=0.0, atol=tolerance)
    )
    expected: np.ndarray | None = None
    reference_mode = "unavailable"
    if len(exact) == 1:
        expected = reference_values[int(exact[0])]
        reference_mode = "exact_reference_condition"
    elif len(exact) > 1:
        raise RuntimeError("Reference stress coordinates must be unique")
    else:
        same_temperature = np.flatnonzero(
            np.isclose(
                temperatures,
                target_temperature,
                rtol=0.0,
                atol=tolerance,
            )
        )
        lower = same_temperature[socs[same_temperature] < target_soc]
        upper = same_temperature[socs[same_temperature] > target_soc]
        if len(lower) and len(upper):
            lower_index = int(lower[np.argmax(socs[lower])])
            upper_index = int(upper[np.argmin(socs[upper])])
            weight = (target_soc - socs[lower_index]) / (
                socs[upper_index] - socs[lower_index]
            )
            expected = reference_values[lower_index] + weight * (
                reference_values[upper_index] - reference_values[lower_index]
            )
            reference_mode = "same_temperature_soc_interpolation"
        else:
            same_soc = np.flatnonzero(
                np.isclose(socs, target_soc, rtol=0.0, atol=tolerance)
            )
            lower = same_soc[temperatures[same_soc] < target_temperature]
            upper = same_soc[temperatures[same_soc] > target_temperature]
            if len(lower) and len(upper):
                lower_index = int(lower[np.argmax(temperatures[lower])])
                upper_index = int(upper[np.argmin(temperatures[upper])])
                weight = (target_temperature - temperatures[lower_index]) / (
                    temperatures[upper_index] - temperatures[lower_index]
                )
                expected = reference_values[lower_index] + weight * (
                    reference_values[upper_index] - reference_values[lower_index]
                )
                reference_mode = "same_soc_temperature_interpolation"

    model_config = dict(config["model"])
    point_margin = 2.0 * float(model_config["robust_loss_scale_pp"])
    step_margin = point_margin
    if expected is None:
        details: dict[str, object] = {
            "prefix_supported": False,
            "prefix_support_rule": (
                "joint_local_stress_slice_interpolation_with_locked_residual_limits"
            ),
            "prefix_reference_mode": reference_mode,
            "prefix_local_stress_support": False,
            "prefix_support_margin_pp": point_margin,
            "prefix_step_support_margin_pp": step_margin,
            "prefix_point_residual_supported": None,
            "prefix_step_residual_supported": None,
            "prefix_max_point_deviation_pp": None,
            "prefix_max_step_deviation_pp": None,
        }
        return False, details

    values = np.asarray(
        [float(row["capacity_retention_pct"]) for row in request["prefix"]],
        dtype=float,
    )
    residual = values - expected
    max_point_deviation = float(np.max(np.abs(residual)))
    max_step_deviation = float(np.max(np.abs(np.diff(residual))))
    point_supported = max_point_deviation <= point_margin + tolerance
    step_supported = max_step_deviation <= step_margin + tolerance
    details = {
        "prefix_supported": point_supported and step_supported,
        "prefix_support_rule": (
            "joint_local_stress_slice_interpolation_with_locked_residual_limits"
        ),
        "prefix_reference_mode": reference_mode,
        "prefix_local_stress_support": True,
        "prefix_support_margin_pp": point_margin,
        "prefix_step_support_margin_pp": step_margin,
        "prefix_point_residual_supported": point_supported,
        "prefix_step_residual_supported": step_supported,
        "prefix_max_point_deviation_pp": max_point_deviation,
        "prefix_max_step_deviation_pp": max_step_deviation,
    }
    return point_supported and step_supported, details


def _request_rejection(
    request: Mapping[str, object],
    *,
    reference_observations: pd.DataFrame,
    reference_state: CalendarV4ReferenceState,
    status: str,
    reason: str,
    domain_supported: bool,
    prefix_support: Mapping[str, object],
    failure_type: str | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    operational = conservative_issuance_decision(
        specialist_gate_ready=False,
        specialist_fit_succeeded=False,
        fallback_fit_succeeded=False,
        residual_support_ok=True,
        residual_cap_hit=False,
        calibration_multiplier=None,
        calibration_evidence_independent=False,
        sufficient_same_route_calibration=False,
        calibration_horizon_matched=True,
        domain_supported=domain_supported,
        independent_long_term_evidence_required=True,
        independent_long_term_evidence_available=False,
        interval_width_pp=None,
        max_interval_width_pp=None,
    )
    operational_reasons = [item.value for item in operational.abstention_reasons]
    if reason not in operational_reasons:
        operational_reasons.append(reason)
    fallback_reasons = [reason]
    if not set(OPERATIONAL_BASE_REASONS).issubset(operational_reasons):
        raise RuntimeError("Request rejection lost the public evidence boundary")
    locked_days = np.asarray(LOCKED_CALENDAR_ELAPSED_DAYS, dtype=float)
    prefix_end_days = float(locked_days[PREFIX_END_INDEX])
    rows: list[dict[str, object]] = []
    for row in request["forecast"]:
        assert isinstance(row, Mapping)
        index = int(row["forecast_index"])
        elapsed_days = float(locked_days[index])
        rows.append(
            {
                "request_id": str(request["request_id"]),
                "model_id": EXPERIMENT_ID,
                "forecast_index": index,
                "elapsed_days": elapsed_days,
                "forecast_horizon_days": elapsed_days - prefix_end_days,
                "predicted_capacity_retention_pct": None,
                "predictive_sd_pp": None,
                "residual_correction_pp": None,
                "mean_route": operational.mean_route.value,
                "mean_fallback_reasons": ";".join(fallback_reasons),
                "domain_supported": domain_supported,
                "calibration_horizon_matched": True,
                "diagnostic_interval_status": DIAGNOSTIC_UNAVAILABLE,
                "diagnostic_abstention_reasons": reason,
                "diagnostic_lower_pct": None,
                "diagnostic_upper_pct": None,
                "operational_issuance_status": operational.issuance_status.value,
            }
        )
    forecast = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    request_hash = _canonical_json_sha256(request)
    prediction_hash = _canonical_json_sha256(
        {
            "request_sha256": request_hash,
            "training_state_sha256": reference_state.training_state_sha256,
            "calibration_state_sha256": reference_state.calibration_state_sha256,
            "status": status,
            "reason": reason,
            "failure_type": failure_type,
            "prefix_support": dict(prefix_support),
        }
    )
    reference_ids = set(TRAINING_CONDITION_IDS) | set(CALIBRATION_CONDITION_IDS)
    reference_max_days = float(
        pd.to_numeric(
            reference_observations.loc[
                reference_observations["condition_id"].astype(str).isin(reference_ids),
                "elapsed_days",
            ]
        ).max()
    )
    decision: dict[str, object] = {
        "schema_version": "lifetwin.calendar_prefix_decision.v1",
        "status": status,
        "request_id": str(request["request_id"]),
        "model_id": EXPERIMENT_ID,
        "lifetwin_version": __version__,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "mean_prediction": {
            "status": "unavailable",
            "route": operational.mean_route.value,
            "fallback_reasons": fallback_reasons,
            "activation_gate_ready": False,
            "failure_type": failure_type,
        },
        "diagnostic_interval": {
            "status": DIAGNOSTIC_UNAVAILABLE,
            "role": "retrospective_reused_data_diagnostic_only",
            "requested_coverage": float(request["requested_coverage"]),
            "formal_coverage_claim_allowed": False,
            "calibration_condition_count": 0,
            "calibration_order_statistic_rank": 0,
            "calibration_multiplier": None,
            "abstention_reasons": [reason],
        },
        "operational_decision": {
            "issuance_status": operational.issuance_status.value,
            "abstention_reasons": operational_reasons,
            "lower_pct": None,
            "upper_pct": None,
        },
        "firewall": {
            "target_future_outcomes_used": False,
            "future_outcome_fields_accepted": False,
            "reference_test_condition_outcomes_used": False,
            "target_prefix_observation_count": PREFIX_CHECKUPS,
        },
        "support": {
            "domain_supported": domain_supported,
            **dict(prefix_support),
            "calibration_horizon_matched": True,
            "prefix_end_checkup_index": PREFIX_END_INDEX,
            "forecast_start_checkup_index": FORECAST_START_INDEX,
            "forecast_end_checkup_index": FORECAST_END_INDEX,
            "reference_observation_max_days": reference_max_days,
            "residual_support_horizon_days": float(
                reference_state.residual_fit.support_horizon_days
            ),
            "support_boundary_tolerance_days": (
                RESIDUAL_SUPPORT_BOUNDARY_ATOL_DAYS
            ),
            "claim_15_to_25_year_allowed": False,
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
        "provenance": {
            "request_sha256": request_hash,
            "config_sha256": reference_state.config_sha256,
            "training_state_sha256": reference_state.training_state_sha256,
            "calibration_state_sha256": (
                reference_state.calibration_state_sha256
            ),
            "prediction_state_sha256": prediction_hash,
            "forecast_content_sha256": _frame_sha256(forecast),
            "training_condition_count": len(TRAINING_CONDITION_IDS),
            "calibration_condition_count": len(CALIBRATION_CONDITION_IDS),
        },
    }
    return decision, forecast


def predict_calendar_prefix(
    request: Mapping[str, object],
    *,
    reference_observations: pd.DataFrame,
    config: Mapping[str, object],
    schema: Mapping[str, object] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Generate a public-data research forecast and a conservative decision."""

    parsed = validate_calendar_prefix_request(request, schema=schema)
    reference_state = fit_calendar_v4_reference_state(
        reference_observations, config=config
    )
    if not _request_in_reference_domain(parsed, reference_state):
        return _request_rejection(
            parsed,
            reference_observations=reference_observations,
            reference_state=reference_state,
            status="request_rejected_domain_unsupported",
            reason="domain_unsupported",
            domain_supported=False,
            prefix_support={
                "prefix_supported": None,
                "prefix_support_rule": "not_evaluated_outside_stress_domain",
                "prefix_reference_mode": "not_evaluated",
                "prefix_local_stress_support": None,
                "prefix_support_margin_pp": None,
                "prefix_step_support_margin_pp": None,
                "prefix_point_residual_supported": None,
                "prefix_step_residual_supported": None,
                "prefix_max_point_deviation_pp": None,
                "prefix_max_step_deviation_pp": None,
            },
        )
    prefix_supported, prefix_support = _prefix_in_reference_support(
        parsed,
        reference_observations=reference_observations,
        config=config,
    )
    if not prefix_supported:
        return _request_rejection(
            parsed,
            reference_observations=reference_observations,
            reference_state=reference_state,
            status="request_rejected_prefix_outside_reference_support",
            reason="prefix_outside_reference_support",
            domain_supported=True,
            prefix_support=prefix_support,
        )
    try:
        predicted = predict_calendar_v4_condition(
            _target_condition(parsed),
            reference_state=reference_state,
            config=config,
        )
    except Exception as exc:
        return _request_rejection(
            parsed,
            reference_observations=reference_observations,
            reference_state=reference_state,
            status="request_rejected_model_fit_failed",
            reason="model_fit_failed",
            domain_supported=True,
            prefix_support=prefix_support,
            failure_type=type(exc).__name__,
        )
    coverage = float(parsed["requested_coverage"])
    quantile = _calibration_quantile(
        reference_state.calibration_quantiles,
        route=predicted.mean_route,
        coverage=coverage,
    )
    multiplier_value = quantile["multiplier"]
    multiplier = None if pd.isna(multiplier_value) else float(multiplier_value)
    horizons = (
        predicted.future["elapsed_days"].to_numpy(dtype=float)
        - predicted.prefix_end_days
    )
    horizon_matched = bool(
        np.all(horizons >= -RESIDUAL_SUPPORT_BOUNDARY_ATOL_DAYS)
        and np.all(
            horizons
            <= float(reference_state.residual_fit.support_horizon_days)
            + RESIDUAL_SUPPORT_BOUNDARY_ATOL_DAYS
        )
        and predicted.future["checkup_index"].astype(int).tolist()
        == list(FORECAST_INDICES)
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
    diagnostic_available = not diagnostic_reasons
    if diagnostic_available:
        assert multiplier is not None
        radius = multiplier * predicted.predictive_sd_pp
        diagnostic_lower = np.clip(
            predicted.predicted_retention_pct - radius, 0.0, 100.0
        )
        diagnostic_upper = np.clip(
            predicted.predicted_retention_pct + radius, 0.0, 100.0
        )
        max_width = float(np.max(diagnostic_upper - diagnostic_lower))
    else:
        diagnostic_lower = np.full(len(FORECAST_INDICES), np.nan)
        diagnostic_upper = np.full(len(FORECAST_INDICES), np.nan)
        max_width = None

    operational = conservative_issuance_decision(
        specialist_gate_ready=predicted.activation_gate_ready,
        specialist_fit_succeeded=(
            "specialist_fit_failed" not in predicted.mean_fallback_reasons
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
        interval_width_pp=max_width,
        max_interval_width_pp=None,
    )
    operational_reasons = tuple(
        reason.value for reason in operational.abstention_reasons
    )
    if not set(OPERATIONAL_BASE_REASONS).issubset(operational_reasons):
        raise RuntimeError("Public demo must retain the evidence abstention reasons")

    expose_mean = predicted.domain_supported and horizon_matched
    rows: list[dict[str, object]] = []
    fallback_reasons = ";".join(predicted.mean_fallback_reasons) or "none"
    diagnostic_reason_text = ";".join(diagnostic_reasons) or "none"
    for position, coordinate in enumerate(predicted.future.itertuples(index=False)):
        rows.append(
            {
                "request_id": str(parsed["request_id"]),
                "model_id": EXPERIMENT_ID,
                "forecast_index": int(coordinate.checkup_index),
                "elapsed_days": float(coordinate.elapsed_days),
                "forecast_horizon_days": float(horizons[position]),
                "predicted_capacity_retention_pct": (
                    float(predicted.predicted_retention_pct[position])
                    if expose_mean
                    else None
                ),
                "predictive_sd_pp": (
                    float(predicted.predictive_sd_pp[position])
                    if expose_mean
                    else None
                ),
                "residual_correction_pp": (
                    float(predicted.residual_correction_pp[position])
                    if expose_mean
                    else None
                ),
                "mean_route": predicted.mean_route,
                "mean_fallback_reasons": fallback_reasons,
                "domain_supported": predicted.domain_supported,
                "calibration_horizon_matched": horizon_matched,
                "diagnostic_interval_status": (
                    DIAGNOSTIC_AVAILABLE
                    if diagnostic_available
                    else DIAGNOSTIC_UNAVAILABLE
                ),
                "diagnostic_abstention_reasons": diagnostic_reason_text,
                "diagnostic_lower_pct": (
                    float(diagnostic_lower[position])
                    if diagnostic_available
                    else None
                ),
                "diagnostic_upper_pct": (
                    float(diagnostic_upper[position])
                    if diagnostic_available
                    else None
                ),
                "operational_issuance_status": operational.issuance_status.value,
            }
        )
    forecast = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    request_hash = _canonical_json_sha256(parsed)
    forecast_hash = _frame_sha256(forecast)
    status = (
        "research_forecast_generated_operationally_abstained"
        if expose_mean
        else "request_rejected_domain_or_horizon_unsupported"
    )
    decision: dict[str, object] = {
        "schema_version": "lifetwin.calendar_prefix_decision.v1",
        "status": status,
        "request_id": str(parsed["request_id"]),
        "model_id": EXPERIMENT_ID,
        "lifetwin_version": __version__,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "mean_prediction": {
            "status": "available" if expose_mean else "unavailable",
            "route": predicted.mean_route,
            "fallback_reasons": list(predicted.mean_fallback_reasons),
            "activation_gate_ready": predicted.activation_gate_ready,
            "failure_type": None,
        },
        "diagnostic_interval": {
            "status": (
                DIAGNOSTIC_AVAILABLE
                if diagnostic_available
                else DIAGNOSTIC_UNAVAILABLE
            ),
            "role": "retrospective_reused_data_diagnostic_only",
            "requested_coverage": coverage,
            "formal_coverage_claim_allowed": False,
            "calibration_condition_count": int(
                quantile["calibration_condition_count"]
            ),
            "calibration_order_statistic_rank": int(
                quantile["order_statistic_rank"]
            ),
            "calibration_multiplier": multiplier,
            "abstention_reasons": diagnostic_reasons,
        },
        "operational_decision": {
            "issuance_status": operational.issuance_status.value,
            "abstention_reasons": list(operational_reasons),
            "lower_pct": None,
            "upper_pct": None,
        },
        "firewall": {
            "target_future_outcomes_used": False,
            "future_outcome_fields_accepted": False,
            "reference_test_condition_outcomes_used": False,
            "target_prefix_observation_count": PREFIX_CHECKUPS,
        },
        "support": {
            "domain_supported": predicted.domain_supported,
            **prefix_support,
            "calibration_horizon_matched": horizon_matched,
            "prefix_end_checkup_index": PREFIX_END_INDEX,
            "forecast_start_checkup_index": FORECAST_START_INDEX,
            "forecast_end_checkup_index": FORECAST_END_INDEX,
            "reference_observation_max_days": float(
                pd.to_numeric(
                    reference_observations.loc[
                        reference_observations["condition_id"]
                        .astype(str)
                        .isin(
                            set(TRAINING_CONDITION_IDS)
                            | set(CALIBRATION_CONDITION_IDS)
                        ),
                        "elapsed_days",
                    ]
                ).max()
            ),
            "residual_support_horizon_days": float(
                reference_state.residual_fit.support_horizon_days
            ),
            "support_boundary_tolerance_days": (
                RESIDUAL_SUPPORT_BOUNDARY_ATOL_DAYS
            ),
            "claim_15_to_25_year_allowed": False,
        },
        "claim_boundary": list(CLAIM_BOUNDARY),
        "provenance": {
            "request_sha256": request_hash,
            "config_sha256": reference_state.config_sha256,
            "training_state_sha256": reference_state.training_state_sha256,
            "calibration_state_sha256": (
                reference_state.calibration_state_sha256
            ),
            "prediction_state_sha256": predicted.prediction_state_sha256,
            "forecast_content_sha256": forecast_hash,
            "training_condition_count": len(TRAINING_CONDITION_IDS),
            "calibration_condition_count": len(CALIBRATION_CONDITION_IDS),
        },
    }
    return decision, forecast
