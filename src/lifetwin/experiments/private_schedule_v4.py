"""Declared-schedule challenger for private dual-clock forecasts.

The schedule is a prediction-time input.  It contains no capacity outcome and
is sealed alongside the resulting curve.  V4 preserves the V3 posterior for
the observed prefix, then adjusts future basis increments using condition
priors for each declared operating segment.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    PrivateDualClockPriorV3Error,
    predict_private_dual_clock_prior_capsule,
)
from lifetwin.models.hierarchical_cycle_prior import (
    DualClockKernelPrior,
    dual_clock_basis_coordinates,
    dual_clock_condition_prior_coefficients,
    dual_clock_prior_coefficients,
)
from lifetwin.validation.private_cycle_adapter import PARTITIONED_PREFIX_COLUMNS


SCHEDULE_MODE_ID = "v4_declared_schedule_delta_prior"
ELAPSED_SCHEDULE_MODE_ID = "v4_1_explicit_elapsed_dual_clock"
BOUNDED_SCHEDULE_MODE_ID = "v4_2_support_gated_bounded_delta"
SCHEDULE_ADAPTATION_WEIGHT = 1.0
BOUNDED_SCHEDULE_MAX_WEIGHT = 0.25
BOUNDED_SCHEDULE_INTERVAL_FRACTION = 0.25
FORECAST_SCHEDULE_COLUMNS = (
    "partition",
    "cell_id",
    "condition_id",
    "landmark_visit_count",
    "scenario_id",
    "schedule_role",
    "schedule_source",
    "declared_at_elapsed_days",
    "forecast_elapsed_days",
    "forecast_equivalent_full_cycles",
    "planned_temperature_c",
    "planned_min_soc_pct",
    "planned_max_soc_pct",
    "planned_charge_c_rate",
    "planned_discharge_c_rate",
)
SCHEDULE_ROLES = ("deployment_candidate", "oracle_upper_bound")
SCHEDULE_SOURCES = (
    "declared_operating_plan",
    "operator_scenario",
    "realized_future_schedule",
)


class PrivateScheduleV4Error(ValueError):
    """Raised when a declared future schedule violates the V4 firewall."""


def canonicalize_private_forecast_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Validate outcome-free schedule fields and return a canonical frame."""
    if tuple(schedule.columns) != FORECAST_SCHEDULE_COLUMNS:
        raise PrivateScheduleV4Error("Forecast schedule columns changed")
    if schedule.empty or schedule.isna().any().any():
        raise PrivateScheduleV4Error("Forecast schedule is empty or incomplete")
    data = schedule.copy()
    string_columns = (
        "partition",
        "cell_id",
        "condition_id",
        "scenario_id",
        "schedule_role",
        "schedule_source",
    )
    for column in string_columns:
        data[column] = data[column].astype(str)
        if (data[column].str.strip() == "").any():
            raise PrivateScheduleV4Error(f"{column} contains empty values")
    numeric_columns = tuple(
        column for column in FORECAST_SCHEDULE_COLUMNS if column not in string_columns
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column].to_numpy(dtype=float)).all():
            raise PrivateScheduleV4Error(f"{column} contains non-finite values")
    landmarks = data["landmark_visit_count"].to_numpy(dtype=float)
    if (landmarks < 3).any() or not np.equal(landmarks, np.floor(landmarks)).all():
        raise PrivateScheduleV4Error("Schedule landmarks must be integers >= 3")
    data["landmark_visit_count"] = landmarks.astype(int)
    if (
        len(set(data["schedule_role"])) != 1
        or data.iloc[0]["schedule_role"] not in SCHEDULE_ROLES
    ):
        raise PrivateScheduleV4Error("Forecast schedule must have one supported role")
    if (
        len(set(data["schedule_source"])) != 1
        or data.iloc[0]["schedule_source"] not in SCHEDULE_SOURCES
    ):
        raise PrivateScheduleV4Error("Forecast schedule must have one supported source")
    role = str(data.iloc[0]["schedule_role"])
    source = str(data.iloc[0]["schedule_source"])
    if (role == "oracle_upper_bound") != (source == "realized_future_schedule"):
        raise PrivateScheduleV4Error(
            "Realized future schedules must be isolated as oracle upper bounds"
        )
    if (data["declared_at_elapsed_days"] < 0.0).any():
        raise PrivateScheduleV4Error("Schedule declaration time cannot be negative")
    if (data["forecast_elapsed_days"] <= 0.0).any() or (
        data["forecast_equivalent_full_cycles"] <= 0.0
    ).any():
        raise PrivateScheduleV4Error("Forecast coordinates must be positive")
    if (data["planned_temperature_c"] < -60.0).any() or (
        data["planned_temperature_c"] > 80.0
    ).any():
        raise PrivateScheduleV4Error("Planned temperature is outside [-60, 80] C")
    if (
        (data["planned_min_soc_pct"] < 0.0).any()
        or (data["planned_max_soc_pct"] > 100.0).any()
        or (data["planned_min_soc_pct"] >= data["planned_max_soc_pct"]).any()
    ):
        raise PrivateScheduleV4Error("Planned SOC bounds are invalid")
    if (data["planned_charge_c_rate"] <= 0.0).any() or (
        data["planned_discharge_c_rate"] <= 0.0
    ).any():
        raise PrivateScheduleV4Error("Planned C-rates must be positive")
    duplicate_key = [
        "cell_id",
        "landmark_visit_count",
        "forecast_equivalent_full_cycles",
    ]
    if data.duplicated(duplicate_key).any():
        raise PrivateScheduleV4Error("Forecast schedule coordinates are duplicated")
    return data.sort_values(duplicate_key, kind="stable", ignore_index=True).loc[
        :, FORECAST_SCHEDULE_COLUMNS
    ]


def _forecast_grid(
    prefix: pd.DataFrame, model_config: Mapping[str, object]
) -> np.ndarray:
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    end = float(model_config["score_end_equivalent_full_cycles"])
    step = float(model_config["forecast_grid_step_equivalent_full_cycles"])
    first = math.ceil((x0 + 1e-12) / step) * step
    later = np.arange(first, end + step * 0.5, step, dtype=float)
    return later[later > x0]


def validate_private_forecast_schedule(
    schedule: pd.DataFrame,
    target_prefixes: pd.DataFrame,
    model_config: Mapping[str, object],
) -> pd.DataFrame:
    """Bind a schedule to every sealed target prefix and frozen EFC grid."""
    data = canonicalize_private_forecast_schedule(schedule)
    if tuple(target_prefixes.columns) != PARTITIONED_PREFIX_COLUMNS:
        raise PrivateScheduleV4Error("Target prefix columns changed")
    partitions = set(target_prefixes["partition"].astype(str))
    if len(partitions) != 1 or set(data["partition"]) != partitions:
        raise PrivateScheduleV4Error("Schedule partition differs from target prefixes")
    landmarks = {int(value) for value in model_config["landmark_visit_counts"]}
    expected_keys = {
        (str(cell_id), landmark)
        for cell_id in sorted(target_prefixes["cell_id"].astype(str).unique())
        for landmark in landmarks
    }
    observed_keys = {
        (str(row.cell_id), int(row.landmark_visit_count))
        for row in data.loc[:, ["cell_id", "landmark_visit_count"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    if observed_keys != expected_keys:
        raise PrivateScheduleV4Error(
            "Schedule does not cover every sealed cell and landmark"
        )
    role = str(data.iloc[0]["schedule_role"])
    for (cell_id, landmark), group in data.groupby(
        ["cell_id", "landmark_visit_count"], sort=True
    ):
        prefix = target_prefixes.loc[
            (target_prefixes["cell_id"].astype(str) == str(cell_id))
            & (target_prefixes["landmark_visit_count"] == int(landmark))
        ].sort_values("visit_index", kind="stable")
        if len(prefix) != int(landmark):
            raise PrivateScheduleV4Error("Schedule prefix landmark is not exact")
        if set(group["condition_id"]) != set(prefix["condition_id"].astype(str)):
            raise PrivateScheduleV4Error("Schedule condition identity changed")
        for column in ("scenario_id", "declared_at_elapsed_days"):
            if group[column].nunique() != 1:
                raise PrivateScheduleV4Error(f"{column} changes within a schedule")
        last_days = float(prefix.iloc[-1]["elapsed_days"])
        last_efc = float(prefix.iloc[-1]["equivalent_full_cycles"])
        if (
            role == "deployment_candidate"
            and float(group.iloc[0]["declared_at_elapsed_days"]) > last_days + 1e-12
        ):
            raise PrivateScheduleV4Error(
                "Deployment schedule was declared after its prediction landmark"
            )
        elapsed = group["forecast_elapsed_days"].to_numpy(dtype=float)
        exposure = group["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
        if (
            elapsed[0] <= last_days
            or exposure[0] <= last_efc
            or (np.diff(elapsed) <= 0.0).any()
            or (np.diff(exposure) <= 0.0).any()
        ):
            raise PrivateScheduleV4Error(
                "Schedule coordinates must increase beyond the observed prefix"
            )
        expected_grid = _forecast_grid(
            prefix.loc[:, RPT_TRAJECTORY_COLUMNS], model_config
        )
        if exposure.shape != expected_grid.shape or not np.allclose(
            exposure, expected_grid, rtol=0.0, atol=1e-10
        ):
            raise PrivateScheduleV4Error(
                "Schedule EFC coordinates differ from the frozen forecast grid"
            )
    return data


def _schedule_support_diagnostics(
    prefix: pd.DataFrame,
    plan: pd.DataFrame,
    model: DualClockKernelPrior,
) -> tuple[list[float], np.ndarray]:
    """Return condition distances and segment duty rates without outcomes."""
    previous_days = float(prefix.iloc[-1]["elapsed_days"])
    previous_efc = float(prefix.iloc[-1]["equivalent_full_cycles"])
    distances: list[float] = []
    duty_rates: list[float] = []
    for row in plan.itertuples(index=False):
        delta_days = float(row.forecast_elapsed_days) - previous_days
        delta_efc = float(row.forecast_equivalent_full_cycles) - previous_efc
        if delta_days <= 0.0 or delta_efc <= 0.0:
            raise PrivateScheduleV4Error("Schedule segment exposure must be positive")
        duty_rate = delta_efc / delta_days
        condition = np.asarray(
            [
                float(row.planned_temperature_c),
                (float(row.planned_max_soc_pct) - float(row.planned_min_soc_pct))
                / 100.0,
                float(row.planned_discharge_c_rate),
                math.log(duty_rate),
            ]
        )
        _, condition_distances = dual_clock_condition_prior_coefficients(
            condition, model
        )
        distances.append(float(np.min(condition_distances)))
        duty_rates.append(duty_rate)
        previous_days = float(row.forecast_elapsed_days)
        previous_efc = float(row.forecast_equivalent_full_cycles)
    return distances, np.asarray(duty_rates, dtype=float)


def predict_private_dual_clock_elapsed_schedule_capsule(
    prefix: pd.DataFrame,
    schedule: pd.DataFrame,
    capsule: Mapping[str, object],
    *,
    strict_ood: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Use declared future elapsed time without condition-prior curve shifts.

    Planned temperature, SOC and rate fields remain sealed provenance and OOD
    diagnostics. Only the two future clock coordinates enter the predictor.
    """
    plan = canonicalize_private_forecast_schedule(schedule)
    if plan[["cell_id", "landmark_visit_count"]].drop_duplicates().shape[0] != 1:
        raise PrivateScheduleV4Error(
            "Capsule prediction needs exactly one schedule key"
        )
    ordered = prefix.loc[:, RPT_TRAJECTORY_COLUMNS].sort_values(
        "visit_index", kind="stable"
    )
    forecast_efc = plan["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
    forecast_days = plan["forecast_elapsed_days"].to_numpy(dtype=float)
    result, metadata = predict_private_dual_clock_prior_capsule(
        ordered,
        forecast_efc,
        capsule,
        forecast_elapsed_days=forecast_days,
        strict_ood=False,
    )
    landmark = int(metadata["selected_landmark_visit_count"])
    model_bundle = capsule["landmark_models"][str(landmark)]
    model = DualClockKernelPrior.from_dict(model_bundle["dual_clock_prior"])
    schedule_distances, duty_rates = _schedule_support_diagnostics(ordered, plan, model)
    threshold = float(metadata["condition_ood_threshold"])
    maximum_distance = max(schedule_distances)
    schedule_supported = maximum_distance <= threshold
    prefix_supported = str(metadata["evidence_status"]) == "supported"
    if strict_ood and (not prefix_supported or not schedule_supported):
        raise PrivateDualClockPriorV3Error(
            "Private target prefix or declared schedule is outside capsule support"
        )
    metadata.update(
        {
            "schedule_mode_id": ELAPSED_SCHEDULE_MODE_ID,
            "future_schedule_source": str(plan.iloc[0]["schedule_source"]),
            "schedule_role": str(plan.iloc[0]["schedule_role"]),
            "schedule_scenario_id": str(plan.iloc[0]["scenario_id"]),
            "maximum_schedule_condition_distance": maximum_distance,
            "schedule_adjustment_max_abs_pp": 0.0,
            "schedule_segment_duty_rate_min_efc_per_day": float(np.min(duty_rates)),
            "schedule_segment_duty_rate_max_efc_per_day": float(np.max(duty_rates)),
            "schedule_features_used_by_model": [
                "forecast_elapsed_days",
                "forecast_equivalent_full_cycles",
            ],
            "schedule_features_used_for_support_diagnostics": [
                "planned_temperature_c",
                "planned_min_soc_pct",
                "planned_max_soc_pct",
                "planned_discharge_c_rate",
                "segment_efc_per_day",
            ],
            "planned_charge_c_rate_used_by_model": False,
            "evidence_status": (
                "supported"
                if prefix_supported and schedule_supported
                else "schedule_or_prefix_outside_training_support"
            ),
            "primary_evidence_eligible": str(plan.iloc[0]["schedule_role"])
            == "deployment_candidate",
        }
    )
    return result, metadata


def predict_private_dual_clock_schedule_capsule(
    prefix: pd.DataFrame,
    schedule: pd.DataFrame,
    capsule: Mapping[str, object],
    *,
    strict_ood: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply a declared piecewise operating plan to a frozen V3 capsule."""
    plan = canonicalize_private_forecast_schedule(schedule)
    if plan[["cell_id", "landmark_visit_count"]].drop_duplicates().shape[0] != 1:
        raise PrivateScheduleV4Error(
            "Capsule prediction needs exactly one schedule key"
        )
    ordered = prefix.loc[:, RPT_TRAJECTORY_COLUMNS].sort_values(
        "visit_index", kind="stable"
    )
    forecast_efc = plan["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
    forecast_days = plan["forecast_elapsed_days"].to_numpy(dtype=float)
    baseline, metadata = predict_private_dual_clock_prior_capsule(
        ordered,
        forecast_efc,
        capsule,
        forecast_elapsed_days=forecast_days,
        strict_ood=False,
    )
    landmark = int(metadata["selected_landmark_visit_count"])
    model_bundle = capsule["landmark_models"][str(landmark)]
    model = DualClockKernelPrior.from_dict(model_bundle["dual_clock_prior"])
    prefix_prior, _ = dual_clock_prior_coefficients(ordered, model)
    previous_days = float(ordered.iloc[-1]["elapsed_days"])
    previous_efc = float(ordered.iloc[-1]["equivalent_full_cycles"])
    previous_basis = dual_clock_basis_coordinates(
        [previous_days], [previous_efc], model
    )[0]
    cumulative_adjustment = 0.0
    adjustments: list[float] = []
    schedule_distances, _ = _schedule_support_diagnostics(ordered, plan, model)
    for row in plan.itertuples(index=False):
        delta_days = float(row.forecast_elapsed_days) - previous_days
        delta_efc = float(row.forecast_equivalent_full_cycles) - previous_efc
        if delta_days <= 0.0 or delta_efc <= 0.0:
            raise PrivateScheduleV4Error("Schedule segment exposure must be positive")
        duty_rate = delta_efc / delta_days
        condition = np.asarray(
            [
                float(row.planned_temperature_c),
                (float(row.planned_max_soc_pct) - float(row.planned_min_soc_pct))
                / 100.0,
                float(row.planned_discharge_c_rate),
                math.log(duty_rate),
            ]
        )
        planned_prior, _ = dual_clock_condition_prior_coefficients(condition, model)
        current_basis = dual_clock_basis_coordinates(
            [float(row.forecast_elapsed_days)],
            [float(row.forecast_equivalent_full_cycles)],
            model,
        )[0]
        coefficient_delta = SCHEDULE_ADAPTATION_WEIGHT * (planned_prior - prefix_prior)
        cumulative_adjustment += float(
            coefficient_delta @ (current_basis - previous_basis)
        )
        adjustments.append(cumulative_adjustment)
        previous_days = float(row.forecast_elapsed_days)
        previous_efc = float(row.forecast_equivalent_full_cycles)
        previous_basis = current_basis
    adjustment = np.asarray(adjustments, dtype=float)
    lower_clip, upper_clip = (float(value) for value in capsule["prediction_clip_pct"])
    result = baseline.copy()
    for column in (
        "predicted_capacity_retention_pct",
        "diagnostic_lower_capacity_retention_pct",
        "diagnostic_upper_capacity_retention_pct",
    ):
        result[column] = np.clip(
            result[column].to_numpy(dtype=float) - adjustment,
            lower_clip,
            upper_clip,
        )
    threshold = float(metadata["condition_ood_threshold"])
    maximum_distance = max(schedule_distances)
    schedule_supported = maximum_distance <= threshold
    prefix_supported = str(metadata["evidence_status"]) == "supported"
    if strict_ood and (not prefix_supported or not schedule_supported):
        raise PrivateDualClockPriorV3Error(
            "Private target prefix or declared schedule is outside capsule support"
        )
    metadata.update(
        {
            "schedule_mode_id": SCHEDULE_MODE_ID,
            "future_schedule_source": str(plan.iloc[0]["schedule_source"]),
            "schedule_role": str(plan.iloc[0]["schedule_role"]),
            "schedule_scenario_id": str(plan.iloc[0]["scenario_id"]),
            "schedule_adaptation_weight": SCHEDULE_ADAPTATION_WEIGHT,
            "maximum_schedule_condition_distance": maximum_distance,
            "schedule_adjustment_max_abs_pp": float(np.max(np.abs(adjustment))),
            "schedule_features_used_by_model": [
                "forecast_elapsed_days",
                "forecast_equivalent_full_cycles",
                "planned_temperature_c",
                "planned_min_soc_pct",
                "planned_max_soc_pct",
                "planned_discharge_c_rate",
            ],
            "planned_charge_c_rate_used_by_model": False,
            "evidence_status": (
                "supported"
                if prefix_supported and schedule_supported
                else "schedule_or_prefix_outside_training_support"
            ),
            "primary_evidence_eligible": str(plan.iloc[0]["schedule_role"])
            == "deployment_candidate",
        }
    )
    return result, metadata


def predict_private_dual_clock_bounded_schedule_capsule(
    prefix: pd.DataFrame,
    schedule: pd.DataFrame,
    capsule: Mapping[str, object],
    *,
    strict_ood: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Apply a support-gated condition delta bounded by LOCO residual width."""
    baseline, metadata = predict_private_dual_clock_elapsed_schedule_capsule(
        prefix, schedule, capsule, strict_ood=strict_ood
    )
    full, full_metadata = predict_private_dual_clock_schedule_capsule(
        prefix, schedule, capsule, strict_ood=strict_ood
    )
    center = baseline["predicted_capacity_retention_pct"].to_numpy(dtype=float)
    raw_delta = full["predicted_capacity_retention_pct"].to_numpy(dtype=float) - center
    diagnostic_half_width = np.maximum(
        center
        - baseline["diagnostic_lower_capacity_retention_pct"].to_numpy(dtype=float),
        baseline["diagnostic_upper_capacity_retention_pct"].to_numpy(dtype=float)
        - center,
    )
    threshold = float(metadata["condition_ood_threshold"])
    distance = float(metadata["maximum_schedule_condition_distance"])
    support_fraction = (
        max(0.0, min(1.0, 1.0 - distance / threshold)) if threshold > 0.0 else 0.0
    )
    gate_weight = BOUNDED_SCHEDULE_MAX_WEIGHT * support_fraction
    bound = BOUNDED_SCHEDULE_INTERVAL_FRACTION * diagnostic_half_width
    applied_delta = np.clip(gate_weight * raw_delta, -bound, bound)
    lower_clip, upper_clip = (float(value) for value in capsule["prediction_clip_pct"])
    result = baseline.copy()
    for column in (
        "predicted_capacity_retention_pct",
        "diagnostic_lower_capacity_retention_pct",
        "diagnostic_upper_capacity_retention_pct",
    ):
        result[column] = np.clip(
            baseline[column].to_numpy(dtype=float) + applied_delta,
            lower_clip,
            upper_clip,
        )
    metadata.update(
        {
            "schedule_mode_id": BOUNDED_SCHEDULE_MODE_ID,
            "schedule_adaptation_weight": gate_weight,
            "schedule_support_fraction": support_fraction,
            "schedule_adjustment_max_abs_pp": float(np.max(np.abs(applied_delta))),
            "unbounded_schedule_adjustment_max_abs_pp": float(
                np.max(np.abs(raw_delta))
            ),
            "schedule_adjustment_bound_max_pp": float(np.max(bound)),
            "schedule_adjustment_bound_source": (
                "training_inner_loco_diagnostic_interval_width"
            ),
            "bounded_schedule_max_weight": BOUNDED_SCHEDULE_MAX_WEIGHT,
            "bounded_schedule_interval_fraction": (BOUNDED_SCHEDULE_INTERVAL_FRACTION),
            "schedule_features_used_by_model": full_metadata[
                "schedule_features_used_by_model"
            ],
            "schedule_features_used_for_support_diagnostics": [
                "planned_temperature_c",
                "planned_min_soc_pct",
                "planned_max_soc_pct",
                "planned_discharge_c_rate",
                "segment_efc_per_day",
            ],
            "planned_charge_c_rate_used_by_model": False,
        }
    )
    return result, metadata


__all__ = [
    "BOUNDED_SCHEDULE_INTERVAL_FRACTION",
    "BOUNDED_SCHEDULE_MAX_WEIGHT",
    "BOUNDED_SCHEDULE_MODE_ID",
    "ELAPSED_SCHEDULE_MODE_ID",
    "FORECAST_SCHEDULE_COLUMNS",
    "SCHEDULE_ADAPTATION_WEIGHT",
    "SCHEDULE_MODE_ID",
    "PrivateScheduleV4Error",
    "canonicalize_private_forecast_schedule",
    "predict_private_dual_clock_bounded_schedule_capsule",
    "predict_private_dual_clock_elapsed_schedule_capsule",
    "predict_private_dual_clock_schedule_capsule",
    "validate_private_forecast_schedule",
]
