from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


CALENDAR_MODEL_NAME = "empirical_stress_surface_sqrt_time_v1"
CALENDAR_NULL_MODEL_NAME = "condition_agnostic_sqrt_time_v1"
PARAMETER_NAMES = (
    "log_k_at_25c_soc50",
    "inverse_temperature_coefficient",
    "soc_linear_coefficient",
    "soc_quadratic_coefficient",
    "temperature_soc_interaction",
)


@dataclass(frozen=True)
class EmpiricalStressSurface:
    parameters: tuple[float, ...]
    training_condition_ids: tuple[str, ...]
    maximum_training_hours: float
    maximum_supported_hours: float
    optimizer_success: bool
    optimizer_cost: float
    optimizer_evaluations: int

    @property
    def model_name(self) -> str:
        return CALENDAR_MODEL_NAME

    def parameter_map(self) -> dict[str, float]:
        return dict(zip(PARAMETER_NAMES, self.parameters, strict=True))


@dataclass(frozen=True)
class ConditionAgnosticSqrtModel:
    k_pct_per_sqrt_hour: float
    training_condition_ids: tuple[str, ...]
    maximum_training_hours: float
    maximum_supported_hours: float

    @property
    def model_name(self) -> str:
        return CALENDAR_NULL_MODEL_NAME


def _validate_model_frame(frame: pd.DataFrame, *, minimum_conditions: int) -> pd.DataFrame:
    required = {
        "condition_id",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_hours",
        "capacity_loss_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing calendar model columns: {missing}")
    if frame.empty:
        raise ValueError("Calendar model frame cannot be empty")
    ordered = frame.copy()
    numeric_columns = [
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_hours",
        "capacity_loss_pct",
    ]
    ordered[numeric_columns] = ordered[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if ordered[numeric_columns].isna().any().any():
        raise ValueError("Calendar model values must be numeric")
    if not np.isfinite(ordered[numeric_columns].to_numpy()).all():
        raise ValueError("Calendar model values must be finite")
    if (ordered["elapsed_hours"] < 0).any():
        raise ValueError("Elapsed hours cannot be negative")
    if not ordered["storage_soc_fraction"].between(0.0, 1.0).all():
        raise ValueError("Storage SOC must be a fraction in [0, 1]")
    if ordered["condition_id"].nunique() < minimum_conditions:
        raise ValueError(f"At least {minimum_conditions} independent conditions are required")
    if "checkup_index" in ordered and ordered.duplicated(
        ["condition_id", "checkup_index"]
    ).any():
        raise ValueError("Duplicate condition/checkup rows are not allowed")
    return ordered.sort_values(
        ["condition_id", "elapsed_hours"],
        kind="stable",
    ).reset_index(drop=True)


def _stress_features(frame: pd.DataFrame) -> np.ndarray:
    temperature_kelvin = frame["temperature_c"].to_numpy(dtype=float) + 273.15
    if np.any(temperature_kelvin <= 0):
        raise ValueError("Temperature in Kelvin must be positive")
    inverse_temperature = 1000.0 * (1.0 / 298.15 - 1.0 / temperature_kelvin)
    centered_soc = frame["storage_soc_fraction"].to_numpy(dtype=float) - 0.5
    return np.column_stack(
        [
            np.ones(len(frame)),
            inverse_temperature,
            centered_soc,
            centered_soc**2,
            inverse_temperature * centered_soc,
        ]
    )


def _condition_balancing_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("condition_id")["condition_id"].transform("size")
    return np.sqrt(1.0 / counts.to_numpy(dtype=float))


def _initial_parameters(frame: pd.DataFrame) -> np.ndarray:
    rows: list[dict[str, float]] = []
    for _, condition in frame.groupby("condition_id", sort=True):
        positive = condition.loc[condition["elapsed_hours"] > 0]
        root_time = np.sqrt(positive["elapsed_hours"].to_numpy(dtype=float))
        loss = positive["capacity_loss_pct"].to_numpy(dtype=float)
        denominator = float(root_time @ root_time)
        k_value = max(float(root_time @ loss) / denominator, 1e-5)
        first = positive.iloc[0]
        rows.append(
            {
                "temperature_c": float(first["temperature_c"]),
                "storage_soc_fraction": float(first["storage_soc_fraction"]),
                "log_k": float(np.log(k_value)),
            }
        )
    condition_rates = pd.DataFrame(rows)
    design = _stress_features(condition_rates)
    coefficients, *_ = np.linalg.lstsq(
        design,
        condition_rates["log_k"].to_numpy(dtype=float),
        rcond=None,
    )
    lower = np.array([-12.0, -12.0, -12.0, -12.0, -12.0])
    upper = np.array([2.0, 12.0, 12.0, 12.0, 12.0])
    return np.clip(coefficients, lower + 1e-8, upper - 1e-8)


def fit_empirical_stress_surface(
    frame: pd.DataFrame,
    *,
    minimum_conditions: int = 6,
    robust_loss_scale_pp: float = 0.25,
    maximum_prediction_hours: float | None = None,
) -> EmpiricalStressSurface:
    """Fit a positive T/SOC stress surface with a fixed square-root time law."""
    ordered = _validate_model_frame(frame, minimum_conditions=minimum_conditions)
    fit_frame = ordered.loc[ordered["elapsed_hours"] > 0].copy()
    design = _stress_features(fit_frame)
    root_time = np.sqrt(fit_frame["elapsed_hours"].to_numpy(dtype=float))
    observed_loss = fit_frame["capacity_loss_pct"].to_numpy(dtype=float)
    weights = _condition_balancing_weights(fit_frame)

    def residual(parameters: np.ndarray) -> np.ndarray:
        log_k = np.clip(design @ parameters, -30.0, 10.0)
        predicted = np.exp(log_k) * root_time
        return weights * (predicted - observed_loss)

    fitted = least_squares(
        residual,
        _initial_parameters(fit_frame),
        bounds=(
            np.array([-12.0, -12.0, -12.0, -12.0, -12.0]),
            np.array([2.0, 12.0, 12.0, 12.0, 12.0]),
        ),
        loss="soft_l1",
        f_scale=float(robust_loss_scale_pp),
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Calendar stress-surface fit failed: {fitted.message}")
    maximum_training_hours = float(ordered["elapsed_hours"].max())
    if maximum_prediction_hours is None:
        maximum_prediction_hours = maximum_training_hours
    maximum_prediction_hours = float(maximum_prediction_hours)
    if (
        not np.isfinite(maximum_prediction_hours)
        or maximum_prediction_hours < maximum_training_hours
    ):
        raise ValueError(
            "Maximum prediction hours must be finite and cannot precede training support"
        )
    return EmpiricalStressSurface(
        parameters=tuple(float(value) for value in fitted.x),
        training_condition_ids=tuple(sorted(ordered["condition_id"].astype(str).unique())),
        maximum_training_hours=maximum_training_hours,
        maximum_supported_hours=maximum_prediction_hours,
        optimizer_success=bool(fitted.success),
        optimizer_cost=float(fitted.cost),
        optimizer_evaluations=int(fitted.nfev),
    )


def fit_condition_agnostic_sqrt_model(
    frame: pd.DataFrame,
    *,
    minimum_conditions: int = 2,
    maximum_prediction_hours: float | None = None,
) -> ConditionAgnosticSqrtModel:
    ordered = _validate_model_frame(frame, minimum_conditions=minimum_conditions)
    fit_frame = ordered.loc[ordered["elapsed_hours"] > 0].copy()
    root_time = np.sqrt(fit_frame["elapsed_hours"].to_numpy(dtype=float))
    loss = fit_frame["capacity_loss_pct"].to_numpy(dtype=float)
    weights = _condition_balancing_weights(fit_frame)
    weighted_time = weights * root_time
    weighted_loss = weights * loss
    denominator = float(weighted_time @ weighted_time)
    if denominator <= 0:
        raise ValueError("Positive elapsed-time support is required")
    k_value = max(float(weighted_time @ weighted_loss) / denominator, 0.0)
    maximum_training_hours = float(ordered["elapsed_hours"].max())
    if maximum_prediction_hours is None:
        maximum_prediction_hours = maximum_training_hours
    maximum_prediction_hours = float(maximum_prediction_hours)
    if (
        not np.isfinite(maximum_prediction_hours)
        or maximum_prediction_hours < maximum_training_hours
    ):
        raise ValueError(
            "Maximum prediction hours must be finite and cannot precede training support"
        )
    return ConditionAgnosticSqrtModel(
        k_pct_per_sqrt_hour=k_value,
        training_condition_ids=tuple(sorted(ordered["condition_id"].astype(str).unique())),
        maximum_training_hours=maximum_training_hours,
        maximum_supported_hours=maximum_prediction_hours,
    )


def _validate_prediction_horizon(
    elapsed_hours: np.ndarray,
    maximum_supported_hours: float,
) -> None:
    if np.any(elapsed_hours < 0) or not np.isfinite(elapsed_hours).all():
        raise ValueError("Prediction elapsed hours must be finite and non-negative")
    tolerance = max(1e-9, maximum_supported_hours * 1e-10)
    if np.any(elapsed_hours > maximum_supported_hours + tolerance):
        requested = float(elapsed_hours.max())
        raise ValueError(
            "Calendar projection beyond observed support is prohibited: "
            f"requested={requested:.6f} h, supported={maximum_supported_hours:.6f} h"
        )


def predict_stress_surface_loss(
    model: EmpiricalStressSurface,
    frame: pd.DataFrame,
) -> np.ndarray:
    required = {"temperature_c", "storage_soc_fraction", "elapsed_hours"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    prediction_frame = frame.copy()
    prediction_frame[list(required)] = prediction_frame[list(required)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if prediction_frame[list(required)].isna().any().any():
        raise ValueError("Prediction inputs must be numeric")
    if not prediction_frame["storage_soc_fraction"].between(0.0, 1.0).all():
        raise ValueError("Prediction SOC must be in [0, 1]")
    elapsed = prediction_frame["elapsed_hours"].to_numpy(dtype=float)
    _validate_prediction_horizon(elapsed, model.maximum_supported_hours)
    log_k = _stress_features(prediction_frame) @ np.asarray(model.parameters)
    return np.exp(np.clip(log_k, -30.0, 10.0)) * np.sqrt(elapsed)


def predict_condition_agnostic_loss(
    model: ConditionAgnosticSqrtModel,
    elapsed_hours: Sequence[float] | np.ndarray,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_hours, dtype=float)
    _validate_prediction_horizon(elapsed, model.maximum_supported_hours)
    return model.k_pct_per_sqrt_hour * np.sqrt(elapsed)


def select_prefix(frame: pd.DataFrame, prefix_checkups: int) -> pd.DataFrame:
    if prefix_checkups < 2:
        raise ValueError("A prefix must include the initial checkup and one later checkup")
    if "checkup_index" not in frame:
        raise ValueError("Prefix selection requires checkup_index")
    selected = frame.loc[pd.to_numeric(frame["checkup_index"]) < prefix_checkups]
    if selected.empty:
        raise ValueError("No observations are available in the requested prefix")
    counts = selected.groupby("condition_id").size()
    if (counts < prefix_checkups).any():
        raise ValueError("Every selected condition must contain the full prefix")
    return selected.copy()


def estimate_empirical_bayes_ridge(
    model: EmpiricalStressSurface,
    training_frame: pd.DataFrame,
    *,
    prefix_checkups: int,
    ridge_bounds: tuple[float, float] = (0.01, 100.0),
) -> dict[str, object]:
    prefix = select_prefix(training_frame, prefix_checkups)
    scales: list[float] = []
    denominators: list[float] = []
    residuals: list[float] = []
    for _, condition in prefix.groupby("condition_id", sort=True):
        positive = condition.loc[condition["elapsed_hours"] > 0]
        base = predict_stress_surface_loss(model, positive)
        observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
        denominator = float(base @ base)
        if denominator <= 0:
            continue
        scale = max(float(base @ observed) / denominator, 0.0)
        scales.append(scale)
        denominators.append(denominator)
        residuals.extend((observed - scale * base).tolist())
    if len(scales) < 3 or not residuals:
        raise ValueError("At least three training conditions are required for prefix updating")
    residual_variance = max(float(np.mean(np.square(residuals))), 1e-8)
    raw_scale_variance = float(np.var(scales, ddof=1))
    mean_measurement_scale_variance = float(
        np.mean([residual_variance / value for value in denominators])
    )
    between_condition_variance = max(
        raw_scale_variance - mean_measurement_scale_variance,
        1e-6,
    )
    raw_ridge = residual_variance / between_condition_variance
    ridge = float(np.clip(raw_ridge, ridge_bounds[0], ridge_bounds[1]))
    return {
        "ridge": ridge,
        "raw_ridge": float(raw_ridge),
        "residual_variance_pp2": residual_variance,
        "between_condition_scale_variance": between_condition_variance,
        "training_scale_count": len(scales),
        "training_scales": [float(value) for value in scales],
    }


def estimate_target_scale(
    model: EmpiricalStressSurface,
    target_prefix: pd.DataFrame,
    *,
    ridge: float,
    scale_bounds: tuple[float, float] = (0.0, 4.0),
) -> float:
    positive = target_prefix.loc[target_prefix["elapsed_hours"] > 0]
    if len(positive) < 2:
        raise ValueError("At least two positive-time target checkups are required")
    base = predict_stress_surface_loss(model, positive)
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    denominator = float(base @ base) + float(ridge)
    posterior_scale = (float(base @ observed) + float(ridge)) / denominator
    return float(np.clip(posterior_scale, scale_bounds[0], scale_bounds[1]))


def estimate_prefix_only_k(target_prefix: pd.DataFrame) -> float:
    positive = target_prefix.loc[target_prefix["elapsed_hours"] > 0]
    if len(positive) < 2:
        raise ValueError("At least two positive-time target checkups are required")
    root_time = np.sqrt(positive["elapsed_hours"].to_numpy(dtype=float))
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    denominator = float(root_time @ root_time)
    return max(float(root_time @ observed) / denominator, 0.0)
