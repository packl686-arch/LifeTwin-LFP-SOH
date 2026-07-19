from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


TARGET_SQRT_METHOD = "target_prefix_only_sqrt_time_v1"
TARGET_POWER_METHOD = "target_prefix_only_power_law_v1"
TARGET_SQRT_LINEAR_METHOD = "target_prefix_only_sqrt_plus_linear_v1"
HIERARCHICAL_POWER_METHOD = "hierarchical_power_law_prefix_update_v2"
METHOD_NAMES = (
    TARGET_SQRT_METHOD,
    TARGET_POWER_METHOD,
    TARGET_SQRT_LINEAR_METHOD,
    HIERARCHICAL_POWER_METHOD,
)

STRESS_FEATURE_NAMES = (
    "intercept",
    "inverse_temperature",
    "centered_soc",
    "centered_soc_squared",
    "inverse_temperature_x_centered_soc",
)
POWER_PARAMETER_NAMES = ("log_amplitude", "time_exponent")


@dataclass(frozen=True)
class PowerLawFit:
    log_amplitude: float
    time_exponent: float
    optimizer_cost: float
    optimizer_evaluations: int
    parameter_covariance: tuple[tuple[float, float], tuple[float, float]] | None

    def parameter_map(self) -> dict[str, float]:
        return {
            "log_amplitude": self.log_amplitude,
            "time_exponent": self.time_exponent,
        }


@dataclass(frozen=True)
class HierarchicalPowerPrior:
    surface_coefficients: tuple[tuple[float, ...], tuple[float, ...]]
    parameter_scales: tuple[float, float]
    observation_scale_pp: float
    training_condition_ids: tuple[str, ...]
    training_observation_count: int
    maximum_training_days: float
    condition_parameters: tuple[tuple[str, float, float], ...]

    def prior_mean(self, frame: pd.DataFrame) -> np.ndarray:
        coefficients = np.asarray(self.surface_coefficients, dtype=float)
        return coefficients @ stress_features_for_condition(frame)


def _validate_frame(
    frame: pd.DataFrame,
    *,
    minimum_conditions: int = 1,
) -> pd.DataFrame:
    required = {
        "condition_id",
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "capacity_loss_pct",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing calendar V2 columns: {missing}")
    if frame.empty:
        raise ValueError("Calendar V2 frame cannot be empty")
    ordered = frame.copy()
    numeric = [
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "capacity_loss_pct",
    ]
    ordered[numeric] = ordered[numeric].apply(pd.to_numeric, errors="coerce")
    if ordered[numeric].isna().any().any():
        raise ValueError("Calendar V2 values must be numeric")
    if not np.isfinite(ordered[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Calendar V2 values must be finite")
    if (ordered["elapsed_days"] < 0.0).any():
        raise ValueError("Elapsed days cannot be negative")
    if not ordered["storage_soc_fraction"].between(0.0, 1.0).all():
        raise ValueError("Storage SOC must be a fraction in [0, 1]")
    if ordered["condition_id"].astype(str).nunique() < minimum_conditions:
        raise ValueError(
            f"At least {minimum_conditions} independent conditions are required"
        )
    return ordered.sort_values(
        ["condition_id", "elapsed_days"],
        kind="stable",
    ).reset_index(drop=True)


def _positive_time(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame.loc[frame["elapsed_days"] > 0.0].copy()
    if len(positive) < 2:
        raise ValueError("At least two positive-time observations are required")
    return positive


def stress_features_for_condition(frame: pd.DataFrame) -> np.ndarray:
    ordered = _validate_frame(frame)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("Stress features require exactly one condition")
    temperature = ordered["temperature_c"].to_numpy(dtype=float)
    soc = ordered["storage_soc_fraction"].to_numpy(dtype=float)
    if not np.allclose(temperature, temperature[0], rtol=0.0, atol=1e-12):
        raise ValueError("Temperature must be constant within a calendar condition")
    if not np.allclose(soc, soc[0], rtol=0.0, atol=1e-12):
        raise ValueError("Storage SOC must be constant within a calendar condition")
    temperature_kelvin = float(temperature[0]) + 273.15
    if temperature_kelvin <= 0.0:
        raise ValueError("Temperature in Kelvin must be positive")
    inverse_temperature = 1000.0 * (
        1.0 / 298.15 - 1.0 / temperature_kelvin
    )
    centered_soc = float(soc[0]) - 0.5
    return np.asarray(
        [
            1.0,
            inverse_temperature,
            centered_soc,
            centered_soc**2,
            inverse_temperature * centered_soc,
        ],
        dtype=float,
    )


def fit_power_law(
    frame: pd.DataFrame,
    *,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
    robust_loss_scale_pp: float = 0.25,
) -> PowerLawFit:
    ordered = _validate_frame(frame)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("A power-law fit requires exactly one condition")
    positive = _positive_time(ordered)
    elapsed = positive["elapsed_days"].to_numpy(dtype=float)
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    lower_exponent, upper_exponent = map(float, exponent_bounds)
    if not 0.0 < lower_exponent < upper_exponent:
        raise ValueError("Power-law exponent bounds must be positive and ordered")
    if robust_loss_scale_pp <= 0.0:
        raise ValueError("Power-law robust loss scale must be positive")

    root_time = np.sqrt(elapsed)
    denominator = float(root_time @ root_time)
    initial_amplitude = max(float(root_time @ observed) / denominator, 1e-6)
    initial_log_amplitude = float(np.clip(np.log(initial_amplitude), -12.0, 5.0))

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (
            np.exp(parameters[0]) * np.power(elapsed, parameters[1]) - observed
        )

    fitted = least_squares(
        residual,
        np.asarray([initial_log_amplitude, 0.5]),
        bounds=(
            np.asarray([-12.0, lower_exponent]),
            np.asarray([5.0, upper_exponent]),
        ),
        loss="soft_l1",
        f_scale=float(robust_loss_scale_pp),
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Power-law fit failed: {fitted.message}")
    return PowerLawFit(
        log_amplitude=float(fitted.x[0]),
        time_exponent=float(fitted.x[1]),
        optimizer_cost=float(fitted.cost),
        optimizer_evaluations=int(fitted.nfev),
        parameter_covariance=None,
    )


def fit_sqrt_rate(frame: pd.DataFrame) -> float:
    ordered = _positive_time(_validate_frame(frame))
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("A square-root fit requires exactly one condition")
    root_time = np.sqrt(ordered["elapsed_days"].to_numpy(dtype=float))
    observed = ordered["capacity_loss_pct"].to_numpy(dtype=float)
    denominator = float(root_time @ root_time)
    if denominator <= 0.0:
        raise ValueError("Positive elapsed-time support is required")
    return max(float(root_time @ observed) / denominator, 0.0)


def fit_sqrt_linear_coefficients(
    frame: pd.DataFrame,
    *,
    time_scale_days: float = 365.0,
    robust_loss_scale_pp: float = 0.25,
) -> tuple[float, float]:
    ordered = _positive_time(_validate_frame(frame))
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("A sqrt-plus-linear fit requires exactly one condition")
    if time_scale_days <= 0.0 or robust_loss_scale_pp <= 0.0:
        raise ValueError("Sqrt-plus-linear scales must be positive")
    scaled_time = ordered["elapsed_days"].to_numpy(dtype=float) / time_scale_days
    design = np.column_stack([np.sqrt(scaled_time), scaled_time])
    observed = ordered["capacity_loss_pct"].to_numpy(dtype=float)
    initial, *_ = np.linalg.lstsq(design, observed, rcond=None)
    initial = np.clip(initial, 0.0, 100.0)

    fitted = least_squares(
        lambda parameters: design @ parameters - observed,
        initial,
        bounds=(np.zeros(2), np.full(2, 100.0)),
        loss="soft_l1",
        f_scale=float(robust_loss_scale_pp),
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Sqrt-plus-linear fit failed: {fitted.message}")
    return float(fitted.x[0]), float(fitted.x[1])


def fit_hierarchical_power_prior(
    training_frame: pd.DataFrame,
    *,
    minimum_conditions: int = 6,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
    robust_loss_scale_pp: float = 0.25,
    stress_surface_ridge: float = 1.0,
    parameter_scale_floors: tuple[float, float] = (0.1, 0.05),
    observation_scale_floor_pp: float = 0.1,
) -> HierarchicalPowerPrior:
    ordered = _validate_frame(
        training_frame,
        minimum_conditions=minimum_conditions,
    )
    if stress_surface_ridge < 0.0:
        raise ValueError("Stress-surface ridge must be non-negative")
    scale_floors = np.asarray(parameter_scale_floors, dtype=float)
    if scale_floors.shape != (2,) or np.any(scale_floors <= 0.0):
        raise ValueError("Two positive parameter scale floors are required")
    if observation_scale_floor_pp <= 0.0:
        raise ValueError("Observation scale floor must be positive")

    feature_rows: list[np.ndarray] = []
    parameter_rows: list[np.ndarray] = []
    residuals: list[float] = []
    condition_parameters: list[tuple[str, float, float]] = []
    for condition_id, condition in ordered.groupby("condition_id", sort=True):
        fitted = fit_power_law(
            condition,
            exponent_bounds=exponent_bounds,
            robust_loss_scale_pp=robust_loss_scale_pp,
        )
        parameters = np.asarray(
            [fitted.log_amplitude, fitted.time_exponent],
            dtype=float,
        )
        feature_rows.append(stress_features_for_condition(condition))
        parameter_rows.append(parameters)
        condition_parameters.append(
            (
                str(condition_id),
                fitted.log_amplitude,
                fitted.time_exponent,
            )
        )
        positive = _positive_time(condition)
        predicted = predict_power_loss(
            fitted,
            positive["elapsed_days"].to_numpy(dtype=float),
        )
        residuals.extend(
            (
                predicted
                - positive["capacity_loss_pct"].to_numpy(dtype=float)
            ).tolist()
        )

    design = np.vstack(feature_rows)
    condition_parameter_array = np.vstack(parameter_rows)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    system = design.T @ design + float(stress_surface_ridge) * penalty
    coefficients = np.linalg.solve(
        system,
        design.T @ condition_parameter_array,
    )
    parameter_residuals = condition_parameter_array - design @ coefficients
    parameter_scales = np.maximum(
        np.std(parameter_residuals, axis=0, ddof=1),
        scale_floors,
    )
    observation_scale = max(
        float(np.sqrt(np.mean(np.square(residuals)))),
        float(observation_scale_floor_pp),
    )
    return HierarchicalPowerPrior(
        surface_coefficients=tuple(
            tuple(float(value) for value in coefficients[:, index])
            for index in range(2)
        ),
        parameter_scales=(
            float(parameter_scales[0]),
            float(parameter_scales[1]),
        ),
        observation_scale_pp=observation_scale,
        training_condition_ids=tuple(
            sorted(ordered["condition_id"].astype(str).unique())
        ),
        training_observation_count=len(ordered),
        maximum_training_days=float(ordered["elapsed_days"].max()),
        condition_parameters=tuple(condition_parameters),
    )


def update_hierarchical_power_law(
    prior: HierarchicalPowerPrior,
    target_prefix: pd.DataFrame,
    *,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
) -> PowerLawFit:
    ordered = _validate_frame(target_prefix)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("A hierarchical update requires exactly one target condition")
    positive = _positive_time(ordered)
    elapsed = positive["elapsed_days"].to_numpy(dtype=float)
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    lower_exponent, upper_exponent = map(float, exponent_bounds)
    prior_mean = prior.prior_mean(ordered)
    prior_mean[0] = float(np.clip(prior_mean[0], -12.0, 5.0))
    prior_mean[1] = float(
        np.clip(prior_mean[1], lower_exponent, upper_exponent)
    )
    prior_scales = np.asarray(prior.parameter_scales, dtype=float)

    def posterior_residual(parameters: np.ndarray) -> np.ndarray:
        predicted = np.exp(parameters[0]) * np.power(elapsed, parameters[1])
        data_residual = (predicted - observed) / prior.observation_scale_pp
        prior_residual = (parameters - prior_mean) / prior_scales
        return np.concatenate([data_residual, prior_residual])

    fitted = least_squares(
        posterior_residual,
        prior_mean,
        bounds=(
            np.asarray([-12.0, lower_exponent]),
            np.asarray([5.0, upper_exponent]),
        ),
        loss="linear",
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Hierarchical power-law update failed: {fitted.message}")
    precision = fitted.jac.T @ fitted.jac
    covariance = np.linalg.pinv(precision, hermitian=True)
    if not np.isfinite(covariance).all():
        raise RuntimeError("Hierarchical posterior covariance is not finite")
    return PowerLawFit(
        log_amplitude=float(fitted.x[0]),
        time_exponent=float(fitted.x[1]),
        optimizer_cost=float(fitted.cost),
        optimizer_evaluations=int(fitted.nfev),
        parameter_covariance=(
            (float(covariance[0, 0]), float(covariance[0, 1])),
            (float(covariance[1, 0]), float(covariance[1, 1])),
        ),
    )


def predict_power_loss(
    fitted: PowerLawFit,
    elapsed_days: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    if np.any(elapsed < 0.0) or not np.isfinite(elapsed).all():
        raise ValueError("Prediction elapsed days must be finite and non-negative")
    return np.exp(fitted.log_amplitude) * np.power(elapsed, fitted.time_exponent)


def predict_sqrt_loss(
    rate: float,
    elapsed_days: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    if rate < 0.0 or np.any(elapsed < 0.0) or not np.isfinite(elapsed).all():
        raise ValueError("Square-root prediction inputs must be finite and non-negative")
    return float(rate) * np.sqrt(elapsed)


def predict_sqrt_linear_loss(
    coefficients: tuple[float, float],
    elapsed_days: np.ndarray | pd.Series | list[float],
    *,
    time_scale_days: float = 365.0,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    values = np.asarray(coefficients, dtype=float)
    if (
        values.shape != (2,)
        or np.any(values < 0.0)
        or time_scale_days <= 0.0
        or np.any(elapsed < 0.0)
        or not np.isfinite(elapsed).all()
    ):
        raise ValueError("Sqrt-plus-linear prediction inputs are invalid")
    scaled = elapsed / float(time_scale_days)
    return values[0] * np.sqrt(scaled) + values[1] * scaled
