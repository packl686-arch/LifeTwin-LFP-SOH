from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from lifetwin.models.calendar_v2 import (
    fit_power_law,
    stress_features_for_condition,
)


TARGET_ACTIVATION_METHOD = "target_prefix_activation_offset_power_tau7_v1"
HIERARCHICAL_ACTIVATION_METHOD = (
    "hierarchical_activation_offset_power_tau7_v3"
)
GATED_TARGET_ACTIVATION_METHOD = (
    "mechanism_gated_target_activation_offset_hybrid_v1"
)
GATED_HIERARCHICAL_ACTIVATION_METHOD = (
    "mechanism_gated_hierarchical_activation_offset_hybrid_v1"
)

ACTIVATION_PARAMETER_NAMES = (
    "log_amplitude",
    "time_exponent",
    "activation_offset_pp",
)


@dataclass(frozen=True)
class ActivationOffsetFit:
    log_amplitude: float
    time_exponent: float
    activation_offset_pp: float
    activation_timescale_days: float
    optimizer_cost: float
    optimizer_evaluations: int
    parameter_covariance: tuple[tuple[float, ...], ...] | None

    def parameter_map(self) -> dict[str, float]:
        return {
            "log_amplitude": self.log_amplitude,
            "time_exponent": self.time_exponent,
            "activation_offset_pp": self.activation_offset_pp,
            "activation_timescale_days": self.activation_timescale_days,
        }


@dataclass(frozen=True)
class ActivationOffsetPrior:
    surface_coefficients: tuple[tuple[float, ...], ...]
    parameter_scales: tuple[float, float, float]
    observation_scale_pp: float
    activation_timescale_days: float
    training_condition_ids: tuple[str, ...]
    training_observation_count: int
    maximum_training_days: float
    condition_parameters: tuple[tuple[str, float, float, float], ...]

    def prior_mean(self, frame: pd.DataFrame) -> np.ndarray:
        coefficients = np.asarray(self.surface_coefficients, dtype=float)
        return coefficients @ stress_features_for_condition(frame)


@dataclass(frozen=True)
class ActivationGate:
    ready: bool
    negative_loss_evidence: bool
    positive_time_observation_count: int
    minimum_capacity_loss_pct: float
    minimum_positive_time_observations: int
    negative_loss_threshold_pp: float


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
        raise ValueError(f"Missing Calendar V3 activation columns: {missing}")
    if frame.empty:
        raise ValueError("Calendar V3 activation frame cannot be empty")
    ordered = frame.copy()
    numeric = [
        "temperature_c",
        "storage_soc_fraction",
        "elapsed_days",
        "capacity_loss_pct",
    ]
    ordered[numeric] = ordered[numeric].apply(pd.to_numeric, errors="coerce")
    if ordered[numeric].isna().any().any():
        raise ValueError("Calendar V3 activation values must be numeric")
    if not np.isfinite(ordered[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Calendar V3 activation values must be finite")
    if (ordered["elapsed_days"] < 0.0).any():
        raise ValueError("Elapsed days cannot be negative")
    if ordered["condition_id"].astype(str).nunique() < minimum_conditions:
        raise ValueError(
            f"At least {minimum_conditions} independent conditions are required"
        )
    return ordered.sort_values(
        ["condition_id", "elapsed_days"], kind="stable"
    ).reset_index(drop=True)


def _positive_time(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame.loc[frame["elapsed_days"] > 0.0].copy()
    if len(positive) < 3:
        raise ValueError(
            "At least three positive-time observations are required for activation fit"
        )
    return positive


def activation_basis(
    elapsed_days: np.ndarray | pd.Series | list[float],
    *,
    timescale_days: float,
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    if (
        timescale_days <= 0.0
        or np.any(elapsed < 0.0)
        or not np.isfinite(elapsed).all()
    ):
        raise ValueError("Activation basis inputs must be finite and non-negative")
    return 1.0 - np.exp(-elapsed / float(timescale_days))


def predict_activation_offset_loss(
    fitted: ActivationOffsetFit,
    elapsed_days: np.ndarray | pd.Series | list[float],
) -> np.ndarray:
    elapsed = np.asarray(elapsed_days, dtype=float)
    if np.any(elapsed < 0.0) or not np.isfinite(elapsed).all():
        raise ValueError("Activation prediction times must be finite and non-negative")
    irreversible = np.exp(fitted.log_amplitude) * np.power(
        elapsed, fitted.time_exponent
    )
    activation = fitted.activation_offset_pp * activation_basis(
        elapsed, timescale_days=fitted.activation_timescale_days
    )
    return irreversible - activation


def fit_activation_offset_power_law(
    frame: pd.DataFrame,
    *,
    activation_timescale_days: float = 7.0,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
    activation_offset_bounds_pp: tuple[float, float] = (0.0, 10.0),
    robust_loss_scale_pp: float = 0.25,
) -> ActivationOffsetFit:
    ordered = _validate_frame(frame)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("An activation-offset fit requires exactly one condition")
    positive = _positive_time(ordered)
    elapsed = positive["elapsed_days"].to_numpy(dtype=float)
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    lower_exponent, upper_exponent = map(float, exponent_bounds)
    lower_offset, upper_offset = map(float, activation_offset_bounds_pp)
    if (
        not 0.0 < lower_exponent < upper_exponent
        or lower_offset < 0.0
        or lower_offset >= upper_offset
        or activation_timescale_days <= 0.0
        or robust_loss_scale_pp <= 0.0
    ):
        raise ValueError("Activation-offset model bounds are invalid")
    base = fit_power_law(
        ordered,
        exponent_bounds=exponent_bounds,
        robust_loss_scale_pp=robust_loss_scale_pp,
    )
    initial_offset = float(
        np.clip(
            max(-float(positive["capacity_loss_pct"].min()), 0.0),
            lower_offset,
            upper_offset,
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        irreversible = np.exp(parameters[0]) * np.power(elapsed, parameters[1])
        activation = parameters[2] * activation_basis(
            elapsed, timescale_days=activation_timescale_days
        )
        return irreversible - activation - observed

    fitted = least_squares(
        residual,
        np.asarray(
            [base.log_amplitude, base.time_exponent, initial_offset], dtype=float
        ),
        bounds=(
            np.asarray([-12.0, lower_exponent, lower_offset]),
            np.asarray([5.0, upper_exponent, upper_offset]),
        ),
        loss="soft_l1",
        f_scale=float(robust_loss_scale_pp),
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Activation-offset fit failed: {fitted.message}")
    return ActivationOffsetFit(
        log_amplitude=float(fitted.x[0]),
        time_exponent=float(fitted.x[1]),
        activation_offset_pp=float(fitted.x[2]),
        activation_timescale_days=float(activation_timescale_days),
        optimizer_cost=float(fitted.cost),
        optimizer_evaluations=int(fitted.nfev),
        parameter_covariance=None,
    )


def fit_hierarchical_activation_offset_prior(
    training_frame: pd.DataFrame,
    *,
    activation_timescale_days: float = 7.0,
    minimum_conditions: int = 6,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
    activation_offset_bounds_pp: tuple[float, float] = (0.0, 10.0),
    robust_loss_scale_pp: float = 0.25,
    stress_surface_ridge: float = 1.0,
    parameter_scale_floors: tuple[float, float, float] = (0.1, 0.05, 0.1),
    observation_scale_floor_pp: float = 0.1,
) -> ActivationOffsetPrior:
    ordered = _validate_frame(
        training_frame, minimum_conditions=minimum_conditions
    )
    scale_floors = np.asarray(parameter_scale_floors, dtype=float)
    if (
        stress_surface_ridge < 0.0
        or scale_floors.shape != (3,)
        or np.any(scale_floors <= 0.0)
        or observation_scale_floor_pp <= 0.0
    ):
        raise ValueError("Activation hierarchy scales are invalid")
    feature_rows: list[np.ndarray] = []
    parameter_rows: list[np.ndarray] = []
    residuals: list[float] = []
    condition_parameters: list[tuple[str, float, float, float]] = []
    for condition_id, condition in ordered.groupby("condition_id", sort=True):
        fitted = fit_activation_offset_power_law(
            condition,
            activation_timescale_days=activation_timescale_days,
            exponent_bounds=exponent_bounds,
            activation_offset_bounds_pp=activation_offset_bounds_pp,
            robust_loss_scale_pp=robust_loss_scale_pp,
        )
        parameters = np.asarray(
            [
                fitted.log_amplitude,
                fitted.time_exponent,
                fitted.activation_offset_pp,
            ],
            dtype=float,
        )
        feature_rows.append(stress_features_for_condition(condition))
        parameter_rows.append(parameters)
        condition_parameters.append(
            (
                str(condition_id),
                fitted.log_amplitude,
                fitted.time_exponent,
                fitted.activation_offset_pp,
            )
        )
        positive = _positive_time(condition)
        predicted = predict_activation_offset_loss(
            fitted, positive["elapsed_days"].to_numpy(dtype=float)
        )
        residuals.extend(
            (
                predicted
                - positive["capacity_loss_pct"].to_numpy(dtype=float)
            ).tolist()
        )
    design = np.vstack(feature_rows)
    parameters = np.vstack(parameter_rows)
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    system = design.T @ design + float(stress_surface_ridge) * penalty
    coefficients = np.linalg.solve(system, design.T @ parameters)
    parameter_residuals = parameters - design @ coefficients
    parameter_scales = np.maximum(
        np.std(parameter_residuals, axis=0, ddof=1), scale_floors
    )
    observation_scale = max(
        float(np.sqrt(np.mean(np.square(residuals)))),
        float(observation_scale_floor_pp),
    )
    return ActivationOffsetPrior(
        surface_coefficients=tuple(
            tuple(float(value) for value in coefficients[:, index])
            for index in range(3)
        ),
        parameter_scales=tuple(float(value) for value in parameter_scales),
        observation_scale_pp=observation_scale,
        activation_timescale_days=float(activation_timescale_days),
        training_condition_ids=tuple(
            sorted(ordered["condition_id"].astype(str).unique())
        ),
        training_observation_count=len(ordered),
        maximum_training_days=float(ordered["elapsed_days"].max()),
        condition_parameters=tuple(condition_parameters),
    )


def update_hierarchical_activation_offset(
    prior: ActivationOffsetPrior,
    target_prefix: pd.DataFrame,
    *,
    exponent_bounds: tuple[float, float] = (0.05, 1.5),
    activation_offset_bounds_pp: tuple[float, float] = (0.0, 10.0),
) -> ActivationOffsetFit:
    ordered = _validate_frame(target_prefix)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("An activation hierarchy update requires one condition")
    positive = _positive_time(ordered)
    elapsed = positive["elapsed_days"].to_numpy(dtype=float)
    observed = positive["capacity_loss_pct"].to_numpy(dtype=float)
    lower_exponent, upper_exponent = map(float, exponent_bounds)
    lower_offset, upper_offset = map(float, activation_offset_bounds_pp)
    prior_mean = prior.prior_mean(ordered)
    prior_mean = np.clip(
        prior_mean,
        np.asarray([-12.0, lower_exponent, lower_offset]),
        np.asarray([5.0, upper_exponent, upper_offset]),
    )
    prior_scales = np.asarray(prior.parameter_scales, dtype=float)

    def posterior_residual(parameters: np.ndarray) -> np.ndarray:
        irreversible = np.exp(parameters[0]) * np.power(elapsed, parameters[1])
        activation = parameters[2] * activation_basis(
            elapsed, timescale_days=prior.activation_timescale_days
        )
        data_residual = (
            irreversible - activation - observed
        ) / prior.observation_scale_pp
        prior_residual = (parameters - prior_mean) / prior_scales
        return np.concatenate([data_residual, prior_residual])

    fitted = least_squares(
        posterior_residual,
        prior_mean,
        bounds=(
            np.asarray([-12.0, lower_exponent, lower_offset]),
            np.asarray([5.0, upper_exponent, upper_offset]),
        ),
        loss="linear",
        max_nfev=5000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"Activation hierarchy update failed: {fitted.message}")
    covariance = np.linalg.pinv(fitted.jac.T @ fitted.jac, hermitian=True)
    if not np.isfinite(covariance).all():
        raise RuntimeError("Activation hierarchy covariance is not finite")
    return ActivationOffsetFit(
        log_amplitude=float(fitted.x[0]),
        time_exponent=float(fitted.x[1]),
        activation_offset_pp=float(fitted.x[2]),
        activation_timescale_days=prior.activation_timescale_days,
        optimizer_cost=float(fitted.cost),
        optimizer_evaluations=int(fitted.nfev),
        parameter_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
    )


def activation_mechanism_gate(
    target_prefix: pd.DataFrame,
    *,
    minimum_positive_time_observations: int = 7,
    negative_loss_threshold_pp: float = 0.0,
) -> ActivationGate:
    ordered = _validate_frame(target_prefix)
    if ordered["condition_id"].astype(str).nunique() != 1:
        raise ValueError("Activation mechanism gate requires exactly one condition")
    if minimum_positive_time_observations < 3:
        raise ValueError("Activation gate requires at least three positive-time points")
    if negative_loss_threshold_pp < 0.0:
        raise ValueError("Negative-loss threshold is expressed as a non-negative margin")
    positive = ordered.loc[ordered["elapsed_days"] > 0.0]
    if positive.empty:
        raise ValueError(
            "Activation mechanism gate requires at least one positive-time point"
        )
    minimum_loss = float(positive["capacity_loss_pct"].min())
    negative_evidence = minimum_loss < -float(negative_loss_threshold_pp)
    ready = bool(
        negative_evidence and len(positive) >= minimum_positive_time_observations
    )
    return ActivationGate(
        ready=ready,
        negative_loss_evidence=bool(negative_evidence),
        positive_time_observation_count=len(positive),
        minimum_capacity_loss_pct=minimum_loss,
        minimum_positive_time_observations=int(
            minimum_positive_time_observations
        ),
        negative_loss_threshold_pp=float(negative_loss_threshold_pp),
    )
