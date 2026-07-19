from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import norm

from lifetwin.models.calendar_v2 import PowerLawFit


LAPLACE_INTERVAL_METHOD = "laplace_gaussian_pointwise_v1"
PREFIX_CONFORMAL_INTERVAL_METHOD = (
    "training_prefix_condition_max_conformal_v1"
)
INTERVAL_METHODS = (
    LAPLACE_INTERVAL_METHOD,
    PREFIX_CONFORMAL_INTERVAL_METHOD,
)


@dataclass(frozen=True)
class FiniteSampleQuantile:
    requested_coverage: float
    calibration_count: int
    order_statistic_rank: int
    multiplier: float | None
    status: str

    @property
    def available(self) -> bool:
        return self.multiplier is not None


def power_law_predictive_sd(
    fitted: PowerLawFit,
    elapsed_days: np.ndarray | list[float],
    *,
    observation_scale_pp: float,
    scale_floor_pp: float,
) -> np.ndarray:
    """Delta-method predictive SD for a condition-mean power-law trajectory."""
    elapsed = np.asarray(elapsed_days, dtype=float)
    if (
        elapsed.ndim != 1
        or elapsed.size == 0
        or np.any(elapsed <= 0.0)
        or not np.isfinite(elapsed).all()
    ):
        raise ValueError("Predictive-scale times must be finite and positive")
    if observation_scale_pp <= 0.0 or scale_floor_pp <= 0.0:
        raise ValueError("Predictive-scale floors must be positive")
    if fitted.parameter_covariance is None:
        raise ValueError("Power-law posterior covariance is required")
    covariance = np.asarray(fitted.parameter_covariance, dtype=float)
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("Power-law posterior covariance must be finite 2x2")
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
        raise ValueError("Power-law posterior covariance must be positive semidefinite")

    predicted_loss = np.exp(fitted.log_amplitude) * np.power(
        elapsed,
        fitted.time_exponent,
    )
    gradient = np.column_stack(
        [predicted_loss, predicted_loss * np.log(elapsed)]
    )
    parameter_variance = np.einsum(
        "ij,jk,ik->i",
        gradient,
        covariance,
        gradient,
    )
    total_variance = (
        np.maximum(parameter_variance, 0.0) + float(observation_scale_pp) ** 2
    )
    return np.maximum(np.sqrt(total_variance), float(scale_floor_pp))


def gaussian_central_multiplier(coverage: float) -> float:
    if not 0.0 < coverage < 1.0:
        raise ValueError("Interval coverage must lie strictly between zero and one")
    return float(norm.ppf(0.5 * (1.0 + float(coverage))))


def finite_sample_higher_quantile(
    scores: np.ndarray | list[float],
    *,
    coverage: float,
) -> FiniteSampleQuantile:
    """Split-conformal order statistic, including the honest infinity case."""
    values = np.asarray(scores, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or np.any(values < 0.0)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Calibration scores must be finite non-negative values")
    if not 0.0 < coverage < 1.0:
        raise ValueError("Interval coverage must lie strictly between zero and one")
    count = int(values.size)
    rank = int(math.ceil((count + 1) * float(coverage)))
    if rank > count:
        return FiniteSampleQuantile(
            requested_coverage=float(coverage),
            calibration_count=count,
            order_statistic_rank=rank,
            multiplier=None,
            status="unavailable_finite_sample_full_physical_range",
        )
    ordered = np.sort(values, kind="stable")
    return FiniteSampleQuantile(
        requested_coverage=float(coverage),
        calibration_count=count,
        order_statistic_rank=rank,
        multiplier=float(ordered[rank - 1]),
        status="finite_order_statistic",
    )


def physical_interval(
    predicted_retention_pct: np.ndarray | list[float],
    predictive_sd_pp: np.ndarray | list[float],
    *,
    multiplier: float | None,
    physical_bounds_pct: tuple[float, float] = (0.0, 100.0),
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_retention_pct, dtype=float)
    scale = np.asarray(predictive_sd_pp, dtype=float)
    if (
        predicted.shape != scale.shape
        or predicted.ndim != 1
        or predicted.size == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError("Interval predictions and scales must be aligned and finite")
    lower_bound, upper_bound = map(float, physical_bounds_pct)
    if not lower_bound < upper_bound:
        raise ValueError("Physical interval bounds must be ordered")
    if multiplier is None:
        return (
            np.full_like(predicted, lower_bound),
            np.full_like(predicted, upper_bound),
        )
    if multiplier < 0.0 or not math.isfinite(multiplier):
        raise ValueError("Interval multiplier must be finite and non-negative")
    radius = float(multiplier) * scale
    return (
        np.clip(predicted - radius, lower_bound, upper_bound),
        np.clip(predicted + radius, lower_bound, upper_bound),
    )


def interval_score(
    observed: np.ndarray | list[float],
    lower: np.ndarray | list[float],
    upper: np.ndarray | list[float],
    *,
    coverage: float,
) -> np.ndarray:
    truth = np.asarray(observed, dtype=float)
    low = np.asarray(lower, dtype=float)
    high = np.asarray(upper, dtype=float)
    if (
        truth.shape != low.shape
        or truth.shape != high.shape
        or truth.ndim != 1
        or not np.isfinite(truth).all()
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or np.any(low > high)
    ):
        raise ValueError("Interval-score inputs must be aligned, finite, and ordered")
    if not 0.0 < coverage < 1.0:
        raise ValueError("Interval coverage must lie strictly between zero and one")
    alpha = 1.0 - float(coverage)
    width = high - low
    below = (truth < low).astype(float)
    above = (truth > high).astype(float)
    return (
        width
        + (2.0 / alpha) * (low - truth) * below
        + (2.0 / alpha) * (truth - high) * above
    )
