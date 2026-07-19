from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm


@dataclass(frozen=True)
class AFTFitSummary:
    success: bool
    iterations: int
    objective: float
    event_count: int
    censored_count: int

    def to_dict(self) -> dict[str, bool | int | float]:
        return asdict(self)


def _as_feature_matrix(values: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Features must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Features must contain only finite values")
    return matrix


def _as_durations(values: Sequence[float] | np.ndarray, row_count: int) -> np.ndarray:
    durations = np.asarray(values, dtype=float)
    if durations.ndim != 1 or len(durations) != row_count:
        raise ValueError("Durations must be one-dimensional and match feature rows")
    if not np.isfinite(durations).all() or (durations <= 0).any():
        raise ValueError("Durations must contain finite positive values")
    return durations


def _as_censoring(
    values: Sequence[bool] | np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    if values is None:
        return np.zeros(row_count, dtype=bool)
    censoring = np.asarray(values)
    if censoring.ndim != 1 or len(censoring) != row_count:
        raise ValueError("is_censored must be one-dimensional and match feature rows")
    if censoring.dtype != np.bool_:
        raise ValueError("is_censored must contain boolean values")
    return censoring.astype(bool, copy=False)


class LogNormalAFT:
    """Log-normal accelerated failure-time regression with right censoring."""

    def __init__(
        self,
        *,
        l2_penalty: float = 1e-4,
        max_iterations: int = 5000,
        minimum_sigma: float = 1e-3,
        maximum_sigma: float = 10.0,
    ) -> None:
        if not math.isfinite(l2_penalty) or l2_penalty < 0:
            raise ValueError("l2_penalty must be finite and non-negative")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if (
            not math.isfinite(minimum_sigma)
            or not math.isfinite(maximum_sigma)
            or not 0 < minimum_sigma < maximum_sigma
        ):
            raise ValueError("Expected 0 < minimum_sigma < maximum_sigma")
        self.l2_penalty = float(l2_penalty)
        self.max_iterations = int(max_iterations)
        self.minimum_sigma = float(minimum_sigma)
        self.maximum_sigma = float(maximum_sigma)

    def fit(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        durations: Sequence[float] | np.ndarray,
        *,
        is_censored: Sequence[bool] | np.ndarray | None = None,
    ) -> LogNormalAFT:
        matrix = _as_feature_matrix(features)
        time = _as_durations(durations, len(matrix))
        censored = _as_censoring(is_censored, len(matrix))
        observed = ~censored
        if observed.sum() < 2:
            raise ValueError("At least two observed EOL events are required")

        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (matrix - mean) / scale
        design = np.column_stack((np.ones(len(matrix)), standardized))
        log_time = np.log(time)

        initial_beta, *_ = np.linalg.lstsq(
            design[observed], log_time[observed], rcond=None
        )
        residual = log_time[observed] - design[observed] @ initial_beta
        initial_sigma = float(np.std(residual, ddof=0))
        if not math.isfinite(initial_sigma) or initial_sigma < 0.05:
            initial_sigma = 0.25
        initial_sigma = float(
            np.clip(initial_sigma, self.minimum_sigma, self.maximum_sigma)
        )
        initial = np.concatenate((initial_beta, [math.log(initial_sigma)]))
        log_sigma_bounds = (
            math.log(self.minimum_sigma),
            math.log(self.maximum_sigma),
        )

        def objective_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            beta = parameters[:-1]
            log_sigma = float(parameters[-1])
            sigma = math.exp(log_sigma)
            location = design @ beta
            z_score = (log_time - location) / sigma

            row_loss = np.empty(len(matrix), dtype=float)
            location_gradient = np.empty(len(matrix), dtype=float)
            sigma_gradient = np.empty(len(matrix), dtype=float)

            observed_z = z_score[observed]
            row_loss[observed] = (
                log_time[observed]
                + log_sigma
                + 0.5 * observed_z**2
                + 0.5 * math.log(2 * math.pi)
            )
            location_gradient[observed] = -observed_z / sigma
            sigma_gradient[observed] = 1.0 - observed_z**2

            if censored.any():
                censored_z = z_score[censored]
                log_survival = log_ndtr(-censored_z)
                row_loss[censored] = -log_survival
                log_density = -0.5 * censored_z**2 - 0.5 * math.log(2 * math.pi)
                inverse_mills = np.exp(log_density - log_survival)
                location_gradient[censored] = -inverse_mills / sigma
                sigma_gradient[censored] = -inverse_mills * censored_z

            penalty = 0.5 * self.l2_penalty * float(np.dot(beta[1:], beta[1:]))
            objective = float(row_loss.sum() + penalty)
            beta_gradient = design.T @ location_gradient
            beta_gradient[1:] += self.l2_penalty * beta[1:]
            gradient = np.concatenate((beta_gradient, [sigma_gradient.sum()]))
            if not math.isfinite(objective) or not np.isfinite(gradient).all():
                return 1e100, np.zeros_like(parameters)
            return objective, gradient

        result = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(None, None)] * design.shape[1] + [log_sigma_bounds],
            options={
                "maxiter": self.max_iterations,
                "maxls": 100,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Log-normal AFT optimization failed: {result.message}")

        scaled_beta = result.x[:-1]
        self.n_features_in_ = matrix.shape[1]
        self.feature_mean_ = mean
        self.feature_scale_ = scale
        self.scaled_coef_ = scaled_beta[1:].copy()
        self.scaled_intercept_ = float(scaled_beta[0])
        self.coef_ = self.scaled_coef_ / scale
        self.intercept_ = float(
            self.scaled_intercept_ - np.dot(mean / scale, self.scaled_coef_)
        )
        self.sigma_ = float(math.exp(result.x[-1]))
        self.fit_summary_ = AFTFitSummary(
            success=True,
            iterations=int(result.nit),
            objective=float(result.fun),
            event_count=int(observed.sum()),
            censored_count=int(censored.sum()),
        )
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("LogNormalAFT must be fitted before prediction")

    def predict_log_location(
        self, features: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        self._check_fitted()
        matrix = _as_feature_matrix(features)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, found {matrix.shape[1]}"
            )
        return self.intercept_ + matrix @ self.coef_

    @staticmethod
    def _exp_lifetime(log_lifetime: np.ndarray) -> np.ndarray:
        if (log_lifetime > math.log(np.finfo(float).max)).any():
            raise OverflowError("Predicted lifetime exceeds floating-point range")
        return np.exp(log_lifetime)

    def predict_median(
        self, features: Sequence[Sequence[float]] | np.ndarray
    ) -> np.ndarray:
        return self._exp_lifetime(self.predict_log_location(features))

    def predict_quantile(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        quantile: float | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        location = self.predict_log_location(features)
        quantiles = np.asarray(quantile, dtype=float)
        if quantiles.ndim > 1 or not np.isfinite(quantiles).all():
            raise ValueError("quantile must be a finite scalar or one-dimensional array")
        if ((quantiles <= 0) | (quantiles >= 1)).any():
            raise ValueError("quantile values must lie strictly between zero and one")
        z_score = norm.ppf(quantiles)
        if quantiles.ndim == 0:
            return self._exp_lifetime(location + self.sigma_ * float(z_score))
        return self._exp_lifetime(
            location[:, np.newaxis] + self.sigma_ * z_score[np.newaxis, :]
        )

    def survival_probability(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        time: float | Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        location = self.predict_log_location(features)
        evaluation_time = np.asarray(time, dtype=float)
        if evaluation_time.ndim == 0:
            evaluation_time = np.full(len(location), float(evaluation_time))
        if evaluation_time.ndim != 1 or len(evaluation_time) != len(location):
            raise ValueError("time must be scalar or one-dimensional and match rows")
        if not np.isfinite(evaluation_time).all() or (evaluation_time <= 0).any():
            raise ValueError("time must contain finite positive values")
        z_score = (np.log(evaluation_time) - location) / self.sigma_
        return ndtr(-z_score)

    def negative_log_likelihood(
        self,
        features: Sequence[Sequence[float]] | np.ndarray,
        durations: Sequence[float] | np.ndarray,
        *,
        is_censored: Sequence[bool] | np.ndarray | None = None,
        average: bool = True,
    ) -> float:
        location = self.predict_log_location(features)
        time = _as_durations(durations, len(location))
        censored = _as_censoring(is_censored, len(location))
        observed = ~censored
        log_time = np.log(time)
        z_score = (log_time - location) / self.sigma_
        loss = np.empty(len(location), dtype=float)
        loss[observed] = (
            log_time[observed]
            + math.log(self.sigma_)
            + 0.5 * z_score[observed] ** 2
            + 0.5 * math.log(2 * math.pi)
        )
        loss[censored] = -log_ndtr(-z_score[censored])
        return float(loss.mean() if average else loss.sum())
