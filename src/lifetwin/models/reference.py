from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
from scipy.stats import norm


def _finite_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


@dataclass(frozen=True)
class GaussianOffsetPosterior:
    """Conjugate posterior for a target-domain log-lifetime intercept."""

    reference_count: int
    residual_sigma: float
    prior_mean: float
    prior_std: float | None
    posterior_mean: float
    posterior_std: float
    update_method: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_normal_prior(
        cls,
        log_residuals: Sequence[float] | np.ndarray,
        *,
        residual_sigma: float,
        prior_std: float,
        prior_mean: float = 0.0,
    ) -> GaussianOffsetPosterior:
        residuals = _finite_vector(log_residuals, name="log_residuals")
        if not math.isfinite(residual_sigma) or residual_sigma <= 0:
            raise ValueError("residual_sigma must be finite and positive")
        if not math.isfinite(prior_std) or prior_std <= 0:
            raise ValueError("prior_std must be finite and positive")
        if not math.isfinite(prior_mean):
            raise ValueError("prior_mean must be finite")
        observation_precision = len(residuals) / residual_sigma**2
        prior_precision = 1 / prior_std**2
        posterior_variance = 1 / (prior_precision + observation_precision)
        posterior_mean = posterior_variance * (
            prior_mean * prior_precision + residuals.sum() / residual_sigma**2
        )
        return cls(
            reference_count=len(residuals),
            residual_sigma=float(residual_sigma),
            prior_mean=float(prior_mean),
            prior_std=float(prior_std),
            posterior_mean=float(posterior_mean),
            posterior_std=float(math.sqrt(posterior_variance)),
            update_method="normal_prior_shrinkage",
        )

    @classmethod
    def from_flat_prior(
        cls,
        log_residuals: Sequence[float] | np.ndarray,
        *,
        residual_sigma: float,
    ) -> GaussianOffsetPosterior:
        residuals = _finite_vector(log_residuals, name="log_residuals")
        if not math.isfinite(residual_sigma) or residual_sigma <= 0:
            raise ValueError("residual_sigma must be finite and positive")
        posterior_std = residual_sigma / math.sqrt(len(residuals))
        return cls(
            reference_count=len(residuals),
            residual_sigma=float(residual_sigma),
            prior_mean=0.0,
            prior_std=None,
            posterior_mean=float(residuals.mean()),
            posterior_std=float(posterior_std),
            update_method="flat_prior_unshrunk_mean",
        )

    @property
    def predictive_sigma(self) -> float:
        return float(math.hypot(self.residual_sigma, self.posterior_std))

    def predict_log_location(
        self, base_log_location: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        base = _finite_vector(base_log_location, name="base_log_location")
        return base + self.posterior_mean

    def predict_quantile(
        self,
        base_log_location: Sequence[float] | np.ndarray,
        quantiles: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        location = self.predict_log_location(base_log_location)
        levels = np.asarray(quantiles, dtype=float)
        if levels.ndim != 1 or len(levels) == 0 or not np.isfinite(levels).all():
            raise ValueError("quantiles must be a non-empty finite vector")
        if ((levels <= 0) | (levels >= 1)).any():
            raise ValueError("quantiles must lie strictly between zero and one")
        return np.exp(
            location[:, np.newaxis]
            + self.predictive_sigma * norm.ppf(levels)[np.newaxis, :]
        )

    def negative_log_likelihood(
        self,
        base_log_location: Sequence[float] | np.ndarray,
        durations: Sequence[float] | np.ndarray,
    ) -> float:
        location = self.predict_log_location(base_log_location)
        time = _finite_vector(durations, name="durations")
        if len(time) != len(location):
            raise ValueError("durations must match prediction rows")
        if (time <= 0).any():
            raise ValueError("durations must be positive")
        log_time = np.log(time)
        sigma = self.predictive_sigma
        z_score = (log_time - location) / sigma
        loss = (
            log_time
            + math.log(sigma)
            + 0.5 * z_score**2
            + 0.5 * math.log(2 * math.pi)
        )
        return float(loss.mean())


def prior_predictive_quantiles(
    base_log_location: Sequence[float] | np.ndarray,
    *,
    residual_sigma: float,
    prior_std: float,
    quantiles: Sequence[float] | np.ndarray,
) -> np.ndarray:
    base = _finite_vector(base_log_location, name="base_log_location")
    if not math.isfinite(residual_sigma) or residual_sigma <= 0:
        raise ValueError("residual_sigma must be finite and positive")
    if not math.isfinite(prior_std) or prior_std <= 0:
        raise ValueError("prior_std must be finite and positive")
    levels = np.asarray(quantiles, dtype=float)
    if levels.ndim != 1 or len(levels) == 0 or not np.isfinite(levels).all():
        raise ValueError("quantiles must be a non-empty finite vector")
    if ((levels <= 0) | (levels >= 1)).any():
        raise ValueError("quantiles must lie strictly between zero and one")
    sigma = math.hypot(residual_sigma, prior_std)
    return np.exp(base[:, np.newaxis] + sigma * norm.ppf(levels)[np.newaxis, :])


def prior_predictive_negative_log_likelihood(
    base_log_location: Sequence[float] | np.ndarray,
    durations: Sequence[float] | np.ndarray,
    *,
    residual_sigma: float,
    prior_std: float,
) -> float:
    base = _finite_vector(base_log_location, name="base_log_location")
    time = _finite_vector(durations, name="durations")
    if len(time) != len(base):
        raise ValueError("durations must match prediction rows")
    if (time <= 0).any():
        raise ValueError("durations must be positive")
    sigma = math.hypot(residual_sigma, prior_std)
    log_time = np.log(time)
    z_score = (log_time - base) / sigma
    loss = (
        log_time
        + math.log(sigma)
        + 0.5 * z_score**2
        + 0.5 * math.log(2 * math.pi)
    )
    return float(loss.mean())
