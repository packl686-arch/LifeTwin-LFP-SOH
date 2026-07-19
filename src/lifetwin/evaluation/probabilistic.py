from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class QuantileEvaluation:
    quantile: float
    pinball_loss: float
    empirical_cdf: float
    calibration_error: float
    empirical_cdf_ci_lower: float
    empirical_cdf_ci_upper: float
    sample_count: int
    censored_excluded: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class IntervalEvaluation:
    nominal_coverage: float
    empirical_coverage: float
    coverage_error: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    mean_width: float
    mean_relative_width: float
    sample_count: int
    censored_excluded: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class GroupIntervalEvaluation:
    nominal_coverage: float
    simultaneous_group_coverage: float
    coverage_error: float
    coverage_ci_lower: float
    coverage_ci_upper: float
    group_count: int
    sample_count: int
    censored_excluded: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


class ConformalCalibrationError(ValueError):
    """Raised when a finite-sample conformal radius is not identifiable."""


def wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z_score: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Expected 0 <= successes <= trials and trials >= 1")
    proportion = successes / trials
    z_squared = z_score**2
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    half_width = (
        z_score
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z_squared / (4 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _one_dimensional_finite(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _observed_mask(
    is_censored: Sequence[bool] | np.ndarray | None,
    row_count: int,
) -> tuple[np.ndarray, int]:
    if is_censored is None:
        return np.ones(row_count, dtype=bool), 0
    censoring = np.asarray(is_censored)
    if censoring.ndim != 1 or len(censoring) != row_count:
        raise ValueError("is_censored must be one-dimensional and match rows")
    if censoring.dtype != np.bool_:
        raise ValueError("is_censored must contain boolean values")
    observed = ~censoring
    excluded = int(censoring.sum())
    if not observed.any():
        raise ValueError("At least one uncensored event is required for this metric")
    return observed, excluded


def evaluate_quantile(
    y_true: Sequence[float] | np.ndarray,
    y_quantile: Sequence[float] | np.ndarray,
    quantile: float,
    *,
    is_censored: Sequence[bool] | np.ndarray | None = None,
) -> QuantileEvaluation:
    """Evaluate a lifetime quantile on observed EOL events only.

    This deliberately excludes right-censored rows instead of presenting a naive
    score as an IPCW-corrected survival metric.
    """
    truth = _one_dimensional_finite(y_true, name="y_true")
    prediction = _one_dimensional_finite(y_quantile, name="y_quantile")
    if len(prediction) != len(truth):
        raise ValueError("y_true and y_quantile must have equal length")
    if (truth <= 0).any() or (prediction <= 0).any():
        raise ValueError("Lifetime values and predictions must be positive")
    if not math.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    observed, excluded = _observed_mask(is_censored, len(truth))
    error = truth[observed] - prediction[observed]
    pinball = np.maximum(quantile * error, (quantile - 1) * error)
    below = truth[observed] <= prediction[observed]
    empirical_cdf = float(np.mean(below))
    cdf_lower, cdf_upper = wilson_score_interval(int(below.sum()), len(below))
    return QuantileEvaluation(
        quantile=float(quantile),
        pinball_loss=float(pinball.mean()),
        empirical_cdf=empirical_cdf,
        calibration_error=empirical_cdf - float(quantile),
        empirical_cdf_ci_lower=cdf_lower,
        empirical_cdf_ci_upper=cdf_upper,
        sample_count=int(observed.sum()),
        censored_excluded=excluded,
    )


def evaluate_prediction_interval(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    nominal_coverage: float,
    is_censored: Sequence[bool] | np.ndarray | None = None,
) -> IntervalEvaluation:
    truth = _one_dimensional_finite(y_true, name="y_true")
    lower_bound = _one_dimensional_finite(lower, name="lower")
    upper_bound = _one_dimensional_finite(upper, name="upper")
    if len(lower_bound) != len(truth) or len(upper_bound) != len(truth):
        raise ValueError("Truth and interval bounds must have equal length")
    if (truth <= 0).any() or (lower_bound <= 0).any() or (upper_bound <= 0).any():
        raise ValueError("Lifetime values and interval bounds must be positive")
    if (lower_bound > upper_bound).any():
        raise ValueError("lower cannot exceed upper")
    if not math.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must lie strictly between zero and one")
    observed, excluded = _observed_mask(is_censored, len(truth))
    covered = (truth[observed] >= lower_bound[observed]) & (
        truth[observed] <= upper_bound[observed]
    )
    width = upper_bound[observed] - lower_bound[observed]
    empirical = float(covered.mean())
    coverage_lower, coverage_upper = wilson_score_interval(
        int(covered.sum()), len(covered)
    )
    return IntervalEvaluation(
        nominal_coverage=float(nominal_coverage),
        empirical_coverage=empirical,
        coverage_error=empirical - float(nominal_coverage),
        coverage_ci_lower=coverage_lower,
        coverage_ci_upper=coverage_upper,
        mean_width=float(width.mean()),
        mean_relative_width=float(np.mean(width / truth[observed])),
        sample_count=int(observed.sum()),
        censored_excluded=excluded,
    )


def conformal_log_radius(
    y_true: Sequence[float] | np.ndarray,
    median_prediction: Sequence[float] | np.ndarray,
    *,
    coverage: float = 0.8,
    is_censored: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Finite-sample split-conformal radius for a symmetric log-life interval."""
    truth = _one_dimensional_finite(y_true, name="y_true")
    median = _one_dimensional_finite(median_prediction, name="median_prediction")
    if len(median) != len(truth):
        raise ValueError("y_true and median_prediction must have equal length")
    if (truth <= 0).any() or (median <= 0).any():
        raise ValueError("Lifetime values and predictions must be positive")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    observed, _ = _observed_mask(is_censored, len(truth))
    scores = np.abs(np.log(truth[observed]) - np.log(median[observed]))
    return _finite_sample_radius(scores, coverage)


def _finite_sample_radius(scores: np.ndarray, coverage: float) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    if ordered.ndim != 1 or len(ordered) == 0 or not np.isfinite(ordered).all():
        raise ValueError("Conformal scores must be a non-empty finite vector")
    rank = math.ceil((len(ordered) + 1) * coverage)
    if rank > len(ordered):
        raise ConformalCalibrationError(
            f"{len(ordered)} calibration units are insufficient for finite "
            f"coverage={coverage}; the conformal quantile would be infinite"
        )
    return float(ordered[rank - 1])


def _group_vector(
    values: Sequence[object] | np.ndarray,
    row_count: int,
) -> np.ndarray:
    groups = np.asarray(values, dtype=object)
    if groups.ndim != 1 or len(groups) != row_count:
        raise ValueError("group_ids must be one-dimensional and match rows")
    if any(value is None or value != value for value in groups):
        raise ValueError("group_ids cannot contain null values")
    return groups


def group_max_conformal_log_radius(
    y_true: Sequence[float] | np.ndarray,
    median_prediction: Sequence[float] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
    *,
    coverage: float = 0.8,
    is_censored: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Calibrate on per-group maximum log residuals for new-group coverage."""
    truth = _one_dimensional_finite(y_true, name="y_true")
    median = _one_dimensional_finite(median_prediction, name="median_prediction")
    if len(median) != len(truth):
        raise ValueError("y_true and median_prediction must have equal length")
    if (truth <= 0).any() or (median <= 0).any():
        raise ValueError("Lifetime values and predictions must be positive")
    if not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must lie strictly between zero and one")
    groups = _group_vector(group_ids, len(truth))
    observed, _ = _observed_mask(is_censored, len(truth))
    row_scores = np.abs(np.log(truth) - np.log(median))
    maximum_by_group: dict[object, float] = {}
    for group, score, include in zip(groups, row_scores, observed, strict=True):
        if include:
            maximum_by_group[group] = max(maximum_by_group.get(group, 0.0), score)
    return _finite_sample_radius(
        np.asarray(list(maximum_by_group.values()), dtype=float),
        coverage,
    )


def evaluate_group_prediction_interval(
    y_true: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    group_ids: Sequence[object] | np.ndarray,
    *,
    nominal_coverage: float,
    is_censored: Sequence[bool] | np.ndarray | None = None,
) -> GroupIntervalEvaluation:
    """Evaluate whether every observed row in each group is covered."""
    truth = _one_dimensional_finite(y_true, name="y_true")
    lower_bound = _one_dimensional_finite(lower, name="lower")
    upper_bound = _one_dimensional_finite(upper, name="upper")
    if len(lower_bound) != len(truth) or len(upper_bound) != len(truth):
        raise ValueError("Truth and interval bounds must have equal length")
    if (truth <= 0).any() or (lower_bound <= 0).any() or (upper_bound <= 0).any():
        raise ValueError("Lifetime values and interval bounds must be positive")
    if (lower_bound > upper_bound).any():
        raise ValueError("lower cannot exceed upper")
    if not math.isfinite(nominal_coverage) or not 0 < nominal_coverage < 1:
        raise ValueError("nominal_coverage must lie strictly between zero and one")
    groups = _group_vector(group_ids, len(truth))
    observed, excluded = _observed_mask(is_censored, len(truth))
    row_covered = (truth >= lower_bound) & (truth <= upper_bound)
    covered_by_group: dict[object, bool] = {}
    for group, covered, include in zip(groups, row_covered, observed, strict=True):
        if include:
            covered_by_group[group] = covered_by_group.get(group, True) and bool(
                covered
            )
    simultaneous = float(np.mean(list(covered_by_group.values())))
    successes = sum(covered_by_group.values())
    coverage_lower, coverage_upper = wilson_score_interval(
        successes, len(covered_by_group)
    )
    return GroupIntervalEvaluation(
        nominal_coverage=float(nominal_coverage),
        simultaneous_group_coverage=simultaneous,
        coverage_error=simultaneous - float(nominal_coverage),
        coverage_ci_lower=coverage_lower,
        coverage_ci_upper=coverage_upper,
        group_count=len(covered_by_group),
        sample_count=int(observed.sum()),
        censored_excluded=excluded,
    )


def log_symmetric_interval(
    median_prediction: Sequence[float] | np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    median = _one_dimensional_finite(
        median_prediction, name="median_prediction"
    )
    if (median <= 0).any():
        raise ValueError("median_prediction must be positive")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and non-negative")
    return median * math.exp(-radius), median * math.exp(radius)
