"""Pure statistical primitives for the frozen V2 long-horizon protocol.

This module deliberately contains no file access, random-number generation, or
formal experiment orchestration.  Its inputs are already-fitted, label-free
candidate summaries or explicit development/calibration arrays.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit
from scipy.stats import beta as beta_distribution
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


FORECAST_DIMENSION = 8
SHAPE_SIGNATURE_STEP_PP = 0.25
PREFIX_DAYS = (
    0.0,
    7.0,
    14.0,
    30.0,
    60.0,
    90.0,
    120.0,
    180.0,
    270.0,
    365.0,
    540.0,
    730.0,
)
FORECAST_DAYS = (
    1095.75,
    1461.0,
    1826.25,
    2556.75,
    3652.5,
    5478.75,
    7305.0,
    9131.25,
)
PREFIX_FEATURE_NAMES = (
    "successful_structure_family_count",
    "fit_failure_count",
    "best_prefix_rmse_pp",
    "unique_shape_25y_q90_minus_q10_pp",
    "mean_over_horizons_unique_shape_iqr_pp",
    "effective_unique_shape_count",
    "parameter_boundary_hit_fraction",
    "leave_day730_out_absolute_prediction_error_pp",
    "center_25y_retention_pct",
    "center_minus_sqrt_25y_pp",
    "observed_q365_minus_q730_pp",
    "observed_q0_minus_q90_pp",
    "slope_180_365_minus_slope_365_730_pp_per_year",
    "nonnegative_25y_minus_10y_unique_shape_q90_q10_growth_pp",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class V2ModelError(ValueError):
    """Raised when an input violates a frozen V2 statistical primitive."""


@dataclass(frozen=True)
class VariantSummary:
    """One already-credible candidate variant.

    ``parameter_values`` and ``parameter_bounds`` contain fitted scalars only.
    Fixed grid coordinates can be present, but must be named in
    ``fixed_parameters`` so they are excluded from the boundary statistic.
    """

    forecast: tuple[float, ...]
    prefix_rmse_pp: float
    parameter_values: tuple[tuple[str, float], ...] = ()
    parameter_bounds: tuple[tuple[str, float, float], ...] = ()
    fixed_parameters: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibraryForecastResult:
    forecast: tuple[float, ...]
    successful_family_count: int
    hard_eligible: bool
    family_representatives: tuple[tuple[str, tuple[float, ...]], ...]
    support_vectors: tuple[tuple[float, ...], ...]
    support_weights: tuple[float, ...]


@dataclass(frozen=True)
class PrefixFeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


@dataclass(frozen=True)
class StandardizerState:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    zero_variance: tuple[bool, ...]

    def transform(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = _finite_matrix(features, columns=len(self.mean), name="features")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        transformed = (matrix - mean) / scale
        transformed[:, np.asarray(self.zero_variance, dtype=bool)] = 0.0
        return transformed


@dataclass(frozen=True)
class LogisticRiskState:
    feature_names: tuple[str, ...]
    standardizer: StandardizerState
    intercept: float
    coefficients: tuple[float, ...]

    def decision_function(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        standardized = self.standardizer.transform(features)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        return np.sum(standardized * coefficients, axis=1) + self.intercept

    def predict_probability(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return expit(self.decision_function(features))


@dataclass(frozen=True)
class IsotonicState:
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]

    def predict(self, raw_scores: Sequence[float]) -> np.ndarray:
        scores = _finite_vector(raw_scores, name="raw_scores")
        return np.interp(
            scores,
            np.asarray(self.x_thresholds, dtype=np.float64),
            np.asarray(self.y_thresholds, dtype=np.float64),
        )


@dataclass(frozen=True)
class ConformalExpansionState:
    coverage: float
    calibration_count: int
    order_statistic_index: int
    expansion_pp: float


@dataclass(frozen=True)
class RankingResult:
    order: tuple[int, ...]
    ranks: tuple[int, ...]
    issued: tuple[bool, ...]


def _finite_vector(
    values: Sequence[float],
    *,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise V2ModelError(f"{name} must be one-dimensional")
    if expected_length is not None and array.size != expected_length:
        raise V2ModelError(f"{name} must contain exactly {expected_length} values")
    if not np.isfinite(array).all():
        raise V2ModelError(f"{name} must contain only finite values")
    return array


def _finite_matrix(
    values: Sequence[Sequence[float]],
    *,
    name: str,
    columns: int | None = None,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise V2ModelError(f"{name} must be two-dimensional")
    if columns is not None and matrix.shape[1] != columns:
        raise V2ModelError(f"{name} must have exactly {columns} columns")
    if not np.isfinite(matrix).all():
        raise V2ModelError(f"{name} must contain only finite values")
    return matrix


def canonical_float64_vector_bytes(vector: Sequence[float]) -> bytes:
    """Return the protocol's canonical big-endian float64 vector bytes."""

    array = _finite_vector(
        vector, name="forecast vector", expected_length=FORECAST_DIMENSION
    )
    return np.asarray(array, dtype=">f8").tobytes(order="C")


def quantized_shape_signature(
    vector: Sequence[float],
) -> tuple[int, ...]:
    """Return the eight-coordinate 0.25 pp round-half-even signature."""

    array = _finite_vector(
        vector, name="forecast vector", expected_length=FORECAST_DIMENSION
    )
    scaled = np.rint(array / SHAPE_SIGNATURE_STEP_PP)
    limits = np.iinfo(np.int64)
    if np.any(scaled < limits.min) or np.any(scaled > limits.max):
        raise V2ModelError("quantized signature is outside signed int64 range")
    return tuple(int(value) for value in scaled.astype(np.int64))


def deduplicate_vectors(
    vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Deduplicate by canonical raw bytes and return canonical byte order."""

    unique: dict[bytes, tuple[float, ...]] = {}
    for vector in vectors:
        array = _finite_vector(
            vector, name="forecast vector", expected_length=FORECAST_DIMENSION
        )
        key = canonical_float64_vector_bytes(array)
        unique.setdefault(key, tuple(float(value) for value in array))
    return tuple(unique[key] for key in sorted(unique))


def signature_representatives(
    vectors: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Collapse exact duplicates and then 0.25 pp signature groups."""

    grouped: dict[tuple[int, ...], list[tuple[float, ...]]] = {}
    for vector in deduplicate_vectors(vectors):
        grouped.setdefault(quantized_shape_signature(vector), []).append(vector)

    representatives: list[tuple[float, ...]] = []
    for signature in sorted(grouped):
        group = np.asarray(grouped[signature], dtype=np.float64)
        representative = np.quantile(group, 0.5, axis=0, method="linear")
        representatives.append(tuple(float(value) for value in representative))
    return tuple(representatives)


def inverted_cdf_weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    probability: float,
    *,
    tie_keys: Sequence[bytes] | None = None,
) -> float:
    """Compute the frozen normalized inverted empirical-CDF quantile."""

    value_array = _finite_vector(values, name="values")
    weight_array = _finite_vector(weights, name="weights")
    if value_array.size == 0 or value_array.size != weight_array.size:
        raise V2ModelError("values and weights must have the same positive length")
    if np.any(weight_array < 0.0) or not float(weight_array.sum()) > 0.0:
        raise V2ModelError("weights must be nonnegative with positive total")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise V2ModelError("probability must be finite and in [0, 1]")
    if tie_keys is None:
        keys = tuple(b"" for _ in range(value_array.size))
    else:
        keys = tuple(tie_keys)
        if len(keys) != value_array.size or any(
            not isinstance(key, bytes) for key in keys
        ):
            raise V2ModelError("tie_keys must contain one bytes key per value")

    order = sorted(
        range(value_array.size),
        key=lambda index: (float(value_array[index]), keys[index]),
    )
    if probability == 1.0:
        return float(value_array[order[-1]])
    threshold = probability * float(weight_array.sum())
    cumulative = 0.0
    for index in order:
        cumulative += float(weight_array[index])
        if cumulative >= threshold:
            return float(value_array[index])
    return float(value_array[order[-1]])


def coordinatewise_weighted_quantile(
    vectors: Sequence[Sequence[float]],
    weights: Sequence[float],
    probability: float,
) -> tuple[float, ...]:
    matrix = _finite_matrix(vectors, name="support vectors", columns=FORECAST_DIMENSION)
    if matrix.shape[0] == 0:
        raise V2ModelError("support vectors must not be empty")
    weight_array = _finite_vector(
        weights, name="support weights", expected_length=matrix.shape[0]
    )
    tie_keys = tuple(canonical_float64_vector_bytes(row) for row in matrix)
    return tuple(
        inverted_cdf_weighted_quantile(
            matrix[:, column],
            weight_array,
            probability,
            tie_keys=tie_keys,
        )
        for column in range(matrix.shape[1])
    )


def family_balanced_support(
    family_vectors: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Build the two-level family-balanced empirical support."""

    by_family: list[tuple[str, tuple[tuple[float, ...], ...]]] = []
    for family_id in sorted(family_vectors):
        representatives = signature_representatives(family_vectors[family_id])
        if representatives:
            by_family.append((family_id, representatives))
    successful_count = len(by_family)
    if successful_count == 0:
        return (), ()

    vectors: list[tuple[float, ...]] = []
    weights: list[float] = []
    for _, representatives in by_family:
        family_weight = 1.0 / successful_count
        signature_weight = family_weight / len(representatives)
        vectors.extend(representatives)
        weights.extend(signature_weight for _ in representatives)
    return tuple(vectors), tuple(weights)


def family_representative(
    vectors: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    representatives = signature_representatives(vectors)
    if not representatives:
        raise V2ModelError("a family representative requires a nonempty family")
    weights = tuple(1.0 / len(representatives) for _ in representatives)
    return coordinatewise_weighted_quantile(representatives, weights, 0.5)


def build_library_forecast(
    family_vectors: Mapping[str, Sequence[Sequence[float]]],
    sqrt_forecast: Sequence[float],
) -> LibraryForecastResult:
    """Aggregate family representatives or apply the frozen sqrt fallback."""

    sqrt_array = _finite_vector(
        sqrt_forecast,
        name="sqrt_forecast",
        expected_length=FORECAST_DIMENSION,
    )
    family_representatives = tuple(
        (family_id, family_representative(family_vectors[family_id]))
        for family_id in sorted(family_vectors)
        if len(family_vectors[family_id]) > 0
    )
    support_vectors, support_weights = family_balanced_support(family_vectors)
    successful_count = len(family_representatives)

    if successful_count < 2:
        library = tuple(float(value) for value in sqrt_array)
        hard_eligible = False
    else:
        representatives = [vector for _, vector in family_representatives]
        grouped_representatives = signature_representatives(representatives)
        matrix = np.asarray(grouped_representatives, dtype=np.float64)
        library_array = np.quantile(matrix, 0.5, axis=0, method="linear")
        library = tuple(float(value) for value in library_array)
        hard_eligible = True

    return LibraryForecastResult(
        forecast=library,
        successful_family_count=successful_count,
        hard_eligible=hard_eligible,
        family_representatives=family_representatives,
        support_vectors=support_vectors,
        support_weights=support_weights,
    )


def fit_center_blend_beta(
    library_forecasts: Sequence[Sequence[float]],
    sqrt_forecasts: Sequence[Sequence[float]],
    latent_truth: Sequence[Sequence[float]],
    *,
    ridge_penalty: float = 0.01,
) -> float:
    """Fit the single frozen center coefficient in closed form."""

    library = _finite_matrix(
        library_forecasts,
        name="library_forecasts",
        columns=FORECAST_DIMENSION,
    )
    sqrt = _finite_matrix(
        sqrt_forecasts,
        name="sqrt_forecasts",
        columns=FORECAST_DIMENSION,
    )
    truth = _finite_matrix(
        latent_truth,
        name="latent_truth",
        columns=FORECAST_DIMENSION,
    )
    if library.shape != sqrt.shape or library.shape != truth.shape:
        raise V2ModelError("center-development arrays must have identical shapes")
    if library.size == 0:
        raise V2ModelError("center-development arrays must not be empty")
    if not math.isfinite(ridge_penalty) or ridge_penalty < 0.0:
        raise V2ModelError("ridge_penalty must be finite and nonnegative")

    x = (library - sqrt).ravel()
    if np.all(x == 0.0):
        return 0.0
    y = (truth - sqrt).ravel()
    denominator = float(x @ x) + ridge_penalty * x.size
    if not denominator > 0.0 or not math.isfinite(denominator):
        raise V2ModelError("center coefficient denominator is invalid")
    beta = float(x @ y) / denominator
    return float(np.clip(beta, 0.0, 1.0))


def blend_center_forecast(
    sqrt_forecast: Sequence[float],
    library_forecast: Sequence[float],
    beta: float,
) -> tuple[float, ...]:
    sqrt = _finite_vector(
        sqrt_forecast, name="sqrt_forecast", expected_length=FORECAST_DIMENSION
    )
    library = _finite_vector(
        library_forecast,
        name="library_forecast",
        expected_length=FORECAST_DIMENSION,
    )
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise V2ModelError("beta must be finite and in [0, 1]")
    return tuple(float(value) for value in sqrt + beta * (library - sqrt))


def _variant_boundary_fraction(variant: VariantSummary) -> float:
    values = dict(variant.parameter_values)
    bounds = {name: (lower, upper) for name, lower, upper in variant.parameter_bounds}
    if len(values) != len(variant.parameter_values):
        raise V2ModelError("parameter_values contains duplicate names")
    if len(bounds) != len(variant.parameter_bounds):
        raise V2ModelError("parameter_bounds contains duplicate names")
    fixed = set(variant.fixed_parameters)
    fitted_names = sorted(set(values) - fixed)
    if not fitted_names:
        return 0.0

    hits = 0
    for name in fitted_names:
        if name not in bounds:
            raise V2ModelError(f"missing finite bounds for fitted parameter {name!r}")
        value = float(values[name])
        lower, upper = (float(item) for item in bounds[name])
        if not all(math.isfinite(item) for item in (value, lower, upper)):
            raise V2ModelError("parameter values and bounds must be finite")
        if upper < lower or not lower <= value <= upper:
            raise V2ModelError("fitted parameter must lie inside ordered bounds")
        tolerance = 1e-6 * max(1.0, upper - lower)
        if min(value - lower, upper - value) <= tolerance:
            hits += 1
    return hits / len(fitted_names)


def _variant_metadata_key(variant: VariantSummary) -> tuple[object, ...]:
    return (
        float(variant.prefix_rmse_pp),
        tuple(sorted(variant.parameter_values)),
        tuple(sorted(variant.parameter_bounds)),
        tuple(sorted(variant.fixed_parameters)),
    )


def _unique_variants(
    variants: Sequence[VariantSummary],
) -> tuple[VariantSummary, ...]:
    unique: dict[bytes, VariantSummary] = {}
    for variant in variants:
        forecast = _finite_vector(
            variant.forecast,
            name="variant forecast",
            expected_length=FORECAST_DIMENSION,
        )
        if not math.isfinite(variant.prefix_rmse_pp):
            raise V2ModelError("prefix_rmse_pp must be finite")
        key = canonical_float64_vector_bytes(forecast)
        previous = unique.get(key)
        if previous is not None:
            if _variant_metadata_key(previous) != _variant_metadata_key(variant):
                raise V2ModelError(
                    "exact duplicate forecasts have conflicting metadata"
                )
            continue
        unique[key] = variant
    return tuple(unique[key] for key in sorted(unique))


def parameter_boundary_hit_fraction(
    family_variants: Mapping[str, Sequence[VariantSummary]],
) -> float:
    """Compute family- and signature-balanced boundary saturation."""

    family_values: list[float] = []
    for family_id in sorted(family_variants):
        variants = _unique_variants(family_variants[family_id])
        if not variants:
            continue
        by_signature: dict[tuple[int, ...], list[float]] = {}
        for variant in variants:
            signature = quantized_shape_signature(variant.forecast)
            by_signature.setdefault(signature, []).append(
                _variant_boundary_fraction(variant)
            )
        signature_values = [
            float(np.mean(by_signature[signature]))
            for signature in sorted(by_signature)
        ]
        family_values.append(float(np.mean(signature_values)))
    if not family_values:
        raise V2ModelError("boundary fraction requires a successful family")
    return float(np.mean(family_values))


def _value_at_day(
    days: np.ndarray,
    values: np.ndarray,
    day: float,
    *,
    name: str,
) -> float:
    matches = np.flatnonzero(days == day)
    if matches.size != 1:
        raise V2ModelError(f"{name} must contain day {day} exactly once")
    return float(values[matches[0]])


def leave_day730_out_sqrt_error(
    prefix_days: Sequence[float],
    observed_retention_pct: Sequence[float],
) -> float:
    """Fit the bounded sqrt baseline through day 540 and score day 730."""

    days = _finite_vector(prefix_days, name="prefix_days")
    observed = _finite_vector(
        observed_retention_pct,
        name="observed_retention_pct",
        expected_length=days.size,
    )
    if np.unique(days).size != days.size:
        raise V2ModelError("prefix_days must be unique")
    target = _value_at_day(days, observed, 730.0, name="prefix_days")
    fit_mask = days <= 540.0
    if np.any(days[fit_mask] < 0.0):
        raise V2ModelError("prefix days must be nonnegative")
    x = np.sqrt(days[fit_mask] / 365.25)
    y = 100.0 - observed[fit_mask]
    denominator = float(x @ x)
    coefficient = 0.0 if denominator == 0.0 else float(x @ y) / denominator
    coefficient = float(np.clip(coefficient, 0.0, 5.0))
    prediction = 100.0 - coefficient * math.sqrt(730.0 / 365.25)
    return abs(prediction - target)


def extract_prefix_features(
    *,
    prefix_days: Sequence[float],
    observed_retention_pct: Sequence[float],
    family_variants: Mapping[str, Sequence[VariantSummary]],
    sqrt_forecast: Sequence[float],
    center_forecast: Sequence[float],
    declared_family_count: int = 7,
) -> PrefixFeatureVector:
    """Extract the fourteen frozen prefix-and-structure features."""

    days = _finite_vector(
        prefix_days,
        name="prefix_days",
        expected_length=len(PREFIX_DAYS),
    )
    observed = _finite_vector(
        observed_retention_pct,
        name="observed_retention_pct",
        expected_length=len(PREFIX_DAYS),
    )
    if tuple(float(day) for day in days) != PREFIX_DAYS:
        raise V2ModelError("prefix_days must equal the frozen ordered grid")
    sqrt = _finite_vector(
        sqrt_forecast, name="sqrt_forecast", expected_length=FORECAST_DIMENSION
    )
    center = _finite_vector(
        center_forecast,
        name="center_forecast",
        expected_length=FORECAST_DIMENSION,
    )
    if (
        isinstance(declared_family_count, bool)
        or not isinstance(declared_family_count, int)
        or declared_family_count <= 0
    ):
        raise V2ModelError("declared_family_count must be a positive integer")

    unique_by_family = {
        family_id: _unique_variants(variants)
        for family_id, variants in family_variants.items()
    }
    successful = {
        family_id: variants
        for family_id, variants in unique_by_family.items()
        if variants
    }
    successful_count = len(successful)
    if successful_count == 0:
        raise V2ModelError("prefix features require at least one successful family")
    if successful_count > declared_family_count:
        raise V2ModelError("successful families exceed declared_family_count")

    family_vectors = {
        family_id: tuple(variant.forecast for variant in variants)
        for family_id, variants in successful.items()
    }
    support_vectors, support_weights = family_balanced_support(family_vectors)
    support = np.asarray(support_vectors, dtype=np.float64)
    q10 = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.10)
    )
    q25 = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.25)
    )
    q75 = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.75)
    )
    q90 = np.asarray(
        coordinatewise_weighted_quantile(support_vectors, support_weights, 0.90)
    )
    del support

    family_representatives = [
        family_representative(family_vectors[family_id])
        for family_id in sorted(family_vectors)
    ]
    unique_family_representatives = deduplicate_vectors(family_representatives)
    effective_shape_count = len(
        {quantized_shape_signature(vector) for vector in unique_family_representatives}
    )
    best_rmse = min(
        variant.prefix_rmse_pp
        for variants in successful.values()
        for variant in variants
    )
    boundary_fraction = parameter_boundary_hit_fraction(successful)
    leave_out_error = leave_day730_out_sqrt_error(days, observed)

    q0_value = _value_at_day(days, observed, 0.0, name="prefix_days")
    q90_value = _value_at_day(days, observed, 90.0, name="prefix_days")
    q180_value = _value_at_day(days, observed, 180.0, name="prefix_days")
    q365_value = _value_at_day(days, observed, 365.0, name="prefix_days")
    q730_value = _value_at_day(days, observed, 730.0, name="prefix_days")
    slope_180_365 = (q365_value - q180_value) / (365.0 - 180.0) * 365.25
    slope_365_730 = (q730_value - q365_value) / (730.0 - 365.0) * 365.25

    width_25y = float(q90[-1] - q10[-1])
    width_10y = float(q90[4] - q10[4])
    values = (
        float(successful_count),
        float(declared_family_count - successful_count),
        float(best_rmse),
        width_25y,
        float(np.mean(q75 - q25)),
        float(effective_shape_count),
        boundary_fraction,
        leave_out_error,
        float(center[-1]),
        float(center[-1] - sqrt[-1]),
        q365_value - q730_value,
        q0_value - q90_value,
        slope_180_365 - slope_365_730,
        max(0.0, width_25y - width_10y),
    )
    if not np.isfinite(values).all():
        raise V2ModelError("extracted prefix features must all be finite")
    return PrefixFeatureVector(PREFIX_FEATURE_NAMES, values)


def fit_standardizer(
    features: Sequence[Sequence[float]],
) -> StandardizerState:
    matrix = _finite_matrix(features, name="features")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise V2ModelError("features must have positive row and column counts")
    mean = np.mean(matrix, axis=0)
    population_std = np.std(matrix, axis=0, ddof=0)
    zero_variance = population_std == 0.0
    scale = population_std.copy()
    scale[zero_variance] = 1.0
    return StandardizerState(
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        zero_variance=tuple(bool(value) for value in zero_variance),
    )


def fit_logistic_risk_state(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    *,
    feature_names: Sequence[str] | None = None,
) -> LogisticRiskState:
    """Fit the frozen standardized L2 logistic danger score."""

    matrix = _finite_matrix(features, name="features")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise V2ModelError("features must have positive row and column counts")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size != matrix.shape[0]:
        raise V2ModelError("labels must contain one value per feature row")
    if not np.isin(label_array, (0, 1)).all() or np.unique(label_array).size != 2:
        raise V2ModelError("labels must be binary and contain both classes")
    labels_int = label_array.astype(np.int64)

    if feature_names is None:
        names = tuple(f"feature_{index}" for index in range(matrix.shape[1]))
    else:
        names = tuple(feature_names)
        if len(names) != matrix.shape[1] or len(set(names)) != len(names):
            raise V2ModelError("feature_names must be unique and match columns")

    standardizer = fit_standardizer(matrix)
    standardized = standardizer.transform(matrix)
    estimator = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        class_weight=None,
        max_iter=10000,
        tol=1e-10,
        random_state=None,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(standardized, labels_int)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise V2ModelError("logistic regression did not converge")

    coefficients = np.asarray(estimator.coef_[0], dtype=np.float64)
    zero_variance = np.asarray(standardizer.zero_variance, dtype=bool)
    if np.any(np.abs(coefficients[zero_variance]) > 1e-12):
        raise V2ModelError("a zero-variance feature received a nonzero coefficient")
    coefficients[zero_variance] = 0.0
    intercept = float(estimator.intercept_[0])
    if not np.isfinite(coefficients).all() or not math.isfinite(intercept):
        raise V2ModelError("logistic state contains nonfinite values")
    return LogisticRiskState(
        feature_names=names,
        standardizer=standardizer,
        intercept=intercept,
        coefficients=tuple(float(value) for value in coefficients),
    )


def fit_isotonic_state(
    raw_scores: Sequence[float],
    labels: Sequence[int],
) -> IsotonicState:
    scores = _finite_vector(raw_scores, name="raw_scores")
    label_array = np.asarray(labels)
    if label_array.ndim != 1 or label_array.size != scores.size:
        raise V2ModelError("labels must contain one value per raw score")
    if scores.size == 0 or not np.isin(label_array, (0, 1)).all():
        raise V2ModelError("isotonic labels must be nonempty and binary")
    if np.unique(label_array).size != 2:
        raise V2ModelError("isotonic labels must contain both classes")

    estimator = IsotonicRegression(out_of_bounds="clip")
    estimator.fit(scores, label_array.astype(np.float64))
    x_thresholds = np.asarray(estimator.X_thresholds_, dtype=np.float64)
    y_thresholds = np.asarray(estimator.y_thresholds_, dtype=np.float64)
    if not np.isfinite(x_thresholds).all() or not np.isfinite(y_thresholds).all():
        raise V2ModelError("isotonic state contains nonfinite values")
    return IsotonicState(
        x_thresholds=tuple(float(value) for value in x_thresholds),
        y_thresholds=tuple(float(value) for value in y_thresholds),
    )


def simultaneous_nonconformity_scores(
    lower: Sequence[Sequence[float]],
    upper: Sequence[Sequence[float]],
    truth: Sequence[Sequence[float]],
) -> np.ndarray:
    lower_array = _finite_matrix(lower, name="lower", columns=FORECAST_DIMENSION)
    upper_array = _finite_matrix(upper, name="upper", columns=FORECAST_DIMENSION)
    truth_array = _finite_matrix(truth, name="truth", columns=FORECAST_DIMENSION)
    if lower_array.shape != upper_array.shape or lower_array.shape != truth_array.shape:
        raise V2ModelError("lower, upper, and truth must have identical shapes")
    if lower_array.shape[0] == 0 or np.any(lower_array > upper_array):
        raise V2ModelError("interval rows must be nonempty and ordered")
    violations = np.maximum.reduce(
        (
            lower_array - truth_array,
            truth_array - upper_array,
            np.zeros_like(truth_array),
        )
    )
    return np.max(violations, axis=1)


def finite_sample_order_statistic_index(
    calibration_count: int,
    coverage: float,
) -> int:
    if (
        isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count <= 0
    ):
        raise V2ModelError("calibration_count must be a positive integer")
    if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
        raise V2ModelError("coverage must be finite and strictly between 0 and 1")
    index = math.ceil((calibration_count + 1) * coverage)
    if index > calibration_count:
        raise V2ModelError(
            "requested finite-sample coverage is undefined at this sample size"
        )
    return index


def fit_conformal_expansion(
    nonconformity_scores: Sequence[float],
    *,
    coverage: float = 0.90,
) -> ConformalExpansionState:
    scores = _finite_vector(nonconformity_scores, name="nonconformity_scores")
    if scores.size == 0 or np.any(scores < 0.0):
        raise V2ModelError("nonconformity scores must be nonempty and nonnegative")
    index = finite_sample_order_statistic_index(int(scores.size), coverage)
    sorted_scores = np.sort(scores, kind="stable")
    expansion = float(sorted_scores[index - 1])
    return ConformalExpansionState(
        coverage=coverage,
        calibration_count=int(scores.size),
        order_statistic_index=index,
        expansion_pp=expansion,
    )


def expand_intervals(
    lower: Sequence[Sequence[float]],
    upper: Sequence[Sequence[float]],
    expansion_pp: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower_array = _finite_matrix(lower, name="lower", columns=FORECAST_DIMENSION)
    upper_array = _finite_matrix(upper, name="upper", columns=FORECAST_DIMENSION)
    if lower_array.shape != upper_array.shape or np.any(lower_array > upper_array):
        raise V2ModelError("lower and upper must have identical, ordered shapes")
    if not math.isfinite(expansion_pp) or expansion_pp < 0.0:
        raise V2ModelError("expansion_pp must be finite and nonnegative")
    return lower_array - expansion_pp, upper_array + expansion_pp


def rank_for_issuance(
    raw_danger_scores: Sequence[float],
    tie_hashes: Sequence[str],
    issue_count: int,
    *,
    occurrence_ordinals: Sequence[int] | None = None,
) -> RankingResult:
    """Rank ascending danger, then ascending SHA256, then occurrence ordinal."""

    scores = _finite_vector(raw_danger_scores, name="raw_danger_scores")
    hashes = tuple(tie_hashes)
    if len(hashes) != scores.size or any(
        _SHA256_PATTERN.fullmatch(value) is None for value in hashes
    ):
        raise V2ModelError("tie_hashes must contain one 64-hex SHA256 per score")
    if (
        isinstance(issue_count, bool)
        or not isinstance(issue_count, int)
        or not 0 <= issue_count <= scores.size
    ):
        raise V2ModelError("issue_count must be an integer in [0, row_count]")

    if occurrence_ordinals is None:
        if len(set(value.lower() for value in hashes)) != len(hashes):
            raise V2ModelError(
                "duplicate predictor content is forbidden without occurrence ordinals"
            )
        ordinals = tuple(0 for _ in hashes)
    else:
        ordinals = tuple(occurrence_ordinals)
        if (
            len(ordinals) != scores.size
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in ordinals
            )
            or len(set(zip((value.lower() for value in hashes), ordinals)))
            != len(hashes)
        ):
            raise V2ModelError(
                "occurrence ordinals must form unique nonnegative hash pairs"
            )

    order = tuple(
        sorted(
            range(scores.size),
            key=lambda index: (
                float(scores[index]),
                bytes.fromhex(hashes[index]),
                ordinals[index],
            ),
        )
    )
    ranks = [0] * scores.size
    for rank, index in enumerate(order, start=1):
        ranks[index] = rank
    issued = tuple(rank <= issue_count for rank in ranks)
    return RankingResult(order, tuple(ranks), issued)


def one_sided_clopper_pearson_lower(successes: int, trials: int) -> float:
    _validate_binomial_counts(successes, trials)
    if successes == 0:
        return 0.0
    return float(beta_distribution.ppf(0.05, successes, trials - successes + 1))


def two_sided_clopper_pearson(
    successes: int,
    trials: int,
) -> tuple[float, float]:
    _validate_binomial_counts(successes, trials)
    lower = (
        0.0
        if successes == 0
        else float(beta_distribution.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta_distribution.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def _validate_binomial_counts(successes: int, trials: int) -> None:
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, int)
        or not isinstance(trials, int)
        or trials <= 0
        or not 0 <= successes <= trials
    ):
        raise V2ModelError(
            "successes and trials must be integers with 0 <= successes <= trials"
        )
