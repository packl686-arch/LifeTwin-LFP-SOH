from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np

from lifetwin.models.calendar_v3_activation import (
    ActivationOffsetFit,
    activation_basis,
)


RESIDUAL_BASIS_NAME = "landmark_anchored_bounded_ridge_v1"
_COVARIANCE_SYMMETRY_ATOL = 1e-12
_COVARIANCE_PSD_ATOL = 1e-10


class ResidualSupportError(ValueError):
    """Raised instead of extrapolating a residual correction beyond support."""


class MeanPredictionRoute(StrEnum):
    SPECIALIST = "hierarchical_activation_residual"
    FALLBACK = "hierarchical_power_fallback"
    UNAVAILABLE = "unavailable"


class MeanFallbackReason(StrEnum):
    SPECIALIST_GATE_NOT_READY = "specialist_gate_not_ready"
    SPECIALIST_FIT_FAILED = "specialist_fit_failed"
    RESIDUAL_OUTSIDE_SUPPORT = "residual_outside_support"
    RESIDUAL_CAP_HIT = "residual_cap_hit"
    FALLBACK_FIT_FAILED = "fallback_fit_failed"


class IssuanceStatus(StrEnum):
    ISSUED = "issued"
    ABSTAINED = "abstained"


class AbstentionReason(StrEnum):
    MEAN_UNAVAILABLE = "mean_unavailable"
    CALIBRATION_UNAVAILABLE = "calibration_unavailable"
    CALIBRATION_EVIDENCE_NOT_INDEPENDENT = (
        "calibration_evidence_not_independent"
    )
    INSUFFICIENT_SAME_ROUTE_CALIBRATION = (
        "insufficient_same_route_calibration"
    )
    HORIZON_MISMATCH = "horizon_mismatch"
    DOMAIN_UNSUPPORTED = "domain_unsupported"
    INDEPENDENT_LONG_TERM_EVIDENCE_MISSING = (
        "independent_long_term_evidence_missing"
    )
    RESIDUAL_OUTSIDE_SUPPORT = "residual_outside_support"
    RESIDUAL_CAP_HIT = "residual_cap_hit"
    INTERVAL_WIDTH_INVALID = "interval_width_invalid"
    INTERVAL_TOO_WIDE = "interval_too_wide"


@dataclass(frozen=True)
class BoundedResidualFit:
    coefficients: tuple[float, float]
    ridge_penalty: float
    support_horizon_days: float
    correction_cap_pp: float
    training_observation_count: int
    training_condition_ids: tuple[str, ...]
    landmark_days: float
    observed_max_horizon_days: float
    upstream_training_state_sha256: str
    residual_training_state_sha256: str
    basis_name: str = RESIDUAL_BASIS_NAME


@dataclass(frozen=True)
class ResidualCorrectionPrediction:
    raw_correction_pp: tuple[float, ...]
    correction_pp: tuple[float, ...]
    cap_hit: tuple[bool, ...]
    support_horizon_days: float

    @property
    def any_cap_hit(self) -> bool:
        return any(self.cap_hit)


@dataclass(frozen=True)
class ConservativeIssuanceDecision:
    mean_route: MeanPredictionRoute
    mean_fallback_reasons: tuple[MeanFallbackReason, ...]
    issuance_status: IssuanceStatus
    abstention_reasons: tuple[AbstentionReason, ...]
    interval_width_pp: float | None
    max_interval_width_pp: float | None

    @property
    def issued(self) -> bool:
        return self.issuance_status is IssuanceStatus.ISSUED


def _finite_positive(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and positive")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return converted


def _strict_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _sha256(value: object, *, name: str) -> str:
    converted = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", converted) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return converted


def _condition_labels(
    values: Sequence[str],
    *,
    expected_length: int,
) -> tuple[str, ...]:
    labels = tuple(values)
    if len(labels) != expected_length or any(
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or value.casefold() == "nan"
        for value in labels
    ):
        raise ValueError(
            "training_condition_ids must contain one canonical string per residual"
        )
    unique = tuple(sorted(set(labels)))
    if len(unique) < 2:
        raise ValueError("Residual learning requires at least two training conditions")
    return unique


def _residual_training_sha256(
    *,
    horizon: np.ndarray,
    residual: np.ndarray,
    labels: tuple[str, ...],
    landmark_days: float,
    support_horizon_days: float,
    correction_cap_pp: float,
    ridge_penalty: float,
    upstream_training_state_sha256: str,
) -> str:
    payload = {
        "rows": [
            {
                "condition_id": label,
                "horizon_days_hex": float(horizon_value).hex(),
                "residual_pp_hex": float(residual_value).hex(),
            }
            for horizon_value, residual_value, label in zip(
                horizon,
                residual,
                labels,
                strict=True,
            )
        ],
        "landmark_days_hex": float(landmark_days).hex(),
        "support_horizon_days_hex": float(support_horizon_days).hex(),
        "correction_cap_pp_hex": float(correction_cap_pp).hex(),
        "ridge_penalty_hex": float(ridge_penalty).hex(),
        "upstream_training_state_sha256": upstream_training_state_sha256,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str,
    allow_empty: bool = False,
) -> np.ndarray:
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or (not allow_empty and raw.size == 0):
        qualifier = "one-dimensional" if allow_empty else "non-empty one-dimensional"
        raise ValueError(f"{name} must be a {qualifier} array")
    if any(isinstance(value, (bool, np.bool_)) for value in raw):
        raise ValueError(f"{name} cannot contain booleans")
    try:
        array = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _activation_parameter_covariance(fitted: ActivationOffsetFit) -> np.ndarray:
    if fitted.parameter_covariance is None:
        raise ValueError("Activation parameter covariance is required")
    covariance = np.asarray(fitted.parameter_covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
        raise ValueError("Activation parameter covariance must be finite 3x3")
    if not np.allclose(
        covariance,
        covariance.T,
        rtol=0.0,
        atol=_COVARIANCE_SYMMETRY_ATOL,
    ):
        raise ValueError("Activation parameter covariance must be symmetric")
    covariance = 0.5 * (covariance + covariance.T)
    if float(np.linalg.eigvalsh(covariance).min()) < -_COVARIANCE_PSD_ATOL:
        raise ValueError(
            "Activation parameter covariance must be positive semidefinite"
        )
    return covariance


def activation_loss_parameter_gradient(
    fitted: ActivationOffsetFit,
    elapsed_days: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Gradient of activation-offset capacity loss in parameter order.

    The columns are log-amplitude, time exponent, and activation offset. The
    retention gradient has the opposite sign, which yields the same variance.
    """
    elapsed = _finite_vector(elapsed_days, name="elapsed_days")
    if np.any(elapsed <= 0.0):
        raise ValueError("elapsed_days must be strictly positive")
    parameters = np.asarray(
        [
            fitted.log_amplitude,
            fitted.time_exponent,
            fitted.activation_offset_pp,
            fitted.activation_timescale_days,
        ],
        dtype=float,
    )
    if not np.isfinite(parameters).all() or fitted.activation_timescale_days <= 0.0:
        raise ValueError("Activation fit parameters must be finite with positive tau")
    irreversible = np.exp(fitted.log_amplitude) * np.power(
        elapsed,
        fitted.time_exponent,
    )
    basis = activation_basis(
        elapsed,
        timescale_days=fitted.activation_timescale_days,
    )
    gradient = np.column_stack(
        (irreversible, irreversible * np.log(elapsed), -basis)
    )
    if not np.isfinite(gradient).all():
        raise ValueError("Activation predictive gradient is not finite")
    return gradient


def activation_offset_predictive_sd(
    fitted: ActivationOffsetFit,
    elapsed_days: Sequence[float] | np.ndarray,
    *,
    observation_scale_pp: float,
    scale_floor_pp: float,
) -> np.ndarray:
    """Delta-method pointwise SD for a hierarchical activation-offset fit."""
    covariance = _activation_parameter_covariance(fitted)
    observation_scale = _finite_positive(
        observation_scale_pp,
        name="observation_scale_pp",
    )
    scale_floor = _finite_positive(scale_floor_pp, name="scale_floor_pp")
    gradient = activation_loss_parameter_gradient(fitted, elapsed_days)
    parameter_variance = np.einsum(
        "ij,jk,ik->i",
        gradient,
        covariance,
        gradient,
    )
    if float(parameter_variance.min()) < -_COVARIANCE_PSD_ATOL:
        raise ValueError("Activation predictive variance cannot be negative")
    total_variance = np.maximum(parameter_variance, 0.0) + observation_scale**2
    predictive_sd = np.sqrt(total_variance)
    if not np.isfinite(predictive_sd).all():
        raise ValueError("Activation predictive SD is not finite")
    return np.maximum(predictive_sd, scale_floor)


def landmark_anchored_residual_basis(
    horizon_days: Sequence[float] | np.ndarray,
    *,
    support_horizon_days: float,
) -> np.ndarray:
    """Two bounded bases that are exactly zero at the landmark (horizon zero)."""
    support = _finite_positive(
        support_horizon_days,
        name="support_horizon_days",
    )
    horizon = _finite_vector(horizon_days, name="horizon_days", allow_empty=True)
    if np.any(horizon < 0.0) or np.any(horizon > support):
        raise ResidualSupportError(
            "Residual horizons must remain within [0, support_horizon_days]"
        )
    normalized = horizon / support
    return np.column_stack(
        (
            normalized,
            1.0 - np.exp(-3.0 * normalized),
        )
    )


def fit_bounded_residual_correction(
    horizon_days: Sequence[float] | np.ndarray,
    residual_pp: Sequence[float] | np.ndarray,
    *,
    support_horizon_days: float,
    correction_cap_pp: float,
    ridge_penalty: float,
    training_condition_ids: Sequence[str],
    landmark_days: float,
    upstream_training_state_sha256: str,
) -> BoundedResidualFit:
    horizon = _finite_vector(horizon_days, name="horizon_days")
    residual = _finite_vector(residual_pp, name="residual_pp")
    if horizon.shape != residual.shape:
        raise ValueError("horizon_days and residual_pp must have equal length")
    support = _finite_positive(
        support_horizon_days,
        name="support_horizon_days",
    )
    positive_horizon = horizon[horizon > 0.0]
    if positive_horizon.size == 0:
        raise ValueError("Residual learning requires a positive observed horizon")
    observed_max = float(positive_horizon.max())
    if not math.isclose(support, observed_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "support_horizon_days must equal the maximum observed training horizon"
        )
    cap = _finite_positive(correction_cap_pp, name="correction_cap_pp")
    ridge = _finite_positive(ridge_penalty, name="ridge_penalty")
    if isinstance(landmark_days, (bool, np.bool_)):
        raise ValueError("landmark_days must be finite and non-negative")
    landmark = float(landmark_days)
    if not math.isfinite(landmark) or landmark < 0.0:
        raise ValueError("landmark_days must be finite and non-negative")
    raw_condition_ids = tuple(training_condition_ids)
    condition_ids = _condition_labels(
        training_condition_ids,
        expected_length=len(horizon),
    )
    informative_condition_ids = {
        label
        for label, horizon_value in zip(
            raw_condition_ids,
            horizon,
            strict=True,
        )
        if horizon_value > 0.0
    }
    if len(informative_condition_ids) < 2:
        raise ValueError(
            "Residual learning requires two conditions with positive horizons"
        )
    upstream_state_sha256 = _sha256(
        upstream_training_state_sha256,
        name="upstream_training_state_sha256",
    )
    design = landmark_anchored_residual_basis(
        horizon,
        support_horizon_days=support,
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        raise ValueError("Residual horizons do not identify both anchored basis terms")
    system = design.T @ design + ridge * np.eye(design.shape[1])
    coefficients = np.linalg.solve(system, design.T @ residual)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Residual ridge coefficients are not finite")
    residual_state_sha256 = _residual_training_sha256(
        horizon=horizon,
        residual=residual,
        labels=raw_condition_ids,
        landmark_days=landmark,
        support_horizon_days=support,
        correction_cap_pp=cap,
        ridge_penalty=ridge,
        upstream_training_state_sha256=upstream_state_sha256,
    )
    return BoundedResidualFit(
        coefficients=(float(coefficients[0]), float(coefficients[1])),
        ridge_penalty=ridge,
        support_horizon_days=support,
        correction_cap_pp=cap,
        training_observation_count=len(horizon),
        training_condition_ids=condition_ids,
        landmark_days=landmark,
        observed_max_horizon_days=observed_max,
        upstream_training_state_sha256=upstream_state_sha256,
        residual_training_state_sha256=residual_state_sha256,
    )


def predict_bounded_residual_correction(
    fitted: BoundedResidualFit,
    horizon_days: Sequence[float] | np.ndarray,
) -> ResidualCorrectionPrediction:
    if fitted.basis_name != RESIDUAL_BASIS_NAME:
        raise ValueError(f"Unsupported residual basis: {fitted.basis_name}")
    coefficients = np.asarray(fitted.coefficients, dtype=float)
    if coefficients.shape != (2,) or not np.isfinite(coefficients).all():
        raise ValueError("Residual coefficients must be a finite length-two vector")
    support = _finite_positive(
        fitted.support_horizon_days,
        name="support_horizon_days",
    )
    observed_max = _finite_positive(
        fitted.observed_max_horizon_days,
        name="observed_max_horizon_days",
    )
    if not math.isclose(support, observed_max, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Residual support exceeds its observed training horizon")
    cap = _finite_positive(fitted.correction_cap_pp, name="correction_cap_pp")
    _finite_positive(fitted.ridge_penalty, name="ridge_penalty")
    if isinstance(fitted.landmark_days, (bool, np.bool_)):
        raise ValueError("Residual landmark_days must be finite and non-negative")
    landmark = float(fitted.landmark_days)
    if not math.isfinite(landmark) or landmark < 0.0:
        raise ValueError("Residual landmark_days must be finite and non-negative")
    _sha256(
        fitted.upstream_training_state_sha256,
        name="upstream_training_state_sha256",
    )
    _sha256(
        fitted.residual_training_state_sha256,
        name="residual_training_state_sha256",
    )
    if (
        isinstance(fitted.training_observation_count, (bool, np.bool_))
        or not isinstance(fitted.training_observation_count, (int, np.integer))
        or fitted.training_observation_count < 1
    ):
        raise ValueError("Residual fit must record at least one training observation")
    if not all(
        isinstance(value, str) and value.strip()
        for value in fitted.training_condition_ids
    ):
        raise ValueError("Residual fit condition IDs must be non-empty strings")
    expected_condition_ids = tuple(sorted(set(fitted.training_condition_ids)))
    if (
        len(expected_condition_ids) < 2
        or fitted.training_condition_ids != expected_condition_ids
        or any(not value for value in fitted.training_condition_ids)
        or fitted.training_observation_count < len(expected_condition_ids)
    ):
        raise ValueError("Residual fit must record at least two training conditions")
    design = landmark_anchored_residual_basis(
        horizon_days,
        support_horizon_days=support,
    )
    raw = design @ coefficients
    cap_hit = np.abs(raw) > cap
    correction = np.clip(raw, -cap, cap)
    return ResidualCorrectionPrediction(
        raw_correction_pp=tuple(float(value) for value in raw),
        correction_pp=tuple(float(value) for value in correction),
        cap_hit=tuple(bool(value) for value in cap_hit),
        support_horizon_days=support,
    )


def conservative_issuance_decision(
    *,
    specialist_gate_ready: bool,
    specialist_fit_succeeded: bool,
    fallback_fit_succeeded: bool,
    residual_support_ok: bool,
    residual_cap_hit: bool,
    calibration_multiplier: float | None,
    calibration_evidence_independent: bool,
    sufficient_same_route_calibration: bool,
    calibration_horizon_matched: bool,
    domain_supported: bool,
    independent_long_term_evidence_required: bool,
    independent_long_term_evidence_available: bool,
    interval_width_pp: float | None,
    max_interval_width_pp: float | None = None,
) -> ConservativeIssuanceDecision:
    """Choose a mean route separately from whether an interval may be issued."""
    specialist_gate_ready = _strict_bool(
        specialist_gate_ready,
        name="specialist_gate_ready",
    )
    specialist_fit_succeeded = _strict_bool(
        specialist_fit_succeeded,
        name="specialist_fit_succeeded",
    )
    fallback_fit_succeeded = _strict_bool(
        fallback_fit_succeeded,
        name="fallback_fit_succeeded",
    )
    residual_support_ok = _strict_bool(
        residual_support_ok,
        name="residual_support_ok",
    )
    residual_cap_hit = _strict_bool(
        residual_cap_hit,
        name="residual_cap_hit",
    )
    calibration_evidence_independent = _strict_bool(
        calibration_evidence_independent,
        name="calibration_evidence_independent",
    )
    sufficient_same_route_calibration = _strict_bool(
        sufficient_same_route_calibration,
        name="sufficient_same_route_calibration",
    )
    calibration_horizon_matched = _strict_bool(
        calibration_horizon_matched,
        name="calibration_horizon_matched",
    )
    domain_supported = _strict_bool(domain_supported, name="domain_supported")
    independent_long_term_evidence_required = _strict_bool(
        independent_long_term_evidence_required,
        name="independent_long_term_evidence_required",
    )
    independent_long_term_evidence_available = _strict_bool(
        independent_long_term_evidence_available,
        name="independent_long_term_evidence_available",
    )
    fallback_reasons: list[MeanFallbackReason] = []
    if not specialist_gate_ready:
        fallback_reasons.append(MeanFallbackReason.SPECIALIST_GATE_NOT_READY)
    if not specialist_fit_succeeded:
        fallback_reasons.append(MeanFallbackReason.SPECIALIST_FIT_FAILED)
    if not residual_support_ok:
        fallback_reasons.append(MeanFallbackReason.RESIDUAL_OUTSIDE_SUPPORT)
    if residual_cap_hit:
        fallback_reasons.append(MeanFallbackReason.RESIDUAL_CAP_HIT)

    specialist_available = not fallback_reasons
    if specialist_available:
        mean_route = MeanPredictionRoute.SPECIALIST
    elif fallback_fit_succeeded:
        mean_route = MeanPredictionRoute.FALLBACK
    else:
        mean_route = MeanPredictionRoute.UNAVAILABLE
        fallback_reasons.append(MeanFallbackReason.FALLBACK_FIT_FAILED)

    if max_interval_width_pp is not None:
        maximum_width = _finite_positive(
            max_interval_width_pp,
            name="max_interval_width_pp",
        )
    else:
        maximum_width = None

    if isinstance(interval_width_pp, (bool, np.bool_)):
        raise ValueError("interval_width_pp must be numeric or null")
    raw_width = None if interval_width_pp is None else float(interval_width_pp)
    abstention_reasons: list[AbstentionReason] = []
    if mean_route is MeanPredictionRoute.UNAVAILABLE:
        abstention_reasons.append(AbstentionReason.MEAN_UNAVAILABLE)
    if isinstance(calibration_multiplier, (bool, np.bool_)):
        raise ValueError("calibration_multiplier must be numeric or null")
    if (
        calibration_multiplier is None
        or not math.isfinite(float(calibration_multiplier))
        or float(calibration_multiplier) < 0.0
    ):
        abstention_reasons.append(AbstentionReason.CALIBRATION_UNAVAILABLE)
    if not calibration_evidence_independent:
        abstention_reasons.append(
            AbstentionReason.CALIBRATION_EVIDENCE_NOT_INDEPENDENT
        )
    if not sufficient_same_route_calibration:
        abstention_reasons.append(
            AbstentionReason.INSUFFICIENT_SAME_ROUTE_CALIBRATION
        )
    if not calibration_horizon_matched:
        abstention_reasons.append(AbstentionReason.HORIZON_MISMATCH)
    if not domain_supported:
        abstention_reasons.append(AbstentionReason.DOMAIN_UNSUPPORTED)
    if (
        independent_long_term_evidence_required
        and not independent_long_term_evidence_available
    ):
        abstention_reasons.append(
            AbstentionReason.INDEPENDENT_LONG_TERM_EVIDENCE_MISSING
        )
    if not residual_support_ok:
        abstention_reasons.append(AbstentionReason.RESIDUAL_OUTSIDE_SUPPORT)
    if residual_cap_hit:
        abstention_reasons.append(AbstentionReason.RESIDUAL_CAP_HIT)
    if raw_width is None or not math.isfinite(raw_width) or raw_width < 0.0:
        abstention_reasons.append(AbstentionReason.INTERVAL_WIDTH_INVALID)
        width = None
    else:
        width = raw_width
    if width is not None and maximum_width is not None and width > maximum_width:
        abstention_reasons.append(AbstentionReason.INTERVAL_TOO_WIDE)

    issuance_status = (
        IssuanceStatus.ISSUED
        if not abstention_reasons
        else IssuanceStatus.ABSTAINED
    )
    return ConservativeIssuanceDecision(
        mean_route=mean_route,
        mean_fallback_reasons=tuple(fallback_reasons),
        issuance_status=issuance_status,
        abstention_reasons=tuple(abstention_reasons),
        interval_width_pp=width,
        max_interval_width_pp=maximum_width,
    )
