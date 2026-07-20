from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from lifetwin.models.calendar_v3_activation import (
    ActivationOffsetFit,
    predict_activation_offset_loss,
)
from lifetwin.models.calendar_v4_hybrid import (
    AbstentionReason,
    BoundedResidualFit,
    IssuanceStatus,
    MeanFallbackReason,
    MeanPredictionRoute,
    ResidualSupportError,
    activation_offset_predictive_sd,
    conservative_issuance_decision,
    fit_bounded_residual_correction,
    landmark_anchored_residual_basis,
    predict_bounded_residual_correction,
)


def _activation_fit(
    covariance: tuple[tuple[float, ...], ...] | None = (
        (0.04, 0.002, 0.0),
        (0.002, 0.01, 0.001),
        (0.0, 0.001, 0.09),
    ),
) -> ActivationOffsetFit:
    return ActivationOffsetFit(
        log_amplitude=-2.3,
        time_exponent=0.62,
        activation_offset_pp=0.8,
        activation_timescale_days=7.0,
        optimizer_cost=0.0,
        optimizer_evaluations=1,
        parameter_covariance=covariance,
    )


def test_activation_predictive_sd_matches_finite_difference_gradient() -> None:
    fitted = _activation_fit()
    elapsed = np.asarray([1.0, 10.0, 120.0])
    covariance = np.asarray(fitted.parameter_covariance, dtype=float)
    finite_difference = np.empty((len(elapsed), 3), dtype=float)
    step = 1e-6
    parameter_names = (
        "log_amplitude",
        "time_exponent",
        "activation_offset_pp",
    )
    for column, parameter in enumerate(parameter_names):
        plus = replace(fitted, **{parameter: getattr(fitted, parameter) + step})
        minus = replace(fitted, **{parameter: getattr(fitted, parameter) - step})
        finite_difference[:, column] = (
            predict_activation_offset_loss(plus, elapsed)
            - predict_activation_offset_loss(minus, elapsed)
        ) / (2.0 * step)
    observation_scale = 0.2
    expected = np.sqrt(
        np.einsum(
            "ij,jk,ik->i",
            finite_difference,
            covariance,
            finite_difference,
        )
        + observation_scale**2
    )
    observed = activation_offset_predictive_sd(
        fitted,
        elapsed,
        observation_scale_pp=observation_scale,
        scale_floor_pp=0.01,
    )
    np.testing.assert_allclose(observed, expected, rtol=2e-7, atol=1e-9)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (
            ((1.0, 0.2, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            "symmetric",
        ),
        (
            ((1.0, 0.0, 0.0), (0.0, -0.1, 0.0), (0.0, 0.0, 1.0)),
            "positive semidefinite",
        ),
        (
            ((1.0, 0.0, 0.0), (0.0, np.nan, 0.0), (0.0, 0.0, 1.0)),
            "finite 3x3",
        ),
    ],
)
def test_activation_predictive_sd_rejects_invalid_covariance(
    covariance: tuple[tuple[float, ...], ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        activation_offset_predictive_sd(
            _activation_fit(covariance),
            [10.0],
            observation_scale_pp=0.2,
            scale_floor_pp=0.1,
        )


def test_activation_predictive_sd_requires_covariance_and_positive_time() -> None:
    with pytest.raises(ValueError, match="covariance is required"):
        activation_offset_predictive_sd(
            _activation_fit(None),
            [10.0],
            observation_scale_pp=0.2,
            scale_floor_pp=0.1,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        activation_offset_predictive_sd(
            _activation_fit(),
            [0.0],
            observation_scale_pp=0.2,
            scale_floor_pp=0.1,
        )


def test_residual_basis_and_prediction_are_exactly_landmark_anchored() -> None:
    basis = landmark_anchored_residual_basis(
        [0.0, 5.0, 20.0],
        support_horizon_days=20.0,
    )
    np.testing.assert_array_equal(basis[0], np.zeros(2))
    assert np.all((basis >= 0.0) & (basis <= 1.0))

    fitted = fit_bounded_residual_correction(
        [0.0, 5.0, 10.0, 20.0],
        [7.0, 0.3, 0.5, 0.8],
        support_horizon_days=20.0,
        correction_cap_pp=2.0,
        ridge_penalty=0.5,
        training_condition_ids=["A", "A", "B", "B"],
        landmark_days=136.0,
        upstream_training_state_sha256="a" * 64,
    )
    prediction = predict_bounded_residual_correction(fitted, [0.0, 10.0])
    assert prediction.raw_correction_pp[0] == 0.0
    assert prediction.correction_pp[0] == 0.0
    assert not prediction.cap_hit[0]


def test_residual_fit_is_deterministic_and_reports_cap_hits() -> None:
    arguments = {
        "horizon_days": [0.0, 2.0, 5.0, 10.0],
        "residual_pp": [0.0, 50.0, 60.0, 70.0],
        "support_horizon_days": 10.0,
        "correction_cap_pp": 1.0,
        "ridge_penalty": 1e-6,
        "training_condition_ids": ["A", "A", "B", "B"],
        "landmark_days": 136.0,
        "upstream_training_state_sha256": "a" * 64,
    }
    first = fit_bounded_residual_correction(**arguments)
    second = fit_bounded_residual_correction(**arguments)
    assert first == second
    changed = fit_bounded_residual_correction(
        **{**arguments, "residual_pp": [0.0, 50.0, 60.0, 70.1]}
    )
    assert changed.residual_training_state_sha256 != (
        first.residual_training_state_sha256
    )
    prediction = predict_bounded_residual_correction(first, [0.0, 5.0, 10.0])
    assert prediction.any_cap_hit
    assert max(abs(value) for value in prediction.correction_pp) <= 1.0
    assert any(
        abs(raw) > abs(applied)
        for raw, applied in zip(
            prediction.raw_correction_pp,
            prediction.correction_pp,
            strict=True,
        )
    )


def test_residual_model_refuses_extrapolation_outside_declared_support() -> None:
    fitted = fit_bounded_residual_correction(
        [0.0, 2.0, 5.0],
        [0.0, 0.1, 0.2],
        support_horizon_days=5.0,
        correction_cap_pp=1.0,
        ridge_penalty=0.1,
        training_condition_ids=["A", "A", "B"],
        landmark_days=136.0,
        upstream_training_state_sha256="a" * 64,
    )
    with pytest.raises(ResidualSupportError, match="within"):
        predict_bounded_residual_correction(fitted, [5.0001])
    with pytest.raises(ValueError, match="maximum observed training horizon"):
        fit_bounded_residual_correction(
            [0.0, 2.0, 6.0],
            [0.0, 0.1, 0.2],
            support_horizon_days=5.0,
            correction_cap_pp=1.0,
            ridge_penalty=0.1,
            training_condition_ids=["A", "A", "B"],
            landmark_days=136.0,
            upstream_training_state_sha256="a" * 64,
        )


def _decision(**overrides: object):
    arguments: dict[str, object] = {
        "specialist_gate_ready": True,
        "specialist_fit_succeeded": True,
        "fallback_fit_succeeded": True,
        "residual_support_ok": True,
        "residual_cap_hit": False,
        "calibration_multiplier": 1.5,
        "calibration_evidence_independent": True,
        "sufficient_same_route_calibration": True,
        "calibration_horizon_matched": True,
        "domain_supported": True,
        "independent_long_term_evidence_required": False,
        "independent_long_term_evidence_available": False,
        "interval_width_pp": 4.0,
    }
    arguments.update(overrides)
    return conservative_issuance_decision(**arguments)


def test_issuance_separates_safe_mean_fallback_from_abstention() -> None:
    specialist = _decision()
    assert specialist.mean_route is MeanPredictionRoute.SPECIALIST
    assert specialist.issued

    fallback = _decision(specialist_gate_ready=False)
    assert fallback.mean_route is MeanPredictionRoute.FALLBACK
    assert fallback.mean_fallback_reasons == (
        MeanFallbackReason.SPECIALIST_GATE_NOT_READY,
    )
    assert fallback.issuance_status is IssuanceStatus.ISSUED
    assert fallback.abstention_reasons == ()


@pytest.mark.parametrize(
    ("override", "expected_reason"),
    [
        ({"calibration_multiplier": None}, AbstentionReason.CALIBRATION_UNAVAILABLE),
        ({"calibration_multiplier": np.inf}, AbstentionReason.CALIBRATION_UNAVAILABLE),
        (
            {"calibration_evidence_independent": False},
            AbstentionReason.CALIBRATION_EVIDENCE_NOT_INDEPENDENT,
        ),
        (
            {"sufficient_same_route_calibration": False},
            AbstentionReason.INSUFFICIENT_SAME_ROUTE_CALIBRATION,
        ),
        ({"calibration_horizon_matched": False}, AbstentionReason.HORIZON_MISMATCH),
        ({"domain_supported": False}, AbstentionReason.DOMAIN_UNSUPPORTED),
        (
            {
                "independent_long_term_evidence_required": True,
                "independent_long_term_evidence_available": False,
            },
            AbstentionReason.INDEPENDENT_LONG_TERM_EVIDENCE_MISSING,
        ),
        ({"interval_width_pp": np.nan}, AbstentionReason.INTERVAL_WIDTH_INVALID),
    ],
)
def test_issuance_abstains_at_each_uncertainty_boundary(
    override: dict[str, object],
    expected_reason: AbstentionReason,
) -> None:
    decision = _decision(**override)
    assert decision.issuance_status is IssuanceStatus.ABSTAINED
    assert expected_reason in decision.abstention_reasons
    if expected_reason is AbstentionReason.INTERVAL_WIDTH_INVALID:
        assert decision.interval_width_pp is None


def test_residual_cap_or_support_failure_falls_back_and_abstains() -> None:
    cap = _decision(residual_cap_hit=True)
    assert cap.mean_route is MeanPredictionRoute.FALLBACK
    assert MeanFallbackReason.RESIDUAL_CAP_HIT in cap.mean_fallback_reasons
    assert cap.abstention_reasons == (AbstentionReason.RESIDUAL_CAP_HIT,)

    support = _decision(residual_support_ok=False)
    assert support.mean_route is MeanPredictionRoute.FALLBACK
    assert MeanFallbackReason.RESIDUAL_OUTSIDE_SUPPORT in (
        support.mean_fallback_reasons
    )
    assert support.abstention_reasons == (
        AbstentionReason.RESIDUAL_OUTSIDE_SUPPORT,
    )


def test_missing_specialist_and_fallback_yields_unavailable_mean() -> None:
    decision = _decision(
        specialist_gate_ready=False,
        specialist_fit_succeeded=False,
        fallback_fit_succeeded=False,
    )
    assert decision.mean_route is MeanPredictionRoute.UNAVAILABLE
    assert decision.mean_fallback_reasons == (
        MeanFallbackReason.SPECIALIST_GATE_NOT_READY,
        MeanFallbackReason.SPECIALIST_FIT_FAILED,
        MeanFallbackReason.FALLBACK_FIT_FAILED,
    )
    assert decision.abstention_reasons == (AbstentionReason.MEAN_UNAVAILABLE,)


def test_width_limit_is_only_applied_when_explicitly_supplied() -> None:
    assert _decision(interval_width_pp=500.0).issued
    assert _decision(
        interval_width_pp=5.0,
        max_interval_width_pp=5.0,
    ).issued
    rejected = _decision(
        interval_width_pp=5.0001,
        max_interval_width_pp=5.0,
    )
    assert rejected.abstention_reasons == (AbstentionReason.INTERVAL_TOO_WIDE,)
    with pytest.raises(ValueError, match="max_interval_width_pp"):
        _decision(max_interval_width_pp=0.0)


@pytest.mark.parametrize(
    "override",
    [
        {"specialist_gate_ready": "False"},
        {"calibration_horizon_matched": np.nan},
        {"domain_supported": 1},
        {"independent_long_term_evidence_available": None},
    ],
)
def test_safety_flags_require_real_booleans(override: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        _decision(**override)


def test_wire_enums_stringify_to_their_stable_values() -> None:
    assert str(MeanPredictionRoute.SPECIALIST) == "hierarchical_activation_residual"
    assert str(IssuanceStatus.ABSTAINED) == "abstained"
    assert str(AbstentionReason.HORIZON_MISMATCH) == "horizon_mismatch"


def test_residual_support_is_observed_not_caller_declared() -> None:
    arguments = {
        "horizon_days": [0.0, 2.0, 5.0],
        "residual_pp": [0.0, 0.1, 0.2],
        "correction_cap_pp": 1.0,
        "ridge_penalty": 0.1,
        "training_condition_ids": ["A", "A", "B"],
        "landmark_days": 136.0,
        "upstream_training_state_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="maximum observed training horizon"):
        fit_bounded_residual_correction(
            **arguments,
            support_horizon_days=100.0,
        )
    with pytest.raises(ValueError, match="identify both anchored basis terms"):
        fit_bounded_residual_correction(
            **{
                **arguments,
                "horizon_days": [0.0, 5.0, 5.0],
                "support_horizon_days": 5.0,
            }
        )
    with pytest.raises(ValueError, match="two conditions with positive horizons"):
        fit_bounded_residual_correction(
            **{
                **arguments,
                "training_condition_ids": ["B", "A", "A"],
                "support_horizon_days": 5.0,
            }
        )


@pytest.mark.parametrize(
    "condition_ids",
    [
        [None, "A", "B"],
        ["A", " A ", "B"],
        ["A", "", "B"],
    ],
)
def test_residual_condition_ids_require_canonical_strings(
    condition_ids: list[object],
) -> None:
    with pytest.raises(ValueError, match="canonical string"):
        fit_bounded_residual_correction(
            [0.0, 2.0, 5.0],
            [0.0, 0.1, 0.2],
            support_horizon_days=5.0,
            correction_cap_pp=1.0,
            ridge_penalty=0.1,
            training_condition_ids=condition_ids,
            landmark_days=136.0,
            upstream_training_state_sha256="a" * 64,
        )


@pytest.mark.parametrize(
    ("horizon", "residual"),
    [
        ([False, 2.0, 5.0], [0.0, 0.1, 0.2]),
        ([0.0, 2.0, 5.0], [False, 0.1, 0.2]),
    ],
)
def test_residual_vectors_reject_boolean_values(horizon, residual) -> None:
    with pytest.raises(ValueError, match="cannot contain booleans"):
        fit_bounded_residual_correction(
            horizon,
            residual,
            support_horizon_days=5.0,
            correction_cap_pp=1.0,
            ridge_penalty=0.1,
            training_condition_ids=["A", "A", "B"],
            landmark_days=136.0,
            upstream_training_state_sha256="a" * 64,
        )


def test_invalid_manually_constructed_residual_fit_is_rejected() -> None:
    fitted = BoundedResidualFit(
        coefficients=(0.0, np.nan),
        ridge_penalty=1.0,
        support_horizon_days=10.0,
        correction_cap_pp=1.0,
        training_observation_count=1,
        training_condition_ids=("A", "B"),
        landmark_days=136.0,
        observed_max_horizon_days=10.0,
        upstream_training_state_sha256="a" * 64,
        residual_training_state_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="coefficients"):
        predict_bounded_residual_correction(fitted, [1.0])
