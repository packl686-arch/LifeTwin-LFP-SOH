from __future__ import annotations

import math

import numpy as np
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DIMENSION,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    VariantSummary,
    V2ModelError,
    blend_center_forecast,
    build_library_forecast,
    canonical_float64_vector_bytes,
    coordinatewise_weighted_quantile,
    deduplicate_vectors,
    expand_intervals,
    extract_prefix_features,
    family_balanced_support,
    finite_sample_order_statistic_index,
    fit_center_blend_beta,
    fit_conformal_expansion,
    fit_isotonic_state,
    fit_logistic_risk_state,
    fit_standardizer,
    inverted_cdf_weighted_quantile,
    one_sided_clopper_pearson_lower,
    parameter_boundary_hit_fraction,
    quantized_shape_signature,
    rank_for_issuance,
    signature_representatives,
    simultaneous_nonconformity_scores,
    two_sided_clopper_pearson,
)


def _constant_vector(value: float) -> tuple[float, ...]:
    return (value,) * FORECAST_DIMENSION


def _variant(
    value: float,
    *,
    rmse: float = 0.5,
    parameter: float | None = None,
    fixed: bool = False,
) -> VariantSummary:
    if parameter is None:
        return VariantSummary(_constant_vector(value), rmse)
    return VariantSummary(
        _constant_vector(value),
        rmse,
        parameter_values=(("a", parameter),),
        parameter_bounds=(("a", 0.0, 1.0),),
        fixed_parameters=("a",) if fixed else (),
    )


def test_canonical_bytes_deduplicate_exact_vectors_only() -> None:
    positive_zero = (0.0,) + _constant_vector(1.0)[:-1]
    negative_zero = (-0.0,) + _constant_vector(1.0)[:-1]

    assert canonical_float64_vector_bytes(positive_zero) != (
        canonical_float64_vector_bytes(negative_zero)
    )
    unique = deduplicate_vectors((positive_zero, positive_zero, negative_zero))
    assert len(unique) == 2


def test_shape_signature_uses_numpy_round_half_even() -> None:
    vector = (0.125, 0.375, 0.625, 0.875, -0.125, -0.375, -0.625, -0.875)

    assert quantized_shape_signature(vector) == (0, 2, 2, 4, 0, -2, -2, -4)


def test_signature_representative_uses_linear_median_after_exact_dedup() -> None:
    lower = _constant_vector(10.01)
    upper = _constant_vector(10.11)

    representatives = signature_representatives((upper, lower, lower))

    assert np.asarray(representatives) == pytest.approx(
        np.asarray((_constant_vector(10.06),))
    )


def test_two_level_support_balances_families_not_variant_count() -> None:
    family_vectors = {
        "grid": (
            _constant_vector(10.0),
            _constant_vector(20.0),
            _constant_vector(30.0),
        ),
        "single": (_constant_vector(100.0),),
    }
    vectors, weights = family_balanced_support(family_vectors)

    assert len(vectors) == 4
    assert sum(weights[:3]) == pytest.approx(0.5)
    assert weights[3] == pytest.approx(0.5)
    assert coordinatewise_weighted_quantile(vectors, weights, 0.5) == (
        _constant_vector(30.0)
    )


def test_inverted_cdf_weighted_quantile_has_no_interpolation() -> None:
    values = (10.0, 20.0, 30.0)
    weights = (0.2, 0.3, 0.5)

    assert inverted_cdf_weighted_quantile(values, weights, 0.2) == 10.0
    assert inverted_cdf_weighted_quantile(values, weights, 0.200001) == 20.0
    assert inverted_cdf_weighted_quantile(values, weights, 1.0) == 30.0
    with pytest.raises(V2ModelError, match="nonnegative"):
        inverted_cdf_weighted_quantile(values, (0.2, -0.1, 0.9), 0.5)


def test_library_forecast_is_invariant_to_duplicate_and_member_order() -> None:
    sqrt = _constant_vector(70.0)
    first = {
        "family_b": (_constant_vector(100.0),),
        "family_a": (
            _constant_vector(90.0),
            _constant_vector(80.0),
            _constant_vector(80.0),
        ),
    }
    reordered = {
        "family_a": (
            _constant_vector(80.0),
            _constant_vector(90.0),
        ),
        "family_b": (_constant_vector(100.0),),
    }

    result = build_library_forecast(first, sqrt)
    other = build_library_forecast(reordered, sqrt)

    assert result.hard_eligible
    assert result.successful_family_count == 2
    assert result.forecast == _constant_vector(90.0)
    assert result.forecast == other.forecast
    assert result.support_vectors == other.support_vectors
    assert result.support_weights == other.support_weights


def test_library_forecast_falls_back_with_fewer_than_two_families() -> None:
    sqrt = tuple(float(index) for index in range(FORECAST_DIMENSION))

    empty = build_library_forecast({}, sqrt)
    one = build_library_forecast({"only": (_constant_vector(50.0),)}, sqrt)

    assert not empty.hard_eligible
    assert not one.hard_eligible
    assert empty.forecast == sqrt
    assert one.forecast == sqrt
    assert empty.support_vectors == ()
    assert len(one.support_vectors) == 1


def test_center_ridge_closed_form_zero_direction_and_clipping() -> None:
    sqrt = np.full((2, FORECAST_DIMENSION), 80.0)
    library = sqrt + 1.0
    truth = sqrt + 0.5

    fitted = fit_center_blend_beta(library, sqrt, truth)

    assert fitted == pytest.approx(0.5 / 1.01)
    assert fit_center_blend_beta(sqrt, sqrt, truth) == 0.0
    assert fit_center_blend_beta(library, sqrt, sqrt + 2.0) == 1.0
    assert fit_center_blend_beta(library, sqrt, sqrt - 2.0) == 0.0
    assert blend_center_forecast(
        _constant_vector(80.0), _constant_vector(90.0), 0.25
    ) == _constant_vector(82.5)
    with pytest.raises(V2ModelError, match="finite"):
        fit_center_blend_beta(
            np.full((1, FORECAST_DIMENSION), np.nan),
            sqrt[:1],
            truth[:1],
        )


def test_boundary_fraction_balances_signatures_and_families() -> None:
    variants = {
        "bounded": (
            _variant(80.0, parameter=0.0),
            _variant(80.1, parameter=0.5),
            _variant(90.0, parameter=0.5),
        ),
        "persistence": (_variant(100.0),),
    }

    # bounded signature means are (1 + 0) / 2 and 0, so family value is .25;
    # the zero-parameter persistence family contributes zero.
    assert parameter_boundary_hit_fraction(variants) == pytest.approx(0.125)
    fixed_only = {"grid": (_variant(80.0, parameter=0.0, fixed=True),)}
    assert parameter_boundary_hit_fraction(fixed_only) == 0.0


def test_exact_duplicate_variant_metadata_conflict_is_rejected() -> None:
    variants = {
        "family": (
            _variant(80.0, rmse=0.2),
            _variant(80.0, rmse=0.3),
        )
    }

    with pytest.raises(V2ModelError, match="conflicting metadata"):
        parameter_boundary_hit_fraction(variants)


def test_extracts_fourteen_prefix_features_in_frozen_order() -> None:
    observed = tuple(100.0 - day / 365.25 for day in PREFIX_DAYS)
    family_variants = {
        "family_a": (
            VariantSummary(
                (95.0, 94.0, 93.0, 92.0, 91.0, 90.0, 89.0, 88.0),
                0.2,
                parameter_values=(("a", 0.0),),
                parameter_bounds=(("a", 0.0, 1.0),),
            ),
        ),
        "family_b": (
            VariantSummary(
                (85.0, 84.0, 83.0, 82.0, 81.0, 80.0, 79.0, 78.0),
                0.4,
            ),
        ),
    }
    sqrt = (90.0, 89.0, 88.0, 87.0, 86.0, 85.0, 84.0, 84.0)
    center = (89.0, 88.0, 87.0, 86.0, 85.0, 84.0, 83.0, 82.0)

    result = extract_prefix_features(
        prefix_days=PREFIX_DAYS,
        observed_retention_pct=observed,
        family_variants=family_variants,
        sqrt_forecast=sqrt,
        center_forecast=center,
    )
    features = result.as_dict()

    assert result.names == PREFIX_FEATURE_NAMES
    assert len(result.values) == 14
    assert features["successful_structure_family_count"] == 2.0
    assert features["fit_failure_count"] == 5.0
    assert features["best_prefix_rmse_pp"] == 0.2
    assert features["unique_shape_25y_q90_minus_q10_pp"] == 10.0
    assert features["mean_over_horizons_unique_shape_iqr_pp"] == 10.0
    assert features["effective_unique_shape_count"] == 2.0
    assert features["parameter_boundary_hit_fraction"] == 0.5
    assert features["center_25y_retention_pct"] == 82.0
    assert features["center_minus_sqrt_25y_pp"] == -2.0
    assert features["observed_q365_minus_q730_pp"] == pytest.approx(365.0 / 365.25)
    assert features["observed_q0_minus_q90_pp"] == pytest.approx(90.0 / 365.25)
    assert features["slope_180_365_minus_slope_365_730_pp_per_year"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert features["nonnegative_25y_minus_10y_unique_shape_q90_q10_growth_pp"] == 0.0
    assert np.isfinite(result.values).all()


def test_standardizer_maps_constant_feature_to_exact_zero() -> None:
    features = np.asarray(((1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0)))

    state = fit_standardizer(features)
    transformed = state.transform(features)

    assert state.zero_variance == (False, True)
    assert np.array_equal(transformed[:, 1], np.zeros(4))
    assert np.mean(transformed[:, 0]) == pytest.approx(0.0)
    assert np.std(transformed[:, 0], ddof=0) == pytest.approx(1.0)


def test_logistic_state_orients_larger_score_as_more_dangerous() -> None:
    features = np.asarray(((1.0, 5.0), (2.0, 5.0), (3.0, 5.0), (4.0, 5.0)))
    labels = (0, 0, 1, 1)

    state = fit_logistic_risk_state(
        features, labels, feature_names=("signal", "constant")
    )
    scores = state.decision_function(features)
    probabilities = state.predict_probability(features)

    assert state.coefficients[1] == 0.0
    assert np.all(np.diff(scores) > 0.0)
    assert np.all(np.diff(probabilities) > 0.0)
    with pytest.raises(V2ModelError, match="both classes"):
        fit_logistic_risk_state(features, (0, 0, 0, 0))


def test_isotonic_state_is_monotone_and_clips_out_of_bounds() -> None:
    state = fit_isotonic_state((0.0, 1.0, 2.0, 3.0), (0, 0, 1, 1))

    predictions = state.predict((-10.0, 0.5, 1.5, 2.5, 10.0))

    assert np.all(np.diff(predictions) >= 0.0)
    assert predictions[0] == 0.0
    assert predictions[-1] == 1.0


def test_conformal_score_and_finite_sample_order_statistic() -> None:
    lower = np.zeros((4, FORECAST_DIMENSION))
    upper = np.ones((4, FORECAST_DIMENSION))
    truth = np.asarray(
        (
            _constant_vector(0.5),
            _constant_vector(2.0),
            _constant_vector(-2.0),
            _constant_vector(4.0),
        )
    )

    scores = simultaneous_nonconformity_scores(lower, upper, truth)
    state = fit_conformal_expansion(scores, coverage=0.8)
    expanded_lower, expanded_upper = expand_intervals(lower, upper, state.expansion_pp)

    assert np.array_equal(scores, np.asarray((0.0, 1.0, 2.0, 3.0)))
    assert state.order_statistic_index == 4
    assert state.expansion_pp == 3.0
    assert np.array_equal(expanded_lower, lower - 3.0)
    assert np.array_equal(expanded_upper, upper + 3.0)
    assert finite_sample_order_statistic_index(900, 0.90) == 811
    with pytest.raises(V2ModelError, match="undefined"):
        finite_sample_order_statistic_index(4, 0.90)


def test_ranking_issues_lowest_score_and_breaks_ties_by_hash() -> None:
    hashes = ("0" * 64, "1" * 64, "2" * 64)

    result = rank_for_issuance((1.0, 0.0, 1.0), hashes, 2)

    assert result.order == (1, 0, 2)
    assert result.ranks == (2, 1, 3)
    assert result.issued == (True, True, False)
    with pytest.raises(V2ModelError, match="duplicate predictor"):
        rank_for_issuance((0.0, 0.0), (hashes[0], hashes[0]), 1)
    bootstrap = rank_for_issuance(
        (0.0, 0.0),
        (hashes[0], hashes[0]),
        1,
        occurrence_ordinals=(1, 0),
    )
    assert bootstrap.order == (1, 0)


def test_clopper_pearson_rules_include_frozen_boundary_cases() -> None:
    assert one_sided_clopper_pearson_lower(0, 10) == 0.0
    assert one_sided_clopper_pearson_lower(10, 10) == pytest.approx(
        0.05 ** (1.0 / 10.0)
    )
    assert two_sided_clopper_pearson(10, 10) == pytest.approx(
        (0.025 ** (1.0 / 10.0), 1.0)
    )
    assert two_sided_clopper_pearson(0, 10) == pytest.approx(
        (0.0, 1.0 - 0.025 ** (1.0 / 10.0))
    )
    with pytest.raises(V2ModelError, match="0 <= successes"):
        two_sided_clopper_pearson(11, 10)


def test_all_primitives_reject_nonfinite_values() -> None:
    with pytest.raises(V2ModelError, match="finite"):
        quantized_shape_signature((math.nan,) + _constant_vector(1.0)[:-1])
    with pytest.raises(V2ModelError, match="finite"):
        fit_isotonic_state((0.0, math.inf), (0, 1))
    with pytest.raises(V2ModelError, match="finite"):
        rank_for_issuance((0.0, math.nan), ("0" * 64, "1" * 64), 1)
