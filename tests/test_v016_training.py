from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import inspect
from typing import Any, Mapping

import numpy as np
import pytest

from lifetwin.experiments import calendar_long_horizon_v016_training as training
from lifetwin.experiments import calendar_long_horizon_v019_training as v019_training
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    LogisticRiskState,
    StandardizerState,
    coordinatewise_weighted_quantile,
    family_balanced_support,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    DECLARED_STRUCTURE_FAMILIES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CALIBRATION_COUNT,
    CalibrationDevelopmentState,
    RiskDevelopmentState,
)
from lifetwin.experiments.calendar_long_horizon_v019_state import (
    serialize_calibration_mask_commitment_json_v024,
)


def _logistic_state(
    feature_names: tuple[str, ...],
    *,
    active_indices: tuple[int, ...],
) -> LogisticRiskState:
    dimension = len(feature_names)
    coefficients = np.zeros(dimension, dtype=np.float64)
    for index in active_indices:
        coefficients[index] = 1.0 / max(1, len(active_indices))
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(False,) * dimension,
        ),
        intercept=0.0,
        coefficients=tuple(float(value) for value in coefficients),
    )


def _risk_state() -> RiskDevelopmentState:
    return RiskDevelopmentState(
        prefix_only_risk=_logistic_state(
            PREFIX_FEATURE_NAMES,
            active_indices=(2,),
        ),
        visible_stress_risk=_logistic_state(
            VISIBLE_STRESS_FEATURE_NAMES,
            active_indices=(2, len(PREFIX_FEATURE_NAMES)),
        ),
        placebo_risk=_logistic_state(
            PLACEBO_FEATURE_NAMES,
            active_indices=(2,),
        ),
        arm_a_plus_s_plan_risk=_logistic_state(
            ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
            active_indices=(2,),
        ),
        strongest_single_feature_name=PREFIX_FEATURE_NAMES[2],
        strongest_single_feature_orientation=1,
        strongest_single_feature_auroc=0.75,
        development_cluster_count=600,
        eligible_cluster_count=600,
        positive_label_count=120,
        negative_label_count=480,
    )


def _cancellation_state(feature_names: tuple[str, ...]) -> LogisticRiskState:
    dimension = len(feature_names)
    coefficients = (1e16, 1.0, -1e16, *((1.0,) * (dimension - 3)))
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(False,) * dimension,
        ),
        intercept=0.0,
        coefficients=coefficients,
    )


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _support_band(
    supports: Mapping[str, list[list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    vectors, weights = family_balanced_support(supports)
    lower = np.asarray(
        coordinatewise_weighted_quantile(vectors, weights, 0.05),
        dtype=np.float64,
    )
    upper = np.asarray(
        coordinatewise_weighted_quantile(vectors, weights, 0.95),
        dtype=np.float64,
    )
    return lower, upper


def _two_family_support() -> dict[str, list[list[float]]]:
    return {
        DECLARED_STRUCTURE_FAMILIES[0]: [[75.0] * 8],
        DECLARED_STRUCTURE_FAMILIES[1]: [[85.0] * 8],
    }


def _calibration_inputs() -> dict[str, Any]:
    risk_state = _risk_state()
    cluster_ids = np.asarray(
        [f"v021_c_{index:04d}" for index in range(CALIBRATION_COUNT)],
        dtype=object,
    )
    prefix = np.zeros(
        (CALIBRATION_COUNT, len(PREFIX_FEATURE_NAMES)),
        dtype=np.float64,
    )
    family_counts = np.full(CALIBRATION_COUNT, 2, dtype=np.int64)
    prefix[:, 0] = family_counts
    prefix[:, 1] = 7 - family_counts
    prefix[:120, 2] = 2.0 + np.arange(120) / 10_000.0
    prefix[120:, 2] = -2.0 + np.arange(780) / 10_000.0

    real_operating = np.zeros((CALIBRATION_COUNT, 8), dtype=np.float64)
    real_operating[:, 0] = np.linspace(-0.2, 0.2, CALIBRATION_COUNT)
    placebo_operating = np.zeros((CALIBRATION_COUNT, 8), dtype=np.float64)
    real_stress = real_operating.copy()
    placebo_features = placebo_operating.copy()
    center = np.full((CALIBRATION_COUNT, 8), 80.0, dtype=np.float64)
    targets = np.full((CALIBRATION_COUNT, 8), 80.0, dtype=np.float64)
    targets[:120, -1] = 90.0

    raw_prefix = risk_state.prefix_only_risk.decision_function(prefix)
    raw_visible = risk_state.visible_stress_risk.decision_function(
        np.column_stack((prefix, real_stress))
    )
    support_template = _two_family_support()
    support_rows = [
        {
            family: [vector.copy() for vector in vectors]
            for family, vectors in support_template.items()
        }
        for _ in range(CALIBRATION_COUNT)
    ]
    lower_row, upper_row = _support_band(support_template)
    lower = np.tile(lower_row, (CALIBRATION_COUNT, 1))
    upper = np.tile(upper_row, (CALIBRATION_COUNT, 1))
    baselines = {
        "target_prefix_persistence": np.full(
            (CALIBRATION_COUNT, 8), 80.0, dtype=np.float64
        ),
        "target_prefix_sqrt_time": np.full(
            (CALIBRATION_COUNT, 8), 81.0, dtype=np.float64
        ),
        "target_prefix_bounded_power_law": np.full(
            (CALIBRATION_COUNT, 8), 82.0, dtype=np.float64
        ),
    }
    return {
        "risk_state": risk_state,
        "cluster_ids": cluster_ids,
        "arm_a_predictor_content_sha256": np.asarray(
            [_digest(f"arm-a-{index}") for index in range(CALIBRATION_COUNT)],
            dtype=object,
        ),
        "arm_b_predictor_content_sha256": np.asarray(
            [_digest(f"arm-b-{index}") for index in range(CALIBRATION_COUNT)],
            dtype=object,
        ),
        "placebo_predictor_content_sha256": np.asarray(
            [_digest(f"placebo-{index}") for index in range(CALIBRATION_COUNT)],
            dtype=object,
        ),
        "prefix_days": np.tile(
            np.asarray(PREFIX_DAYS, dtype=np.float64),
            (CALIBRATION_COUNT, 1),
        ),
        "prefix_observations_pct": np.tile(
            np.linspace(100.0, 98.0, len(PREFIX_DAYS), dtype=np.float64),
            (CALIBRATION_COUNT, 1),
        ),
        "forecast_days": np.tile(
            np.asarray(FORECAST_DAYS, dtype=np.float64),
            (CALIBRATION_COUNT, 1),
        ),
        "real_operating_fields": real_operating,
        "placebo_operating_fields": placebo_operating,
        "real_stress_features": real_stress,
        "placebo_features": placebo_features,
        "successful_structure_family_count": family_counts,
        "structural_family_supports_pct": support_rows,
        "frozen_center_forecasts_pct": center,
        "prefix_features": prefix,
        "raw_prefix_risk_scores": raw_prefix,
        "raw_visible_risk_scores": raw_visible,
        "base_interval_lower_pct": lower,
        "base_interval_upper_pct": upper,
        "mean_baseline_forecasts_pct": baselines,
        "latent_target_cluster_ids": cluster_ids.copy(),
        "latent_target_forecast_days": np.tile(
            np.asarray(FORECAST_DAYS, dtype=np.float64),
            (CALIBRATION_COUNT, 1),
        ),
        "latent_targets_pct": targets,
    }


def _pretruth_kwargs(inputs: dict[str, Any]) -> dict[str, Any]:
    truth_inputs = {
        "latent_target_cluster_ids",
        "latent_target_forecast_days",
        "latent_targets_pct",
    }
    return {key: value for key, value in inputs.items() if key not in truth_inputs}


def _commit(inputs: dict[str, Any]) -> training.V021PretruthMaskCommitment:
    return training.derive_calibration_mask_commitment_v021(**_pretruth_kwargs(inputs))


def _fit(
    inputs: dict[str, Any],
    *,
    commitment: training.V021PretruthMaskCommitment | None = None,
) -> tuple[CalibrationDevelopmentState, training.V021CalibrationAudit]:
    frozen = _commit(inputs) if commitment is None else commitment
    return training.fit_calibration_development_state_v021(
        pretruth_commitment=frozen,
        **inputs,
    )


def _set_one_family(inputs: dict[str, Any], index: int) -> None:
    support = {
        DECLARED_STRUCTURE_FAMILIES[6]: [
            [78.0 + horizon / 100.0 for horizon in range(8)]
        ]
    }
    inputs["structural_family_supports_pct"][index] = support
    inputs["successful_structure_family_count"][index] = 1
    inputs["prefix_features"][index, 0] = 1
    inputs["prefix_features"][index, 1] = 6
    lower, upper = _support_band(support)
    inputs["base_interval_lower_pct"][index] = lower
    inputs["base_interval_upper_pct"][index] = upper


def test_risk_scores_are_bit_exact_for_rowwise_and_batch_evaluation() -> None:
    model = _cancellation_state(PREFIX_FEATURE_NAMES)
    rows = np.vstack(
        (
            np.ones(len(PREFIX_FEATURE_NAMES), dtype=np.float64),
            np.arange(1, len(PREFIX_FEATURE_NAMES) + 1, dtype=np.float64),
            np.linspace(-1.0, 1.0, len(PREFIX_FEATURE_NAMES), dtype=np.float64),
        )
    )

    rowwise = np.asarray([model.decision_function((row,))[0] for row in rows])
    batched = model.decision_function(rows)

    assert np.array_equal(rowwise, batched)


def _v024_cancellation_inputs() -> dict[str, Any]:
    inputs = _calibration_inputs()
    risk_state = replace(
        inputs["risk_state"],
        prefix_only_risk=_cancellation_state(PREFIX_FEATURE_NAMES),
        visible_stress_risk=_cancellation_state(VISIBLE_STRESS_FEATURE_NAMES),
    )
    inputs["risk_state"] = risk_state
    inputs["raw_prefix_risk_scores"] = np.asarray(
        [
            risk_state.prefix_only_risk.decision_function((row,))[0]
            for row in inputs["prefix_features"]
        ]
    )
    visible = np.column_stack(
        (inputs["prefix_features"], inputs["real_stress_features"])
    )
    inputs["raw_visible_risk_scores"] = np.asarray(
        [risk_state.visible_stress_risk.decision_function((row,))[0] for row in visible]
    )
    return inputs


def test_v024_primary_score_verifier_and_commitment_are_reproducible() -> None:
    inputs = _v024_cancellation_inputs()
    v019_training._verify_primary_scores(
        risk_state=inputs["risk_state"],
        prefix=inputs["prefix_features"],
        real_stress=inputs["real_stress_features"],
        supplied_prefix=inputs["raw_prefix_risk_scores"],
        supplied_visible=inputs["raw_visible_risk_scores"],
    )

    raw = serialize_calibration_mask_commitment_json_v024(
        v019_training.derive_calibration_mask_commitment_v024(
            **_pretruth_kwargs(inputs)
        )
    )
    repeated = serialize_calibration_mask_commitment_json_v024(
        v019_training.derive_calibration_mask_commitment_v024(
            **_pretruth_kwargs(inputs)
        )
    )

    assert raw == repeated
    assert hashlib.sha256(raw).digest() == hashlib.sha256(repeated).digest()


@pytest.mark.parametrize("mutation", ("row", "value", "nan", "inf", "-inf"))
def test_v024_primary_score_mutations_fail_closed(mutation: str) -> None:
    inputs = _v024_cancellation_inputs()
    supplied = inputs["raw_prefix_risk_scores"].copy()
    if mutation == "row":
        supplied[[0, 1]] = supplied[[1, 0]]
    elif mutation == "value":
        supplied[0] = np.nextafter(supplied[0], np.inf)
    else:
        supplied[0] = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[mutation]

    with pytest.raises(v019_training.V024CalibrationError, match="risk-state"):
        v019_training._verify_primary_scores(
            risk_state=inputs["risk_state"],
            prefix=inputs["prefix_features"],
            real_stress=inputs["real_stress_features"],
            supplied_prefix=supplied,
            supplied_visible=inputs["raw_visible_risk_scores"],
        )


def test_v024_calibration_key_mismatch_fails_closed() -> None:
    inputs = _v024_cancellation_inputs()
    inputs["cluster_ids"][1] = inputs["cluster_ids"][0]

    with pytest.raises(v019_training.V024CalibrationError, match="unique"):
        v019_training.derive_calibration_mask_commitment_v024(
            **_pretruth_kwargs(inputs)
        )


def _set_zero_family(inputs: dict[str, Any], index: int) -> None:
    inputs["structural_family_supports_pct"][index] = {}
    inputs["successful_structure_family_count"][index] = 0
    inputs["prefix_features"][index, 0] = 0
    inputs["prefix_features"][index, 1] = 7
    inputs["base_interval_lower_pct"][index] = np.nan
    inputs["base_interval_upper_pct"][index] = np.nan


def _permuted(inputs: dict[str, Any], order: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"risk_state": inputs["risk_state"]}
    for key, value in inputs.items():
        if key == "risk_state":
            continue
        if key == "structural_family_supports_pct":
            result[key] = [value[int(index)] for index in order]
        elif key == "mean_baseline_forecasts_pct":
            result[key] = {
                model_id: forecasts[order] for model_id, forecasts in value.items()
            }
        else:
            result[key] = value[order]
    return result


def test_pretruth_api_is_truth_free_and_caller_cannot_assert_mask() -> None:
    derive_parameters = set(
        inspect.signature(training.derive_calibration_mask_commitment_v021).parameters
    )
    fit_parameters = set(
        inspect.signature(training.fit_calibration_development_state_v021).parameters
    )
    forbidden = ("truth", "target", "label", "outcome", "catastrophic", "future")
    assert not any(
        token in parameter.lower()
        for parameter in derive_parameters
        for token in forbidden
    )
    assert {"hard_eligible", "hard_ineligibility_reasons"}.isdisjoint(
        derive_parameters | fit_parameters
    )
    assert {
        "cluster_ids",
        "arm_a_predictor_content_sha256",
        "arm_b_predictor_content_sha256",
        "placebo_predictor_content_sha256",
        "prefix_days",
        "prefix_observations_pct",
        "forecast_days",
        "real_operating_fields",
        "placebo_operating_fields",
        "real_stress_features",
        "placebo_features",
        "successful_structure_family_count",
        "structural_family_supports_pct",
        "frozen_center_forecasts_pct",
        "prefix_features",
        "raw_prefix_risk_scores",
        "raw_visible_risk_scores",
    }.issubset(derive_parameters)

    inputs = _calibration_inputs()
    missing = _pretruth_kwargs(inputs)
    missing.pop("placebo_features")
    with pytest.raises(TypeError, match="placebo_features"):
        training.derive_calibration_mask_commitment_v021(**missing)

    before = _commit(inputs)
    inputs["latent_targets_pct"][:] = -999.0
    assert _commit(inputs) == before


def test_commitment_is_immutable_and_binds_all_900_unique_ids() -> None:
    commitment = _commit(_calibration_inputs())
    assert commitment.source_calibration_count == 900
    assert commitment.eligible_count == 900
    assert len(commitment.rows) == 900
    assert len({row.cluster_id for row in commitment.rows}) == 900
    assert (
        training.calibration_eligibility_mask_sha256_v021(commitment)
        == commitment.eligibility_mask_sha256
    )
    assert commitment.canonical_bytes().endswith(b"\n")
    assert (
        hashlib.sha256(commitment.canonical_bytes()).hexdigest()
        == commitment.canonical_byte_sha256
    )
    with pytest.raises(FrozenInstanceError):
        commitment.eligibility_mask_sha256 = "0" * 64  # type: ignore[misc]


def test_duplicate_cluster_identity_is_rejected() -> None:
    inputs = _calibration_inputs()
    inputs["cluster_ids"][1] = inputs["cluster_ids"][0]
    with pytest.raises(training.V021CalibrationError, match="unique"):
        _commit(inputs)


def test_identity_binding_blocks_duplicate_row_mask_exchange_attack() -> None:
    left = _calibration_inputs()
    right = _calibration_inputs()
    for inputs in (left, right):
        for key in (
            "arm_a_predictor_content_sha256",
            "arm_b_predictor_content_sha256",
            "placebo_predictor_content_sha256",
        ):
            inputs[key][899] = inputs[key][898]
        for key in (
            "prefix_days",
            "prefix_observations_pct",
            "forecast_days",
            "real_operating_fields",
            "placebo_operating_fields",
            "real_stress_features",
            "placebo_features",
            "frozen_center_forecasts_pct",
            "prefix_features",
            "raw_prefix_risk_scores",
            "raw_visible_risk_scores",
            "base_interval_lower_pct",
            "base_interval_upper_pct",
        ):
            inputs[key][899] = inputs[key][898]
        for forecasts in inputs["mean_baseline_forecasts_pct"].values():
            forecasts[899] = forecasts[898]
        inputs["structural_family_supports_pct"][899] = {
            family: [vector.copy() for vector in vectors]
            for family, vectors in inputs["structural_family_supports_pct"][898].items()
        }
    _set_one_family(left, 898)
    _set_one_family(right, 899)

    left_commitment = _commit(left)
    right_commitment = _commit(right)
    assert left_commitment.eligible_count == right_commitment.eligible_count == 899
    assert left_commitment.eligibility_mask_sha256 != (
        right_commitment.eligibility_mask_sha256
    )
    with pytest.raises(training.V021CalibrationError, match="commitment differs"):
        _fit(right, commitment=left_commitment)


def test_one_family_uses_genuine_support_and_stays_in_full_conformal_pool() -> None:
    inputs = _calibration_inputs()
    _set_one_family(inputs, 17)
    commitment = _commit(inputs)
    committed = {row.cluster_id: row for row in commitment.rows}
    row = committed[str(inputs["cluster_ids"][17])]
    assert row.eligible is False
    assert row.successful_structure_family_ids == (DECLARED_STRUCTURE_FAMILIES[6],)
    assert row.ineligibility_reasons == ("insufficient_structure_families",)

    state, audit = _fit(inputs, commitment=commitment)
    assert audit.risk_isotonic_ineligible_one_family_count == 1
    assert audit.risk_isotonic_eligible_count == 899
    assert state.conformal.calibration_count == 900
    assert state.conformal.order_statistic_index == 811


def test_one_family_sqrt_band_masquerade_is_rejected() -> None:
    inputs = _calibration_inputs()
    _set_one_family(inputs, 17)
    sqrt_band = inputs["mean_baseline_forecasts_pct"]["target_prefix_sqrt_time"][17]
    inputs["base_interval_lower_pct"][17] = sqrt_band
    inputs["base_interval_upper_pct"][17] = sqrt_band
    with pytest.raises(training.V021CalibrationError, match="support quantiles"):
        _commit(inputs)


def test_zero_family_is_terminal_before_truth_fit() -> None:
    inputs = _calibration_inputs()
    _set_zero_family(inputs, 23)
    with pytest.raises(training.V021CalibrationTerminalInconclusive) as captured:
        _commit(inputs)
    assert captured.value.reason_code == "CALIBRATION_ZERO_FAMILY_NO_BAND"
    assert captured.value.offending_row_indices == (23,)


def test_changed_label_free_evidence_cannot_reuse_pretruth_commitment() -> None:
    inputs = _calibration_inputs()
    commitment = _commit(inputs)
    inputs["placebo_features"][899, 0] = np.nan
    with pytest.raises(training.V021CalibrationError, match="commitment differs"):
        _fit(inputs, commitment=commitment)


@pytest.mark.parametrize(
    ("eligible_count", "expected_reason"),
    (
        (854, "CALIBRATION_RISK_ELIGIBLE_BELOW_855"),
        (855, None),
    ),
)
def test_eligibility_boundary_is_exactly_854_vs_855(
    eligible_count: int,
    expected_reason: str | None,
) -> None:
    inputs = _calibration_inputs()
    inputs["placebo_features"][eligible_count:, 0] = np.nan
    if expected_reason is None:
        commitment = _commit(inputs)
        assert commitment.eligible_count == 855
    else:
        with pytest.raises(training.V021CalibrationTerminalInconclusive) as captured:
            _commit(inputs)
        assert captured.value.reason_code == expected_reason


def test_class_count_is_enforced_only_after_pretruth_commitment() -> None:
    inputs = _calibration_inputs()
    inputs["placebo_features"][855:, 0] = np.nan
    commitment = _commit(inputs)
    inputs["latent_targets_pct"][:, -1] = 80.0
    inputs["latent_targets_pct"][:59, -1] = 90.0
    inputs["latent_targets_pct"][899, -1] = 90.0

    with pytest.raises(training.V021CalibrationTerminalInconclusive) as captured:
        _fit(inputs, commitment=commitment)
    assert captured.value.reason_code == "CALIBRATION_RISK_POSITIVE_BELOW_60"
    assert "positive=59" in str(captured.value)


def test_source_count_other_than_900_is_typed_terminal() -> None:
    inputs = _calibration_inputs()
    inputs["prefix_days"] = inputs["prefix_days"][:-1]
    with pytest.raises(training.V021CalibrationTerminalInconclusive) as captured:
        _commit(inputs)
    assert captured.value.reason_code == "CALIBRATION_SOURCE_COUNT_NOT_900"


def test_raw_primary_scores_are_reproduced_and_bound() -> None:
    inputs = _calibration_inputs()
    inputs["raw_prefix_risk_scores"][8] += 0.25
    with pytest.raises(training.V021CalibrationError, match="risk-state"):
        _commit(inputs)


def test_finite_reproducible_score_cannot_be_downgraded_to_nan() -> None:
    inputs = _calibration_inputs()
    inputs["raw_prefix_risk_scores"][8] = np.nan
    with pytest.raises(training.V021CalibrationError, match="risk-state"):
        _commit(inputs)


def test_nonfinite_band_is_typed_terminal() -> None:
    inputs = _calibration_inputs()
    inputs["base_interval_lower_pct"][23, 4] = np.nan
    with pytest.raises(training.V021CalibrationTerminalInconclusive) as captured:
        _commit(inputs)
    assert captured.value.reason_code == "CALIBRATION_BAND_NONFINITE_OR_UNORDERED"


def test_state_commitment_and_fit_are_permutation_invariant() -> None:
    inputs = _calibration_inputs()
    _set_one_family(inputs, 5)
    commitment = _commit(inputs)
    expected_state, expected_audit = _fit(inputs, commitment=commitment)
    order = np.random.default_rng(20260726).permutation(CALIBRATION_COUNT)
    permuted = _permuted(inputs, order)

    observed_commitment = _commit(permuted)
    observed_state, observed_audit = _fit(
        permuted,
        commitment=commitment,
    )
    assert observed_commitment == commitment
    assert observed_state == expected_state
    assert observed_audit == expected_audit


def test_truth_rows_are_joined_by_id_not_by_position() -> None:
    inputs = _calibration_inputs()
    commitment = _commit(inputs)
    expected_state, expected_audit = _fit(inputs, commitment=commitment)
    order = np.random.default_rng(20260727).permutation(CALIBRATION_COUNT)
    inputs["latent_target_cluster_ids"] = inputs["latent_target_cluster_ids"][order]
    inputs["latent_target_forecast_days"] = inputs["latent_target_forecast_days"][order]
    inputs["latent_targets_pct"] = inputs["latent_targets_pct"][order]

    observed_state, observed_audit = _fit(inputs, commitment=commitment)

    assert observed_state == expected_state
    assert observed_audit == expected_audit


def test_truth_join_rejects_duplicate_missing_id_and_wrong_grid() -> None:
    duplicate = _calibration_inputs()
    commitment = _commit(duplicate)
    duplicate["latent_target_cluster_ids"][1] = duplicate["latent_target_cluster_ids"][
        0
    ]
    with pytest.raises(training.V021CalibrationError, match="unique"):
        _fit(duplicate, commitment=commitment)

    wrong_grid = _calibration_inputs()
    commitment = _commit(wrong_grid)
    wrong_grid["latent_target_forecast_days"][7, -1] += 1.0
    with pytest.raises(training.V021CalibrationError, match="exact forecast grid"):
        _fit(wrong_grid, commitment=commitment)


def test_audit_count_registry_remains_exact() -> None:
    assert {field.name for field in fields(training.V021CalibrationAudit)} == {
        "source_calibration_count",
        "risk_isotonic_eligible_count",
        "risk_isotonic_ineligible_zero_family_count",
        "risk_isotonic_ineligible_one_family_count",
        "risk_isotonic_ineligible_other_count",
        "risk_isotonic_positive_label_count",
        "risk_isotonic_negative_label_count",
        "mean_baseline_count",
        "conformal_calibration_count",
        "conformal_order_statistic_index",
        "eligibility_mask_sha256",
    }
