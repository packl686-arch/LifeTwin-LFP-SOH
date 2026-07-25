from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import inspect

import numpy as np
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_json_bytes,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    PREFIX_FEATURE_NAMES,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_CONFIG_BYTE_SHA256,
    FROZEN_PROTOCOL_ID,
)
from lifetwin.experiments.calendar_long_horizon_v015_training import (
    CALIBRATION_COUNT,
    CENTER_COMPLETENESS_INTERPRETATION,
    CENTER_DEVELOPMENT_COUNT,
    CONFORMAL_ORDER_STATISTIC_INDEX,
    MEAN_BASELINE_IDS,
    RISK_DEVELOPMENT_COUNT,
    CalibrationDevelopmentState,
    CenterDevelopmentState,
    FrozenTrainingState,
    RiskDevelopmentState,
    V015TrainingError,
    build_calibration_manifest,
    build_model_state_payload,
    build_training_manifest,
    construct_frozen_label_free_state,
    center_state_sha256,
    default_software_versions,
    deserialize_model_state_json,
    fit_calibration_development_state,
    fit_center_development_state,
    fit_risk_development_state,
    make_probe_state,
    risk_state_sha256,
    serialize_model_state_json,
    validate_calibration_manifest,
    validate_training_manifest,
    verify_calibration_manifest_state_hashes,
    verify_training_manifest_state_hashes,
)


def _center_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row = np.linspace(98.0, 82.0, 8)
    sqrt = np.tile(row, (CENTER_DEVELOPMENT_COUNT, 1))
    library = sqrt - 2.0
    latent = sqrt - 1.0
    return library, sqrt, latent


def _risk_arrays() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260723)
    labels = np.r_[np.ones(300), np.zeros(300)]
    prefix = rng.normal(size=(RISK_DEVELOPMENT_COUNT, len(PREFIX_FEATURE_NAMES)))
    prefix[:, 0] = labels + 2.0
    prefix[:, 1] = 5.0 - labels
    prefix[:, 2] = labels
    prefix[:, 3] = 4.0
    visible = rng.normal(size=(RISK_DEVELOPMENT_COUNT, 8))
    placebo = rng.normal(size=(RISK_DEVELOPMENT_COUNT, 8))
    planned = rng.normal(size=RISK_DEVELOPMENT_COUNT)
    center = np.full(RISK_DEVELOPMENT_COUNT, 90.0)
    latent = np.where(labels == 1, 84.0, 89.0)
    eligible = np.ones(RISK_DEVELOPMENT_COUNT, dtype=bool)
    return {
        "prefix_features": prefix,
        "visible_stress_features": visible,
        "placebo_features": placebo,
        "planned_stress_index": planned,
        "frozen_center_25y_pct": center,
        "latent_target_25y_pct": latent,
        "common_pool_eligible": eligible,
    }


def _calibration_arrays() -> dict[str, object]:
    rng = np.random.default_rng(20260724)
    labels = np.r_[np.ones(450), np.zeros(450)]
    prefix = rng.normal(size=(CALIBRATION_COUNT, len(PREFIX_FEATURE_NAMES)))
    prefix[:, 0] = labels + 2.0
    prefix[:, 1] = 5.0 - labels
    prefix[:, 2] = labels
    prefix[:, 3] = 4.0
    targets = np.full((CALIBRATION_COUNT, 8), 90.0)
    targets[:, -1] = np.where(labels == 1, 84.0, 89.0)
    center = np.full(CALIBRATION_COUNT, 90.0)
    lower = targets - 0.5
    upper = targets + 0.5
    baselines = {
        "target_prefix_persistence": targets + 1.0,
        "target_prefix_sqrt_time": targets + 2.0,
        "target_prefix_bounded_power_law": targets + 1.0,
    }
    return {
        "prefix_features": prefix,
        "visible_stress_features": rng.normal(size=(CALIBRATION_COUNT, 8)),
        "frozen_center_25y_pct": center,
        "latent_targets_pct": targets,
        "base_interval_lower_pct": lower,
        "base_interval_upper_pct": upper,
        "mean_baseline_forecasts_pct": baselines,
    }


@lru_cache(maxsize=1)
def _states() -> tuple[
    CenterDevelopmentState,
    RiskDevelopmentState,
    CalibrationDevelopmentState,
]:
    library, sqrt, latent = _center_fixture()
    center = fit_center_development_state(
        library_forecasts_pct=library,
        sqrt_forecasts_pct=sqrt,
        latent_targets_pct=latent,
    )
    risk = fit_risk_development_state(**_risk_arrays())
    calibration = fit_calibration_development_state(
        risk_state=risk, **_calibration_arrays()
    )
    return center, risk, calibration


def _hashes(letter: str) -> dict[str, str]:
    return {"fixture_array": letter * 64}


def _payload() -> dict[str, object]:
    center, risk, calibration = _states()
    return build_model_state_payload(
        FrozenTrainingState(center, risk, calibration),
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        calibration_input_hashes=_hashes("c"),
        software_versions=default_software_versions(),
        created_utc="2026-07-23T08:00:00Z",
    )


def test_center_fit_requires_exact_complete_600_by_8_and_records_choice() -> None:
    library, sqrt, latent = _center_fixture()
    state = fit_center_development_state(
        library_forecasts_pct=library,
        sqrt_forecasts_pct=sqrt,
        latent_targets_pct=latent,
    )
    assert state.beta == pytest.approx(0.49875311720698257)
    assert state.completeness_rule == CENTER_COMPLETENESS_INTERPRETATION

    with pytest.raises(V015TrainingError, match="shape"):
        fit_center_development_state(
            library_forecasts_pct=library[:-1],
            sqrt_forecasts_pct=sqrt[:-1],
            latent_targets_pct=latent[:-1],
        )
    bad = latent.copy()
    bad[0, 0] = np.nan
    with pytest.raises(V015TrainingError, match="finite"):
        fit_center_development_state(
            library_forecasts_pct=library,
            sqrt_forecasts_pct=sqrt,
            latent_targets_pct=bad,
        )


def test_risk_fit_uses_common_pool_and_fits_all_frozen_heads() -> None:
    arrays = _risk_arrays()
    arrays["common_pool_eligible"][:10] = False
    arrays["prefix_features"][:10, 3] = np.nan
    arrays["frozen_center_25y_pct"][:10] = np.nan
    state = fit_risk_development_state(**arrays)
    assert state.development_cluster_count == 600
    assert state.eligible_cluster_count == 590
    assert state.positive_label_count == 290
    assert state.negative_label_count == 300
    assert state.prefix_only_risk.feature_names == PREFIX_FEATURE_NAMES
    assert len(state.visible_stress_risk.feature_names) == 22
    assert len(state.placebo_risk.feature_names) == 22
    assert len(state.arm_a_plus_s_plan_risk.feature_names) == 15
    assert state.strongest_single_feature_name == PREFIX_FEATURE_NAMES[0]
    assert state.strongest_single_feature_orientation == 1
    assert state.strongest_single_feature_auroc == 1.0
    assert state.prefix_only_risk.standardizer.zero_variance[3] is True
    assert state.prefix_only_risk.coefficients[3] == 0.0


def test_risk_fit_rejects_nonfinite_eligible_and_small_class() -> None:
    nonfinite = _risk_arrays()
    nonfinite["visible_stress_features"][0, 0] = np.nan
    with pytest.raises(V015TrainingError, match="eligible pool"):
        fit_risk_development_state(**nonfinite)

    small = _risk_arrays()
    small["latent_target_25y_pct"][:] = 89.0
    small["latent_target_25y_pct"][:59] = 84.0
    with pytest.raises(V015TrainingError, match="at least 60"):
        fit_risk_development_state(**small)


def test_risk_fit_rejects_non_boolean_pool_and_wrong_row_count() -> None:
    arrays = _risk_arrays()
    arrays["common_pool_eligible"] = np.ones(600, dtype=np.int64)
    with pytest.raises(V015TrainingError, match="strict booleans"):
        fit_risk_development_state(**arrays)

    arrays = _risk_arrays()
    arrays["prefix_features"] = arrays["prefix_features"][:-1]
    with pytest.raises(V015TrainingError, match="shape"):
        fit_risk_development_state(**arrays)

    arrays = _risk_arrays()
    arrays["prefix_features"][0, 0] = 1.0
    with pytest.raises(V015TrainingError, match="family count"):
        fit_risk_development_state(**arrays)


def test_calibration_is_exact_900_selects_lexical_tie_and_uses_k811() -> None:
    state = fit_calibration_development_state(
        risk_state=_states()[1], **_calibration_arrays()
    )
    assert state.calibration_cluster_count == 900
    assert state.positive_label_count == 450
    assert state.negative_label_count == 450
    assert state.selected_mean_baseline == "target_prefix_bounded_power_law"
    assert set(state.baseline_iae_by_id()) == set(MEAN_BASELINE_IDS)
    assert state.conformal.calibration_count == 900
    assert state.conformal.order_statistic_index == CONFORMAL_ORDER_STATISTIC_INDEX
    assert state.conformal.expansion_pp == 0.0


def test_training_states_are_row_permutation_invariant() -> None:
    permutation_600 = np.random.default_rng(11).permutation(RISK_DEVELOPMENT_COUNT)
    library, sqrt, latent = _center_fixture()
    row_adjustment = np.linspace(0.0, 0.2, CENTER_DEVELOPMENT_COUNT)[:, None]
    library = library - row_adjustment
    latent = latent - 0.5 * row_adjustment
    center_first = fit_center_development_state(
        library_forecasts_pct=library,
        sqrt_forecasts_pct=sqrt,
        latent_targets_pct=latent,
    )
    center_permuted = fit_center_development_state(
        library_forecasts_pct=library[permutation_600],
        sqrt_forecasts_pct=sqrt[permutation_600],
        latent_targets_pct=latent[permutation_600],
    )
    assert center_state_sha256(center_first) == center_state_sha256(center_permuted)

    risk_inputs = _risk_arrays()
    risk_first = fit_risk_development_state(**risk_inputs)
    risk_permuted = fit_risk_development_state(
        **{name: values[permutation_600] for name, values in risk_inputs.items()}
    )
    assert risk_state_sha256(risk_first) == risk_state_sha256(risk_permuted)

    calibration_inputs = _calibration_arrays()
    permutation_900 = np.random.default_rng(12).permutation(CALIBRATION_COUNT)
    permuted_calibration: dict[str, object] = {}
    for name, values in calibration_inputs.items():
        if name == "mean_baseline_forecasts_pct":
            assert isinstance(values, dict)
            permuted_calibration[name] = {
                model_id: forecasts[permutation_900]
                for model_id, forecasts in values.items()
            }
        else:
            assert isinstance(values, np.ndarray)
            permuted_calibration[name] = values[permutation_900]
    calibration_first = fit_calibration_development_state(
        risk_state=risk_first,
        **calibration_inputs,
    )
    calibration_permuted = fit_calibration_development_state(
        risk_state=risk_first,
        **permuted_calibration,
    )
    assert calibration_first == calibration_permuted


def test_calibration_baseline_selection_uses_frozen_trapezoid_iae() -> None:
    arrays = _calibration_arrays()
    targets = arrays["latent_targets_pct"]
    mean_better_but_iae_worse = targets.copy()
    mean_better_but_iae_worse[:, 5] += 1.0
    mean_worse_but_iae_better = targets.copy()
    mean_worse_but_iae_better[:, 0] += 0.75
    mean_worse_but_iae_better[:, 7] += 0.75
    clearly_worse = targets + 3.0
    assert np.mean(np.abs(mean_better_but_iae_worse - targets)) < np.mean(
        np.abs(mean_worse_but_iae_better - targets)
    )
    arrays["mean_baseline_forecasts_pct"] = {
        "target_prefix_persistence": mean_better_but_iae_worse,
        "target_prefix_sqrt_time": mean_worse_but_iae_better,
        "target_prefix_bounded_power_law": clearly_worse,
    }
    state = fit_calibration_development_state(risk_state=_states()[1], **arrays)
    assert state.selected_mean_baseline == "target_prefix_sqrt_time"
    metrics = state.baseline_iae_by_id()
    assert metrics["target_prefix_sqrt_time"] < metrics["target_prefix_persistence"]


def test_calibration_rejects_incomplete_nonfinite_and_small_class() -> None:
    incomplete = _calibration_arrays()
    incomplete["prefix_features"] = incomplete["prefix_features"][:-1]
    with pytest.raises(V015TrainingError, match="shape"):
        fit_calibration_development_state(risk_state=_states()[1], **incomplete)

    nonfinite = _calibration_arrays()
    nonfinite["base_interval_upper_pct"][0, 0] = np.inf
    with pytest.raises(V015TrainingError, match="finite"):
        fit_calibration_development_state(risk_state=_states()[1], **nonfinite)

    small = _calibration_arrays()
    small["latent_targets_pct"][:, -1] = 89.0
    small["latent_targets_pct"][:59, -1] = 84.0
    with pytest.raises(V015TrainingError, match="at least 60"):
        fit_calibration_development_state(risk_state=_states()[1], **small)


def test_constructed_pipeline_state_is_fully_validated() -> None:
    center, risk, calibration = _states()
    state = construct_frozen_label_free_state(center, risk, calibration)
    assert state.center_beta == center.beta
    assert state.conformal.calibration_count == 900
    assert state.visible_stress_risk == risk.visible_stress_risk


def test_probe_state_is_valid_but_has_no_trained_signal() -> None:
    state = make_probe_state(1.0)
    assert state.center_beta == 1.0
    assert set(state.prefix_only_risk.coefficients) == {0.0}
    assert set(state.visible_stress_risk.coefficients) == {0.0}
    assert state.conformal.calibration_count == 900
    assert state.conformal.order_statistic_index == 811
    with pytest.raises(V015TrainingError, match="center_beta"):
        make_probe_state(float("nan"))


def test_model_state_codec_is_canonical_and_roundtrips_exactly() -> None:
    center, risk, calibration = _states()
    training = FrozenTrainingState(center, risk, calibration)
    raw = serialize_model_state_json(
        training,
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        calibration_input_hashes=_hashes("c"),
        software_versions=default_software_versions(),
        created_utc="2026-07-23T08:00:00Z",
    )
    decoded = deserialize_model_state_json(raw)
    assert decoded.training_state == training
    assert decoded.frozen_label_free_state.center_beta == center.beta
    assert raw == canonical_json_bytes(_payload())
    assert _payload()["config_sha256"] == FROZEN_CONFIG_BYTE_SHA256
    assert _payload()["protocol_id"] == FROZEN_PROTOCOL_ID


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda payload: payload["center_state"].__setitem__("unknown", 1),
            "keys changed",
        ),
        (
            lambda payload: payload["feature_orders"]["prefix_only"].reverse(),
            "feature_orders",
        ),
        (
            lambda payload: payload.__setitem__("config_sha256", "0" * 64),
            "config_sha256",
        ),
        (
            lambda payload: payload["risk_states"].__setitem__(
                "positive_label_count", 1
            ),
            "counts",
        ),
        (
            lambda payload: payload["software_versions"].__setitem__("numpy", "0.0.0"),
            "software_versions",
        ),
        (
            lambda payload: payload["software_versions"].__setitem__(
                "extra-package", "1.0"
            ),
            "software_versions",
        ),
    ],
)
def test_model_state_nested_tampering_is_rejected(mutation, message: str) -> None:
    payload = deepcopy(_payload())
    mutation(payload)
    with pytest.raises(V015TrainingError, match=message):
        deserialize_model_state_json(canonical_json_bytes(payload))


def test_model_state_rejects_noncanonical_bytes_and_nonfinite_values() -> None:
    raw = canonical_json_bytes(_payload())
    with pytest.raises(V015TrainingError, match="not canonical"):
        deserialize_model_state_json(raw.rstrip(b"\n"))

    payload = deepcopy(_payload())
    payload["center_state"]["beta"] = np.nan
    with pytest.raises(Exception):
        canonical_json_bytes(payload)


def test_manifest_builders_freeze_allowlists_hashes_and_opened_files() -> None:
    center, risk, calibration = _states()
    training = build_training_manifest(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=center,
        risk_state=risk,
        created_utc="2026-07-23T08:01:00Z",
    )
    validate_training_manifest(training)
    assert training["opened_truth_files"] == [
        "center_development_truth.csv",
        "risk_development_truth.csv",
    ]
    assert training["forbidden_v1_evidence_matches"] == []

    calibration_manifest = build_calibration_manifest(
        calibration_input_hashes=_hashes("c"),
        calibration_state=calibration,
        created_utc="2026-07-23T08:02:00Z",
    )
    validate_calibration_manifest(calibration_manifest)
    assert calibration_manifest["opened_truth_files"] == [
        "calibration_truth.csv",
        "center_development_truth.csv",
        "risk_development_truth.csv",
    ]
    assert (
        calibration_manifest["selected_mean_baseline"]
        == calibration.selected_mean_baseline
    )


def test_manifest_tampering_is_rejected() -> None:
    center, risk, calibration = _states()
    training = build_training_manifest(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=center,
        risk_state=risk,
        created_utc="2026-07-23T08:01:00Z",
    )
    training["opened_truth_files"].append("test_truth.csv")
    with pytest.raises(V015TrainingError, match="opened_truth_files"):
        validate_training_manifest(training)

    manifest = build_calibration_manifest(
        calibration_input_hashes=_hashes("c"),
        calibration_state=calibration,
        created_utc="2026-07-23T08:02:00Z",
    )
    manifest["calibration_input_hashes"]["fixture_array"] = "A" * 64
    with pytest.raises(V015TrainingError, match="lowercase SHA256"):
        validate_calibration_manifest(manifest)

    valid_training = build_training_manifest(
        center_development_input_hashes=_hashes("a"),
        risk_development_input_hashes=_hashes("b"),
        center_state=center,
        risk_state=risk,
        created_utc="2026-07-23T08:01:00Z",
    )
    valid_training["center_state_sha256"] = "d" * 64
    with pytest.raises(V015TrainingError, match="center state hash mismatch"):
        verify_training_manifest_state_hashes(
            valid_training, center_state=center, risk_state=risk
        )

    valid_calibration = build_calibration_manifest(
        calibration_input_hashes=_hashes("c"),
        calibration_state=calibration,
        created_utc="2026-07-23T08:02:00Z",
    )
    valid_calibration["conformal_state_sha256"] = "e" * 64
    with pytest.raises(V015TrainingError, match="conformal state hash mismatch"):
        verify_calibration_manifest_state_hashes(
            valid_calibration, calibration_state=calibration
        )


def test_public_fit_apis_have_only_explicit_array_level_inputs() -> None:
    forbidden = ("frame", "path", "family", "pair", "score_object", "truth")
    for function in (
        fit_center_development_state,
        fit_risk_development_state,
        fit_calibration_development_state,
    ):
        names = tuple(inspect.signature(function).parameters)
        assert not any(token in name for name in names for token in forbidden)
