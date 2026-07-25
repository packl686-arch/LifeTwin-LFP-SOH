from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from lifetwin.experiments import calendar_long_horizon_v015_protocol as protocol_v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "synthetic_long_horizon_identifiability_v2.json"
)


@pytest.fixture()
def protocol() -> protocol_v2.ValidatedV015Protocol:
    return protocol_v2.load_frozen_protocol_config(CONFIG_PATH)


def _neutral_operating() -> protocol_v2.OperatingCovariates:
    return protocol_v2.OperatingCovariates(
        past_mean_temperature_c=27.5,
        past_mean_soc_fraction=0.55,
        past_mean_dod_fraction=0.55,
        past_efc_per_year=275.0,
        planned_mean_temperature_c=27.5,
        planned_mean_soc_fraction=0.55,
        planned_mean_dod_fraction=0.55,
        planned_efc_per_year=275.0,
    )


def _operating_with_plan(
    planned: tuple[float, float, float, float],
) -> protocol_v2.OperatingCovariates:
    return protocol_v2.OperatingCovariates(
        past_mean_temperature_c=27.5,
        past_mean_soc_fraction=0.55,
        past_mean_dod_fraction=0.55,
        past_efc_per_year=275.0,
        planned_mean_temperature_c=planned[0],
        planned_mean_soc_fraction=planned[1],
        planned_mean_dod_fraction=planned[2],
        planned_efc_per_year=planned[3],
        placebo_controls=(0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4, -0.4),
    )


def test_frozen_config_commitments_counts_and_schema(
    protocol: protocol_v2.ValidatedV015Protocol,
) -> None:
    raw = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == protocol_v2.FROZEN_CONFIG_BYTE_SHA256
    assert protocol.config_sha256 == protocol_v2.FROZEN_CONFIG_CANONICAL_SHA256
    assert protocol.protocol_id == protocol_v2.FROZEN_PROTOCOL_ID
    assert protocol.prefix_days[-1] == 730.0
    assert protocol.forecast_days[-1] == 9131.25
    assert tuple(protocol.family_map()) == protocol_v2.TRUTH_FAMILY_IDS
    assert (
        sum(
            sum(family_counts.values())
            for family_counts in protocol.partition_count_map().values()
        )
        == 4950
    )
    assert len(protocol.seed_root_map()) == 13
    assert len(set(protocol.seed_root_map().values())) == 13


def test_semantic_validation_rejects_count_schema_and_seed_drift(
    protocol: protocol_v2.ValidatedV015Protocol,
) -> None:
    changed_count = protocol.config()
    changed_count["design_partitions"]["test"]["total_clusters"] = 1901
    with pytest.raises(protocol_v2.V015ProtocolError, match="total"):
        protocol_v2.validate_protocol_config(changed_count)

    changed_schema = protocol.config()
    changed_schema["firewall_and_artifacts"]["artifact_schemas"]["prefix_pack.csv"][
        "columns"
    ][-1] = "changed_column"
    with pytest.raises(protocol_v2.V015ProtocolError, match="prefix schema"):
        protocol_v2.validate_protocol_config(changed_schema)

    changed_seed = protocol.config()
    roots = changed_seed["design_partitions"]["seed_roots"]
    roots["audit"] = roots["test"]
    with pytest.raises(protocol_v2.V015ProtocolError, match="unique"):
        protocol_v2.validate_protocol_config(changed_seed)

    changed_seed_value = protocol.config()
    changed_seed_value["design_partitions"]["seed_roots"]["audit"] += 1000
    with pytest.raises(protocol_v2.V015ProtocolError, match="seed-root values"):
        protocol_v2.validate_protocol_config(changed_seed_value)


def test_generic_seed_derivation_matches_declared_hash_rule() -> None:
    material = (
        "fixture_protocol|123456|fixture_partition|fixture_family|7|fixture_stream"
    )
    expected = int(hashlib.sha256(material.encode()).hexdigest()[:16], 16) % (2**63 - 1)
    observed = protocol_v2.derive_stream_seed(
        "fixture_protocol",
        123456,
        "fixture_partition",
        "fixture_family",
        7,
        "fixture_stream",
    )
    assert observed == expected
    assert (
        protocol_v2.derive_stream_seed(
            "fixture_protocol",
            123457,
            "fixture_partition",
            "fixture_family",
            7,
            "fixture_stream",
        )
        != observed
    )


@pytest.mark.parametrize(
    ("family", "parameters"),
    [
        ("single_power", {"a": 0.5, "b": 0.6}),
        (
            "dual_power",
            {"a1": 0.4, "b1": 0.5, "a2": 0.1, "b2": 1.1},
        ),
        (
            "saturating_plus_slow",
            {
                "a_sat": 2.0,
                "tau_sat_days": 300.0,
                "b_sat": 1.0,
                "a_slow": 0.2,
                "b_slow": 0.5,
            },
        ),
        (
            "early_activation_plus_power",
            {
                "a": 0.5,
                "b": 0.6,
                "activation_amplitude_pp": 0.7,
                "tau_rise_days": 10.0,
                "tau_decay_days": 200.0,
            },
        ),
        (
            "late_knee",
            {
                "a": 0.5,
                "b": 0.6,
                "k_pp_per_day": 0.002,
                "t_knee_days": 2500.0,
                "w_days": 90.0,
            },
        ),
        (
            "linear_drift_plus_power",
            {"a": 0.4, "b": 0.5, "c": 0.1},
        ),
        (
            "smooth_broken_power",
            {
                "a": 0.4,
                "b_early": 0.5,
                "b_late": 1.2,
                "transition_tau_days": 2000.0,
                "sharpness": 4.0,
            },
        ),
        (
            "saturating_logistic_knee",
            {
                "a": 0.4,
                "b": 0.5,
                "knee_amplitude_pp": 10.0,
                "t_knee_days": 2500.0,
                "w_days": 150.0,
            },
        ),
    ],
)
def test_all_eight_truth_families_are_finite_and_day_zero_normalized(
    family: str,
    parameters: dict[str, float],
) -> None:
    days = np.array([0.0, 730.0, 9131.25])
    loss = protocol_v2.evaluate_base_loss(family, parameters, days)
    retention = protocol_v2.evaluate_truth_retention(
        family,
        parameters,
        _neutral_operating(),
        0.1,
        days,
    )
    assert loss[0] == 0.0
    assert retention[0] == 100.0
    assert np.isfinite(loss).all()
    assert np.isfinite(retention).all()


def test_stable_transforms_handle_large_scalar_and_vector_inputs() -> None:
    np.testing.assert_allclose(
        protocol_v2.stable_sigmoid(np.array([-1000.0, 0.0, 1000.0])),
        np.array([0.0, 0.5, 1.0]),
        atol=1e-15,
    )
    assert protocol_v2.stable_sigmoid(-1000.0).item() == 0.0
    np.testing.assert_allclose(
        protocol_v2.stable_softplus(np.array([-1000.0, 1000.0])),
        np.array([0.0, 1000.0]),
        atol=1e-15,
    )


def test_operating_stress_is_causal_at_the_prefix_boundary() -> None:
    days = np.array([0.0, 365.0, 730.0, 1095.75, 9131.25])
    parameters = {"a": 0.5, "b": 0.6}
    neutral = _neutral_operating()
    hot_plan = _operating_with_plan((40.0, 0.9, 0.9, 450.0))
    neutral_curve = protocol_v2.evaluate_truth_retention(
        "single_power", parameters, neutral, 0.2, days
    )
    hot_curve = protocol_v2.evaluate_truth_retention(
        "single_power", parameters, hot_plan, 0.2, days
    )
    np.testing.assert_array_equal(neutral_curve[:3], hot_curve[:3])
    assert np.all(hot_curve[3:] < neutral_curve[3:])
    assert protocol_v2.stress_index(15.0, 0.2, 0.2, 100.0) == -1.0
    assert protocol_v2.stress_index(27.5, 0.55, 0.55, 275.0) == 0.0
    assert protocol_v2.stress_index(40.0, 0.9, 0.9, 450.0) == 1.0


def test_truth_admissibility_enforces_support_change_and_monotonicity(
    protocol: protocol_v2.ValidatedV015Protocol,
) -> None:
    decreasing = np.linspace(100.0, 80.0, len(protocol.combined_days))
    assert protocol_v2.truth_is_admissible(protocol, "single_power", decreasing)

    upward = decreasing.copy()
    upward[4] = upward[3] + 0.5
    assert not protocol_v2.truth_is_admissible(protocol, "single_power", upward)
    assert protocol_v2.truth_is_admissible(
        protocol, "early_activation_plus_power", upward
    )

    abrupt = decreasing.copy()
    abrupt[5] = abrupt[4] - 12.1
    assert not protocol_v2.truth_is_admissible(
        protocol, "early_activation_plus_power", abrupt
    )


def test_ar1_noise_recurrence_and_day_zero_normalization() -> None:
    innovations = np.array([1.0, 0.0, -1.0])
    errors = protocol_v2.ar1_observation_errors(0.2, 0.5, innovations)
    expected = np.array(
        [
            0.2,
            0.1,
            0.05 - 0.2 * math.sqrt(1.0 - 0.5**2),
        ]
    )
    np.testing.assert_allclose(errors, expected)
    observed = protocol_v2.apply_observation_noise(
        np.array([100.0, 99.0, 98.0]),
        protocol_v2.ObservationNoise(
            sigma_pp=0.2,
            rho=0.5,
            errors_pp=tuple(errors),
        ),
    )
    assert observed[0] == 100.0
    np.testing.assert_allclose(observed[1:], np.array([98.9, 97.67679491924312]))


@pytest.mark.parametrize(
    ("mechanism", "mechanism_parameters"),
    [
        (
            "piecewise_linear_knee",
            {"k_pp_per_day": 0.002, "t_knee_days": 2000.0},
        ),
        (
            "compact_smoothstep",
            {
                "amplitude_pp": 6.75,
                "t_start_days": 1500.0,
                "duration_days": 800.0,
            },
        ),
    ],
)
def test_intrinsic_pair_hand_fixtures_have_identical_prefixes(
    protocol: protocol_v2.ValidatedV015Protocol,
    mechanism: str,
    mechanism_parameters: dict[str, float],
) -> None:
    left, right = protocol_v2.evaluate_intrinsic_pair_retention(
        {"a": 0.4, "b": 0.5},
        _neutral_operating(),
        0.1,
        protocol.combined_days,
        mechanism=mechanism,
        mechanism_parameters=mechanism_parameters,
    )
    prefix_count = len(protocol.prefix_days)
    np.testing.assert_array_equal(
        left[:prefix_count],
        right[:prefix_count],
    )
    assert abs(left[-1] - right[-1]) >= 5.0
    assert np.isfinite(left).all()
    assert np.isfinite(right).all()


def test_stress_plan_pair_hand_fixture_shares_only_the_visible_prefix(
    protocol: protocol_v2.ValidatedV015Protocol,
) -> None:
    low_plan = _operating_with_plan((20.0, 0.3, 0.3, 150.0))
    high_plan = _operating_with_plan((35.0, 0.8, 0.8, 400.0))
    low, high = protocol_v2.evaluate_stress_plan_pair_retention(
        "single_power",
        {"a": 0.5, "b": 0.6},
        low_plan,
        high_plan,
        0.2,
        protocol.combined_days,
    )
    prefix_count = len(protocol.prefix_days)
    np.testing.assert_array_equal(low[:prefix_count], high[:prefix_count])
    assert np.all(high[prefix_count:] < low[prefix_count:])


def test_operating_covariates_reject_bad_hand_fixtures() -> None:
    with pytest.raises(protocol_v2.V015ProtocolError, match="sixteen finite"):
        protocol_v2.OperatingCovariates(
            27.5,
            0.55,
            0.55,
            275.0,
            27.5,
            0.55,
            0.55,
            275.0,
            placebo_controls=(0.0,) * 7,
        )
    with pytest.raises(protocol_v2.V015ProtocolError, match="sixteen finite"):
        protocol_v2.OperatingCovariates(
            27.5,
            0.55,
            0.55,
            275.0,
            float("nan"),
            0.55,
            0.55,
            275.0,
        )


def test_validation_inputs_are_copied_from_the_fixture(
    protocol: protocol_v2.ValidatedV015Protocol,
) -> None:
    first = protocol.config()
    second = deepcopy(first)
    first["protocol_id"] = "mutated_fixture"
    assert second["protocol_id"] == protocol_v2.FROZEN_PROTOCOL_ID
    assert protocol.protocol_id == protocol_v2.FROZEN_PROTOCOL_ID
