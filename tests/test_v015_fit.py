from __future__ import annotations

import inspect
import json
import math

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_long_horizon_synthetic as v1
from lifetwin.experiments import calendar_long_horizon_v015_fit as fit
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    V015FitError,
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    canonical_csv_bytes,
    load_artifact_contract,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FORECAST_COORDINATE_COLUMNS,
    FROZEN_PROTOCOL_ID,
    PREFIX_COLUMNS,
)


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


def _early_curve(days: np.ndarray) -> np.ndarray:
    years = days / 365.25
    activation = 0.45 * (1.0 - np.exp(-days / 25.0)) * np.exp(-days / 260.0)
    return 100.0 - 0.75 * np.power(years, 0.58) + activation


def _fixture_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = _early_curve(np.asarray(PREFIX_DAYS, dtype=float))
    prefix = pd.DataFrame(
        {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": "test",
            "cluster_id": "hand-fixture-001",
            "prefix_day": PREFIX_DAYS,
            "observed_retention_pct": observed,
        },
        columns=PREFIX_COLUMNS,
    )
    coordinates = pd.DataFrame(
        {
            "protocol_id": FROZEN_PROTOCOL_ID,
            "partition": "test",
            "cluster_id": "hand-fixture-001",
            "forecast_day": FORECAST_DAYS,
        },
        columns=FORECAST_COORDINATE_COLUMNS,
    )
    return prefix, coordinates


@pytest.fixture(scope="module")
def fitted_fixture() -> tuple[pd.DataFrame, pd.DataFrame, fit.V015FitResult]:
    prefix, coordinates = _fixture_tables()
    return prefix, coordinates, fit_structure_library(prefix, coordinates)


def test_public_fit_api_has_only_the_two_label_free_inputs() -> None:
    parameters = tuple(inspect.signature(fit_structure_library).parameters)
    assert parameters == ("prefix_pack", "forecast_coordinates")
    forbidden = {"truth", "path", "family", "pair"}
    assert not any(
        token in parameter.lower() for parameter in parameters for token in forbidden
    )


def test_frozen_v1_config_is_double_hash_verified() -> None:
    protocol = fit._legacy_protocol()
    assert v1.FROZEN_CONFIG_BYTE_SHA256 == (
        "503ec964bb2015fe3460433749d1b0d79f89187fc3dcd1c3809f9d4da2ffc319"
    )
    assert protocol.config_sha256 == (
        "6ad1e6dc1caa089ce0b9ee2c4e739a56c44f42f65436294649261a7676d4e320"
    )


def test_frozen_v1_commitment_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit._legacy_protocol.cache_clear()
    monkeypatch.setattr(v1, "FROZEN_CONFIG_BYTE_SHA256", "0" * 64)
    with pytest.raises(V015FitError, match="commitment changed"):
        fit._legacy_protocol()
    fit._legacy_protocol.cache_clear()


def test_fit_emits_exactly_86_diagnostics_and_eight_forecasts_each(
    fitted_fixture: tuple[pd.DataFrame, pd.DataFrame, fit.V015FitResult],
) -> None:
    _, _, result = fitted_fixture
    diagnostics = result.member_fit_diagnostics
    forecasts = result.member_forecast_bundle
    assert len(diagnostics) == 86
    assert len(forecasts) == 86 * 8
    assert not diagnostics.duplicated(["model_id", "variant_id"]).any()
    assert set(forecasts.groupby(["model_id", "variant_id"]).size()) == {8}
    early = diagnostics.loc[
        diagnostics["model_id"].eq("target_prefix_early_activation_plus_power")
    ]
    assert len(early) == 1
    assert early.iloc[0]["fit_status"] == "succeeded"
    assert early.iloc[0]["credible_variant"]


def test_public_frozen_variant_universe_is_exact_and_validated() -> None:
    assert len(fit.FROZEN_VARIANT_KEYS) == 86
    assert len(fit.FROZEN_VARIANT_KEY_SET) == 86
    assert (
        "target_prefix_persistence",
        "persistence",
    ) in fit.FROZEN_VARIANT_KEY_SET
    assert (
        "target_prefix_late_knee_prior_grid",
        "k=0.004|t=7305|w=365",
    ) in fit.FROZEN_VARIANT_KEY_SET
    fit.validate_frozen_variant_keys(fit.FROZEN_VARIANT_KEYS)

    with pytest.raises(V015FitError, match="exact 86"):
        fit.validate_frozen_variant_keys(fit.FROZEN_VARIANT_KEYS[:-1])
    with pytest.raises(V015FitError, match="duplicate"):
        fit.validate_frozen_variant_keys(
            (*fit.FROZEN_VARIANT_KEYS[:-1], fit.FROZEN_VARIANT_KEYS[0])
        )


def test_fit_tables_satisfy_the_frozen_artifact_schemas(
    fitted_fixture: tuple[pd.DataFrame, pd.DataFrame, fit.V015FitResult],
) -> None:
    _, _, result = fitted_fixture
    contract = load_artifact_contract()
    diagnostic_bytes = canonical_csv_bytes(
        result.member_fit_diagnostics,
        contract.csv_schema("member_fit_diagnostics.csv"),
        contract,
        formal=False,
    )
    forecast_bytes = canonical_csv_bytes(
        result.member_forecast_bundle,
        contract.csv_schema("member_forecast_bundle.csv"),
        contract,
        formal=False,
    )
    assert diagnostic_bytes.startswith(b"protocol_id,partition,cluster_id")
    assert forecast_bytes.startswith(b"protocol_id,partition,cluster_id")
    for payload in result.member_fit_diagnostics["parameters_json"]:
        parsed = json.loads(payload)
        assert payload == json.dumps(
            parsed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )


def test_formula_recomputation_reproduces_every_successful_commitment(
    fitted_fixture: tuple[pd.DataFrame, pd.DataFrame, fit.V015FitResult],
) -> None:
    prefix, coordinates, result = fitted_fixture
    forecasts = result.member_forecast_bundle
    for row in result.member_fit_diagnostics.itertuples(index=False):
        if row.fit_status != "succeeded":
            continue
        parameters = fit.parse_canonical_parameters_json(row.parameters_json)
        recomputed = fit.recompute_variant_commitment(
            model_id=row.model_id,
            variant_id=row.variant_id,
            parameters=parameters,
            prefix_days=prefix["prefix_day"],
            observed_retention_pct=prefix["observed_retention_pct"],
            forecast_days=coordinates["forecast_day"],
        )
        committed_forecast = forecasts.loc[
            forecasts["model_id"].eq(row.model_id)
            & forecasts["variant_id"].eq(row.variant_id)
        ].sort_values("forecast_day")["raw_forecast_retention_pct"]
        assert recomputed.prefix_rmse_pp == row.prefix_rmse_pp
        assert recomputed.prefix_max_abs_residual_pp == row.prefix_max_abs_residual_pp
        np.testing.assert_array_equal(
            recomputed.forecast_retention_pct,
            committed_forecast.to_numpy(float),
        )


def test_late_knee_parameters_are_bound_to_their_variant_id() -> None:
    parameters = {
        "a": 0.5,
        "b": 0.5,
        "k_pp_per_day": 0.001,
        "t_knee_days": 1095.75,
        "w_days": 30.0,
    }
    with pytest.raises(V015FitError, match="do not match variant_id"):
        fit.frozen_parameter_metadata(
            "target_prefix_late_knee_prior_grid",
            "k=0.0005|t=1095.75|w=30",
            parameters,
        )


def test_all_six_shared_members_reproduce_v1_exactly(
    fitted_fixture: tuple[pd.DataFrame, pd.DataFrame, fit.V015FitResult],
) -> None:
    prefix, coordinates, result = fitted_fixture
    legacy = v1.fit_structure_family_variants(
        prefix["prefix_day"].to_numpy(dtype=float),
        prefix["observed_retention_pct"].to_numpy(dtype=float),
        coordinates["forecast_day"].to_numpy(dtype=float),
        fit._legacy_protocol(),
    )
    diagnostics = result.member_fit_diagnostics.set_index(["model_id", "variant_id"])
    forecasts = result.member_forecast_bundle
    assert len(legacy) == 85

    for variant in legacy:
        key = (variant.model_id, variant.variant_id)
        row = diagnostics.loc[key]
        assert row["fit_status"] == ("succeeded" if variant.fit_succeeded else "failed")
        if not variant.fit_succeeded:
            assert row["parameters_json"] == "{}"
            continue
        expected_json = json.dumps(
            dict(variant.parameters),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        assert row["parameters_json"] == expected_json
        assert row["prefix_rmse_pp"] == variant.prefix_rmse_pp
        assert (
            row["prefix_max_abs_residual_pp"] == variant.prefix_max_absolute_residual_pp
        )
        observed = forecasts.loc[
            forecasts["model_id"].eq(variant.model_id)
            & forecasts["variant_id"].eq(variant.variant_id),
            "raw_forecast_retention_pct",
        ].to_numpy(dtype=float)
        np.testing.assert_array_equal(
            observed, np.asarray(variant.forecast_retention_pct, dtype=float)
        )


def _variant(
    *,
    model_id: str,
    parameters: tuple[tuple[str, float], ...],
    rmse: float = 0.2,
    maximum_residual: float = 0.3,
    forecast: tuple[float, ...] = (90.0,) * 8,
) -> v1.CandidateVariant:
    return v1.CandidateVariant(
        model_id=model_id,
        variant_id="fixture",
        parameters=parameters,
        prefix_rmse_pp=rmse,
        prefix_max_absolute_residual_pp=maximum_residual,
        forecast_retention_pct=forecast,
        fit_succeeded=True,
    )


def test_credibility_uses_closed_residual_and_raw_forecast_bounds() -> None:
    base = _variant(
        model_id="target_prefix_sqrt_time",
        parameters=(("c", 0.5),),
        forecast=(40.0, 105.0, 90.0, 90.0, 90.0, 90.0, 90.0, 90.0),
    )
    assert fit._credible_variant(base)
    assert fit._credible_variant(
        _variant(
            model_id=base.model_id,
            parameters=base.parameters,
            rmse=1.0,
            forecast=base.forecast_retention_pct,
        )
    )
    assert fit._credible_variant(
        _variant(
            model_id=base.model_id,
            parameters=base.parameters,
            maximum_residual=1.5,
            forecast=base.forecast_retention_pct,
        )
    )
    assert not fit._credible_variant(
        _variant(
            model_id=base.model_id,
            parameters=base.parameters,
            rmse=math.nextafter(1.0, math.inf),
            forecast=base.forecast_retention_pct,
        )
    )
    assert not fit._credible_variant(
        _variant(
            model_id=base.model_id,
            parameters=base.parameters,
            maximum_residual=math.nextafter(1.5, math.inf),
            forecast=base.forecast_retention_pct,
        )
    )
    assert not fit._credible_variant(
        _variant(
            model_id=base.model_id,
            parameters=base.parameters,
            forecast=(math.nextafter(40.0, -math.inf),) + (90.0,) * 7,
        )
    )


def test_boundary_fraction_uses_actual_bounds_and_excludes_fixed_knee_grid() -> None:
    late = _variant(
        model_id="target_prefix_late_knee_prior_grid",
        parameters=(
            ("a", 0.0),
            ("b", 0.5),
            ("k_pp_per_day", 0.0005),
            ("t_knee_days", 1095.75),
            ("w_days", 30.0),
        ),
    )
    assert fit._parameter_boundary_hit_fraction(late) == 0.5

    early = _variant(
        model_id="target_prefix_early_activation_plus_power",
        parameters=(
            ("a", 0.0),
            ("b", 0.5),
            ("activation_amplitude_pp", 3.0),
            ("tau_rise_days", 30.0),
            ("tau_decay_days", 730.0),
        ),
    )
    assert fit._parameter_boundary_hit_fraction(early) == 0.6

    persistence = _variant(
        model_id="target_prefix_persistence",
        parameters=(("last_retention_pct", 99.0),),
    )
    assert fit._parameter_boundary_hit_fraction(persistence) == 0.0


def test_firewall_rejects_extra_truth_like_columns() -> None:
    prefix, coordinates = _fixture_tables()
    poisoned = prefix.assign(truth_family="late_knee")
    with pytest.raises(V015FitError, match="unknown or missing"):
        fit_structure_library(poisoned, coordinates)
