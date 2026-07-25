from __future__ import annotations

import inspect
from functools import lru_cache
import math
import json

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_io import (
    load_artifact_contract,
    predictor_content_hashes,
)
from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    FROZEN_VARIANT_KEYS,
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_model import (
    FORECAST_DAYS,
    PREFIX_DAYS,
    PREFIX_FEATURE_NAMES,
    ConformalExpansionState,
    IsotonicState,
    LogisticRiskState,
    StandardizerState,
)
from lifetwin.experiments.calendar_long_horizon_v015_pipeline import (
    ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
    DECLARED_STRUCTURE_FAMILIES,
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    FrozenLabelFreeState,
    V015PipelineError,
    _abstention_reasons,
    rank_primary_arms,
    recompute_label_free_pipeline,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
)


_CONTRACT = load_artifact_contract()


@lru_cache(maxsize=1)
def _fitted_member_template() -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = _frame(
        "prefix_pack.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "template",
                "prefix_day": day,
                "observed_retention_pct": (100.0 - 0.8 * math.sqrt(day / 365.25)),
            }
            for day in PREFIX_DAYS
        ],
    )
    coordinates = _frame(
        "forecast_coordinates.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": "template",
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
    )
    result = fit_structure_library(prefix, coordinates)
    return (
        result.member_fit_diagnostics.copy(deep=True),
        result.member_forecast_bundle.copy(deep=True),
    )


def _frame(filename: str, records: list[dict[str, object]]) -> pd.DataFrame:
    columns = _CONTRACT.csv_schema(filename).columns
    return pd.DataFrame(records, columns=columns)


def _state() -> FrozenLabelFreeState:
    prefix_coefficients = np.zeros(len(PREFIX_FEATURE_NAMES), dtype=float)
    prefix_coefficients[0] = 0.1
    prefix_coefficients[8] = 0.02
    visible_coefficients = np.zeros(len(VISIBLE_STRESS_FEATURE_NAMES), dtype=float)
    visible_coefficients[: len(PREFIX_FEATURE_NAMES)] = prefix_coefficients
    visible_coefficients[
        VISIBLE_STRESS_FEATURE_NAMES.index("planned_mean_temperature_c")
    ] = 0.05

    def risk_state(
        names: tuple[str, ...], coefficients: np.ndarray
    ) -> LogisticRiskState:
        standardizer = StandardizerState(
            mean=(0.0,) * len(names),
            scale=(1.0,) * len(names),
            zero_variance=(False,) * len(names),
        )
        return LogisticRiskState(
            feature_names=names,
            standardizer=standardizer,
            intercept=-5.0,
            coefficients=tuple(float(value) for value in coefficients),
        )

    return FrozenLabelFreeState(
        center_beta=0.5,
        prefix_only_risk=risk_state(PREFIX_FEATURE_NAMES, prefix_coefficients),
        visible_stress_risk=risk_state(
            VISIBLE_STRESS_FEATURE_NAMES, visible_coefficients
        ),
        placebo_risk=risk_state(
            PLACEBO_FEATURE_NAMES,
            np.zeros(len(PLACEBO_FEATURE_NAMES), dtype=float),
        ),
        arm_a_plus_s_plan_risk=risk_state(
            ARM_A_PLUS_S_PLAN_FEATURE_NAMES,
            np.zeros(len(ARM_A_PLUS_S_PLAN_FEATURE_NAMES), dtype=float),
        ),
        strongest_single_feature_name=PREFIX_FEATURE_NAMES[0],
        strongest_single_feature_orientation=1,
        prefix_only_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        visible_stress_isotonic=IsotonicState(
            x_thresholds=(-1000.0, 1000.0),
            y_thresholds=(0.0, 1.0),
        ),
        conformal=ConformalExpansionState(
            coverage=0.90,
            calibration_count=900,
            order_statistic_index=811,
            expansion_pp=1.25,
        ),
    )


def _predictor_fixture(
    *,
    cluster_id: str = "c_fixture",
    partition: str = "calibration",
    planned_temperature_c: float = 31.0,
    credible_models: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if credible_models is None:
        credible_models = {
            "target_prefix_persistence",
            "target_prefix_sqrt_time",
            "target_prefix_bounded_power_law",
        }
    prefix = _frame(
        "prefix_pack.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": partition,
                "cluster_id": cluster_id,
                "prefix_day": day,
                "observed_retention_pct": (100.0 - 0.8 * math.sqrt(day / 365.25)),
            }
            for day in PREFIX_DAYS
        ],
    )
    coordinates = _frame(
        "forecast_coordinates.csv",
        [
            {
                "protocol_id": FROZEN_PROTOCOL_ID,
                "partition": partition,
                "cluster_id": cluster_id,
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
    )
    real_values = (
        25.0,
        0.50,
        0.55,
        250.0,
        planned_temperature_c,
        0.60,
        0.65,
        300.0,
    )
    operating_record: dict[str, object] = {
        "protocol_id": FROZEN_PROTOCOL_ID,
        "partition": partition,
        "cluster_id": cluster_id,
        **dict(zip(REAL_OPERATING_FIELDS, real_values, strict=True)),
        **{field: -0.8 + 0.2 * index for index, field in enumerate(PLACEBO_FIELDS)},
    }
    operating = _frame("operating_pack.csv", [operating_record])
    hashes = predictor_content_hashes(prefix, coordinates, operating.iloc[0])

    diagnostics, forecasts = (
        frame.copy(deep=True) for frame in _fitted_member_template()
    )
    for frame in (diagnostics, forecasts):
        frame["partition"] = partition
        frame["cluster_id"] = cluster_id
        frame["canonical_prefix_content_sha256"] = hashes.arm_a

    failed_models = set(DECLARED_STRUCTURE_FAMILIES) - credible_models
    diagnostic_failure = diagnostics["model_id"].isin(failed_models)
    diagnostics.loc[diagnostic_failure, "parameters_json"] = "{}"
    diagnostics.loc[diagnostic_failure, "fit_status"] = "failed"
    diagnostics.loc[diagnostic_failure, "credible_variant"] = False
    diagnostics.loc[
        diagnostic_failure,
        [
            "prefix_rmse_pp",
            "prefix_max_abs_residual_pp",
            "parameter_boundary_hit_fraction",
        ],
    ] = math.nan
    forecasts.loc[
        forecasts["model_id"].isin(failed_models),
        "raw_forecast_retention_pct",
    ] = math.nan
    return prefix, coordinates, operating, diagnostics, forecasts


def _run(
    fixture: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
):
    prefix, coordinates, operating, diagnostics, forecasts = fixture
    return recompute_label_free_pipeline(
        prefix_pack=prefix,
        forecast_coordinates=coordinates,
        operating_pack=operating,
        member_fit_diagnostics=diagnostics,
        member_forecast_bundle=forecasts,
        state=_state(),
    )


def test_pipeline_recomputes_center_features_scores_and_intervals() -> None:
    result = _run(_predictor_fixture())

    assert len(result.prediction_bundle) == len(FORECAST_DAYS)
    assert len(result.feature_bundle) == 1
    assert len(result.primary_risk_bundle) == 9
    assert len(result.decision_bundle) == 2
    feature = result.feature_bundle.iloc[0]
    assert feature["hard_eligible"] is np.True_ or feature["hard_eligible"] is True
    assert len(PREFIX_FEATURE_NAMES) == 14
    assert len(VISIBLE_STRESS_FEATURE_NAMES) == 22
    assert feature["successful_structure_family_count"] == 3.0
    assert feature["fit_failure_count"] == 4.0
    assert feature["parameter_boundary_hit_fraction"] == 0.0

    prediction = result.prediction_bundle.sort_values("forecast_day")
    assert np.isfinite(prediction["center_forecast_pct"]).all()
    assert np.all(
        prediction["base_interval_lower_pct"] <= prediction["center_forecast_pct"]
    )
    assert np.all(
        prediction["center_forecast_pct"] <= prediction["base_interval_upper_pct"]
    )
    assert prediction["calibrated_interval_lower_pct"].to_numpy() == pytest.approx(
        prediction["base_interval_lower_pct"].to_numpy() - 1.25
    )
    assert prediction["calibrated_interval_upper_pct"].to_numpy() == pytest.approx(
        prediction["base_interval_upper_pct"].to_numpy() + 1.25
    )

    arm_a = tuple(float(feature[name]) for name in PREFIX_FEATURE_NAMES)
    arm_b = arm_a + tuple(float(feature[name]) for name in REAL_OPERATING_FIELDS)
    risks = result.primary_risk_bundle.set_index("score_id")
    assert risks.loc["prefix_only", "raw_risk_score"] == pytest.approx(
        _state().prefix_only_risk.decision_function((arm_a,))[0]
    )
    assert risks.loc["visible_stress", "raw_risk_score"] == pytest.approx(
        _state().visible_stress_risk.decision_function((arm_b,))[0]
    )
    assert not result.decision_bundle["issued"].any()
    assert result.decision_bundle["issuance_rank"].isna().all()


def test_operating_plan_changes_only_arm_b_content_and_score() -> None:
    low = _run(_predictor_fixture(planned_temperature_c=20.0))
    high = _run(_predictor_fixture(planned_temperature_c=38.0))
    low_hash = low.predictor_content_bundle.iloc[0]
    high_hash = high.predictor_content_bundle.iloc[0]
    assert low_hash["arm_a_content_sha256"] == high_hash["arm_a_content_sha256"]
    assert low_hash["arm_b_content_sha256"] != high_hash["arm_b_content_sha256"]
    assert np.array_equal(
        low.prediction_bundle["center_forecast_pct"].to_numpy(),
        high.prediction_bundle["center_forecast_pct"].to_numpy(),
    )
    low_risk = low.primary_risk_bundle.set_index("score_id")
    high_risk = high.primary_risk_bundle.set_index("score_id")
    assert (
        low_risk.loc["prefix_only", "raw_risk_score"]
        == (high_risk.loc["prefix_only", "raw_risk_score"])
    )
    assert (
        low_risk.loc["visible_stress", "raw_risk_score"]
        < (high_risk.loc["visible_stress", "raw_risk_score"])
    )


def test_opaque_identity_and_row_order_do_not_change_predictor_outputs() -> None:
    first_fixture = _predictor_fixture(cluster_id="opaque_a")
    second_fixture = _predictor_fixture(cluster_id="opaque_z")
    shuffled = tuple(
        frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
        for frame in second_fixture
    )

    first = _run(first_fixture)
    second = _run(shuffled)

    assert (
        first.predictor_content_bundle.iloc[0]["arm_a_content_sha256"]
        == second.predictor_content_bundle.iloc[0]["arm_a_content_sha256"]
    )
    assert (
        first.predictor_content_bundle.iloc[0]["arm_b_content_sha256"]
        == second.predictor_content_bundle.iloc[0]["arm_b_content_sha256"]
    )
    assert np.array_equal(
        first.prediction_bundle.drop(columns=["cluster_id"]).to_numpy(),
        second.prediction_bundle.drop(columns=["cluster_id"]).to_numpy(),
    )
    pd.testing.assert_frame_equal(
        first.primary_risk_bundle.drop(columns=["cluster_id"]),
        second.primary_risk_bundle.drop(columns=["cluster_id"]),
    )


def test_hard_eligibility_falls_back_when_only_one_family_is_credible() -> None:
    eligible, reasons = _abstention_reasons(
        successful_family_count=1,
        center=(90.0,) * 8,
        prefix_features=(0.0,) * len(PREFIX_FEATURE_NAMES),
        real_operating=(0.0,) * len(REAL_OPERATING_FIELDS),
        placebo_operating=(0.0,) * len(PLACEBO_FIELDS),
    )
    assert not eligible
    assert reasons == "insufficient_structure_families"


def test_common_pool_ranking_uses_one_mask_and_exact_count() -> None:
    hashes = tuple(f"{index:064x}" for index in range(4))
    result = rank_primary_arms(
        prefix_only_scores=(3.0, 1.0, math.nan, 2.0),
        visible_stress_scores=(1.0, 3.0, math.nan, 2.0),
        prefix_only_hashes=hashes,
        visible_stress_hashes=tuple(reversed(hashes)),
        hard_eligible=(True, True, False, True),
        issue_count=2,
    )

    assert result.prefix_only_issued == (False, True, False, True)
    assert result.visible_stress_issued == (True, False, False, True)
    assert result.prefix_only_ranks[2] is None
    assert result.visible_stress_ranks[2] is None
    assert sum(result.prefix_only_issued) == sum(result.visible_stress_issued) == 2
    with pytest.raises(V015PipelineError, match="nonfinite"):
        rank_primary_arms(
            prefix_only_scores=(math.nan,),
            visible_stress_scores=(0.0,),
            prefix_only_hashes=(hashes[0],),
            visible_stress_hashes=(hashes[0],),
            hard_eligible=(True,),
            issue_count=1,
        )


def test_prediction_entry_point_has_no_truth_or_pair_metadata_channel() -> None:
    parameter_names = tuple(inspect.signature(recompute_label_free_pipeline).parameters)
    forbidden = ("truth", "path", "family", "pair", "side")
    assert not any(
        token in parameter.lower()
        for parameter in parameter_names
        for token in forbidden
    )

    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    operating = operating.assign(truth_family="late_knee")
    with pytest.raises(V015PipelineError, match="allowlist"):
        recompute_label_free_pipeline(
            prefix_pack=prefix,
            forecast_coordinates=coordinates,
            operating_pack=operating,
            member_fit_diagnostics=diagnostics,
            member_forecast_bundle=forecasts,
            state=_state(),
        )


def test_credibility_and_prefix_hash_are_recomputed_not_trusted() -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    diagnostics = diagnostics.copy()
    diagnostics.loc[
        diagnostics["model_id"].eq("target_prefix_persistence"),
        "credible_variant",
    ] = False
    with pytest.raises(V015PipelineError, match="credibility rule"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))

    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    forecasts = forecasts.copy()
    forecasts.loc[0, "canonical_prefix_content_sha256"] = "0" * 64
    with pytest.raises(V015PipelineError, match="hash differs"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))


def test_parameter_forecast_and_rmse_tampering_are_rejected() -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    sqrt_mask = diagnostics["model_id"].eq("target_prefix_sqrt_time")
    sqrt_index = diagnostics.index[sqrt_mask][0]

    tampered_parameters = diagnostics.copy()
    payload = json.loads(tampered_parameters.loc[sqrt_index, "parameters_json"])
    payload["c"] += 0.01
    tampered_parameters.loc[sqrt_index, "parameters_json"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    with pytest.raises(V015PipelineError, match="formula recomputation"):
        _run(
            (
                prefix,
                coordinates,
                operating,
                tampered_parameters,
                forecasts,
            )
        )

    tampered_forecast = forecasts.copy()
    forecast_index = tampered_forecast.index[
        tampered_forecast["model_id"].eq("target_prefix_sqrt_time")
    ][0]
    tampered_forecast.loc[forecast_index, "raw_forecast_retention_pct"] += 0.01
    with pytest.raises(V015PipelineError, match="formula recomputation"):
        _run(
            (
                prefix,
                coordinates,
                operating,
                diagnostics,
                tampered_forecast,
            )
        )

    tampered_rmse = diagnostics.copy()
    tampered_rmse.loc[sqrt_index, "prefix_rmse_pp"] += 0.01
    with pytest.raises(V015PipelineError, match="formula recomputation"):
        _run(
            (
                prefix,
                coordinates,
                operating,
                tampered_rmse,
                forecasts,
            )
        )


@pytest.mark.parametrize("target", ["diagnostics", "forecasts"])
def test_missing_frozen_variant_is_rejected(target: str) -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    missing_key = FROZEN_VARIANT_KEYS[-1]
    if target == "diagnostics":
        diagnostics = diagnostics.loc[
            ~(
                diagnostics["model_id"].eq(missing_key[0])
                & diagnostics["variant_id"].eq(missing_key[1])
            )
        ].reset_index(drop=True)
    else:
        forecasts = forecasts.loc[
            ~(
                forecasts["model_id"].eq(missing_key[0])
                & forecasts["variant_id"].eq(missing_key[1])
            )
        ].reset_index(drop=True)
    with pytest.raises(V015PipelineError, match="exact 86"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))


def test_extra_frozen_variant_row_is_rejected() -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    extra_diagnostic = diagnostics.iloc[[0]].copy()
    extra_diagnostic["model_id"] = "undeclared_extra_model"
    extra_diagnostic["variant_id"] = "extra"
    diagnostics = pd.concat([diagnostics, extra_diagnostic], ignore_index=True)
    extra_forecast = forecasts.loc[
        forecasts["model_id"].eq(forecasts.iloc[0]["model_id"])
        & forecasts["variant_id"].eq(forecasts.iloc[0]["variant_id"])
    ].copy()
    extra_forecast["model_id"] = "undeclared_extra_model"
    extra_forecast["variant_id"] = "extra"
    forecasts = pd.concat([forecasts, extra_forecast], ignore_index=True)
    with pytest.raises(V015PipelineError, match="exact 86"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))


def test_swapped_late_knee_variant_ids_are_rejected() -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture(
        credible_models=set(DECLARED_STRUCTURE_FAMILIES)
    )
    late_keys = [
        key
        for key in FROZEN_VARIANT_KEYS
        if key[0] == "target_prefix_late_knee_prior_grid"
    ][:2]
    first_id, second_id = late_keys[0][1], late_keys[1][1]
    diagnostics = diagnostics.copy()
    forecasts = forecasts.copy()
    diagnostics.loc[diagnostics["variant_id"].eq(first_id), "variant_id"] = "__swap__"
    diagnostics.loc[diagnostics["variant_id"].eq(second_id), "variant_id"] = first_id
    diagnostics.loc[diagnostics["variant_id"].eq("__swap__"), "variant_id"] = second_id
    forecasts.loc[forecasts["variant_id"].eq(first_id), "variant_id"] = "__swap__"
    forecasts.loc[forecasts["variant_id"].eq(second_id), "variant_id"] = first_id
    forecasts.loc[forecasts["variant_id"].eq("__swap__"), "variant_id"] = second_id
    with pytest.raises(V015PipelineError, match="variant_id"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))


def test_failed_variant_must_retain_empty_forecasts() -> None:
    prefix, coordinates, operating, diagnostics, forecasts = _predictor_fixture()
    failed_model = "target_prefix_dual_power"
    assert (
        diagnostics.loc[diagnostics["model_id"].eq(failed_model), "fit_status"]
        .eq("failed")
        .all()
    )
    forecasts = forecasts.copy()
    index = forecasts.index[forecasts["model_id"].eq(failed_model)][0]
    forecasts.loc[index, "raw_forecast_retention_pct"] = 90.0
    with pytest.raises(V015PipelineError, match="empty forecasts"):
        _run((prefix, coordinates, operating, diagnostics, forecasts))


def test_test_partition_cannot_silently_reduce_the_frozen_issue_count() -> None:
    with pytest.raises(V015PipelineError, match="smaller than the issue count"):
        _run(_predictor_fixture(partition="test"))
