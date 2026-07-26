from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import math

import pandas as pd
import pytest

from lifetwin.experiments.calendar_long_horizon_v015_fit import (
    fit_structure_library,
)
from lifetwin.experiments.calendar_long_horizon_v015_io import (
    FrozenArtifactContract,
    load_artifact_contract,
    predictor_content_hashes,
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
    PLACEBO_FEATURE_NAMES,
    VISIBLE_STRESS_FEATURE_NAMES,
    FrozenLabelFreeState,
)
from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    FROZEN_PROTOCOL_ID,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
)
from lifetwin.experiments.calendar_long_horizon_v016_pipeline import (
    V021PipelineError,
    _recompute_label_free_pipeline_hand_fixture_v021,
    _suppress_ineligible_probabilities,
    rank_primary_arms_v021,
    recompute_label_free_pipeline_v021,
)
from lifetwin.experiments.calendar_long_horizon_v016_protocol import (
    DEFAULT_V021_AMENDMENT_PATH,
    V021_PROTOCOL_ID,
    load_v021_design,
)


def _v021_contract() -> FrozenArtifactContract:
    design = load_v021_design()
    return replace(
        load_artifact_contract(),
        protocol_id=V021_PROTOCOL_ID,
        config_path=DEFAULT_V021_AMENDMENT_PATH,
        config_byte_sha256=design.config_byte_sha256,
    )


def _risk_state(
    feature_names: tuple[str, ...],
    coefficients: tuple[float, ...] | None = None,
) -> LogisticRiskState:
    dimension = len(feature_names)
    return LogisticRiskState(
        feature_names=feature_names,
        standardizer=StandardizerState(
            mean=(0.0,) * dimension,
            scale=(1.0,) * dimension,
            zero_variance=(False,) * dimension,
        ),
        intercept=-4.0,
        coefficients=coefficients or (0.0,) * dimension,
    )


def _state() -> FrozenLabelFreeState:
    prefix_coefficients = [0.0] * len(PREFIX_FEATURE_NAMES)
    prefix_coefficients[0] = 0.1
    visible_coefficients = [
        *prefix_coefficients,
        *(0.0 for _ in REAL_OPERATING_FIELDS),
    ]
    return FrozenLabelFreeState(
        center_beta=0.5,
        prefix_only_risk=_risk_state(
            PREFIX_FEATURE_NAMES,
            tuple(prefix_coefficients),
        ),
        visible_stress_risk=_risk_state(
            VISIBLE_STRESS_FEATURE_NAMES,
            tuple(visible_coefficients),
        ),
        placebo_risk=_risk_state(PLACEBO_FEATURE_NAMES),
        arm_a_plus_s_plan_risk=_risk_state(ARM_A_PLUS_S_PLAN_FEATURE_NAMES),
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
            coverage=0.9,
            calibration_count=900,
            order_statistic_index=811,
            expansion_pp=1.0,
        ),
    )


def _identity_frame(
    filename: str,
    records: list[dict[str, object]],
) -> pd.DataFrame:
    return pd.DataFrame(records, columns=_v021_contract().csv_schema(filename).columns)


def _hand_fixture() -> tuple[pd.DataFrame, ...]:
    cluster_id = "v021-fixture"
    prefix = _identity_frame(
        "prefix_pack.csv",
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                "prefix_day": day,
                "observed_retention_pct": 100.0 - 0.8 * math.sqrt(day / 365.25),
            }
            for day in PREFIX_DAYS
        ],
    )
    coordinates = _identity_frame(
        "forecast_coordinates.csv",
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                "forecast_day": day,
            }
            for day in FORECAST_DAYS
        ],
    )
    operating = _identity_frame(
        "operating_pack.csv",
        [
            {
                "protocol_id": V021_PROTOCOL_ID,
                "partition": "calibration",
                "cluster_id": cluster_id,
                **dict(
                    zip(
                        REAL_OPERATING_FIELDS,
                        (25.0, 0.5, 0.55, 250.0, 31.0, 0.6, 0.65, 300.0),
                        strict=True,
                    )
                ),
                **{
                    name: -0.8 + 0.2 * index
                    for index, name in enumerate(PLACEBO_FIELDS)
                },
            }
        ],
    )
    base_prefix = prefix.assign(protocol_id=FROZEN_PROTOCOL_ID)
    base_coordinates = coordinates.assign(protocol_id=FROZEN_PROTOCOL_ID)
    fitted = fit_structure_library(base_prefix, base_coordinates)
    hashes = predictor_content_hashes(prefix, coordinates, operating.iloc[0])
    diagnostics = fitted.member_fit_diagnostics.assign(
        protocol_id=V021_PROTOCOL_ID,
        canonical_prefix_content_sha256=hashes.arm_a,
    )
    forecasts = fitted.member_forecast_bundle.assign(
        protocol_id=V021_PROTOCOL_ID,
        canonical_prefix_content_sha256=hashes.arm_a,
    )
    return prefix, coordinates, operating, diagnostics, forecasts


def test_v021_pipeline_adapts_identity_without_mutating_inputs() -> None:
    fixture = _hand_fixture()
    originals = tuple(frame.copy(deep=True) for frame in fixture)
    result = _recompute_label_free_pipeline_hand_fixture_v021(
        prefix_pack=fixture[0],
        forecast_coordinates=fixture[1],
        operating_pack=fixture[2],
        member_fit_diagnostics=fixture[3],
        member_forecast_bundle=fixture[4],
        state=_state(),
        contract=_v021_contract(),
    )

    for observed, expected in zip(fixture, originals, strict=True):
        pd.testing.assert_frame_equal(observed, expected)
    for frame in (
        result.prediction_bundle,
        result.feature_bundle,
        result.primary_risk_bundle,
        result.decision_bundle,
        result.predictor_content_bundle,
    ):
        assert set(frame["protocol_id"]) == {V021_PROTOCOL_ID}
    assert not result.feature_bundle["hard_eligible"].isna().any()


def test_v021_suppresses_probabilities_outside_operational_population() -> None:
    risk = pd.DataFrame(
        {
            "partition": ["calibration"] * 4,
            "cluster_id": ["eligible", "eligible", "one-family", "one-family"],
            "score_id": [
                "prefix_only",
                "visible_stress",
                "prefix_only",
                "visible_stress",
            ],
            "calibrated_catastrophic_probability": [0.2, 0.3, 0.8, 0.9],
        }
    )
    features = pd.DataFrame(
        {
            "partition": ["calibration", "calibration"],
            "cluster_id": ["eligible", "one-family"],
            "hard_eligible": [True, False],
        }
    )
    result = _suppress_ineligible_probabilities(risk, features)
    assert result.loc[result["cluster_id"].eq("eligible")].iloc[:, -1].tolist() == [
        0.2,
        0.3,
    ]
    assert (
        result.loc[
            result["cluster_id"].eq("one-family"),
            "calibrated_catastrophic_probability",
        ]
        .isna()
        .all()
    )


def test_v021_tie_domain_is_protocol_specific_and_deterministic() -> None:
    hashes = ("0" * 64, "f" * 64)
    result = rank_primary_arms_v021(
        prefix_only_scores=(1.0, 1.0),
        visible_stress_scores=(1.0, 1.0),
        prefix_only_hashes=hashes,
        visible_stress_hashes=hashes,
        hard_eligible=(True, True),
        issue_count=1,
    )
    expected_first = min(
        range(2),
        key=lambda index: hashlib.sha256(
            f"{V021_PROTOCOL_ID}|prefix_only|{hashes[index]}".encode("ascii")
        ).hexdigest(),
    )
    assert result.prefix_only_issued[expected_first]
    assert sum(result.prefix_only_issued) == 1


def test_v021_pipeline_rejects_a_v2_contract() -> None:
    fixture = _hand_fixture()
    with pytest.raises(V021PipelineError, match="not V2.1"):
        _recompute_label_free_pipeline_hand_fixture_v021(
            prefix_pack=fixture[0],
            forecast_coordinates=fixture[1],
            operating_pack=fixture[2],
            member_fit_diagnostics=fixture[3],
            member_forecast_bundle=fixture[4],
            state=_state(),
            contract=load_artifact_contract(),
        )


def test_v021_pipeline_surface_has_no_truth_or_training_channel() -> None:
    parameters = inspect.signature(recompute_label_free_pipeline_v021).parameters
    forbidden = ("truth", "target", "label", "family_truth", "sealed", "score")
    assert not any(token in name.lower() for name in parameters for token in forbidden)
