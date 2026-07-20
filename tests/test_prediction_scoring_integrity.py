from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments import calendar_v2_development as v2
from lifetwin.experiments import calendar_v3_activation_development as v3


def _truth() -> pd.DataFrame:
    checkups = np.arange(35)
    return pd.DataFrame(
        {
            "condition_id": "CONDITION_A",
            "checkup_index": checkups,
            "elapsed_days": checkups.astype(float) ** 2,
            "temperature_c": 40.0,
            "storage_soc_fraction": 0.5,
            "capacity_retention_pct": 100.0 - checkups / 10.0,
        }
    )


def _v3_predictions(
    truth: pd.DataFrame,
    *,
    prefix: int = 10,
    scenario: str = "synthetic_v3",
    fold_id: str = "fold_a",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    future = truth.loc[truth["checkup_index"] >= prefix]
    condition_id = str(truth["condition_id"].iloc[0])
    prefix_end_days = float(
        truth.loc[truth["checkup_index"] == prefix - 1, "elapsed_days"].iloc[0]
    )
    validation_horizon_days = float(truth["elapsed_days"].max())
    training_support_days = max(prefix_end_days, 1.0)
    for method in v3.METHOD_NAMES:
        for coordinate in future.itertuples(index=False):
            rows.append(
                {
                    "scenario": scenario,
                    "fold_id": fold_id,
                    "target_condition_id": condition_id,
                    "prefix_checkups": prefix,
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "training_history_policy": v3.GLOBAL_LANDMARK_POLICY,
                    "prefix_end_checkup_index": prefix - 1,
                    "prefix_end_days": prefix_end_days,
                    "temperature_c": float(coordinate.temperature_c),
                    "storage_soc_fraction": float(
                        coordinate.storage_soc_fraction
                    ),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(
                        coordinate.capacity_retention_pct + 0.5
                    ),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == truth["checkup_index"].max()
                    ),
                    "activation_gate_ready": False,
                    "negative_loss_evidence": False,
                    "positive_time_observation_count": prefix - 1,
                    "minimum_prefix_capacity_loss_pct": 0.0,
                    "activation_component_selected": False,
                    "training_support_days": training_support_days,
                    "validation_horizon_days": validation_horizon_days,
                    "time_extrapolation_ratio": (
                        validation_horizon_days / training_support_days
                    ),
                    "training_state_sha256": "a" * 64,
                    "prediction_state_sha256": "b" * 64,
                }
            )
    return pd.DataFrame(rows)[v3.PREDICTION_COLUMNS]


def _v2_predictions(
    truth: pd.DataFrame,
    *,
    prefix: int = 10,
    scenario: str = "synthetic_v2",
    fold_id: str = "fold_a",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    future = truth.loc[truth["checkup_index"] >= prefix]
    condition_id = str(truth["condition_id"].iloc[0])
    prefix_end_days = float(
        truth.loc[truth["checkup_index"] == prefix - 1, "elapsed_days"].iloc[0]
    )
    validation_horizon_days = float(truth["elapsed_days"].max())
    training_support_days = max(prefix_end_days, 1.0)
    for method in v2.METHOD_NAMES:
        for coordinate in future.itertuples(index=False):
            rows.append(
                {
                    "scenario": scenario,
                    "training_history_policy": v2.GLOBAL_LANDMARK_POLICY,
                    "fold_id": fold_id,
                    "target_condition_id": condition_id,
                    "prefix_checkups": prefix,
                    "method": method,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "prefix_end_checkup_index": prefix - 1,
                    "prefix_end_days": prefix_end_days,
                    "temperature_c": float(coordinate.temperature_c),
                    "storage_soc_fraction": float(
                        coordinate.storage_soc_fraction
                    ),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(
                        coordinate.capacity_retention_pct + 0.5
                    ),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == truth["checkup_index"].max()
                    ),
                    "training_support_days": training_support_days,
                    "validation_horizon_days": validation_horizon_days,
                    "time_extrapolation_ratio": (
                        validation_horizon_days / training_support_days
                    ),
                    "training_state_sha256": "a" * 64,
                    "prediction_state_sha256": "b" * 64,
                }
            )
    return pd.DataFrame(rows)[v2.PREDICTION_COLUMNS]


def _score_v3(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return v3.score_calendar_v3_predictions(
        predictions,
        truth,
        frozen_prediction_sha256=v3.calendar_v3_prediction_sha256(predictions),
    )


def _score_v2(predictions: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    return v2.score_calendar_v2_predictions(
        predictions,
        truth,
        frozen_prediction_sha256=v2.calendar_v2_prediction_sha256(predictions),
    )


def _second_condition(truth: pd.DataFrame) -> pd.DataFrame:
    second = truth.copy()
    second["condition_id"] = "CONDITION_B"
    second["temperature_c"] = 25.0
    return second


def _v3_sensitivity_predictions(
    truth: pd.DataFrame,
    *,
    prefix: int = 10,
    scenario: str = "synthetic_v3",
    fold_id: str = "fold_a",
) -> pd.DataFrame:
    condition_id = str(truth["condition_id"].iloc[0])
    future = truth.loc[truth["checkup_index"] >= prefix]
    rows: list[dict[str, object]] = []
    for tau in v3.TAU_SENSITIVITY_DAYS:
        for coordinate in future.itertuples(index=False):
            rows.append(
                {
                    "scenario": scenario,
                    "fold_id": fold_id,
                    "target_condition_id": condition_id,
                    "prefix_checkups": prefix,
                    "activation_timescale_days": tau,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "training_history_policy": v3.GLOBAL_LANDMARK_POLICY,
                    "elapsed_days": float(coordinate.elapsed_days),
                    "predicted_capacity_retention_pct": float(
                        coordinate.capacity_retention_pct + 0.5
                    ),
                    "is_final_checkup": bool(
                        coordinate.checkup_index == truth["checkup_index"].max()
                    ),
                    "activation_gate_ready": False,
                    "activation_component_selected": False,
                    "training_state_sha256": "a" * 64,
                    "prediction_state_sha256": "b" * 64,
                }
            )
    return pd.DataFrame(rows)[v3.SENSITIVITY_COLUMNS]


def _frozen_regular_prediction_pack(
    observations: pd.DataFrame,
    *,
    module,
    pack_factory,
) -> pd.DataFrame:
    profile = observations[
        ["condition_id", "temperature_c", "storage_soc_fraction"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for temperature, targets in profile.groupby("temperature_c", sort=True):
        fold_id = f"temperature_c={float(temperature):g}"
        for condition_id in sorted(targets["condition_id"].astype(str)):
            truth = observations.loc[
                observations["condition_id"].astype(str) == condition_id
            ]
            for prefix in module.EXPECTED_PREFIXES:
                frames.append(
                    pack_factory(
                        truth,
                        prefix=prefix,
                        scenario=module.TEMPERATURE_SCENARIO,
                        fold_id=fold_id,
                    )
                )
    for condition_id in module.SOC_TARGET_CONDITIONS:
        truth = observations.loc[
            observations["condition_id"].astype(str) == condition_id
        ]
        for prefix in module.EXPECTED_PREFIXES:
            frames.append(
                pack_factory(
                    truth,
                    prefix=prefix,
                    scenario=module.SOC_SCENARIO,
                    fold_id=module.SOC_FOLD_ID,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _frozen_sensitivity_prediction_pack(
    observations: pd.DataFrame,
) -> pd.DataFrame:
    profile = observations[
        ["condition_id", "temperature_c", "storage_soc_fraction"]
    ].drop_duplicates()
    frames: list[pd.DataFrame] = []
    for temperature, targets in profile.groupby("temperature_c", sort=True):
        fold_id = f"temperature_c={float(temperature):g}"
        for condition_id in sorted(targets["condition_id"].astype(str)):
            truth = observations.loc[
                observations["condition_id"].astype(str) == condition_id
            ]
            frames.append(
                _v3_sensitivity_predictions(
                    truth,
                    scenario=v3.TEMPERATURE_SCENARIO,
                    fold_id=fold_id,
                )
            )
    for condition_id in v3.SOC_TARGET_CONDITIONS:
        truth = observations.loc[
            observations["condition_id"].astype(str) == condition_id
        ]
        frames.append(
            _v3_sensitivity_predictions(
                truth,
                scenario=v3.SOC_SCENARIO,
                fold_id=v3.SOC_FOLD_ID,
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize(
    ("pack_factory", "score", "version"),
    [
        (_v3_predictions, _score_v3, "V3"),
        (_v2_predictions, _score_v2, "V2"),
    ],
)
@pytest.mark.parametrize(
    ("coordinate_column", "delta"),
    [
        ("elapsed_days", 123.0),
        ("temperature_c", 1.0),
        ("storage_soc_fraction", 0.1),
    ],
)
def test_rehashed_timeline_tampering_is_rejected(
    pack_factory,
    score,
    version: str,
    coordinate_column: str,
    delta: float,
) -> None:
    truth = _truth()
    predictions = pack_factory(truth)
    metrics = score(predictions, truth)
    assert np.allclose(metrics["trajectory_iae_pp"], 0.5)

    tampered = predictions.copy()
    tampered.loc[tampered.index[0], coordinate_column] += delta
    with pytest.raises(ValueError, match=f"Calendar {version} prediction coordinate"):
        score(tampered, truth)


@pytest.mark.parametrize(
    ("pack_factory", "score", "version"),
    [
        (_v3_predictions, _score_v3, "V3"),
        (_v2_predictions, _score_v2, "V2"),
    ],
)
def test_rehashed_final_flag_tampering_is_rejected(
    pack_factory,
    score,
    version: str,
) -> None:
    truth = _truth()
    tampered = pack_factory(truth)
    final_row = tampered.index[tampered["is_final_checkup"]][0]
    tampered.loc[final_row, "is_final_checkup"] = False

    with pytest.raises(ValueError, match=f"Calendar {version} final-checkup flag"):
        score(tampered, truth)


@pytest.mark.parametrize(
    ("pack_factory", "score", "version"),
    [
        (_v3_predictions, _score_v3, "V3"),
        (_v2_predictions, _score_v2, "V2"),
    ],
)
def test_prefix_future_coverage_and_finite_values_are_enforced(
    pack_factory,
    score,
    version: str,
) -> None:
    truth = _truth()
    predictions = pack_factory(truth)

    bad_prefix = predictions.copy()
    bad_prefix["prefix_end_days"] += 1.0
    with pytest.raises(ValueError, match=f"Calendar {version} prefix-end day"):
        score(bad_prefix, truth)

    bad_prefix_index = predictions.copy()
    bad_prefix_index.loc[
        bad_prefix_index.index[0], "prefix_end_checkup_index"
    ] -= 1
    with pytest.raises(ValueError, match=f"Calendar {version} prefix-end index"):
        score(bad_prefix_index, truth)

    pre_prefix_target = predictions.copy()
    first_future = pre_prefix_target["target_checkup_index"] == 10
    pre_prefix_target.loc[first_future, "target_checkup_index"] = 9
    pre_prefix_target.loc[first_future, "elapsed_days"] = 9.0**2
    with pytest.raises(ValueError, match="target precedes its prefix"):
        score(pre_prefix_target, truth)

    incomplete = predictions.loc[
        predictions["target_checkup_index"] != 10
    ].reset_index(drop=True)
    with pytest.raises(ValueError, match="every future checkup"):
        score(incomplete, truth)

    nonfinite = predictions.copy()
    nonfinite.loc[nonfinite.index[0], "predicted_capacity_retention_pct"] = np.inf
    with pytest.raises(ValueError, match="must be finite"):
        score(nonfinite, truth)


@pytest.mark.parametrize(
    ("pack_factory", "score", "version", "method"),
    [
        (_v3_predictions, _score_v3, "V3", v3.GATED_TARGET_ACTIVATION_METHOD),
        (_v2_predictions, _score_v2, "V2", v2.HIERARCHICAL_POWER_METHOD),
    ],
)
def test_rehashed_pack_missing_one_condition_method_trajectory_is_rejected(
    pack_factory,
    score,
    version: str,
    method: str,
) -> None:
    first = _truth()
    second = _second_condition(first)
    truth = pd.concat([first, second], ignore_index=True)
    predictions = pd.concat(
        [pack_factory(first), pack_factory(second)],
        ignore_index=True,
    )
    removed_trajectory = (
        (predictions["target_condition_id"] == "CONDITION_B")
        & (predictions["method"] == method)
    )
    assert removed_trajectory.any()
    tampered = predictions.loc[~removed_trajectory].reset_index(drop=True)

    with pytest.raises(
        ValueError,
        match=f"Calendar {version} prediction support must contain every method",
    ):
        score(tampered, truth)


def test_rehashed_sensitivity_pack_missing_one_condition_tau_is_rejected() -> None:
    first = _truth()
    second = _second_condition(first)
    truth = pd.concat([first, second], ignore_index=True)
    predictions = pd.concat(
        [
            _v3_sensitivity_predictions(first),
            _v3_sensitivity_predictions(second),
        ],
        ignore_index=True,
    )
    frozen = v3.calendar_v3_sensitivity_sha256(predictions)
    metrics = v3.score_calendar_v3_sensitivity(
        predictions,
        truth,
        frozen_prediction_sha256=frozen,
    )
    assert len(metrics) == 2 * len(v3.TAU_SENSITIVITY_DAYS)

    removed_trajectory = (
        (predictions["target_condition_id"] == "CONDITION_B")
        & (predictions["activation_timescale_days"] == v3.TAU_SENSITIVITY_DAYS[0])
    )
    assert removed_trajectory.any()
    tampered = predictions.loc[~removed_trajectory].reset_index(drop=True)
    with pytest.raises(
        ValueError,
        match="sensitivity support must contain every frozen tau",
    ):
        v3.calendar_v3_sensitivity_sha256(tampered)


@pytest.mark.parametrize(
    ("module", "pack_factory", "score", "version"),
    [
        (v3, _v3_predictions, _score_v3, "V3"),
        (v2, _v2_predictions, _score_v2, "V2"),
    ],
)
def test_rehashed_pack_missing_whole_target_is_rejected_by_frozen_protocol(
    observations: pd.DataFrame,
    module,
    pack_factory,
    score,
    version: str,
) -> None:
    predictions = _frozen_regular_prediction_pack(
        observations,
        module=module,
        pack_factory=pack_factory,
    )
    target_mask = (
        (predictions["scenario"] == module.TEMPERATURE_SCENARIO)
        & (predictions["fold_id"] == "temperature_c=60")
        & (predictions["target_condition_id"] == "NAUMANN_CAL_T60_SOC0")
        & (predictions["prefix_checkups"] == module.PRIMARY_PREFIX)
    )
    assert target_mask.any()
    tampered = predictions.loc[~target_mask].reset_index(drop=True)
    frozen = (
        v3.calendar_v3_prediction_sha256(tampered)
        if module is v3
        else v2.calendar_v2_prediction_sha256(tampered)
    )
    assert len(frozen) == 64

    with pytest.raises(
        ValueError,
        match=f"Calendar {version} prediction target coverage",
    ):
        score(tampered, observations)


def test_rehashed_sensitivity_pack_missing_whole_target_is_rejected(
    observations: pd.DataFrame,
) -> None:
    predictions = _frozen_sensitivity_prediction_pack(observations)
    target_mask = (
        (predictions["scenario"] == v3.TEMPERATURE_SCENARIO)
        & (predictions["fold_id"] == "temperature_c=60")
        & (predictions["target_condition_id"] == "NAUMANN_CAL_T60_SOC0")
    )
    assert target_mask.any()
    tampered = predictions.loc[~target_mask].reset_index(drop=True)
    frozen = v3.calendar_v3_sensitivity_sha256(tampered)
    assert len(frozen) == 64

    with pytest.raises(
        ValueError,
        match="Calendar V3 sensitivity target coverage",
    ):
        v3.score_calendar_v3_sensitivity(
            tampered,
            observations,
            frozen_prediction_sha256=frozen,
        )
