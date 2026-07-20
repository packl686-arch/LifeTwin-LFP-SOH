from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

import lifetwin.experiments.calendar_v3_activation_development as experiment
from lifetwin.models.calendar_v2 import HIERARCHICAL_POWER_METHOD
from lifetwin.models.calendar_v3_activation import (
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    GATED_TARGET_ACTIVATION_METHOD,
    HIERARCHICAL_ACTIVATION_METHOD,
    TARGET_ACTIVATION_METHOD,
    activation_mechanism_gate,
)


def _target_state_inputs(
    observations: pd.DataFrame,
    config: Mapping[str, object],
    *,
    prefix_checkups: int,
    target_condition_id: str,
) -> dict[str, object]:
    target_ids = set(experiment.SOC_TARGET_CONDITIONS)
    training = experiment._select_prefix(
        observations.loc[~observations["condition_id"].isin(target_ids)],
        prefix_checkups,
    )
    model = config["model"]
    assert isinstance(model, Mapping)
    return {
        "scenario": experiment.SOC_SCENARIO,
        "fold_id": "fault_injection",
        "target": observations.loc[
            observations["condition_id"] == target_condition_id
        ],
        "prefix_checkups": prefix_checkups,
        "training_state_sha256": "0" * 64,
        "base_prior": experiment._base_prior(training, model),
        "activation_prior": experiment._activation_prior(training, model),
        "config": config,
    }


def _method_values(predictions: pd.DataFrame, method: str) -> np.ndarray:
    return (
        predictions.loc[predictions["method"] == method]
        .sort_values("target_checkup_index", kind="stable")[
            "predicted_capacity_retention_pct"
        ]
        .to_numpy(dtype=float)
    )


def test_gate_rejects_a_prefix_without_positive_time_points() -> None:
    frame = pd.DataFrame(
        {
            "condition_id": ["NO_POSITIVE_TIME"],
            "temperature_c": [25.0],
            "storage_soc_fraction": [0.5],
            "elapsed_days": [0.0],
            "capacity_loss_pct": [0.0],
        }
    )
    with pytest.raises(ValueError, match="at least one positive-time point"):
        activation_mechanism_gate(frame)


def test_gate_false_specialist_failure_does_not_block_v2_fallback(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _target_state_inputs(
        observations,
        calendar_v3_config,
        prefix_checkups=5,
        target_condition_id="NAUMANN_CAL_T40_SOC37.5",
    )

    def fail_target_specialist(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected target specialist failure")

    monkeypatch.setattr(
        experiment,
        "fit_activation_offset_power_law",
        fail_target_specialist,
    )
    predictions, diagnostic, sensitivity = experiment._target_state(**inputs)

    assert not diagnostic["activation_gate_ready"]
    assert diagnostic["fallback_reason"] == "gate_not_ready"
    assert diagnostic["target_activation_fit_status"] == "failed"
    assert "injected target specialist failure" in diagnostic[
        "target_activation_fit_error"
    ]
    baseline = _method_values(predictions, HIERARCHICAL_POWER_METHOD)
    np.testing.assert_array_equal(
        _method_values(predictions, GATED_TARGET_ACTIVATION_METHOD), baseline
    )
    np.testing.assert_array_equal(
        _method_values(predictions, GATED_HIERARCHICAL_ACTIVATION_METHOD), baseline
    )
    np.testing.assert_array_equal(
        _method_values(predictions, TARGET_ACTIVATION_METHOD), baseline
    )
    assert not predictions.loc[
        predictions["method"] == GATED_TARGET_ACTIVATION_METHOD,
        "activation_component_selected",
    ].any()
    assert sensitivity == []


def test_gate_true_specialist_failures_fall_back_and_sensitivity_survives(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _target_state_inputs(
        observations,
        calendar_v3_config,
        prefix_checkups=experiment.PRIMARY_PREFIX,
        target_condition_id="NAUMANN_CAL_T40_SOC12.5",
    )

    def fail_specialist(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected specialist failure")

    monkeypatch.setattr(
        experiment,
        "fit_activation_offset_power_law",
        fail_specialist,
    )
    monkeypatch.setattr(
        experiment,
        "update_hierarchical_activation_offset",
        fail_specialist,
    )
    predictions, diagnostic, sensitivity = experiment._target_state(**inputs)

    assert diagnostic["activation_gate_ready"]
    assert diagnostic["fallback_reason"].startswith("specialist_fit_failed:")
    assert diagnostic["hierarchical_fallback_reason"].startswith(
        "specialist_fit_failed:"
    )
    assert diagnostic["target_activation_fit_status"] == "failed"
    assert diagnostic["hierarchical_activation_fit_status"] == "failed"
    assert diagnostic["sensitivity_fit_failure_count"] == len(
        experiment.TAU_SENSITIVITY_DAYS
    )

    baseline = _method_values(predictions, HIERARCHICAL_POWER_METHOD)
    for method in (
        TARGET_ACTIVATION_METHOD,
        HIERARCHICAL_ACTIVATION_METHOD,
        GATED_TARGET_ACTIVATION_METHOD,
        GATED_HIERARCHICAL_ACTIVATION_METHOD,
    ):
        np.testing.assert_array_equal(_method_values(predictions, method), baseline)
        assert not predictions.loc[
            predictions["method"] == method,
            "activation_component_selected",
        ].any()

    assert len(sensitivity) == len(experiment.TAU_SENSITIVITY_DAYS) * (
        35 - experiment.PRIMARY_PREFIX
    )
    assert not any(row["activation_component_selected"] for row in sensitivity)
