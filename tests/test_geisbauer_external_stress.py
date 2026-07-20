from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
    geisbauer_calendar_observations_sha256,
    load_geisbauer_calendar_observations,
)
from lifetwin.experiments.geisbauer_external_stress import (
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    HIERARCHICAL_POWER_METHOD,
    LONG_TERM_CONFIRMATION_STATUS,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARATOR,
    default_geisbauer_external_stress_protocol,
    generate_geisbauer_external_predictions,
    geisbauer_external_prediction_sha256,
    run_geisbauer_external_stress,
    score_geisbauer_external_predictions,
    validate_geisbauer_external_stress_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = PROJECT_ROOT / "data/external/geisbauer_2022/LFP_Data.csv"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs/experiments/geisbauer_lfp_calendar_external_stress.json"
)


@pytest.fixture(scope="session")
def geisbauer_observations() -> pd.DataFrame:
    observations, audit = load_geisbauer_calendar_observations(TARGET_PATH)
    assert audit["canonical_output_sha256"] == (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    )
    return observations


@pytest.fixture(scope="session")
def geisbauer_protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def geisbauer_run(
    observations: pd.DataFrame,
    geisbauer_observations: pd.DataFrame,
    geisbauer_protocol: dict[str, object],
) -> tuple:
    return run_geisbauer_external_stress(
        observations,
        geisbauer_observations,
        protocol=geisbauer_protocol,
    )


def test_bundled_target_identity_and_locked_protocol(
    geisbauer_observations: pd.DataFrame,
    geisbauer_protocol: dict[str, object],
) -> None:
    assert geisbauer_calendar_observations_sha256(geisbauer_observations) == (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    )
    assert validate_geisbauer_external_stress_protocol(geisbauer_protocol) == (
        default_geisbauer_external_stress_protocol()
    )
    assert TARGET_PATH.stat().st_size == 2_752


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value | {"unexpected": True},
        lambda value: value | {"target_prefix_days": [0, 39, 84]},
        lambda value: value
        | {
            "decision_policy": {
                **value["decision_policy"],
                "independent_long_term_validation_claim_allowed": True,
            }
        },
    ],
)
def test_protocol_changes_fail_closed(mutation) -> None:
    with pytest.raises(ValueError, match="protocol changed"):
        validate_geisbauer_external_stress_protocol(
            mutation(default_geisbauer_external_stress_protocol())
        )


def test_external_screen_is_cell_level_negative_and_claim_bounded(
    geisbauer_run: tuple,
) -> None:
    result, predictions, cell_metrics, condition_summary, comparisons = (
        geisbauer_run
    )
    assert len(predictions) == 120
    assert len(cell_metrics) == 60
    assert len(condition_summary) == 12
    assert result["target_dataset"]["physical_cell_count"] == 15
    assert result["target_dataset"]["condition_count"] == 3
    assert result["target_dataset"]["maximum_observed_days"] == 120.0
    assert result["mechanism_gate"]["gate_ready_physical_cell_count"] == 0
    assert result["mechanism_gate"]["fallback_physical_cell_count"] == 15
    assert not result["mechanism_gate"]["activation_mechanism_tested"]
    assert result["decision"]["long_term_confirmation_status"] == (
        LONG_TERM_CONFIRMATION_STATUS
    )
    assert not result["decision"][
        "independent_long_term_validation_claim_allowed"
    ]
    assert result["descriptive_signal_status"] == (
        "primary_candidate_did_not_outperform_comparator"
    )
    overall = comparisons.set_index("scope").loc["all_cells"]
    assert overall["candidate_method"] == PRIMARY_CANDIDATE
    assert overall["comparator_method"] == PRIMARY_COMPARATOR
    assert overall["candidate_trajectory_iae_pp_mean"] == pytest.approx(
        3.973451,
        abs=1e-6,
    )
    assert overall["comparator_trajectory_iae_pp_mean"] == pytest.approx(
        3.885215,
        abs=1e-6,
    )
    assert overall["mean_paired_delta_iae_pp"] > 0.0


def test_gated_method_is_an_exact_auditable_fallback(
    geisbauer_run: tuple,
) -> None:
    predictions = geisbauer_run[1]
    gated = predictions.loc[
        predictions["method"] == GATED_HIERARCHICAL_ACTIVATION_METHOD
    ].sort_values(["cell_id", "target_checkup_index"], kind="stable")
    fallback = predictions.loc[
        predictions["method"] == HIERARCHICAL_POWER_METHOD
    ].sort_values(["cell_id", "target_checkup_index"], kind="stable")
    np.testing.assert_array_equal(
        gated["predicted_capacity_retention_pct"],
        fallback["predicted_capacity_retention_pct"],
    )
    assert set(gated["fallback_reason"]) == {"specialist_gate_not_ready"}
    assert not gated["activation_component_selected"].any()


def test_target_future_mutation_cannot_change_label_free_predictions(
    observations: pd.DataFrame,
    geisbauer_observations: pd.DataFrame,
    geisbauer_protocol: dict[str, object],
    geisbauer_run: tuple,
) -> None:
    attacked = geisbauer_observations.copy()
    future = attacked["elapsed_days"].isin([84.0, 120.0])
    attacked.loc[future, "capacity_ah"] *= 0.99
    initial = attacked.groupby("cell_id")["capacity_ah"].transform("first")
    attacked["capacity_retention_pct"] = 100.0 * attacked["capacity_ah"] / initial
    attacked["capacity_loss_pct"] = 100.0 - attacked["capacity_retention_pct"]

    attacked_predictions = generate_geisbauer_external_predictions(
        observations,
        attacked,
        protocol=geisbauer_protocol,
    )
    assert geisbauer_calendar_observations_sha256(attacked) != (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    )
    assert geisbauer_external_prediction_sha256(attacked_predictions) == (
        geisbauer_external_prediction_sha256(geisbauer_run[1])
    )


def test_rehashed_prediction_tampering_fails_independent_replay(
    observations: pd.DataFrame,
    geisbauer_observations: pd.DataFrame,
    geisbauer_protocol: dict[str, object],
    geisbauer_run: tuple,
) -> None:
    tampered = geisbauer_run[1].copy()
    first_cell = str(tampered.iloc[0]["cell_id"])
    first_index = int(tampered.iloc[0]["target_checkup_index"])
    matching_fallback_rows = (
        tampered["cell_id"].eq(first_cell)
        & tampered["target_checkup_index"].eq(first_index)
        & tampered["method"].isin(
            [HIERARCHICAL_POWER_METHOD, GATED_HIERARCHICAL_ACTIVATION_METHOD]
        )
    )
    tampered.loc[
        matching_fallback_rows, "predicted_capacity_retention_pct"
    ] += 1.0
    attacker_hash = geisbauer_external_prediction_sha256(tampered)

    with pytest.raises(ValueError, match="independent replay"):
        score_geisbauer_external_predictions(
            tampered,
            observations,
            geisbauer_observations,
            frozen_prediction_sha256=attacker_hash,
            protocol=geisbauer_protocol,
        )


def test_scoring_rejects_a_changed_target_outcome_snapshot(
    observations: pd.DataFrame,
    geisbauer_observations: pd.DataFrame,
    geisbauer_protocol: dict[str, object],
    geisbauer_run: tuple,
) -> None:
    attacked = geisbauer_observations.copy()
    row = attacked.index[-1]
    attacked.loc[row, "capacity_ah"] *= 0.99
    cell_id = attacked.loc[row, "cell_id"]
    initial = float(
        attacked.loc[
            (attacked["cell_id"] == cell_id)
            & (attacked["checkup_index"] == 0),
            "capacity_ah",
        ].iloc[0]
    )
    retention = 100.0 * float(attacked.loc[row, "capacity_ah"]) / initial
    attacked.loc[row, "capacity_retention_pct"] = retention
    attacked.loc[row, "capacity_loss_pct"] = 100.0 - retention

    with pytest.raises(ValueError, match="target outcome snapshot mismatch"):
        score_geisbauer_external_predictions(
            geisbauer_run[1],
            observations,
            attacked,
            frozen_prediction_sha256=geisbauer_external_prediction_sha256(
                geisbauer_run[1]
            ),
            protocol=geisbauer_protocol,
        )
