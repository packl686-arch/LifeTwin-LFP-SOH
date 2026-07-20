from __future__ import annotations

import pandas as pd
import pytest

from lifetwin.experiments.calendar_landmark_readiness import (
    COMMON_SUPPORT_METRIC_COLUMNS,
    CONFIRMATION_STATUS,
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
    default_landmark_readiness_protocol,
    run_landmark_readiness,
    validate_landmark_readiness_protocol,
)
from lifetwin.experiments.calendar_v3_activation_development import (
    EXPECTED_PREFIXES,
    GATE_SCENARIOS,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARATOR,
)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value | {"unexpected": True}, "keys must be exact"),
        (
            lambda value: value
            | {"landmark_prefix_checkups": [5, 8, 10]},
            "Landmark prefixes must remain",
        ),
        (
            lambda value: value
            | {
                "common_support": {
                    **value["common_support"],
                    "start_checkup_index": 13,
                }
            },
            "Common support must remain",
        ),
        (
            lambda value: value
            | {"dataset_relationship": "independent_external_replication"},
            "dataset_relationship must remain",
        ),
    ],
)
def test_protocol_validation_fails_closed(mutation, message: str) -> None:
    protocol = mutation(default_landmark_readiness_protocol())
    with pytest.raises(ValueError, match=message):
        validate_landmark_readiness_protocol(protocol)


def test_default_protocol_returns_independent_copies() -> None:
    first = default_landmark_readiness_protocol()
    second = default_landmark_readiness_protocol()
    first["common_support"]["start_checkup_index"] = 13
    assert second["common_support"]["start_checkup_index"] == 14
    assert validate_landmark_readiness_protocol(second) == second


def test_validated_protocol_does_not_alias_the_callers_nested_state() -> None:
    protocol = default_landmark_readiness_protocol()
    validated = validate_landmark_readiness_protocol(protocol)

    validated["common_support"]["start_checkup_index"] = 13
    validated["landmark_prefix_checkups"].append(99)

    assert protocol["common_support"]["start_checkup_index"] == 14
    assert protocol["landmark_prefix_checkups"] == list(EXPECTED_PREFIXES)


def test_v3_predictions_are_scored_on_identical_authoritative_support(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
) -> None:
    metrics, summary, decision = run_landmark_readiness(
        observations,
        v3_config=calendar_v3_config,
        protocol=default_landmark_readiness_protocol(),
    )

    assert list(metrics.columns) == COMMON_SUPPORT_METRIC_COLUMNS
    assert len(metrics) == 21 * len(EXPECTED_PREFIXES) * 2
    assert set(metrics["prefix_checkups"]) == set(EXPECTED_PREFIXES)
    assert set(metrics["method"]) == {PRIMARY_CANDIDATE, PRIMARY_COMPARATOR}
    assert set(metrics["common_support_start_checkup_index"]) == {14}
    assert set(metrics["common_support_end_checkup_index"]) == {34}
    assert set(metrics["common_support_point_count"]) == {21}
    assert metrics.groupby("prefix_checkups")[
        ["common_support_start_days", "common_support_end_days"]
    ].nunique().eq(1).all().all()
    assert len(summary) == len(EXPECTED_PREFIXES) * len(GATE_SCENARIOS)
    assert decision["retrospective_signal_landmark"] == 10
    assert decision["confirmed_earliest_landmark"] is None
    assert decision["confirmation_status"] == CONFIRMATION_STATUS
    assert decision["authoritative_outcome_sha256"] == (
        EXPECTED_CANONICAL_OUTCOME_SHA256
    )
    assert decision["prediction_input_accepted_from_caller"] is False


def test_landmark_run_rejects_any_outcome_snapshot_mutation(
    observations: pd.DataFrame,
    calendar_v3_config: dict[str, object],
) -> None:
    tampered = observations.copy()
    row = tampered.index[-1]
    condition_id = tampered.loc[row, "condition_id"]
    initial_capacity = float(
        tampered.loc[
            (tampered["condition_id"] == condition_id)
            & (tampered["checkup_index"] == 0),
            "capacity_ah",
        ].iloc[0]
    )
    tampered.loc[row, "capacity_ah"] = float(tampered.loc[row, "capacity_ah"]) + 0.01
    retention = 100.0 * float(tampered.loc[row, "capacity_ah"]) / initial_capacity
    tampered.loc[row, "capacity_retention_pct"] = retention
    tampered.loc[row, "capacity_loss_pct"] = 100.0 - retention

    assert canonical_naumann_outcome_sha256(tampered) != (
        EXPECTED_CANONICAL_OUTCOME_SHA256
    )
    with pytest.raises(ValueError, match="outcome snapshot mismatch"):
        run_landmark_readiness(
            tampered,
            v3_config=calendar_v3_config,
            protocol=default_landmark_readiness_protocol(),
        )
