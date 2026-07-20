from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_v4_hybrid_development import (
    CALIBRATION_CONDITION_IDS,
    DIAGNOSTIC_AVAILABLE,
    DIAGNOSTIC_UNAVAILABLE,
    FALLBACK_ROUTE,
    FORECAST_END_INDEX,
    FORECAST_INDICES,
    FORECAST_START_INDEX,
    OPERATIONAL_ABSTAINED,
    PRIMARY_COVERAGE,
    SPECIALIST_ROUTE,
    TEST_CONDITION_IDS,
    TRAINING_CONDITION_IDS,
    build_calendar_v4_label_free_predictions,
    calendar_v4_prediction_sha256,
    run_calendar_v4_hybrid_development,
    score_calendar_v4_predictions,
    validate_calendar_v4_hybrid_config,
    validate_calendar_v4_split_and_rank,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)


@pytest.fixture(scope="module")
def calendar_v4_config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def calendar_v4_run(
    observations: pd.DataFrame,
    calendar_v4_config: dict[str, object],
) -> tuple:
    return run_calendar_v4_hybrid_development(
        observations,
        config=calendar_v4_config,
    )


def _shift_future_capacity(
    observations: pd.DataFrame,
    condition_ids: tuple[str, ...],
    *,
    delta_retention_pp: float,
) -> pd.DataFrame:
    changed = observations.copy(deep=True)
    condition_column = changed["condition_id"].astype(str)
    future_mask = condition_column.isin(condition_ids) & (
        pd.to_numeric(changed["checkup_index"]) >= FORECAST_START_INDEX
    )
    for condition_id in condition_ids:
        condition_mask = condition_column == condition_id
        initial_capacity = float(
            changed.loc[
                condition_mask
                & (pd.to_numeric(changed["checkup_index"]) == 0),
                "capacity_ah",
            ].iloc[0]
        )
        selected = future_mask & condition_mask
        retention = (
            pd.to_numeric(changed.loc[selected, "capacity_retention_pct"])
            + delta_retention_pp
        )
        changed.loc[selected, "capacity_retention_pct"] = retention
        changed.loc[selected, "capacity_loss_pct"] = 100.0 - retention
        changed.loc[selected, "capacity_ah"] = initial_capacity * retention / 100.0
    return changed


def test_locked_split_rank_and_training_only_residual_provenance(
    observations: pd.DataFrame,
    calendar_v4_run: tuple,
) -> None:
    result, _, residuals, _, _, _, splits = calendar_v4_run
    independently_validated = validate_calendar_v4_split_and_rank(observations)
    pd.testing.assert_frame_equal(splits, independently_validated)
    assert splits.groupby("role").size().to_dict() == {
        "calibration": 6,
        "test": 4,
        "training": 7,
    }
    assert set(splits.loc[splits["role"] == "training", "condition_id"]) == set(
        TRAINING_CONDITION_IDS
    )
    assert set(
        splits.loc[splits["role"] == "calibration", "condition_id"]
    ) == set(CALIBRATION_CONDITION_IDS)
    assert set(splits.loc[splits["role"] == "test", "condition_id"]) == set(
        TEST_CONDITION_IDS
    )

    assert residuals.shape[0] == 7 * len(FORECAST_INDICES)
    assert set(residuals["source_condition_id"]) == set(TRAINING_CONDITION_IDS)
    assert set(residuals["source_condition_role"]) == {"training"}
    assert set(residuals["pseudo_training_condition_count"]) == {6}
    assert residuals.groupby("source_condition_id").size().eq(25).all()
    assert residuals.groupby("source_condition_id")[
        "source_model_state_sha256"
    ].nunique().eq(1).all()
    assert residuals.groupby("source_condition_id")[
        "source_residual_state_sha256"
    ].nunique().eq(1).all()
    for source_id, group in residuals.groupby("source_condition_id"):
        pseudo_ids = set(group["pseudo_training_condition_ids"].iloc[0].split(";"))
        assert source_id not in pseudo_ids
        assert pseudo_ids == set(TRAINING_CONDITION_IDS) - {source_id}
        assert pseudo_ids.isdisjoint(CALIBRATION_CONDITION_IDS)
        assert pseudo_ids.isdisjoint(TEST_CONDITION_IDS)
    assert result["design"]["target_future_outcomes_used_for_prediction"] is False
    assert (
        result["design"]["calibration_outcomes_used_for_mean_or_residual_fit"]
        is False
    )
    assert result["confirmation"]["status"] == "not_confirmed"
    assert result["confirmation"]["15_to_25_year_claim_allowed"] is False


def test_route_stratified_finite_sample_limits_and_honest_abstention(
    calendar_v4_run: tuple,
) -> None:
    result, predictions, _, calibration_scores, quantiles, metrics, _ = (
        calendar_v4_run
    )
    route_counts = calibration_scores.groupby("mean_route").size().to_dict()
    assert route_counts == {FALLBACK_ROUTE: 5, SPECIALIST_ROUTE: 1}

    fallback_80 = quantiles.loc[
        (quantiles["mean_route"] == FALLBACK_ROUTE)
        & np.isclose(quantiles["requested_coverage"], PRIMARY_COVERAGE)
    ].iloc[0]
    assert fallback_80["calibration_condition_count"] == 5
    assert fallback_80["order_statistic_rank"] == 5
    assert np.isfinite(float(fallback_80["multiplier"]))
    specialist_80 = quantiles.loc[
        (quantiles["mean_route"] == SPECIALIST_ROUTE)
        & np.isclose(quantiles["requested_coverage"], PRIMARY_COVERAGE)
    ].iloc[0]
    assert specialist_80["calibration_condition_count"] == 1
    assert specialist_80["order_statistic_rank"] == 2
    assert pd.isna(specialist_80["multiplier"])
    assert quantiles.loc[
        quantiles["requested_coverage"] > PRIMARY_COVERAGE, "multiplier"
    ].isna().all()

    assert set(predictions["operational_issuance_status"]) == {
        OPERATIONAL_ABSTAINED
    }
    assert predictions[
        ["operational_lower_pct", "operational_upper_pct"]
    ].isna().all().all()
    unavailable = predictions.loc[
        predictions["diagnostic_interval_status"] == DIAGNOSTIC_UNAVAILABLE
    ]
    assert unavailable[
        ["diagnostic_lower_pct", "diagnostic_upper_pct", "diagnostic_width_pp"]
    ].isna().all().all()
    available = predictions.loc[
        predictions["diagnostic_interval_status"] == DIAGNOSTIC_AVAILABLE
    ]
    assert available[
        ["diagnostic_lower_pct", "diagnostic_upper_pct", "diagnostic_width_pp"]
    ].notna().all().all()
    primary_metrics = metrics.loc[
        np.isclose(metrics["requested_coverage"], PRIMARY_COVERAGE)
    ]
    assert primary_metrics["diagnostic_interval_status"].value_counts().to_dict() == {
        DIAGNOSTIC_AVAILABLE: 3,
        DIAGNOSTIC_UNAVAILABLE: 1,
    }
    rejected = primary_metrics.loc[
        primary_metrics["diagnostic_interval_status"] == DIAGNOSTIC_UNAVAILABLE
    ]
    assert rejected["diagnostic_simultaneous_covered"].isna().all()
    assert result["calibration"]["formal_coverage_claim_allowed"] is False
    assert result["calibration"]["operational_issued_trajectory_count"] == 0


def test_test_future_labels_cannot_change_frozen_prediction_pack(
    observations: pd.DataFrame,
    calendar_v4_config: dict[str, object],
    calendar_v4_run: tuple,
) -> None:
    _, original, original_residuals, original_scores, original_quantiles, metrics, _ = (
        calendar_v4_run
    )
    attacked_observations = _shift_future_capacity(
        observations,
        TEST_CONDITION_IDS,
        delta_retention_pp=-0.75,
    )
    attacked, residuals, scores, quantiles, _ = (
        build_calendar_v4_label_free_predictions(
            attacked_observations,
            config=calendar_v4_config,
        )
    )
    pd.testing.assert_frame_equal(original, attacked)
    pd.testing.assert_frame_equal(original_residuals, residuals)
    pd.testing.assert_frame_equal(original_scores, scores)
    pd.testing.assert_frame_equal(original_quantiles, quantiles)

    attacked_metrics = score_calendar_v4_predictions(
        original,
        attacked_observations,
        config=calendar_v4_config,
        frozen_prediction_sha256=calendar_v4_prediction_sha256(original),
    )
    assert not np.allclose(
        metrics["point_mae_pp"].to_numpy(dtype=float),
        attacked_metrics["point_mae_pp"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


def test_calibration_future_changes_intervals_but_not_target_mean_or_residual_fit(
    observations: pd.DataFrame,
    calendar_v4_config: dict[str, object],
    calendar_v4_run: tuple,
) -> None:
    _, original, original_residuals, original_scores, original_quantiles, _, _ = (
        calendar_v4_run
    )
    attacked_observations = _shift_future_capacity(
        observations,
        ("NAUMANN_CAL_T40_SOC50",),
        delta_retention_pp=-3.0,
    )
    attacked, residuals, scores, quantiles, _ = (
        build_calendar_v4_label_free_predictions(
            attacked_observations,
            config=calendar_v4_config,
        )
    )
    invariant_columns = [
        *[
            "target_condition_id",
            "prefix_checkups",
            "requested_coverage",
            "target_checkup_index",
        ],
        "mean_route",
        "mean_fallback_reasons",
        "predicted_capacity_retention_pct",
        "predictive_sd_pp",
        "residual_correction_pp",
        "training_state_sha256",
    ]
    pd.testing.assert_frame_equal(
        original[invariant_columns],
        attacked[invariant_columns],
    )
    pd.testing.assert_frame_equal(original_residuals, residuals)
    assert not np.allclose(
        original_scores["maximum_standardized_error"].to_numpy(dtype=float),
        scores["maximum_standardized_error"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
    original_multiplier = original_quantiles.loc[
        (original_quantiles["mean_route"] == FALLBACK_ROUTE)
        & np.isclose(original_quantiles["requested_coverage"], PRIMARY_COVERAGE),
        "multiplier",
    ].iloc[0]
    attacked_multiplier = quantiles.loc[
        (quantiles["mean_route"] == FALLBACK_ROUTE)
        & np.isclose(quantiles["requested_coverage"], PRIMARY_COVERAGE),
        "multiplier",
    ].iloc[0]
    assert not np.isclose(original_multiplier, attacked_multiplier)


def test_rehashed_prediction_tampering_is_rejected_by_independent_regeneration(
    observations: pd.DataFrame,
    calendar_v4_config: dict[str, object],
    calendar_v4_run: tuple,
) -> None:
    predictions = calendar_v4_run[1]
    attacked = predictions.copy(deep=True)
    attacked.loc[0, "residual_correction_pp"] = (
        float(attacked.loc[0, "residual_correction_pp"]) + 0.01
    )
    attacker_hash = calendar_v4_prediction_sha256(attacked)
    with pytest.raises(ValueError, match="independent deterministic regeneration"):
        score_calendar_v4_predictions(
            attacked,
            observations,
            config=calendar_v4_config,
            frozen_prediction_sha256=attacker_hash,
        )


@pytest.mark.parametrize(
    "attack",
    ["coordinate", "operational_bound", "deleted_coordinate"],
)
def test_scorer_rejects_coordinate_bounds_and_support_tampering(
    attack: str,
    observations: pd.DataFrame,
    calendar_v4_config: dict[str, object],
    calendar_v4_run: tuple,
) -> None:
    attacked = calendar_v4_run[1].copy(deep=True)
    if attack == "coordinate":
        attacked.loc[0, "elapsed_days"] = float(attacked.loc[0, "elapsed_days"]) + 1.0
    elif attack == "operational_bound":
        attacked.loc[0, "operational_lower_pct"] = 0.0
    else:
        attacked = attacked.iloc[1:].reset_index(drop=True)
    attacker_hash = calendar_v4_prediction_sha256(attacked)
    with pytest.raises(ValueError):
        score_calendar_v4_predictions(
            attacked,
            observations,
            config=calendar_v4_config,
            frozen_prediction_sha256=attacker_hash,
        )


def test_config_and_forecast_horizon_are_fail_closed(
    calendar_v4_config: dict[str, object],
    calendar_v4_run: tuple,
) -> None:
    attacked_config = json.loads(json.dumps(calendar_v4_config))
    attacked_config["landmark"]["forecast_end_checkup_index"] = 35
    with pytest.raises(ValueError, match="locked retrospective protocol"):
        validate_calendar_v4_hybrid_config(attacked_config)
    predictions = calendar_v4_run[1]
    assert set(predictions["target_checkup_index"].astype(int)) == set(
        range(FORECAST_START_INDEX, FORECAST_END_INDEX + 1)
    )
    assert float(predictions["residual_support_horizon_days"].max()) < 15 * 365
    assert predictions["operational_abstention_reasons"].str.contains(
        "independent_long_term_evidence_missing", regex=False
    ).all()
