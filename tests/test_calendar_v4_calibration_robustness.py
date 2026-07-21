from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np
import pandas as pd
import pytest

from lifetwin.experiments.calendar_v4_calibration_robustness import (
    CANDIDATE_POOL_CONDITION_IDS,
    DIAGNOSTIC_AVAILABLE,
    DIAGNOSTIC_UNAVAILABLE,
    EXPECTED_PARTITION_COUNT,
    FALLBACK_ROUTE,
    PRIMARY_COVERAGE,
    SPECIALIST_ROUTE,
    build_calendar_v4_candidate_predictions,
    calendar_v4_candidate_prediction_sha256,
    run_calendar_v4_calibration_robustness,
    score_calendar_v4_candidate_predictions,
    validate_calendar_v4_calibration_robustness_config,
)
from lifetwin.experiments.calendar_v4_hybrid_development import (
    FORECAST_INDICES,
    FORECAST_START_INDEX,
)
from scripts import run_calendar_v4_calibration_robustness as audit_runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/naumann_calendar_v4_hybrid_development.json"
)
AUDIT_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/naumann_calendar_v4_calibration_robustness.json"
)


@pytest.fixture(scope="module")
def upstream_config() -> dict[str, object]:
    return json.loads(UPSTREAM_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def audit_config() -> dict[str, object]:
    return json.loads(AUDIT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def robustness_run(
    observations: pd.DataFrame,
    upstream_config: dict[str, object],
    audit_config: dict[str, object],
) -> tuple:
    return run_calendar_v4_calibration_robustness(
        observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
    )


@pytest.fixture
def writable_root() -> Path:
    root = PROJECT_ROOT / "artifacts/test-scratch" / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root)


def _shift_candidate_future(
    observations: pd.DataFrame, *, delta_retention_pp: float
) -> pd.DataFrame:
    changed = observations.copy(deep=True)
    condition = changed["condition_id"].astype(str)
    future = condition.isin(CANDIDATE_POOL_CONDITION_IDS) & (
        pd.to_numeric(changed["checkup_index"]) >= FORECAST_START_INDEX
    )
    for condition_id in CANDIDATE_POOL_CONDITION_IDS:
        selected_condition = condition == condition_id
        initial_capacity = float(
            changed.loc[
                selected_condition & pd.to_numeric(changed["checkup_index"]).eq(0),
                "capacity_ah",
            ].iloc[0]
        )
        selected = future & selected_condition
        retention = (
            pd.to_numeric(changed.loc[selected, "capacity_retention_pct"])
            + delta_retention_pp
        )
        changed.loc[selected, "capacity_retention_pct"] = retention
        changed.loc[selected, "capacity_loss_pct"] = 100.0 - retention
        changed.loc[selected, "capacity_ah"] = initial_capacity * retention / 100.0
    return changed


def test_protocol_is_fail_closed_and_predictions_are_label_free(
    audit_config: dict[str, object],
    robustness_run: tuple,
) -> None:
    validate_calendar_v4_calibration_robustness_config(audit_config)
    changed = json.loads(json.dumps(audit_config))
    changed["exhaustive_partition_audit"]["expected_partition_count"] = 209
    with pytest.raises(ValueError, match="locked protocol"):
        validate_calendar_v4_calibration_robustness_config(changed)

    result, predictions = robustness_run[:2]
    assert len(predictions) == len(CANDIDATE_POOL_CONDITION_IDS) * len(FORECAST_INDICES)
    assert not any(
        "true" in column or "observed" in column or "capacity_loss" in column
        for column in predictions.columns
    )
    assert predictions.groupby("candidate_condition_id").size().eq(25).all()
    assert (
        predictions.groupby("candidate_condition_id")["mean_route"]
        .nunique()
        .eq(1)
        .all()
    )
    assert (
        result["future_label_firewall"][
            "candidate_prediction_pack_frozen_before_candidate_scoring"
        ]
        is True
    )
    assert result["interpretation"]["operational_interval_issued"] is False


def test_route_support_and_finite_sample_limits_are_exposed(
    robustness_run: tuple,
) -> None:
    (
        _,
        _,
        condition_scores,
        baseline_routes,
        _,
        loco_routes,
        _,
        partition_catalog,
        partition_routes,
        partition_conditions,
        summary,
    ) = robustness_run
    assert condition_scores["mean_route"].value_counts().to_dict() == {
        FALLBACK_ROUTE: 8,
        SPECIALIST_ROUTE: 2,
    }
    baseline_80 = baseline_routes.loc[
        np.isclose(baseline_routes["requested_coverage"], PRIMARY_COVERAGE)
    ].set_index("mean_route")
    assert baseline_80.loc[FALLBACK_ROUTE, "calibration_condition_count"] == 5
    assert baseline_80.loc[FALLBACK_ROUTE, "order_statistic_rank"] == 5
    assert np.isfinite(float(baseline_80.loc[FALLBACK_ROUTE, "multiplier"]))
    assert baseline_80.loc[SPECIALIST_ROUTE, "calibration_condition_count"] == 1
    assert pd.isna(baseline_80.loc[SPECIALIST_ROUTE, "multiplier"])

    assert len(loco_routes) == 6 * 2 * 3
    loco_80 = loco_routes.loc[
        np.isclose(loco_routes["requested_coverage"], PRIMARY_COVERAGE)
    ]
    fallback_loco = loco_80.loc[loco_80["mean_route"] == FALLBACK_ROUTE]
    specialist_loco = loco_80.loc[loco_80["mean_route"] == SPECIALIST_ROUTE]
    assert fallback_loco["multiplier"].notna().all()
    assert specialist_loco["multiplier"].isna().all()
    assert fallback_loco["multiplier"].nunique() >= 2

    assert len(partition_catalog) == EXPECTED_PARTITION_COUNT
    assert partition_catalog["partition_id"].is_unique
    assert set(partition_catalog["fallback_calibration_count"]) == {4, 5, 6}
    assert set(partition_catalog["specialist_calibration_count"]) == {0, 1, 2}
    assert len(partition_routes) == EXPECTED_PARTITION_COUNT * 2 * 3
    assert len(partition_conditions) == EXPECTED_PARTITION_COUNT * 4 * 3

    summary_80 = summary.loc[
        np.isclose(summary["requested_coverage"], PRIMARY_COVERAGE)
    ].set_index("mean_route")
    assert (
        summary_80.loc[FALLBACK_ROUTE, "finite_multiplier_scenario_count"]
        == EXPECTED_PARTITION_COUNT
    )
    assert summary_80.loc[SPECIALIST_ROUTE, "finite_multiplier_scenario_count"] == 0
    higher = summary.loc[summary["requested_coverage"] > PRIMARY_COVERAGE]
    assert higher["finite_multiplier_scenario_count"].eq(0).all()
    required_by_coverage = (
        summary.groupby("requested_coverage")[
            "minimum_calibration_count_for_finite_quantile"
        ]
        .first()
        .to_dict()
    )
    assert required_by_coverage == {0.8: 4, 0.9: 9, 0.95: 19}
    assert (
        summary_80.loc[FALLBACK_ROUTE, "multiplier_min"]
        < summary_80.loc[FALLBACK_ROUTE, "multiplier_max"]
    )
    for row in summary.itertuples(index=False):
        selected = partition_routes.loc[
            partition_routes["mean_route"].astype(str).eq(row.mean_route)
            & np.isclose(
                partition_routes["requested_coverage"].to_numpy(dtype=float),
                row.requested_coverage,
                rtol=0.0,
                atol=1e-12,
            )
            & partition_routes["evaluation_condition_count"].gt(0)
        ]
        assert row.all_evaluation_available_and_covered_scenario_count == int(
            selected["all_evaluation_available_and_covered"].sum()
        )


def test_original_calibration_scores_preserve_v011_semantics(
    robustness_run: tuple,
) -> None:
    candidate_scores = robustness_run[2]
    published = pd.read_csv(
        PROJECT_ROOT / "showcase/evidence_v011/v4/calibration_condition_scores.csv"
    )
    compared = candidate_scores.merge(
        published,
        left_on="candidate_condition_id",
        right_on="calibration_condition_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_audit", "_v011"),
    )
    assert len(compared) == 6
    assert (compared["mean_route_audit"] == compared["mean_route_v011"]).all()
    np.testing.assert_allclose(
        compared["maximum_standardized_error_audit"].to_numpy(dtype=float),
        compared["maximum_standardized_error_v011"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        compared["maximum_absolute_error_pp_audit"].to_numpy(dtype=float),
        compared["maximum_absolute_error_pp_v011"].to_numpy(dtype=float),
        rtol=0.0,
        atol=5e-15,
    )


def test_unavailable_route_intervals_remain_null_and_not_failed_coverage(
    robustness_run: tuple,
) -> None:
    partition_routes = robustness_run[8]
    partition_conditions = robustness_run[9]
    unavailable = partition_conditions.loc[
        partition_conditions["diagnostic_interval_status"] == DIAGNOSTIC_UNAVAILABLE
    ]
    assert (
        unavailable[
            [
                "trajectory_simultaneously_covered",
                "pointwise_coverage_fraction",
                "diagnostic_mean_width_pp",
                "diagnostic_max_width_pp",
                "diagnostic_interval_score_mean",
            ]
        ]
        .isna()
        .all()
        .all()
    )
    available = partition_conditions.loc[
        partition_conditions["diagnostic_interval_status"] == DIAGNOSTIC_AVAILABLE
    ]
    assert not available.empty
    assert (
        available[
            [
                "trajectory_simultaneously_covered",
                "pointwise_coverage_fraction",
                "diagnostic_mean_width_pp",
                "diagnostic_max_width_pp",
                "diagnostic_interval_score_mean",
            ]
        ]
        .notna()
        .all()
        .all()
    )

    with_evaluation = partition_routes.loc[
        partition_routes["evaluation_condition_count"] > 0
    ]
    incomplete = with_evaluation.loc[
        with_evaluation["diagnostic_available_evaluation_count"]
        < with_evaluation["evaluation_condition_count"]
    ]
    assert not incomplete.empty
    assert incomplete["all_evaluation_simultaneously_covered"].isna().all()
    assert incomplete["all_evaluation_available_and_covered"].eq(False).all()

    complete = with_evaluation.loc[
        with_evaluation["diagnostic_available_evaluation_count"]
        == with_evaluation["evaluation_condition_count"]
    ]
    assert not complete.empty
    assert complete["all_evaluation_simultaneously_covered"].notna().all()
    assert (
        complete["all_evaluation_available_and_covered"].astype(bool)
        == complete["all_evaluation_simultaneously_covered"].astype(bool)
    ).all()


def test_candidate_future_labels_cannot_change_prediction_pack(
    observations: pd.DataFrame,
    upstream_config: dict[str, object],
    audit_config: dict[str, object],
    robustness_run: tuple,
) -> None:
    original_predictions = robustness_run[1]
    original_scores = robustness_run[2]
    attacked_observations = _shift_candidate_future(
        observations, delta_retention_pp=-0.5
    )
    attacked_predictions = build_calendar_v4_candidate_predictions(
        attacked_observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
    )
    pd.testing.assert_frame_equal(original_predictions, attacked_predictions)
    attacked_scores = score_calendar_v4_candidate_predictions(
        original_predictions,
        attacked_observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
        frozen_prediction_sha256=calendar_v4_candidate_prediction_sha256(
            original_predictions
        ),
    )
    assert not np.allclose(
        original_scores["maximum_standardized_error"].to_numpy(dtype=float),
        attacked_scores["maximum_standardized_error"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    )


def test_rehashed_prediction_tampering_is_rejected_by_regeneration(
    observations: pd.DataFrame,
    upstream_config: dict[str, object],
    audit_config: dict[str, object],
    robustness_run: tuple,
) -> None:
    attacked = robustness_run[1].copy(deep=True)
    attacked.loc[0, "predicted_capacity_retention_pct"] = (
        float(attacked.loc[0, "predicted_capacity_retention_pct"]) - 0.01
    )
    attacker_hash = calendar_v4_candidate_prediction_sha256(attacked)
    with pytest.raises(ValueError, match="deterministic regeneration"):
        score_calendar_v4_candidate_predictions(
            attacked,
            observations,
            upstream_config=upstream_config,
            audit_config=audit_config,
            frozen_prediction_sha256=attacker_hash,
        )


def test_claim_boundary_keeps_all_results_retrospective(
    robustness_run: tuple,
) -> None:
    result = robustness_run[0]
    assert result["confirmation"]["status"] == "not_confirmed"
    assert result["confirmation"]["15_to_25_year_claim_allowed"] is False
    assert result["input_integrity"]["canonical_outcome_snapshot_verified"] is True
    assert (
        result["input_integrity"]["canonical_outcome_sha256"]
        == result["input_integrity"]["expected_canonical_outcome_sha256"]
    )
    assert result["input_integrity"]["enforcement_scope"] == (
        "top_level_run_only_build_and_score_remain_separately_testable"
    )
    assert result["design"]["partition_outcomes_are_overlapping"] is True
    assert result["design"]["partition_results_are_independent_replications"] is False
    assert result["interpretation"]["formal_coverage_claim_allowed"] is False
    assert result["interpretation"]["coverage_fraction_denominator"] == (
        "overlapping_condition_partition_evaluation_instances"
    )
    assert (
        result["interpretation"][
            "coverage_fraction_is_effective_independent_sample_estimate"
        ]
        is False
    )
    assert (
        result["interpretation"][
            "all_evaluation_simultaneously_covered_is_null_when_any_"
            "interval_unavailable"
        ]
        is True
    )
    assert (
        result["interpretation"]["joint_availability_and_coverage_gate"]
        == "all_evaluation_available_and_covered"
    )
    assert (
        "formal_finite_sample_coverage_on_reused_naumann_data"
        in result["prohibited_claims"]
    )
    assert "not independent trials" in result["claim_boundary"]


def test_top_level_core_run_rejects_canonical_outcome_tampering(
    observations: pd.DataFrame,
    upstream_config: dict[str, object],
    audit_config: dict[str, object],
) -> None:
    attacked = _shift_candidate_future(
        observations,
        delta_retention_pp=-0.5,
    )
    with pytest.raises(ValueError, match="canonical outcome snapshot mismatch"):
        run_calendar_v4_calibration_robustness(
            attacked,
            upstream_config=upstream_config,
            audit_config=audit_config,
        )


def test_runner_writes_complete_non_overwriting_evidence_bundle(
    writable_root: Path,
) -> None:
    output_dir = writable_root / "robustness"
    result = audit_runner.run(
        PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv",
        UPSTREAM_CONFIG_PATH,
        AUDIT_CONFIG_PATH,
        output_dir,
    )
    expected_rows = {
        "candidate_label_free_predictions": 250,
        "candidate_condition_scores": 10,
        "baseline_route_metrics": 6,
        "baseline_condition_metrics": 12,
        "loco_route_metrics": 36,
        "loco_condition_metrics": 72,
        "partition_catalog": 210,
        "partition_route_metrics": 1260,
        "partition_condition_metrics": 2520,
        "sensitivity_summary": 6,
    }
    assert set(result["artifacts"]) == set(expected_rows)
    for name, row_count in expected_rows.items():
        metadata = result["artifacts"][name]
        artifact_path = Path(metadata["path"])
        assert artifact_path.is_file()
        assert metadata["row_count"] == row_count
        assert (
            metadata["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        )
    persisted = json.loads((output_dir / "result.json").read_text("utf-8"))
    assert persisted["confirmation"]["15_to_25_year_claim_allowed"] is False
    with pytest.raises(FileExistsError, match="never overwrites"):
        audit_runner.run(
            PROJECT_ROOT / "data/interim/naumann_calendar_observations.csv",
            UPSTREAM_CONFIG_PATH,
            AUDIT_CONFIG_PATH,
            output_dir,
        )
