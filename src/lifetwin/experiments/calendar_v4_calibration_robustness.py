from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.naumann import (
    NAUMANN_CALENDAR_DATASET_ID,
    NAUMANN_STATISTICAL_UNIT,
    validate_naumann_calendar_observations,
)
from lifetwin.experiments.calendar_v2_development import (
    EXPECTED_DATASET_SNAPSHOT_ID,
    EXPECTED_LABEL_VERSION,
)
from lifetwin.experiments.calendar_landmark_readiness import (
    EXPECTED_CANONICAL_OUTCOME_SHA256,
    canonical_naumann_outcome_sha256,
)
from lifetwin.experiments.calendar_v4_hybrid_development import (
    CALIBRATION_CONDITION_IDS,
    EXPERIMENT_ID as UPSTREAM_EXPERIMENT_ID,
    FALLBACK_ROUTE,
    FORECAST_INDICES,
    PREFIX_CHECKUPS,
    REQUESTED_COVERAGES,
    ROUTES,
    SPECIALIST_ROUTE,
    SPLIT_ID as UPSTREAM_SPLIT_ID,
    TEST_CONDITION_IDS,
    TRAINING_CONDITION_IDS,
    _build_training_state,
    _canonical_json_sha256,
    _condition_frame,
    _domain_hull,
    _predict_condition,
    _reason_string,
    validate_calendar_v4_hybrid_config,
)
from lifetwin.models.calendar_v2_uncertainty import (
    finite_sample_higher_quantile,
    interval_score,
)


EXPERIMENT_ID = "naumann_calendar_v4_calibration_robustness_v1"
DESIGN_STATUS = "retrospective_protocol_locked_after_v011_result_inspection"
SELECTION_STATUS = "post_hoc_robustness_audit"
EVIDENCE_ROLE = "reused_naumann_calibration_partition_sensitivity_only"
CONFIRMATION_STATUS = "not_confirmed"
PRIMARY_COVERAGE = 0.8
EXPECTED_PARTITION_COUNT = 210
EXPECTED_UPSTREAM_PROTOCOL_SHA256 = (
    "a7ab156fba62229b4223cf91c681bac2402563d1b11cc64c29668392a4182661"
)
EXPECTED_INPUT_FILE_SHA256 = (
    "73e7f3c155aed3da7ae637f6b3b91df3eb1fecc5d19f8702af8da810fd62f47c"
)
EMPTY_ROUTE_STATUS = "unavailable_no_same_route_calibration"
DIAGNOSTIC_AVAILABLE = "available"
DIAGNOSTIC_UNAVAILABLE = "unavailable"

CANDIDATE_POOL_CONDITION_IDS = (
    "NAUMANN_CAL_T25_SOC50",
    "NAUMANN_CAL_T40_SOC0",
    "NAUMANN_CAL_T40_SOC12.5",
    "NAUMANN_CAL_T40_SOC25",
    "NAUMANN_CAL_T40_SOC37.5",
    "NAUMANN_CAL_T40_SOC50",
    "NAUMANN_CAL_T40_SOC62.5",
    "NAUMANN_CAL_T40_SOC75",
    "NAUMANN_CAL_T40_SOC87.5",
    "NAUMANN_CAL_T40_SOC100",
)

EXPECTED_PROHIBITED_CLAIMS = (
    "preregistered_or_outcome_blind_analysis",
    "formal_finite_sample_coverage_on_reused_naumann_data",
    "independent_external_validation",
    "independent_partition_replications",
    "individual_cell_uncertainty",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

CANDIDATE_PREDICTION_KEY_COLUMNS = [
    "candidate_condition_id",
    "target_checkup_index",
]
CANDIDATE_PREDICTION_COLUMNS = [
    "experiment_id",
    "upstream_experiment_id",
    *CANDIDATE_PREDICTION_KEY_COLUMNS,
    "original_role",
    "prefix_checkups",
    "prefix_end_days",
    "temperature_c",
    "storage_soc_fraction",
    "elapsed_days",
    "forecast_horizon_days",
    "mean_route",
    "mean_fallback_reasons",
    "activation_gate_ready",
    "domain_supported",
    "residual_support_ok",
    "residual_cap_hit",
    "residual_correction_pp",
    "predicted_capacity_retention_pct",
    "predictive_sd_pp",
    "mechanistic_training_support_days",
    "residual_support_horizon_days",
    "training_state_sha256",
    "condition_prediction_state_sha256",
]

CONDITION_SCORE_COLUMNS = [
    "candidate_condition_id",
    "original_role",
    "mean_route",
    "future_point_count",
    "maximum_standardized_error",
    "maximum_standardized_error_checkup_index",
    "maximum_absolute_error_pp",
    "point_mae_pp",
    "predictive_sd_min_pp",
    "predictive_sd_max_pp",
    "training_state_sha256",
    "condition_prediction_state_sha256",
]

ROUTE_METRIC_COLUMNS = [
    "audit_family",
    "scenario_id",
    "removed_condition_id",
    "mean_route",
    "requested_coverage",
    "calibration_condition_count",
    "evaluation_condition_count",
    "order_statistic_rank",
    "multiplier",
    "quantile_status",
    "diagnostic_available_evaluation_count",
    "diagnostic_covered_evaluation_count",
    "diagnostic_coverage_fraction_among_available",
    "all_evaluation_diagnostic_available",
    "all_evaluation_simultaneously_covered",
    "all_evaluation_available_and_covered",
    "diagnostic_mean_width_pp",
    "diagnostic_max_width_pp",
    "diagnostic_interval_score_mean",
]

CONDITION_METRIC_COLUMNS = [
    "audit_family",
    "scenario_id",
    "removed_condition_id",
    "evaluation_condition_id",
    "mean_route",
    "requested_coverage",
    "calibration_condition_count",
    "order_statistic_rank",
    "multiplier",
    "quantile_status",
    "diagnostic_interval_status",
    "diagnostic_abstention_reasons",
    "trajectory_simultaneously_covered",
    "pointwise_coverage_fraction",
    "diagnostic_mean_width_pp",
    "diagnostic_max_width_pp",
    "diagnostic_interval_score_mean",
]

PARTITION_CATALOG_COLUMNS = [
    "partition_index",
    "partition_id",
    "calibration_condition_ids",
    "evaluation_condition_ids",
    "calibration_condition_count",
    "evaluation_condition_count",
    "fallback_calibration_count",
    "specialist_calibration_count",
    "fallback_evaluation_count",
    "specialist_evaluation_count",
]

SENSITIVITY_SUMMARY_COLUMNS = [
    "mean_route",
    "requested_coverage",
    "scenario_count",
    "scenario_with_evaluation_count",
    "calibration_condition_count_min",
    "calibration_condition_count_max",
    "minimum_calibration_count_for_finite_quantile",
    "best_case_calibration_count_margin",
    "finite_multiplier_scenario_count",
    "finite_multiplier_scenario_fraction",
    "multiplier_min",
    "multiplier_q25",
    "multiplier_median",
    "multiplier_q75",
    "multiplier_max",
    "diagnostic_available_evaluation_count",
    "diagnostic_total_evaluation_count",
    "diagnostic_covered_evaluation_count",
    "diagnostic_coverage_fraction_among_available",
    "all_evaluation_available_and_covered_scenario_count",
    "all_evaluation_available_and_covered_scenario_fraction",
    "mean_width_min_pp",
    "mean_width_q25_pp",
    "mean_width_median_pp",
    "mean_width_q75_pp",
    "mean_width_max_pp",
]


def default_calendar_v4_calibration_robustness_config() -> dict[str, object]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "selection_status": SELECTION_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "evidence_role": EVIDENCE_ROLE,
        "upstream": {
            "experiment_id": UPSTREAM_EXPERIMENT_ID,
            "split_id": UPSTREAM_SPLIT_ID,
            "canonical_protocol_sha256": EXPECTED_UPSTREAM_PROTOCOL_SHA256,
            "input_file_sha256": EXPECTED_INPUT_FILE_SHA256,
            "canonical_outcome_sha256": EXPECTED_CANONICAL_OUTCOME_SHA256,
            "training_condition_ids": list(TRAINING_CONDITION_IDS),
            "original_calibration_condition_ids": list(CALIBRATION_CONDITION_IDS),
            "original_test_condition_ids": list(TEST_CONDITION_IDS),
            "prefix_checkups": PREFIX_CHECKUPS,
            "forecast_checkup_indices": list(FORECAST_INDICES),
        },
        "candidate_pool_condition_ids": list(CANDIDATE_POOL_CONDITION_IDS),
        "condition_scoring": {
            "route_definition": "v4_selected_mean_route_from_prefix_only",
            "score": ("maximum_standardized_error_over_checkups_10_to_34"),
            "future_truth_used_only_after_candidate_prediction_pack_freeze": True,
            "regenerate_prediction_pack_before_scoring": True,
        },
        "leave_one_calibration_condition_out": {
            "baseline_calibration_condition_ids": list(CALIBRATION_CONDITION_IDS),
            "fixed_evaluation_condition_ids": list(TEST_CONDITION_IDS),
            "removal_unit": "one_calibration_condition_mean_trajectory",
            "expected_scenario_count": len(CALIBRATION_CONDITION_IDS),
        },
        "exhaustive_partition_audit": {
            "candidate_pool_condition_count": len(CANDIDATE_POOL_CONDITION_IDS),
            "calibration_condition_count": 6,
            "evaluation_condition_count": 4,
            "enumeration": ("all_lexicographic_six_of_ten_calibration_combinations"),
            "partition_order": ("candidate_condition_id_unicode_codepoint_ascending"),
            "expected_partition_count": EXPECTED_PARTITION_COUNT,
            "mean_training_state_policy": (
                "fixed_original_seven_condition_training_state"
            ),
            "partition_unit": "condition_mean_trajectory",
        },
        "uncertainty": {
            "requested_coverages": list(REQUESTED_COVERAGES),
            "primary_coverage": PRIMARY_COVERAGE,
            "calibration_partition": "selected_mean_route",
            "quantile_rule": ("ceil((n+1)*coverage)_higher_or_unavailable"),
            "interval_rule": (
                "mean_plus_or_minus_multiplier_times_predictive_sd_"
                "clipped_to_physical_bounds"
            ),
            "physical_bounds_pct": [0.0, 100.0],
            "trajectory_coverage_rule": ("all_25_future_checkups_inside_interval"),
            "width_summary_rule": (
                "mean_pointwise_width_over_each_evaluation_trajectory"
            ),
        },
        "reporting": {
            "split_summary_quantiles": [0.0, 0.25, 0.5, 0.75, 1.0],
            "split_summary_quantile_interpolation": "linear",
            "unavailable_values_are_not_imputed": True,
            "empty_route_calibration_is_reported_unavailable": True,
            "joint_coverage_is_null_when_any_interval_is_unavailable": True,
            "joint_availability_and_coverage_gate": (
                "all_evaluation_available_and_covered"
            ),
            "operational_issuance_enabled": False,
            "formal_coverage_claim_allowed": False,
        },
        "confirmation_status": CONFIRMATION_STATUS,
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }


def validate_calendar_v4_calibration_robustness_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise ValueError("Calendar V4 robustness config must be an object")
    parsed = json.loads(json.dumps(dict(config), allow_nan=False))
    expected = default_calendar_v4_calibration_robustness_config()
    if _canonical_json_sha256(parsed) != _canonical_json_sha256(expected):
        raise ValueError(
            "Calendar V4 robustness config differs from the locked protocol"
        )
    return parsed


def _canonical_frame_sha256(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    sort_by: Sequence[str],
) -> str:
    normalized = (
        frame.loc[:, list(columns)]
        .sort_values(list(sort_by), kind="stable")
        .reset_index(drop=True)
    )
    encoded = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _original_role(condition_id: str) -> str:
    if condition_id in CALIBRATION_CONDITION_IDS:
        return "calibration"
    if condition_id in TEST_CONDITION_IDS:
        return "test"
    raise ValueError(f"Unknown V4 candidate condition: {condition_id}")


def build_calendar_v4_candidate_predictions(
    observations: pd.DataFrame,
    *,
    upstream_config: Mapping[str, object],
    audit_config: Mapping[str, object],
) -> pd.DataFrame:
    """Build prefix-only means and scales for all ten non-training conditions."""
    validate_naumann_calendar_observations(observations)
    parsed_upstream = validate_calendar_v4_hybrid_config(upstream_config)
    validate_calendar_v4_calibration_robustness_config(audit_config)
    if _canonical_json_sha256(parsed_upstream) != EXPECTED_UPSTREAM_PROTOCOL_SHA256:
        raise ValueError("Calendar V4 upstream protocol SHA-256 mismatch")
    ordered = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    ).reset_index(drop=True)
    (
        power_prior,
        activation_prior,
        residual_fit,
        _,
        training_state_sha256,
        _,
        _,
    ) = _build_training_state(ordered, parsed_upstream)
    domain_hull = _domain_hull(ordered)
    rows: list[dict[str, object]] = []
    for condition_id in CANDIDATE_POOL_CONDITION_IDS:
        predicted = _predict_condition(
            _condition_frame(ordered, condition_id),
            power_prior=power_prior,
            activation_prior=activation_prior,
            residual_fit=residual_fit,
            training_state_sha256=training_state_sha256,
            config=parsed_upstream,
            domain_hull=domain_hull,
        )
        horizons = (
            predicted.future["elapsed_days"].to_numpy(dtype=float)
            - predicted.prefix_end_days
        )
        for index, coordinate in enumerate(predicted.future.itertuples(index=False)):
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "upstream_experiment_id": UPSTREAM_EXPERIMENT_ID,
                    "candidate_condition_id": condition_id,
                    "target_checkup_index": int(coordinate.checkup_index),
                    "original_role": _original_role(condition_id),
                    "prefix_checkups": PREFIX_CHECKUPS,
                    "prefix_end_days": predicted.prefix_end_days,
                    "temperature_c": float(coordinate.temperature_c),
                    "storage_soc_fraction": float(coordinate.storage_soc_fraction),
                    "elapsed_days": float(coordinate.elapsed_days),
                    "forecast_horizon_days": float(horizons[index]),
                    "mean_route": predicted.mean_route,
                    "mean_fallback_reasons": _reason_string(
                        predicted.mean_fallback_reasons
                    ),
                    "activation_gate_ready": predicted.activation_gate_ready,
                    "domain_supported": predicted.domain_supported,
                    "residual_support_ok": predicted.residual_support_ok,
                    "residual_cap_hit": predicted.residual_cap_hit,
                    "residual_correction_pp": float(
                        predicted.residual_correction_pp[index]
                    ),
                    "predicted_capacity_retention_pct": float(
                        predicted.predicted_retention_pct[index]
                    ),
                    "predictive_sd_pp": float(predicted.predictive_sd_pp[index]),
                    "mechanistic_training_support_days": float(
                        power_prior.maximum_training_days
                    ),
                    "residual_support_horizon_days": float(
                        residual_fit.support_horizon_days
                    ),
                    "training_state_sha256": training_state_sha256,
                    "condition_prediction_state_sha256": (
                        predicted.prediction_state_sha256
                    ),
                }
            )
    predictions = (
        pd.DataFrame(rows, columns=CANDIDATE_PREDICTION_COLUMNS)
        .sort_values(CANDIDATE_PREDICTION_KEY_COLUMNS, kind="stable")
        .reset_index(drop=True)
    )
    calendar_v4_candidate_prediction_sha256(predictions)
    return predictions


def calendar_v4_candidate_prediction_sha256(predictions: pd.DataFrame) -> str:
    missing = sorted(set(CANDIDATE_PREDICTION_COLUMNS) - set(predictions.columns))
    unknown = sorted(set(predictions.columns) - set(CANDIDATE_PREDICTION_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Calendar V4 candidate prediction schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    expected_keys = {
        (condition_id, checkup_index)
        for condition_id in CANDIDATE_POOL_CONDITION_IDS
        for checkup_index in FORECAST_INDICES
    }
    observed_keys = {
        (str(row.candidate_condition_id), int(row.target_checkup_index))
        for row in predictions.itertuples(index=False)
    }
    if (
        observed_keys != expected_keys
        or predictions.duplicated(CANDIDATE_PREDICTION_KEY_COLUMNS).any()
    ):
        raise ValueError("Calendar V4 candidate prediction support changed")
    numeric = predictions[
        [
            "predicted_capacity_retention_pct",
            "predictive_sd_pp",
            "elapsed_days",
            "forecast_horizon_days",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Calendar V4 candidate predictions must be finite")
    if (numeric["predictive_sd_pp"] <= 0.0).any():
        raise ValueError("Calendar V4 candidate scales must be positive")
    if not numeric["predicted_capacity_retention_pct"].between(0.0, 100.0).all():
        raise ValueError("Calendar V4 candidate means must be physical")
    if (
        not predictions["training_state_sha256"]
        .astype(str)
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
        or not predictions["condition_prediction_state_sha256"]
        .astype(str)
        .str.fullmatch(r"[0-9a-f]{64}")
        .all()
    ):
        raise ValueError("Calendar V4 candidate state hashes are invalid")
    return _canonical_frame_sha256(
        predictions,
        columns=CANDIDATE_PREDICTION_COLUMNS,
        sort_by=CANDIDATE_PREDICTION_KEY_COLUMNS,
    )


def _regenerate_and_validate_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    upstream_config: Mapping[str, object],
    audit_config: Mapping[str, object],
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    observed_sha256 = calendar_v4_candidate_prediction_sha256(predictions)
    if observed_sha256 != frozen_prediction_sha256:
        raise ValueError("Calendar V4 candidate prediction pack changed after freeze")
    regenerated = build_calendar_v4_candidate_predictions(
        observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
    )
    submitted = predictions.sort_values(
        CANDIDATE_PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    expected = regenerated.sort_values(
        CANDIDATE_PREDICTION_KEY_COLUMNS, kind="stable"
    ).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            submitted,
            expected,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError(
            "Candidate pack disagrees with deterministic regeneration"
        ) from exc
    truth = observations[
        ["condition_id", "checkup_index", "capacity_retention_pct"]
    ].rename(
        columns={
            "condition_id": "candidate_condition_id",
            "checkup_index": "target_checkup_index",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    scored = submitted.merge(
        truth,
        on=CANDIDATE_PREDICTION_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if scored["true_capacity_retention_pct"].isna().any():
        raise ValueError("Calendar V4 candidate scoring truth is incomplete")
    scored["absolute_error_pp"] = np.abs(
        pd.to_numeric(scored["predicted_capacity_retention_pct"])
        - pd.to_numeric(scored["true_capacity_retention_pct"])
    )
    scored["standardized_error"] = scored["absolute_error_pp"] / pd.to_numeric(
        scored["predictive_sd_pp"]
    )
    return scored


def _condition_scores(scored_points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition_id, group in scored_points.groupby(
        "candidate_condition_id", sort=True
    ):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        if ordered["target_checkup_index"].astype(int).tolist() != list(
            FORECAST_INDICES
        ):
            raise ValueError("Candidate score trajectories must be complete")
        if ordered["mean_route"].nunique() != 1:
            raise ValueError("Candidate route must be trajectory-level")
        standardized = ordered["standardized_error"].to_numpy(dtype=float)
        maximum_index = int(np.argmax(standardized))
        rows.append(
            {
                "candidate_condition_id": str(condition_id),
                "original_role": str(ordered["original_role"].iloc[0]),
                "mean_route": str(ordered["mean_route"].iloc[0]),
                "future_point_count": len(ordered),
                "maximum_standardized_error": float(standardized[maximum_index]),
                "maximum_standardized_error_checkup_index": int(
                    ordered["target_checkup_index"].iloc[maximum_index]
                ),
                "maximum_absolute_error_pp": float(ordered["absolute_error_pp"].max()),
                "point_mae_pp": float(ordered["absolute_error_pp"].mean()),
                "predictive_sd_min_pp": float(ordered["predictive_sd_pp"].min()),
                "predictive_sd_max_pp": float(ordered["predictive_sd_pp"].max()),
                "training_state_sha256": str(ordered["training_state_sha256"].iloc[0]),
                "condition_prediction_state_sha256": str(
                    ordered["condition_prediction_state_sha256"].iloc[0]
                ),
            }
        )
    return (
        pd.DataFrame(rows, columns=CONDITION_SCORE_COLUMNS)
        .sort_values("candidate_condition_id", kind="stable")
        .reset_index(drop=True)
    )


def score_calendar_v4_candidate_predictions(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    upstream_config: Mapping[str, object],
    audit_config: Mapping[str, object],
    frozen_prediction_sha256: str,
) -> pd.DataFrame:
    scored_points = _regenerate_and_validate_predictions(
        predictions,
        observations,
        upstream_config=upstream_config,
        audit_config=audit_config,
        frozen_prediction_sha256=frozen_prediction_sha256,
    )
    return _condition_scores(scored_points)


def _route_quantile(
    condition_scores: pd.DataFrame,
    calibration_ids: Sequence[str],
    *,
    route: str,
    coverage: float,
) -> dict[str, object]:
    selected = condition_scores.loc[
        condition_scores["candidate_condition_id"].astype(str).isin(calibration_ids)
        & condition_scores["mean_route"].astype(str).eq(route),
        "maximum_standardized_error",
    ].to_numpy(dtype=float)
    if selected.size == 0:
        return {
            "calibration_condition_count": 0,
            "order_statistic_rank": int(math.ceil(coverage)),
            "multiplier": None,
            "quantile_status": EMPTY_ROUTE_STATUS,
        }
    quantile = finite_sample_higher_quantile(selected, coverage=coverage)
    return {
        "calibration_condition_count": quantile.calibration_count,
        "order_statistic_rank": quantile.order_statistic_rank,
        "multiplier": quantile.multiplier,
        "quantile_status": quantile.status,
    }


def _condition_interval_metric(
    scored_points: pd.DataFrame,
    *,
    audit_family: str,
    scenario_id: str,
    removed_condition_id: str | None,
    condition_id: str,
    coverage: float,
    quantile: Mapping[str, object],
    physical_bounds_pct: tuple[float, float],
) -> dict[str, object]:
    selected = scored_points.loc[
        scored_points["candidate_condition_id"].astype(str) == condition_id
    ].sort_values("target_checkup_index", kind="stable")
    if len(selected) != len(FORECAST_INDICES):
        raise ValueError("Every evaluated condition must have 25 scored points")
    route = str(selected["mean_route"].iloc[0])
    reasons: list[str] = []
    multiplier = quantile["multiplier"]
    if multiplier is None:
        reasons.append("insufficient_same_route_calibration")
    if not bool(selected["domain_supported"].all()):
        reasons.append("domain_unsupported")
    if not bool(selected["residual_support_ok"].all()):
        reasons.append("residual_outside_support")
    if bool(selected["residual_cap_hit"].any()):
        reasons.append("residual_cap_hit")
    available = not reasons
    if available:
        mean = selected["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        scale = selected["predictive_sd_pp"].to_numpy(dtype=float)
        truth = selected["true_capacity_retention_pct"].to_numpy(dtype=float)
        radius = float(multiplier) * scale
        lower = np.clip(mean - radius, *physical_bounds_pct)
        upper = np.clip(mean + radius, *physical_bounds_pct)
        covered = (truth >= lower) & (truth <= upper)
        widths = upper - lower
        interval_scores = interval_score(
            truth,
            lower,
            upper,
            coverage=coverage,
        )
        simultaneous: bool | None = bool(covered.all())
        pointwise: float | None = float(covered.mean())
        mean_width: float | None = float(widths.mean())
        max_width: float | None = float(widths.max())
        score_mean: float | None = float(interval_scores.mean())
    else:
        simultaneous = None
        pointwise = None
        mean_width = None
        max_width = None
        score_mean = None
    return {
        "audit_family": audit_family,
        "scenario_id": scenario_id,
        "removed_condition_id": removed_condition_id,
        "evaluation_condition_id": condition_id,
        "mean_route": route,
        "requested_coverage": coverage,
        "calibration_condition_count": int(quantile["calibration_condition_count"]),
        "order_statistic_rank": int(quantile["order_statistic_rank"]),
        "multiplier": multiplier,
        "quantile_status": str(quantile["quantile_status"]),
        "diagnostic_interval_status": (
            DIAGNOSTIC_AVAILABLE if available else DIAGNOSTIC_UNAVAILABLE
        ),
        "diagnostic_abstention_reasons": _reason_string(reasons),
        "trajectory_simultaneously_covered": simultaneous,
        "pointwise_coverage_fraction": pointwise,
        "diagnostic_mean_width_pp": mean_width,
        "diagnostic_max_width_pp": max_width,
        "diagnostic_interval_score_mean": score_mean,
    }


def _evaluate_scenario(
    scored_points: pd.DataFrame,
    condition_scores: pd.DataFrame,
    *,
    audit_family: str,
    scenario_id: str,
    calibration_ids: Sequence[str],
    evaluation_ids: Sequence[str],
    removed_condition_id: str | None,
    physical_bounds_pct: tuple[float, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    route_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for route in ROUTES:
        route_evaluation_ids = tuple(
            condition_id
            for condition_id in evaluation_ids
            if str(
                condition_scores.loc[
                    condition_scores["candidate_condition_id"].astype(str)
                    == condition_id,
                    "mean_route",
                ].iloc[0]
            )
            == route
        )
        for coverage in REQUESTED_COVERAGES:
            quantile = _route_quantile(
                condition_scores,
                calibration_ids,
                route=route,
                coverage=coverage,
            )
            metrics = [
                _condition_interval_metric(
                    scored_points,
                    audit_family=audit_family,
                    scenario_id=scenario_id,
                    removed_condition_id=removed_condition_id,
                    condition_id=condition_id,
                    coverage=coverage,
                    quantile=quantile,
                    physical_bounds_pct=physical_bounds_pct,
                )
                for condition_id in route_evaluation_ids
            ]
            condition_rows.extend(metrics)
            available = [
                row
                for row in metrics
                if row["diagnostic_interval_status"] == DIAGNOSTIC_AVAILABLE
            ]
            covered_count = sum(
                bool(row["trajectory_simultaneously_covered"]) for row in available
            )
            evaluation_count = len(metrics)
            available_count = len(available)
            if evaluation_count == 0:
                all_available: bool | None = None
                all_covered: bool | None = None
                all_available_and_covered: bool | None = None
            else:
                all_available = available_count == evaluation_count
                all_covered = (
                    covered_count == evaluation_count if all_available else None
                )
                all_available_and_covered = bool(
                    all_available and covered_count == evaluation_count
                )
            route_rows.append(
                {
                    "audit_family": audit_family,
                    "scenario_id": scenario_id,
                    "removed_condition_id": removed_condition_id,
                    "mean_route": route,
                    "requested_coverage": coverage,
                    "calibration_condition_count": int(
                        quantile["calibration_condition_count"]
                    ),
                    "evaluation_condition_count": evaluation_count,
                    "order_statistic_rank": int(quantile["order_statistic_rank"]),
                    "multiplier": quantile["multiplier"],
                    "quantile_status": str(quantile["quantile_status"]),
                    "diagnostic_available_evaluation_count": available_count,
                    "diagnostic_covered_evaluation_count": covered_count,
                    "diagnostic_coverage_fraction_among_available": (
                        covered_count / available_count if available_count else None
                    ),
                    "all_evaluation_diagnostic_available": all_available,
                    "all_evaluation_simultaneously_covered": all_covered,
                    "all_evaluation_available_and_covered": (all_available_and_covered),
                    "diagnostic_mean_width_pp": (
                        float(
                            np.mean(
                                [
                                    float(row["diagnostic_mean_width_pp"])
                                    for row in available
                                ]
                            )
                        )
                        if available
                        else None
                    ),
                    "diagnostic_max_width_pp": (
                        max(float(row["diagnostic_max_width_pp"]) for row in available)
                        if available
                        else None
                    ),
                    "diagnostic_interval_score_mean": (
                        float(
                            np.mean(
                                [
                                    float(row["diagnostic_interval_score_mean"])
                                    for row in available
                                ]
                            )
                        )
                        if available
                        else None
                    ),
                }
            )
    return route_rows, condition_rows


def _partition_id(index: int, calibration_ids: Sequence[str]) -> str:
    digest = _canonical_json_sha256(
        {"calibration_condition_ids": list(calibration_ids)}
    )
    return f"partition_{index:03d}_{digest[:12]}"


def _run_partition_audits(
    scored_points: pd.DataFrame,
    condition_scores: pd.DataFrame,
    *,
    audit_config: Mapping[str, object],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    physical_bounds = tuple(
        float(value)
        for value in dict(audit_config["uncertainty"])["physical_bounds_pct"]
    )
    baseline_route_rows, baseline_condition_rows = _evaluate_scenario(
        scored_points,
        condition_scores,
        audit_family="original_fixed_split",
        scenario_id="original_fixed_split",
        calibration_ids=CALIBRATION_CONDITION_IDS,
        evaluation_ids=TEST_CONDITION_IDS,
        removed_condition_id=None,
        physical_bounds_pct=physical_bounds,
    )

    loco_route_rows: list[dict[str, object]] = []
    loco_condition_rows: list[dict[str, object]] = []
    for removed_id in CALIBRATION_CONDITION_IDS:
        calibration_ids = tuple(
            condition_id
            for condition_id in CALIBRATION_CONDITION_IDS
            if condition_id != removed_id
        )
        route_rows, condition_rows = _evaluate_scenario(
            scored_points,
            condition_scores,
            audit_family="leave_one_calibration_condition_out",
            scenario_id=f"loco_{removed_id}",
            calibration_ids=calibration_ids,
            evaluation_ids=TEST_CONDITION_IDS,
            removed_condition_id=removed_id,
            physical_bounds_pct=physical_bounds,
        )
        loco_route_rows.extend(route_rows)
        loco_condition_rows.extend(condition_rows)

    route_lookup = condition_scores.set_index("candidate_condition_id")[
        "mean_route"
    ].astype(str)
    candidate_ids = tuple(sorted(CANDIDATE_POOL_CONDITION_IDS))
    catalog_rows: list[dict[str, object]] = []
    partition_route_rows: list[dict[str, object]] = []
    partition_condition_rows: list[dict[str, object]] = []
    for index, selected in enumerate(itertools.combinations(candidate_ids, 6), start=1):
        calibration_ids = tuple(selected)
        calibration_set = set(calibration_ids)
        evaluation_ids = tuple(
            value for value in candidate_ids if value not in calibration_set
        )
        partition_id = _partition_id(index, calibration_ids)
        calibration_routes = route_lookup.loc[list(calibration_ids)]
        evaluation_routes = route_lookup.loc[list(evaluation_ids)]
        catalog_rows.append(
            {
                "partition_index": index,
                "partition_id": partition_id,
                "calibration_condition_ids": ";".join(calibration_ids),
                "evaluation_condition_ids": ";".join(evaluation_ids),
                "calibration_condition_count": len(calibration_ids),
                "evaluation_condition_count": len(evaluation_ids),
                "fallback_calibration_count": int(
                    calibration_routes.eq(FALLBACK_ROUTE).sum()
                ),
                "specialist_calibration_count": int(
                    calibration_routes.eq(SPECIALIST_ROUTE).sum()
                ),
                "fallback_evaluation_count": int(
                    evaluation_routes.eq(FALLBACK_ROUTE).sum()
                ),
                "specialist_evaluation_count": int(
                    evaluation_routes.eq(SPECIALIST_ROUTE).sum()
                ),
            }
        )
        route_rows, condition_rows = _evaluate_scenario(
            scored_points,
            condition_scores,
            audit_family="exhaustive_six_of_ten_partition",
            scenario_id=partition_id,
            calibration_ids=calibration_ids,
            evaluation_ids=evaluation_ids,
            removed_condition_id=None,
            physical_bounds_pct=physical_bounds,
        )
        partition_route_rows.extend(route_rows)
        partition_condition_rows.extend(condition_rows)

    catalog = (
        pd.DataFrame(catalog_rows, columns=PARTITION_CATALOG_COLUMNS)
        .sort_values("partition_index", kind="stable")
        .reset_index(drop=True)
    )
    if len(catalog) != EXPECTED_PARTITION_COUNT:
        raise RuntimeError("Exhaustive partition count differs from the lock")
    baseline_routes = pd.DataFrame(baseline_route_rows, columns=ROUTE_METRIC_COLUMNS)
    baseline_conditions = pd.DataFrame(
        baseline_condition_rows, columns=CONDITION_METRIC_COLUMNS
    )
    loco_routes = pd.DataFrame(loco_route_rows, columns=ROUTE_METRIC_COLUMNS)
    loco_conditions = pd.DataFrame(
        loco_condition_rows, columns=CONDITION_METRIC_COLUMNS
    )
    partition_routes = pd.DataFrame(partition_route_rows, columns=ROUTE_METRIC_COLUMNS)
    partition_conditions = pd.DataFrame(
        partition_condition_rows, columns=CONDITION_METRIC_COLUMNS
    )
    summary = _summarize_partition_sensitivity(partition_routes)
    return (
        baseline_routes,
        baseline_conditions,
        loco_routes,
        loco_conditions,
        catalog,
        partition_routes,
        partition_conditions,
        summary,
    )


def _distribution(values: pd.Series) -> tuple[float | None, ...]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return (None, None, None, None, None)
    quantiles = finite.quantile([0.0, 0.25, 0.5, 0.75, 1.0], interpolation="linear")
    return tuple(float(value) for value in quantiles.to_numpy(dtype=float))


def _summarize_partition_sensitivity(
    route_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (route, coverage), group in route_metrics.groupby(
        ["mean_route", "requested_coverage"], sort=True
    ):
        scenario_count = len(group)
        with_evaluation = group.loc[group["evaluation_condition_count"] > 0]
        finite = group["multiplier"].notna()
        minimum_required = next(
            count
            for count in range(1, 10_001)
            if math.ceil((count + 1) * float(coverage)) <= count
        )
        maximum_calibration_count = int(group["calibration_condition_count"].max())
        multiplier_distribution = _distribution(group["multiplier"])
        width_distribution = _distribution(group["diagnostic_mean_width_pp"])
        available_count = int(group["diagnostic_available_evaluation_count"].sum())
        covered_count = int(group["diagnostic_covered_evaluation_count"].sum())
        total_evaluation_count = int(group["evaluation_condition_count"].sum())
        all_covered_count = int(
            with_evaluation["all_evaluation_available_and_covered"].astype(bool).sum()
        )
        rows.append(
            {
                "mean_route": str(route),
                "requested_coverage": float(coverage),
                "scenario_count": scenario_count,
                "scenario_with_evaluation_count": len(with_evaluation),
                "calibration_condition_count_min": int(
                    group["calibration_condition_count"].min()
                ),
                "calibration_condition_count_max": int(maximum_calibration_count),
                "minimum_calibration_count_for_finite_quantile": (minimum_required),
                "best_case_calibration_count_margin": (
                    maximum_calibration_count - minimum_required
                ),
                "finite_multiplier_scenario_count": int(finite.sum()),
                "finite_multiplier_scenario_fraction": float(finite.mean()),
                "multiplier_min": multiplier_distribution[0],
                "multiplier_q25": multiplier_distribution[1],
                "multiplier_median": multiplier_distribution[2],
                "multiplier_q75": multiplier_distribution[3],
                "multiplier_max": multiplier_distribution[4],
                "diagnostic_available_evaluation_count": available_count,
                "diagnostic_total_evaluation_count": total_evaluation_count,
                "diagnostic_covered_evaluation_count": covered_count,
                "diagnostic_coverage_fraction_among_available": (
                    covered_count / available_count if available_count else None
                ),
                "all_evaluation_available_and_covered_scenario_count": (
                    all_covered_count
                ),
                "all_evaluation_available_and_covered_scenario_fraction": (
                    all_covered_count / len(with_evaluation)
                    if len(with_evaluation)
                    else None
                ),
                "mean_width_min_pp": width_distribution[0],
                "mean_width_q25_pp": width_distribution[1],
                "mean_width_median_pp": width_distribution[2],
                "mean_width_q75_pp": width_distribution[3],
                "mean_width_max_pp": width_distribution[4],
            }
        )
    return (
        pd.DataFrame(rows, columns=SENSITIVITY_SUMMARY_COLUMNS)
        .sort_values(["mean_route", "requested_coverage"], kind="stable")
        .reset_index(drop=True)
    )


def _value_or_none(value: object) -> object:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _record(frame: pd.DataFrame, *, route: str, coverage: float) -> dict[str, object]:
    selected = frame.loc[
        frame["mean_route"].astype(str).eq(route)
        & np.isclose(
            frame["requested_coverage"].to_numpy(dtype=float),
            coverage,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    if len(selected) != 1:
        raise RuntimeError("Expected one route/coverage summary record")
    return {
        column: _value_or_none(value)
        for column, value in selected.iloc[0].to_dict().items()
    }


def run_calendar_v4_calibration_robustness(
    observations: pd.DataFrame,
    *,
    upstream_config: Mapping[str, object],
    audit_config: Mapping[str, object],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    canonical_outcome_sha256 = canonical_naumann_outcome_sha256(observations)
    if canonical_outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError("Calendar V4 robustness canonical outcome snapshot mismatch")
    parsed_audit = validate_calendar_v4_calibration_robustness_config(audit_config)
    predictions = build_calendar_v4_candidate_predictions(
        observations,
        upstream_config=upstream_config,
        audit_config=parsed_audit,
    )
    prediction_sha256 = calendar_v4_candidate_prediction_sha256(predictions)
    scored_points = _regenerate_and_validate_predictions(
        predictions,
        observations,
        upstream_config=upstream_config,
        audit_config=parsed_audit,
        frozen_prediction_sha256=prediction_sha256,
    )
    condition_scores = _condition_scores(scored_points)
    (
        baseline_routes,
        baseline_conditions,
        loco_routes,
        loco_conditions,
        partition_catalog,
        partition_routes,
        partition_conditions,
        sensitivity_summary,
    ) = _run_partition_audits(
        scored_points,
        condition_scores,
        audit_config=parsed_audit,
    )
    baseline_fallback = _record(
        baseline_routes,
        route=FALLBACK_ROUTE,
        coverage=PRIMARY_COVERAGE,
    )
    baseline_specialist = _record(
        baseline_routes,
        route=SPECIALIST_ROUTE,
        coverage=PRIMARY_COVERAGE,
    )
    fallback_summary = _record(
        sensitivity_summary,
        route=FALLBACK_ROUTE,
        coverage=PRIMARY_COVERAGE,
    )
    specialist_summary = _record(
        sensitivity_summary,
        route=SPECIALIST_ROUTE,
        coverage=PRIMARY_COVERAGE,
    )
    fallback_loco = loco_routes.loc[
        loco_routes["mean_route"].astype(str).eq(FALLBACK_ROUTE)
        & np.isclose(
            loco_routes["requested_coverage"].to_numpy(dtype=float),
            PRIMARY_COVERAGE,
            rtol=0.0,
            atol=1e-12,
        )
    ]
    result: dict[str, object] = {
        "status": "retrospective_calibration_robustness_complete_not_confirmed",
        "execution_status": "completed",
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _canonical_json_sha256(parsed_audit),
        "candidate_prediction_pack_sha256": prediction_sha256,
        "input_integrity": {
            "canonical_outcome_sha256": canonical_outcome_sha256,
            "expected_canonical_outcome_sha256": (EXPECTED_CANONICAL_OUTCOME_SHA256),
            "canonical_outcome_snapshot_verified": True,
            "enforcement_scope": (
                "top_level_run_only_build_and_score_remain_separately_testable"
            ),
        },
        "design": {
            "design_status": DESIGN_STATUS,
            "selection_status": SELECTION_STATUS,
            "evidence_role": EVIDENCE_ROLE,
            "upstream_experiment_id": UPSTREAM_EXPERIMENT_ID,
            "upstream_split_id": UPSTREAM_SPLIT_ID,
            "training_condition_count": len(TRAINING_CONDITION_IDS),
            "candidate_condition_count": len(CANDIDATE_POOL_CONDITION_IDS),
            "loco_scenario_count": len(CALIBRATION_CONDITION_IDS),
            "exhaustive_partition_count": len(partition_catalog),
            "partition_outcomes_are_overlapping": True,
            "partition_results_are_independent_replications": False,
        },
        "future_label_firewall": {
            "candidate_prediction_pack_frozen_before_candidate_scoring": True,
            "scorer_regenerates_candidate_prediction_pack": True,
            "candidate_future_outcome_columns_in_prediction_pack": [],
            "protocol_was_locked_after_v011_outcome_inspection": True,
        },
        "original_split_primary_80pct": {
            "fallback": baseline_fallback,
            "specialist": baseline_specialist,
        },
        "leave_one_calibration_condition_out_primary_80pct": {
            "fallback_scenario_count": len(fallback_loco),
            "fallback_finite_multiplier_scenario_count": int(
                fallback_loco["multiplier"].notna().sum()
            ),
            "fallback_multiplier_min": float(fallback_loco["multiplier"].min()),
            "fallback_multiplier_max": float(fallback_loco["multiplier"].max()),
            "fallback_mean_width_min_pp": float(
                fallback_loco["diagnostic_mean_width_pp"].min()
            ),
            "fallback_mean_width_max_pp": float(
                fallback_loco["diagnostic_mean_width_pp"].max()
            ),
            "specialist_finite_multiplier_scenario_count": int(
                loco_routes.loc[
                    loco_routes["mean_route"].astype(str).eq(SPECIALIST_ROUTE)
                    & np.isclose(
                        loco_routes["requested_coverage"].to_numpy(dtype=float),
                        PRIMARY_COVERAGE,
                        rtol=0.0,
                        atol=1e-12,
                    ),
                    "multiplier",
                ]
                .notna()
                .sum()
            ),
        },
        "exhaustive_partition_primary_80pct": {
            "fallback": fallback_summary,
            "specialist": specialist_summary,
        },
        "interpretation": {
            "mean_model_promoted": False,
            "operational_interval_issued": False,
            "formal_coverage_claim_allowed": False,
            "calibration_support_assessment": (
                "specialist_route_structurally_under_calibrated"
            ),
            "partition_sensitivity_assessment": (
                "fallback_interval_width_and_simultaneous_coverage_depend_on_"
                "which_condition_trajectories_enter_calibration"
            ),
            "appropriate_use": (
                "retrospective_route_support_and_partition_sensitivity_diagnostic_only"
            ),
            "coverage_fraction_denominator": (
                "overlapping_condition_partition_evaluation_instances"
            ),
            "coverage_fraction_is_effective_independent_sample_estimate": False,
            "all_evaluation_simultaneously_covered_is_null_when_any_"
            "interval_unavailable": True,
            "joint_availability_and_coverage_gate": (
                "all_evaluation_available_and_covered"
            ),
        },
        "confirmation": {
            "status": CONFIRMATION_STATUS,
            "independent_long_term_dataset_available": False,
            "15_to_25_year_claim_allowed": False,
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
        "claim_boundary": (
            "All results reuse ten overlapping Naumann condition-mean "
            "trajectories through 885 days. The 210 partitions are a "
            "deterministic sensitivity enumeration, not independent trials, "
            "formal coverage validation, or operational uncertainty evidence."
        ),
    }
    return (
        result,
        predictions,
        condition_scores,
        baseline_routes,
        baseline_conditions,
        loco_routes,
        loco_conditions,
        partition_catalog,
        partition_routes,
        partition_conditions,
        sensitivity_summary,
    )
