"""Post-outcome failure audit for the private dual-clock V3 experiment."""

from __future__ import annotations

from collections import Counter
import math
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_dual_clock_prior_v3 import (
    DECISION_COLUMNS,
    EXPERIMENT_ID,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    TARGET_TRUTH_COLUMNS,
    validate_private_dual_clock_prior_v3_config,
)


CELL_AUDIT_COLUMNS = (
    "experiment_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "temperature_c",
    "dod_fraction",
    "discharge_c_rate",
    "prefix_last_elapsed_days",
    "prefix_last_equivalent_full_cycles",
    "prefix_duty_rate_efc_per_day",
    "future_realized_duty_rate_efc_per_day",
    "absolute_log_duty_rate_drift",
    "prefix_capacity_recovery_count",
    "prefix_max_capacity_recovery_pp",
    "prefix_linear_residual_rms_pp",
    "model_prefix_residual_rms_pp",
    "normalized_condition_distance",
    "mean_abs_v3_v1_disagreement_pp",
    "max_abs_v3_v1_disagreement_pp",
    "endpoint_abs_v3_v1_disagreement_pp",
    "v1_trajectory_iae_pp",
    "v3_trajectory_iae_pp",
    "v3_improvement_vs_v1_pp",
    "v3_regressed_vs_v1",
    "risk_flags",
    "primary_failure_hypothesis",
)
CONDITION_AUDIT_COLUMNS = (
    "experiment_id",
    "outer_condition_id",
    "landmark_visit_count",
    "temperature_c",
    "dod_fraction",
    "discharge_c_rate",
    "cell_count",
    "v1_condition_equal_trajectory_iae_pp",
    "v3_condition_equal_trajectory_iae_pp",
    "v3_improvement_vs_v1_pp",
    "v3_improved_cell_fraction",
    "worst_cell_v3_trajectory_iae_pp",
    "mean_abs_v3_v1_disagreement_pp",
    "mean_absolute_log_duty_rate_drift",
    "maximum_prefix_capacity_recovery_pp",
    "mean_normalized_condition_distance",
    "risk_flagged_cell_fraction",
    "dominant_failure_hypothesis",
)


class PrivateDualClockAuditError(ValueError):
    """Raised when private V3 audit inputs or replay evidence are invalid."""


def _integrated_mean_absolute_difference(
    exposure: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if len(exposure) < 2 or float(exposure[-1]) <= float(exposure[0]):
        return float(np.mean(np.abs(first - second)))
    return float(
        np.trapezoid(np.abs(first - second), exposure)
        / (float(exposure[-1]) - float(exposure[0]))
    )


def _prefix_linear_residual(prefix: pd.DataFrame) -> float:
    x = prefix["equivalent_full_cycles"].to_numpy(dtype=float)
    y = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.sqrt(np.mean(np.square(fitted - y))))


def _spearman_or_none(first: pd.Series, second: pd.Series) -> float | None:
    if first.nunique(dropna=True) < 2 or second.nunique(dropna=True) < 2:
        return None
    value = float(first.corr(second, method="spearman"))
    return value if math.isfinite(value) else None


def _model_curve(
    predictions: pd.DataFrame,
    *,
    outer: str,
    cell_id: str,
    landmark: int,
    model_id: str,
) -> pd.DataFrame:
    curve = predictions.loc[
        (predictions["outer_condition_id"] == outer)
        & (predictions["cell_id"] == cell_id)
        & (predictions["landmark_visit_count"] == landmark)
        & (predictions["model_id"] == model_id)
    ].sort_values("forecast_equivalent_full_cycles", kind="stable")
    if curve.empty:
        raise PrivateDualClockAuditError(
            f"Missing {model_id} curve for {outer}/{cell_id}/{landmark}"
        )
    return curve


def _score_value(
    scores: pd.DataFrame,
    *,
    outer: str,
    cell_id: str,
    landmark: int,
    model_id: str,
) -> float:
    selected = scores.loc[
        (scores["outer_condition_id"] == outer)
        & (scores["cell_id"] == cell_id)
        & (scores["landmark_visit_count"] == landmark)
        & (scores["model_id"] == model_id),
        "trajectory_iae_pp",
    ]
    if len(selected) != 1:
        raise PrivateDualClockAuditError("Private audit score identity is not unique")
    return float(selected.iloc[0])


def _risk_thresholds(frame: pd.DataFrame) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for landmark, selected in frame.groupby("landmark_visit_count", sort=True):
        output[int(landmark)] = {
            "model_disagreement_pp": float(
                selected["mean_abs_v3_v1_disagreement_pp"].quantile(
                    0.9, interpolation="higher"
                )
            ),
            "model_prefix_residual_rms_pp": float(
                selected["model_prefix_residual_rms_pp"].quantile(
                    0.9, interpolation="higher"
                )
            ),
        }
    return output


def _classify_risk_rows(
    frame: pd.DataFrame,
    thresholds: Mapping[int, Mapping[str, float]],
) -> pd.DataFrame:
    result = frame.copy()
    flags_column = []
    hypothesis_column = []
    for row in result.itertuples(index=False):
        threshold = thresholds[int(row.landmark_visit_count)]
        flags = []
        if float(row.prefix_max_capacity_recovery_pp) >= 0.2:
            flags.append("prefix_capacity_recovery")
        if float(row.model_prefix_residual_rms_pp) >= float(
            threshold["model_prefix_residual_rms_pp"]
        ):
            flags.append("high_prefix_model_residual")
        if float(row.absolute_log_duty_rate_drift) >= math.log(1.25):
            flags.append("future_schedule_drift")
        if float(row.normalized_condition_distance) >= 0.5:
            flags.append("sparse_condition_support")
        if float(row.mean_abs_v3_v1_disagreement_pp) >= float(
            threshold["model_disagreement_pp"]
        ):
            flags.append("high_model_disagreement")
        flags_column.append(";".join(flags) if flags else "none")
        if not bool(row.v3_regressed_vs_v1):
            hypothesis = "candidate_improved"
        elif "prefix_capacity_recovery" in flags or (
            "high_prefix_model_residual" in flags
        ):
            hypothesis = "prefix_measurement_instability"
        elif "future_schedule_drift" in flags:
            hypothesis = "future_schedule_drift"
        elif "sparse_condition_support" in flags:
            hypothesis = "sparse_condition_support"
        elif "high_model_disagreement" in flags:
            hypothesis = "model_structure_disagreement"
        else:
            hypothesis = "unresolved_baseline_advantage"
        hypothesis_column.append(hypothesis)
    result["risk_flags"] = flags_column
    result["primary_failure_hypothesis"] = hypothesis_column
    return result.loc[:, CELL_AUDIT_COLUMNS]


def audit_private_dual_clock_v3(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    scores: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build outcome-exposed failure diagnostics without changing V3 evidence."""
    frozen = validate_private_dual_clock_prior_v3_config(config)
    if tuple(truth.columns) != TARGET_TRUTH_COLUMNS:
        raise PrivateDualClockAuditError("Private audit truth columns changed")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PrivateDualClockAuditError("Private audit prediction columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise PrivateDualClockAuditError("Private audit decision columns changed")
    if tuple(scores.columns) != SCORE_COLUMNS:
        raise PrivateDualClockAuditError("Private audit score columns changed")
    rows = []
    score_end = float(frozen["score_end_equivalent_full_cycles"])
    for (outer, cell_id), cell in truth.groupby(
        ["outer_condition_id", "cell_id"], sort=True
    ):
        ordered = cell.sort_values("visit_index", kind="stable")
        for landmark in (int(value) for value in frozen["landmark_visit_counts"]):
            prefix = ordered.iloc[:landmark]
            future = ordered.loc[
                (ordered["visit_index"] >= landmark)
                & (ordered["equivalent_full_cycles"] <= score_end)
            ]
            if future.empty:
                raise PrivateDualClockAuditError("Private audit future is empty")
            decision = decisions.loc[
                (decisions["outer_condition_id"] == outer)
                & (decisions["cell_id"] == cell_id)
                & (decisions["landmark_visit_count"] == landmark)
            ]
            if len(decision) != 1:
                raise PrivateDualClockAuditError(
                    "Private audit decision identity is not unique"
                )
            decision_row = decision.iloc[0]
            v1 = _model_curve(
                predictions,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark,
                model_id="v1_condition_ridge_delta",
            )
            v3 = _model_curve(
                predictions,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark,
                model_id="v3_dual_clock_kernel_shrinkage",
            )
            x1 = v1["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
            x3 = v3["forecast_equivalent_full_cycles"].to_numpy(dtype=float)
            if not np.array_equal(x1, x3):
                raise PrivateDualClockAuditError("Private model grids changed")
            y1 = v1["predicted_capacity_retention_pct"].to_numpy(dtype=float)
            y3 = v3["predicted_capacity_retention_pct"].to_numpy(dtype=float)
            last_prefix = prefix.iloc[-1]
            last_future = future.iloc[-1]
            elapsed_delta = float(last_future["elapsed_days"]) - float(
                last_prefix["elapsed_days"]
            )
            exposure_delta = float(
                last_future["equivalent_full_cycles"]
            ) - float(last_prefix["equivalent_full_cycles"])
            realized_duty = max(exposure_delta / max(elapsed_delta, 1e-9), 1e-4)
            prefix_duty = float(decision_row["prefix_duty_rate_efc_per_day"])
            recovery = np.diff(
                prefix["capacity_retention_pct"].to_numpy(dtype=float)
            )
            positive_recovery = recovery[recovery > 0.0]
            v1_iae = _score_value(
                scores,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark,
                model_id="v1_condition_ridge_delta",
            )
            v3_iae = _score_value(
                scores,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark,
                model_id="v3_dual_clock_kernel_shrinkage",
            )
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "outer_condition_id": str(outer),
                    "cell_id": str(cell_id),
                    "landmark_visit_count": landmark,
                    "temperature_c": float(prefix.iloc[0]["temperature_c"]),
                    "dod_fraction": float(prefix.iloc[0]["dod_fraction"]),
                    "discharge_c_rate": float(prefix.iloc[0]["discharge_c_rate"]),
                    "prefix_last_elapsed_days": float(
                        last_prefix["elapsed_days"]
                    ),
                    "prefix_last_equivalent_full_cycles": float(
                        last_prefix["equivalent_full_cycles"]
                    ),
                    "prefix_duty_rate_efc_per_day": prefix_duty,
                    "future_realized_duty_rate_efc_per_day": realized_duty,
                    "absolute_log_duty_rate_drift": float(
                        abs(math.log(realized_duty / prefix_duty))
                    ),
                    "prefix_capacity_recovery_count": int(
                        np.sum(recovery > 0.0)
                    ),
                    "prefix_max_capacity_recovery_pp": (
                        float(np.max(positive_recovery))
                        if len(positive_recovery)
                        else 0.0
                    ),
                    "prefix_linear_residual_rms_pp": _prefix_linear_residual(
                        prefix
                    ),
                    "model_prefix_residual_rms_pp": float(
                        decision_row["prefix_residual_rms_pp"]
                    ),
                    "normalized_condition_distance": float(
                        decision_row["nearest_condition_distance"]
                        / max(float(decision_row["condition_ood_threshold"]), 1e-12)
                    ),
                    "mean_abs_v3_v1_disagreement_pp": (
                        _integrated_mean_absolute_difference(x1, y1, y3)
                    ),
                    "max_abs_v3_v1_disagreement_pp": float(
                        np.max(np.abs(y3 - y1))
                    ),
                    "endpoint_abs_v3_v1_disagreement_pp": float(
                        abs(y3[-1] - y1[-1])
                    ),
                    "v1_trajectory_iae_pp": v1_iae,
                    "v3_trajectory_iae_pp": v3_iae,
                    "v3_improvement_vs_v1_pp": v1_iae - v3_iae,
                    "v3_regressed_vs_v1": bool(v3_iae > v1_iae),
                }
            )
    raw_cells = pd.DataFrame(rows)
    thresholds = _risk_thresholds(raw_cells)
    cells = _classify_risk_rows(raw_cells, thresholds).sort_values(
        ["landmark_visit_count", "outer_condition_id", "cell_id"],
        kind="stable",
        ignore_index=True,
    )
    condition_rows = []
    for (landmark, outer), group in cells.groupby(
        ["landmark_visit_count", "outer_condition_id"], sort=True
    ):
        hypotheses = group.loc[
            group["primary_failure_hypothesis"] != "candidate_improved",
            "primary_failure_hypothesis",
        ].tolist()
        dominant = (
            Counter(hypotheses).most_common(1)[0][0]
            if hypotheses
            else "candidate_improved"
        )
        condition_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "outer_condition_id": str(outer),
                "landmark_visit_count": int(landmark),
                "temperature_c": float(group["temperature_c"].iloc[0]),
                "dod_fraction": float(group["dod_fraction"].iloc[0]),
                "discharge_c_rate": float(group["discharge_c_rate"].iloc[0]),
                "cell_count": len(group),
                "v1_condition_equal_trajectory_iae_pp": float(
                    group["v1_trajectory_iae_pp"].mean()
                ),
                "v3_condition_equal_trajectory_iae_pp": float(
                    group["v3_trajectory_iae_pp"].mean()
                ),
                "v3_improvement_vs_v1_pp": float(
                    group["v3_improvement_vs_v1_pp"].mean()
                ),
                "v3_improved_cell_fraction": float(
                    (group["v3_improvement_vs_v1_pp"] > 0.0).mean()
                ),
                "worst_cell_v3_trajectory_iae_pp": float(
                    group["v3_trajectory_iae_pp"].max()
                ),
                "mean_abs_v3_v1_disagreement_pp": float(
                    group["mean_abs_v3_v1_disagreement_pp"].mean()
                ),
                "mean_absolute_log_duty_rate_drift": float(
                    group["absolute_log_duty_rate_drift"].mean()
                ),
                "maximum_prefix_capacity_recovery_pp": float(
                    group["prefix_max_capacity_recovery_pp"].max()
                ),
                "mean_normalized_condition_distance": float(
                    group["normalized_condition_distance"].mean()
                ),
                "risk_flagged_cell_fraction": float(
                    (group["risk_flags"] != "none").mean()
                ),
                "dominant_failure_hypothesis": dominant,
            }
        )
    conditions = pd.DataFrame(
        condition_rows, columns=CONDITION_AUDIT_COLUMNS
    ).sort_values(
        ["landmark_visit_count", "outer_condition_id"],
        kind="stable",
        ignore_index=True,
    )
    diagnostic_columns = [
        "mean_abs_v3_v1_disagreement_pp",
        "absolute_log_duty_rate_drift",
        "prefix_max_capacity_recovery_pp",
        "model_prefix_residual_rms_pp",
        "normalized_condition_distance",
    ]
    correlations: dict[str, object] = {}
    failure_counts: dict[str, object] = {}
    worst_conditions: dict[str, object] = {}
    for landmark, group in cells.groupby("landmark_visit_count", sort=True):
        correlations[str(int(landmark))] = [
            {
                "diagnostic": column,
                "spearman_rho_with_v3_error": _spearman_or_none(
                    group[column], group["v3_trajectory_iae_pp"]
                ),
                "spearman_rho_with_v3_improvement": _spearman_or_none(
                    group[column], group["v3_improvement_vs_v1_pp"]
                ),
            }
            for column in diagnostic_columns
        ]
        failure_counts[str(int(landmark))] = {
            str(key): int(value)
            for key, value in group["primary_failure_hypothesis"].value_counts(
                sort=False
            ).items()
        }
        worst = conditions.loc[
            conditions["landmark_visit_count"] == landmark
        ].nsmallest(4, "v3_improvement_vs_v1_pp")
        worst_conditions[str(int(landmark))] = worst[
            [
                "outer_condition_id",
                "v3_improvement_vs_v1_pp",
                "dominant_failure_hypothesis",
            ]
        ].to_dict("records")
    summary: dict[str, object] = {
        "schema_version": "lifetwin.private_dual_clock_v3.audit.v1",
        "experiment_id": EXPERIMENT_ID,
        "private_only": True,
        "evidence_role": "outcome_exposed_failure_analysis",
        "cell_row_count": len(cells),
        "condition_row_count": len(conditions),
        "diagnostic_thresholds_by_landmark": thresholds,
        "failure_hypothesis_counts_by_landmark": failure_counts,
        "diagnostic_correlations_by_landmark": correlations,
        "worst_conditions_by_landmark": worst_conditions,
        "cell_rows_sha256": canonical_frame_sha256(cells, CELL_AUDIT_COLUMNS),
        "condition_rows_sha256": canonical_frame_sha256(
            conditions, CONDITION_AUDIT_COLUMNS
        ),
        "claim_boundary": (
            "Post-outcome private diagnostics only; classifications are hypotheses "
            "and cannot upgrade the frozen V3 evidence claim."
        ),
        "public_release_permitted": False,
    }
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return cells, conditions, summary


__all__ = [
    "CELL_AUDIT_COLUMNS",
    "CONDITION_AUDIT_COLUMNS",
    "PrivateDualClockAuditError",
    "audit_private_dual_clock_v3",
]
