"""Reference-calibrated interval and abstention audit for private V3 forecasts."""

from __future__ import annotations

import json
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
    PREDICTION_COLUMNS,
    _core,
    _future,
    _interval_quantiles,
    _interval_width,
    _selected_inner_predictions,
    _validate_replay,
    validate_private_dual_clock_prior_v3_config,
)
from lifetwin.experiments.snl_rpt_loco import (
    REFERENCE_COLUMNS,
    TARGET_TRUTH_COLUMNS,
    _trajectory_iae,
)


CELL_UNCERTAINTY_COLUMNS = (
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "requested_pointwise_coverage",
    "interval_calibration_method",
    "future_observation_count",
    "pointwise_interval_coverage",
    "endpoint_interval_covered",
    "simultaneous_trajectory_covered",
    "mean_full_interval_width_pp",
    "maximum_full_interval_width_pp",
    "pooled_pointwise_interval_coverage",
    "pooled_simultaneous_trajectory_covered",
    "pooled_mean_full_interval_width_pp",
    "v3_trajectory_iae_pp",
    "prefix_linear_residual_rms_pp",
    "prefix_residual_abstention_threshold_pp",
    "issued",
    "abstention_reason",
)
CONDITION_UNCERTAINTY_COLUMNS = (
    "outer_condition_id",
    "landmark_visit_count",
    "cell_count",
    "issued_cell_fraction",
    "condition_pointwise_interval_coverage",
    "condition_pooled_pointwise_interval_coverage",
    "condition_issued_pointwise_interval_coverage",
    "condition_simultaneous_trajectory_coverage",
    "condition_pooled_simultaneous_trajectory_coverage",
    "condition_mean_full_interval_width_pp",
    "condition_pooled_mean_full_interval_width_pp",
    "condition_v3_trajectory_iae_pp",
    "condition_issued_v3_trajectory_iae_pp",
)


class PrivateDualClockUncertaintyAuditError(ValueError):
    """Raised when a private V3 uncertainty audit contract is violated."""


def _linear_prefix_residual(prefix: pd.DataFrame) -> float:
    x = prefix["equivalent_full_cycles"].to_numpy(dtype=float)
    y = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(np.sqrt(np.mean(np.square(fitted - y))))


def _curve(
    predictions: pd.DataFrame,
    *,
    outer: str,
    cell_id: str,
    landmark: int,
) -> pd.DataFrame:
    selected = predictions.loc[
        (predictions["outer_condition_id"] == outer)
        & (predictions["cell_id"] == cell_id)
        & (predictions["landmark_visit_count"] == landmark)
        & (predictions["model_id"] == "v3_dual_clock_kernel_shrinkage")
    ].sort_values("forecast_equivalent_full_cycles", kind="stable")
    if selected.empty:
        raise PrivateDualClockUncertaintyAuditError("V3 curve is missing")
    return selected


def _nullable_mean(values: pd.Series) -> float | None:
    selected = values.dropna()
    return float(selected.mean()) if len(selected) else None


def _condition_balanced_interval_quantiles(
    inner_predictions: list[dict[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    bins = [float(value) for value in config["uncertainty"]["horizon_bins_efc"]]
    coverage = float(config["uncertainty"]["absolute_residual_quantile"])
    by_condition: dict[str, list[tuple[float, float]]] = {}
    for record in inner_predictions:
        condition = str(record["condition_id"])
        prefix = record["prefix"]
        future = record["future"]
        horizon = future["equivalent_full_cycles"].to_numpy(dtype=float) - float(
            prefix.iloc[-1]["equivalent_full_cycles"]
        )
        error = np.abs(
            np.asarray(record["predicted"], dtype=float)
            - future["capacity_retention_pct"].to_numpy(dtype=float)
        )
        by_condition.setdefault(condition, []).extend(
            (float(horizon_value), float(error_value))
            for horizon_value, error_value in zip(horizon, error, strict=True)
        )

    def calibrated_width(condition_quantiles: list[float]) -> float:
        ordered = sorted(condition_quantiles)
        rank = min(len(ordered) - 1, math.ceil((len(ordered) + 1) * coverage) - 1)
        return float(ordered[rank])

    fallback_quantiles = [
        float(
            np.quantile(
                np.asarray([error for _, error in records], dtype=float),
                coverage,
                method="higher",
            )
        )
        for records in by_condition.values()
    ]
    fallback = calibrated_width(fallback_quantiles)
    output = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        condition_quantiles = []
        support_count = 0
        for records in by_condition.values():
            errors = np.asarray(
                [
                    error
                    for horizon, error in records
                    if lower <= horizon < upper
                ],
                dtype=float,
            )
            support_count += len(errors)
            if len(errors):
                condition_quantiles.append(
                    float(np.quantile(errors, coverage, method="higher"))
                )
        output.append(
            {
                "minimum_horizon_efc": lower,
                "maximum_horizon_efc": upper,
                "support_count": support_count,
                "support_condition_count": len(condition_quantiles),
                "absolute_error_quantile_pp": (
                    calibrated_width(condition_quantiles)
                    if condition_quantiles
                    else fallback
                ),
                "calibration_method": (
                    "condition_balanced_finite_condition_rank"
                ),
            }
        )
    return output


def audit_private_dual_clock_uncertainty(
    references: pd.DataFrame,
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Audit intervals and a reference-only prefix-quality abstention rule."""
    frozen = validate_private_dual_clock_prior_v3_config(config)
    if tuple(references.columns) != REFERENCE_COLUMNS:
        raise PrivateDualClockUncertaintyAuditError("Reference columns changed")
    if tuple(truth.columns) != TARGET_TRUTH_COLUMNS:
        raise PrivateDualClockUncertaintyAuditError("Truth columns changed")
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PrivateDualClockUncertaintyAuditError("Prediction columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise PrivateDualClockUncertaintyAuditError("Decision columns changed")
    _validate_replay(predictions, decisions, prediction_manifest, frozen)
    requested_coverage = float(frozen["uncertainty"]["absolute_residual_quantile"])
    score_end = float(frozen["score_end_equivalent_full_cycles"])
    cell_rows = []
    interval_calibration = {}
    for (outer, landmark), outer_decisions in decisions.groupby(
        ["outer_condition_id", "landmark_visit_count"], sort=True
    ):
        landmark_int = int(landmark)
        reference = _core(
            references.loc[references["outer_condition_id"] == outer]
        )
        hyperparameter_values = sorted(
            set(outer_decisions["dual_clock_hyperparameters_json"])
        )
        if len(hyperparameter_values) != 1:
            raise PrivateDualClockUncertaintyAuditError(
                "Outer hyperparameters are not unique"
            )
        hyperparameters = json.loads(hyperparameter_values[0])
        inner = _selected_inner_predictions(
            reference,
            landmark=landmark_int,
            hyperparameters=hyperparameters,
            config=frozen,
        )
        pooled_quantiles = _interval_quantiles(inner, frozen)
        quantiles = _condition_balanced_interval_quantiles(inner, frozen)
        residuals = np.asarray(
            [_linear_prefix_residual(record["prefix"]) for record in inner],
            dtype=float,
        )
        residual_threshold = float(
            np.quantile(residuals, 0.95, method="higher")
        )
        interval_calibration[f"{outer}|{landmark_int}"] = {
            "condition_balanced_interval_quantiles": quantiles,
            "pooled_interval_quantiles": pooled_quantiles,
            "prefix_residual_abstention_threshold_pp": residual_threshold,
            "inner_cell_count": len(inner),
        }
        outer_truth = truth.loc[truth["outer_condition_id"] == outer]
        for cell_id, cell in outer_truth.groupby("cell_id", sort=True):
            ordered = _core(cell).sort_values("visit_index", kind="stable")
            prefix, future = _future(
                ordered, landmark=landmark_int, score_end=score_end
            )
            curve = _curve(
                predictions,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark_int,
            )
            forecast = future["equivalent_full_cycles"].to_numpy(dtype=float)
            actual = future["capacity_retention_pct"].to_numpy(dtype=float)
            predicted = np.interp(
                forecast,
                curve["forecast_equivalent_full_cycles"].to_numpy(dtype=float),
                curve["predicted_capacity_retention_pct"].to_numpy(dtype=float),
            )
            x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
            widths = np.asarray(
                [
                    _interval_width(float(exposure - x0), quantiles)
                    for exposure in forecast
                ],
                dtype=float,
            )
            pooled_widths = np.asarray(
                [
                    _interval_width(float(exposure - x0), pooled_quantiles)
                    for exposure in forecast
                ],
                dtype=float,
            )
            covered = np.abs(predicted - actual) <= widths + 1e-12
            pooled_covered = (
                np.abs(predicted - actual) <= pooled_widths + 1e-12
            )
            decision = outer_decisions.loc[
                outer_decisions["cell_id"] == cell_id
            ]
            if len(decision) != 1:
                raise PrivateDualClockUncertaintyAuditError(
                    "Target decision identity is not unique"
                )
            prefix_residual = _linear_prefix_residual(prefix)
            reasons = []
            if str(decision.iloc[0]["evidence_status"]) != "supported":
                reasons.append("condition_or_duty_ood")
            if prefix_residual > residual_threshold:
                reasons.append("prefix_shape_outside_reference_support")
            issued = not reasons
            cell_rows.append(
                {
                    "outer_condition_id": str(outer),
                    "cell_id": str(cell_id),
                    "landmark_visit_count": landmark_int,
                    "requested_pointwise_coverage": requested_coverage,
                    "interval_calibration_method": (
                        "condition_balanced_finite_condition_rank"
                    ),
                    "future_observation_count": len(future),
                    "pointwise_interval_coverage": float(np.mean(covered)),
                    "endpoint_interval_covered": bool(covered[-1]),
                    "simultaneous_trajectory_covered": bool(np.all(covered)),
                    "mean_full_interval_width_pp": float(2.0 * np.mean(widths)),
                    "maximum_full_interval_width_pp": float(2.0 * np.max(widths)),
                    "pooled_pointwise_interval_coverage": float(
                        np.mean(pooled_covered)
                    ),
                    "pooled_simultaneous_trajectory_covered": bool(
                        np.all(pooled_covered)
                    ),
                    "pooled_mean_full_interval_width_pp": float(
                        2.0 * np.mean(pooled_widths)
                    ),
                    "v3_trajectory_iae_pp": _trajectory_iae(
                        x0, forecast, actual, predicted
                    ),
                    "prefix_linear_residual_rms_pp": prefix_residual,
                    "prefix_residual_abstention_threshold_pp": residual_threshold,
                    "issued": issued,
                    "abstention_reason": ";".join(reasons) if reasons else "none",
                }
            )
    cells = pd.DataFrame(cell_rows, columns=CELL_UNCERTAINTY_COLUMNS).sort_values(
        ["landmark_visit_count", "outer_condition_id", "cell_id"],
        kind="stable",
        ignore_index=True,
    )
    condition_rows = []
    for (outer, landmark), group in cells.groupby(
        ["outer_condition_id", "landmark_visit_count"], sort=True
    ):
        issued = group.loc[group["issued"]]
        condition_rows.append(
            {
                "outer_condition_id": str(outer),
                "landmark_visit_count": int(landmark),
                "cell_count": len(group),
                "issued_cell_fraction": float(group["issued"].mean()),
                "condition_pointwise_interval_coverage": float(
                    group["pointwise_interval_coverage"].mean()
                ),
                "condition_pooled_pointwise_interval_coverage": float(
                    group["pooled_pointwise_interval_coverage"].mean()
                ),
                "condition_issued_pointwise_interval_coverage": (
                    float(issued["pointwise_interval_coverage"].mean())
                    if len(issued)
                    else None
                ),
                "condition_simultaneous_trajectory_coverage": float(
                    group["simultaneous_trajectory_covered"].mean()
                ),
                "condition_pooled_simultaneous_trajectory_coverage": float(
                    group["pooled_simultaneous_trajectory_covered"].mean()
                ),
                "condition_mean_full_interval_width_pp": float(
                    group["mean_full_interval_width_pp"].mean()
                ),
                "condition_pooled_mean_full_interval_width_pp": float(
                    group["pooled_mean_full_interval_width_pp"].mean()
                ),
                "condition_v3_trajectory_iae_pp": float(
                    group["v3_trajectory_iae_pp"].mean()
                ),
                "condition_issued_v3_trajectory_iae_pp": (
                    float(issued["v3_trajectory_iae_pp"].mean())
                    if len(issued)
                    else None
                ),
            }
        )
    conditions = pd.DataFrame(
        condition_rows, columns=CONDITION_UNCERTAINTY_COLUMNS
    ).sort_values(
        ["landmark_visit_count", "outer_condition_id"],
        kind="stable",
        ignore_index=True,
    )
    nullable_columns = (
        "condition_issued_pointwise_interval_coverage",
        "condition_issued_v3_trajectory_iae_pp",
    )
    for column in nullable_columns:
        conditions[column] = conditions[column].astype(object).where(
            conditions[column].notna(), None
        )
    summary_by_landmark = {}
    for landmark, group in conditions.groupby("landmark_visit_count", sort=True):
        summary_by_landmark[str(int(landmark))] = {
            "requested_pointwise_coverage": requested_coverage,
            "primary_interval_calibration_method": (
                "condition_balanced_finite_condition_rank"
            ),
            "condition_equal_pointwise_interval_coverage": float(
                group["condition_pointwise_interval_coverage"].mean()
            ),
            "pooled_condition_equal_pointwise_interval_coverage": float(
                group["condition_pooled_pointwise_interval_coverage"].mean()
            ),
            "condition_equal_simultaneous_trajectory_coverage": float(
                group["condition_simultaneous_trajectory_coverage"].mean()
            ),
            "pooled_condition_equal_simultaneous_trajectory_coverage": float(
                group[
                    "condition_pooled_simultaneous_trajectory_coverage"
                ].mean()
            ),
            "condition_equal_mean_full_interval_width_pp": float(
                group["condition_mean_full_interval_width_pp"].mean()
            ),
            "pooled_condition_equal_mean_full_interval_width_pp": float(
                group["condition_pooled_mean_full_interval_width_pp"].mean()
            ),
            "issued_cell_fraction": float(
                cells.loc[
                    cells["landmark_visit_count"] == landmark, "issued"
                ].mean()
            ),
            "issued_condition_fraction": float(
                (group["issued_cell_fraction"] > 0.0).mean()
            ),
            "issued_condition_equal_pointwise_interval_coverage": _nullable_mean(
                group["condition_issued_pointwise_interval_coverage"]
            ),
            "full_condition_equal_v3_trajectory_iae_pp": float(
                group["condition_v3_trajectory_iae_pp"].mean()
            ),
            "issued_condition_equal_v3_trajectory_iae_pp": _nullable_mean(
                group["condition_issued_v3_trajectory_iae_pp"]
            ),
        }
    summary: dict[str, object] = {
        "schema_version": "lifetwin.private_dual_clock_uncertainty_audit.v1",
        "private_only": True,
        "evidence_role": "outcome_exposed_reference_calibrated_diagnostic",
        "summary_by_landmark": summary_by_landmark,
        "interval_calibration_by_outer_fold": interval_calibration,
        "formal_interval_coverage_claim": False,
        "abstention_policy_status": "diagnostic_not_production_qualified",
        "cell_rows_sha256": canonical_frame_sha256(
            cells, CELL_UNCERTAINTY_COLUMNS
        ),
        "condition_rows_sha256": canonical_frame_sha256(
            conditions, CONDITION_UNCERTAINTY_COLUMNS
        ),
        "claim_boundary": (
            "Primary intervals use per-condition residual quantiles followed by a "
            "finite-condition conservative rank, while pooled intervals are retained "
            "as a comparator. Both remain retrospective diagnostics, not formal "
            "conformal guarantees. The abstention rule requires independent calibration."
        ),
        "public_release_permitted": False,
    }
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return cells, conditions, summary


__all__ = [
    "CELL_UNCERTAINTY_COLUMNS",
    "CONDITION_UNCERTAINTY_COLUMNS",
    "PrivateDualClockUncertaintyAuditError",
    "audit_private_dual_clock_uncertainty",
]
