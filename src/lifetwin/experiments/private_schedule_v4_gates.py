"""Pre-registered promotion gates for the private schedule-aware challenger."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Mapping

import pandas as pd

from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.private_enterprise_cycle import SCORE_COLUMNS
from lifetwin.experiments.private_schedule_v4 import (
    BOUNDED_SCHEDULE_MODE_ID,
    ELAPSED_SCHEDULE_MODE_ID,
    SCHEDULE_MODE_ID,
)


PREREGISTRATION_SCHEMA = "lifetwin.private_enterprise_schedule_v4.preregistration.v1"
AMENDMENT_SCHEMA = "lifetwin.private_enterprise_schedule_v4_1.amendment.v1"
BOUNDED_AMENDMENT_SCHEMA = (
    "lifetwin.private_enterprise_schedule_v4_2.preregistration.v1"
)
GATE_SCHEMA = "lifetwin.private_enterprise_schedule_v4.gates.v1"


class PrivateScheduleV4GateError(ValueError):
    """Raised when a V4 promotion comparison is incomplete or inconsistent."""


def _finite_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PrivateScheduleV4GateError(f"{label} is not numeric") from exc
    if not math.isfinite(number):
        raise PrivateScheduleV4GateError(f"{label} is not finite")
    return number


def _validate_score_bundle(
    scores: pd.DataFrame,
    summary_value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    summary = deepcopy(dict(summary_value))
    expected_summary_hash = summary.pop("summary_content_sha256", None)
    if expected_summary_hash != canonical_json_sha256(summary):
        raise PrivateScheduleV4GateError(f"{label} score summary hash changed")
    summary["summary_content_sha256"] = expected_summary_hash
    if summary.get("score_rows_sha256") != canonical_frame_sha256(
        scores, SCORE_COLUMNS
    ):
        raise PrivateScheduleV4GateError(f"{label} score rows changed")
    key_columns = [
        "experiment_id",
        "adapter_id",
        "dataset_id",
        "partition",
        "cell_id",
        "landmark_visit_count",
    ]
    if scores.duplicated(key_columns).any():
        raise PrivateScheduleV4GateError(f"{label} score keys are duplicated")
    for field in ("experiment_id", "adapter_id", "dataset_id", "partition"):
        values = set(scores[field].astype(str))
        if len(values) > 1 or (values and values != {str(summary.get(field, ""))}):
            raise PrivateScheduleV4GateError(
                f"{label} score identity differs from its summary"
            )
    metric_columns = [
        "trajectory_iae_pp",
        "trajectory_mae_pp",
        "trajectory_rmse_pp",
        "endpoint_absolute_error_pp",
        "pointwise_interval_coverage",
        "mean_full_interval_width_pp",
    ]
    if scores[metric_columns].isna().any().any():
        raise PrivateScheduleV4GateError(f"{label} scores contain missing metrics")
    for column in metric_columns:
        values = scores[column].astype(float)
        if not values.map(math.isfinite).all():
            raise PrivateScheduleV4GateError(
                f"{label} scores contain non-finite metrics"
            )
    if (
        (scores["future_observation_count"].astype(int) <= 0).any()
        or (scores["trajectory_iae_pp"].astype(float) < 0.0).any()
        or (scores["pointwise_interval_coverage"].astype(float) < 0.0).any()
        or (scores["pointwise_interval_coverage"].astype(float) > 1.0).any()
        or (scores["mean_full_interval_width_pp"].astype(float) < 0.0).any()
    ):
        raise PrivateScheduleV4GateError(f"{label} scores violate metric bounds")
    return summary


def _validate_preregistration(value: Mapping[str, object]) -> dict[str, object]:
    prereg = deepcopy(dict(value))
    schema = prereg.get("schema_version")
    expected_identity = {
        PREREGISTRATION_SCHEMA: (
            "preregistered_before_hithium_data_access",
            SCHEDULE_MODE_ID,
        ),
        AMENDMENT_SCHEMA: (
            "preregistered_before_hithium_data_access_after_outcome_exposed_snl_oracle_audit",
            ELAPSED_SCHEDULE_MODE_ID,
        ),
        BOUNDED_AMENDMENT_SCHEMA: (
            "preregistered_before_hithium_data_access_after_outcome_exposed_public_development",
            BOUNDED_SCHEDULE_MODE_ID,
        ),
    }.get(schema)
    if expected_identity is None:
        raise PrivateScheduleV4GateError("V4 preregistration schema changed")
    expected_status, expected_candidate = expected_identity
    if prereg.get("status") != expected_status:
        raise PrivateScheduleV4GateError("V4 preregistration status changed")
    if prereg.get("candidate") != expected_candidate:
        raise PrivateScheduleV4GateError("V4 candidate identity changed")
    gates = prereg.get("decision_gates")
    expected = {
        "noninferiority_margin_pp",
        "desired_relative_improvement_fraction",
        "minimum_improved_condition_fraction",
        "maximum_worst_condition_regression_pp",
        "minimum_issued_fraction",
        "minimum_issued_pointwise_interval_coverage",
    }
    if not isinstance(gates, Mapping) or set(gates) != expected:
        raise PrivateScheduleV4GateError("V4 decision gates changed")
    if any(
        not math.isfinite(float(value)) or float(value) < 0.0
        for value in gates.values()
    ):
        raise PrivateScheduleV4GateError("V4 decision gate is invalid")
    return prereg


def _validate_inputs(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    baseline_summary: Mapping[str, object],
    candidate_summary: Mapping[str, object],
    candidate_mode_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        tuple(baseline_scores.columns) != SCORE_COLUMNS
        or tuple(candidate_scores.columns) != SCORE_COLUMNS
    ):
        raise PrivateScheduleV4GateError("Enterprise score columns changed")
    baseline = _validate_score_bundle(
        baseline_scores, baseline_summary, label="Baseline"
    )
    candidate = _validate_score_bundle(
        candidate_scores, candidate_summary, label="Candidate"
    )
    for summary in (baseline, candidate):
        if summary.get("schema_version") != (
            "lifetwin.private_enterprise_cycle.score_summary.v1"
        ):
            raise PrivateScheduleV4GateError("Enterprise score schema changed")
        if summary.get("evidence_role") != "private_batch_disjoint_calibration":
            raise PrivateScheduleV4GateError(
                "V4 promotion gates may run only on calibration evidence"
            )
    if baseline.get("partition") != candidate.get("partition"):
        raise PrivateScheduleV4GateError("V3 and V4 partitions differ")
    for field in ("experiment_id", "adapter_id", "dataset_id"):
        if baseline.get(field) != candidate.get(field):
            raise PrivateScheduleV4GateError(f"V3 and V4 {field} identities differ")
    if candidate.get("prediction_mode_id") != candidate_mode_id:
        raise PrivateScheduleV4GateError(
            "Candidate does not match the preregistered schedule mode"
        )
    if (
        candidate.get("schedule_role") != "deployment_candidate"
        or candidate.get("primary_evidence_eligible") is not True
    ):
        raise PrivateScheduleV4GateError(
            "Oracle schedules cannot enter the V4 promotion gate"
        )
    return baseline, candidate


def evaluate_private_schedule_v4_gates(
    baseline_scores: pd.DataFrame,
    candidate_scores: pd.DataFrame,
    baseline_summary: Mapping[str, object],
    candidate_summary: Mapping[str, object],
    preregistration: Mapping[str, object],
) -> dict[str, object]:
    """Compare frozen V3 and V4 calibration results without silent exclusions."""
    prereg = _validate_preregistration(preregistration)
    baseline, candidate = _validate_inputs(
        baseline_scores,
        candidate_scores,
        baseline_summary,
        candidate_summary,
        str(prereg["candidate"]),
    )
    gates = prereg["decision_gates"]
    baseline_landmarks = baseline.get("summary_by_landmark")
    candidate_landmarks = candidate.get("summary_by_landmark")
    if (
        not isinstance(baseline_landmarks, Mapping)
        or not isinstance(candidate_landmarks, Mapping)
        or set(baseline_landmarks) != set(candidate_landmarks)
    ):
        raise PrivateScheduleV4GateError("V3 and V4 landmark summaries differ")
    rows: dict[str, object] = {}
    for landmark in sorted(baseline_landmarks, key=int):
        base_summary = baseline_landmarks[landmark]
        cand_summary = candidate_landmarks[landmark]
        base_iae = _finite_float(
            base_summary.get("condition_equal_trajectory_iae_pp"),
            "V3 primary metric",
        )
        cand_iae = _finite_float(
            cand_summary.get("condition_equal_trajectory_iae_pp"),
            "Candidate primary metric",
        )
        if base_iae <= 0.0 or cand_iae < 0.0:
            raise PrivateScheduleV4GateError("V3/V4 primary metric is invalid")
        base_condition = (
            baseline_scores.loc[
                baseline_scores["landmark_visit_count"] == int(landmark)
            ]
            .groupby("condition_id", sort=True)["trajectory_iae_pp"]
            .mean()
        )
        cand_condition = (
            candidate_scores.loc[
                candidate_scores["landmark_visit_count"] == int(landmark)
            ]
            .groupby("condition_id", sort=True)["trajectory_iae_pp"]
            .mean()
        )
        missing_conditions = sorted(
            set(base_condition.index) - set(cand_condition.index)
        )
        unexpected_conditions = sorted(
            set(cand_condition.index) - set(base_condition.index)
        )
        base_cells = {
            (str(row.condition_id), str(row.cell_id))
            for row in baseline_scores.loc[
                baseline_scores["landmark_visit_count"] == int(landmark)
            ].itertuples(index=False)
        }
        cand_cells = {
            (str(row.condition_id), str(row.cell_id))
            for row in candidate_scores.loc[
                candidate_scores["landmark_visit_count"] == int(landmark)
            ].itertuples(index=False)
        }
        missing_cells = sorted(base_cells - cand_cells)
        unexpected_cells = sorted(cand_cells - base_cells)
        matched = base_condition.index.intersection(cand_condition.index)
        deltas = cand_condition.loc[matched] - base_condition.loc[matched]
        improved_count = int((deltas < 0.0).sum())
        condition_count = len(base_condition)
        improved_fraction = improved_count / condition_count if condition_count else 0.0
        worst_regression = (
            float(deltas.max()) if len(deltas) and not missing_conditions else None
        )
        relative_improvement = (base_iae - cand_iae) / base_iae
        checks = {
            "score_population_complete": not (
                missing_conditions
                or unexpected_conditions
                or missing_cells
                or unexpected_cells
            ),
            "noninferiority": cand_iae
            <= base_iae + float(gates["noninferiority_margin_pp"]),
            "relative_improvement": relative_improvement
            >= float(gates["desired_relative_improvement_fraction"]),
            "improved_condition_fraction": improved_fraction
            >= float(gates["minimum_improved_condition_fraction"]),
            "worst_condition_regression": worst_regression is not None
            and worst_regression
            <= float(gates["maximum_worst_condition_regression_pp"]),
            "issued_fraction": float(cand_summary["issued_fraction"])
            >= float(gates["minimum_issued_fraction"]),
            "pointwise_interval_coverage": float(
                cand_summary["condition_equal_pointwise_interval_coverage"]
            )
            >= float(gates["minimum_issued_pointwise_interval_coverage"]),
        }
        rows[landmark] = {
            "baseline_condition_equal_trajectory_iae_pp": base_iae,
            "candidate_condition_equal_trajectory_iae_pp": cand_iae,
            "relative_improvement_fraction": relative_improvement,
            "improved_condition_fraction": improved_fraction,
            "worst_condition_regression_pp": worst_regression,
            "missing_candidate_conditions": missing_conditions,
            "unexpected_candidate_conditions": unexpected_conditions,
            "missing_candidate_cells": [list(item) for item in missing_cells],
            "unexpected_candidate_cells": [list(item) for item in unexpected_cells],
            "candidate_issued_fraction": float(cand_summary["issued_fraction"]),
            "candidate_condition_equal_pointwise_interval_coverage": float(
                cand_summary["condition_equal_pointwise_interval_coverage"]
            ),
            "checks": checks,
            "passed": all(checks.values()),
        }
    result: dict[str, object] = {
        "schema_version": GATE_SCHEMA,
        "protocol_id": prereg["protocol_id"],
        "partition": baseline["partition"],
        "candidate": prereg["candidate"],
        "primary_evidence_eligible": True,
        "by_landmark": rows,
        "promote_v4": all(bool(row["passed"]) for row in rows.values()),
        "promote_candidate": all(bool(row["passed"]) for row in rows.values()),
        "failure_action": "retain_frozen_v3_without_same_cohort_gate_retuning",
        "public_release_permitted": False,
    }
    result["result_content_sha256"] = canonical_json_sha256(result)
    return result


__all__ = [
    "AMENDMENT_SCHEMA",
    "BOUNDED_AMENDMENT_SCHEMA",
    "GATE_SCHEMA",
    "PrivateScheduleV4GateError",
    "evaluate_private_schedule_v4_gates",
]
