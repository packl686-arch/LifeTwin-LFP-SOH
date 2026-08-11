"""Private SNL RPT repeatability development for the V10 qualification step.

The input contains repeated measurements within an already identified RPT visit.
It cannot identify tester bias, chamber bias, long-term reference drift, or a
cross-tester bridge.  The decision therefore separates the observed IID-like
repeatability component from the still-missing full measurement model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from lifetwin.data.snl import RPT_REPEAT_COLUMNS
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


RESIDUAL_COLUMNS = (
    "cell_id",
    "condition_id",
    "visit_index",
    "repeat_index",
    "repeat_count",
    "retention_pct",
    "visit_mean_retention_pct",
    "repeat_residual_pp",
    "measurement_error_proxy_pp",
)
CANDIDATE_SCORE_COLUMNS = (
    "model_id",
    "distribution",
    "degrees_of_freedom",
    "leave_one_cell_out_log_score",
    "valid_fold_count",
    "expected_fold_count",
    "passed_all_folds",
)
CONDITION_SCALE_COLUMNS = (
    "condition_id",
    "physical_cell_count",
    "visit_count",
    "residual_count",
    "scale_pp",
    "p95_absolute_error_proxy_pp",
)


@dataclass(frozen=True)
class RepeatNoiseCandidate:
    model_id: str
    distribution: str
    degrees_of_freedom: float | None = None


@dataclass(frozen=True)
class RepeatNoiseModel:
    model_id: str
    distribution: str
    scale_pp: float
    degrees_of_freedom: float | None = None


def _candidates(config: Mapping[str, object]) -> tuple[RepeatNoiseCandidate, ...]:
    values = tuple(
        RepeatNoiseCandidate(
            model_id=str(item["model_id"]),
            distribution=str(item["distribution"]),
            degrees_of_freedom=(
                float(item["degrees_of_freedom"])
                if "degrees_of_freedom" in item
                else None
            ),
        )
        for item in config["noise_model_selection"]["candidate_families"]
    )
    identifiers = [item.model_id for item in values]
    if len(values) == 0 or len(set(identifiers)) != len(identifiers):
        raise FastChargeV5PairwiseError("V10 noise candidates are empty or duplicated")
    for item in values:
        if item.distribution not in {"gaussian", "student_t"}:
            raise FastChargeV5PairwiseError("V10 noise distribution is unsupported")
        if item.distribution == "student_t" and (
            item.degrees_of_freedom is None or item.degrees_of_freedom <= 2.0
        ):
            raise FastChargeV5PairwiseError("V10 Student-t candidates require df > 2")
    return values


def validate_repeat_measurements(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    if tuple(frame.columns) != RPT_REPEAT_COLUMNS:
        raise FastChargeV5PairwiseError("V10 repeat-measurement columns changed")
    if frame.empty:
        raise FastChargeV5PairwiseError("V10 repeat-measurement input is empty")
    data = frame.copy()
    for column in ("dataset_id", "cell_id", "condition_id", "measurement_time"):
        if data[column].isna().any():
            raise FastChargeV5PairwiseError(f"V10 {column} contains null values")
        data[column] = data[column].astype(str).str.strip()
        if (data[column] == "").any():
            raise FastChargeV5PairwiseError(f"V10 {column} contains empty values")
    integer_columns = (
        "visit_index",
        "repeat_index",
        "source_cycle_index",
        "rpt_cycle_count",
    )
    numeric_columns = (
        *integer_columns,
        "elapsed_days",
        "equivalent_full_cycles",
        "capacity_ah",
        "retention_pct",
        "visit_center_capacity_ah",
        "visit_center_retention_pct",
    )
    numeric = data.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise FastChargeV5PairwiseError("V10 repeat measurements are non-finite")
    for column in numeric_columns:
        data[column] = numeric[column]
    for column in integer_columns:
        values = data[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise FastChargeV5PairwiseError(f"V10 {column} must be integral")
        data[column] = values.astype(np.int64)
    if (data[["capacity_ah", "retention_pct"]] <= 0.0).any().any():
        raise FastChargeV5PairwiseError("V10 capacity and retention must be positive")
    if (data[["visit_index", "repeat_index"]] < 0).any().any():
        raise FastChargeV5PairwiseError(
            "V10 visit and repeat indices must be nonnegative"
        )
    parsed_time = pd.to_datetime(data["measurement_time"], errors="coerce", utc=True)
    if parsed_time.isna().any():
        raise FastChargeV5PairwiseError("V10 measurement timestamps are invalid")
    data["measurement_time"] = parsed_time
    if data.duplicated(["cell_id", "visit_index", "repeat_index"]).any():
        raise FastChargeV5PairwiseError("V10 repeat coordinates are duplicated")

    contract = config["repeat_contract"]
    minimum_cells = int(contract["minimum_physical_cell_count"])
    minimum_visits = int(contract["minimum_visits_per_cell"])
    minimum_repeats = int(contract["minimum_repeats_per_visit"])
    if int(data["cell_id"].nunique()) < minimum_cells:
        raise FastChargeV5PairwiseError("V10 has too few physical cells")
    visits_per_cell = (
        data[["cell_id", "visit_index"]].drop_duplicates().groupby("cell_id").size()
    )
    if int(visits_per_cell.min()) < minimum_visits:
        raise FastChargeV5PairwiseError("V10 has too few RPT visits per cell")
    repeats_per_visit = data.groupby(["cell_id", "visit_index"], sort=True).size()
    if int(repeats_per_visit.min()) < minimum_repeats:
        raise FastChargeV5PairwiseError("V10 has too few repeats per RPT visit")
    return data.sort_values(
        ["condition_id", "cell_id", "visit_index", "repeat_index"],
        kind="stable",
        ignore_index=True,
    )


def repeat_residuals(data: pd.DataFrame) -> pd.DataFrame:
    groups = ["cell_id", "condition_id", "visit_index"]
    result = data.loc[
        :, ["cell_id", "condition_id", "visit_index", "repeat_index", "retention_pct"]
    ].copy()
    result["repeat_count"] = result.groupby(groups, sort=True)[
        "retention_pct"
    ].transform("size")
    result["visit_mean_retention_pct"] = result.groupby(groups, sort=True)[
        "retention_pct"
    ].transform("mean")
    result["repeat_residual_pp"] = (
        result["retention_pct"] - result["visit_mean_retention_pct"]
    )
    result["measurement_error_proxy_pp"] = result["repeat_residual_pp"] / np.sqrt(
        1.0 - 1.0 / result["repeat_count"]
    )
    return result.loc[:, RESIDUAL_COLUMNS].sort_values(
        ["condition_id", "cell_id", "visit_index", "repeat_index"],
        kind="stable",
        ignore_index=True,
    )


def _fit_scale(
    values: np.ndarray,
    candidate: RepeatNoiseCandidate,
    config: Mapping[str, object],
) -> float:
    variance = float(np.mean(np.square(values)))
    if candidate.distribution == "gaussian":
        scale = math.sqrt(variance)
    else:
        assert candidate.degrees_of_freedom is not None
        scale = math.sqrt(
            variance
            * (candidate.degrees_of_freedom - 2.0)
            / candidate.degrees_of_freedom
        )
    return max(float(scale), float(config["noise_model_selection"]["minimum_scale_pp"]))


def _logpdf(
    values: np.ndarray,
    candidate: RepeatNoiseCandidate,
    scale: float,
) -> np.ndarray:
    if candidate.distribution == "gaussian":
        return norm.logpdf(values, loc=0.0, scale=scale)
    assert candidate.degrees_of_freedom is not None
    return t.logpdf(values / scale, df=candidate.degrees_of_freedom) - math.log(scale)


def crossfit_candidate_scores(
    residuals: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    cells = sorted(residuals["cell_id"].unique())
    rows: list[dict[str, object]] = []
    for candidate in _candidates(config):
        total = 0.0
        valid = 0
        for held_out in cells:
            fit = residuals.loc[residuals["cell_id"] != held_out]
            held = residuals.loc[residuals["cell_id"] == held_out]
            if fit.empty or held.empty:
                continue
            scale = _fit_scale(
                fit["measurement_error_proxy_pp"].to_numpy(dtype=float),
                candidate,
                config,
            )
            total += float(
                np.sum(
                    _logpdf(
                        held["measurement_error_proxy_pp"].to_numpy(dtype=float),
                        candidate,
                        scale,
                    )
                )
            )
            valid += 1
        rows.append(
            {
                "model_id": candidate.model_id,
                "distribution": candidate.distribution,
                "degrees_of_freedom": (
                    candidate.degrees_of_freedom
                    if candidate.degrees_of_freedom is not None
                    else 0.0
                ),
                "leave_one_cell_out_log_score": total,
                "valid_fold_count": valid,
                "expected_fold_count": len(cells),
                "passed_all_folds": valid == len(cells),
            }
        )
    return pd.DataFrame(rows, columns=CANDIDATE_SCORE_COLUMNS).sort_values(
        ["leave_one_cell_out_log_score", "model_id"],
        ascending=[False, True],
        kind="stable",
        ignore_index=True,
    )


def _select_candidate(
    scores: pd.DataFrame,
    config: Mapping[str, object],
) -> RepeatNoiseCandidate:
    eligible = scores.loc[scores["passed_all_folds"]]
    if eligible.empty:
        raise FastChargeV5PairwiseError("No V10 noise candidate passed all folds")
    best = float(eligible["leave_one_cell_out_log_score"].max())
    tolerance = float(config["noise_model_selection"]["tie_tolerance_log_score"])
    tied = set(
        eligible.loc[
            eligible["leave_one_cell_out_log_score"] >= best - tolerance,
            "model_id",
        ]
    )
    selected_id = next(
        model_id
        for model_id in config["noise_model_selection"]["tie_break_order"]
        if model_id in tied
    )
    return next(item for item in _candidates(config) if item.model_id == selected_id)


def _order_slopes(data: pd.DataFrame) -> np.ndarray:
    slopes: list[float] = []
    for _, group in data.groupby(["cell_id", "visit_index"], sort=True):
        x = group["repeat_index"].to_numpy(dtype=float)
        y = group["retention_pct"].to_numpy(dtype=float)
        centered = x - float(np.mean(x))
        denominator = float(np.dot(centered, centered))
        if denominator > 0.0:
            slopes.append(float(np.dot(centered, y - float(np.mean(y))) / denominator))
    return np.asarray(slopes, dtype=float)


def _condition_scales(
    residuals: pd.DataFrame,
    selected: RepeatNoiseCandidate,
    config: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition_id, group in residuals.groupby("condition_id", sort=True):
        values = group["measurement_error_proxy_pp"].to_numpy(dtype=float)
        rows.append(
            {
                "condition_id": str(condition_id),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "visit_count": int(
                    group[["cell_id", "visit_index"]].drop_duplicates().shape[0]
                ),
                "residual_count": len(group),
                "scale_pp": _fit_scale(values, selected, config),
                "p95_absolute_error_proxy_pp": float(np.quantile(np.abs(values), 0.95)),
            }
        )
    return pd.DataFrame(rows, columns=CONDITION_SCALE_COLUMNS)


def characterize_snl_rpt_repeatability(
    frame: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, RepeatNoiseModel, dict[str, object]
]:
    data = validate_repeat_measurements(frame, config)
    residuals = repeat_residuals(data)
    scores = crossfit_candidate_scores(residuals, config)
    selected = _select_candidate(scores, config)
    values = residuals["measurement_error_proxy_pp"].to_numpy(dtype=float)
    model = RepeatNoiseModel(
        model_id=selected.model_id,
        distribution=selected.distribution,
        degrees_of_freedom=selected.degrees_of_freedom,
        scale_pp=_fit_scale(values, selected, config),
    )
    condition_scales = _condition_scales(residuals, selected, config)
    order_slopes = _order_slopes(data)
    gates = config["engineering_gates"]
    scale_passed = model.scale_pp <= float(gates["maximum_repeatability_scale_pp"])
    order_passed = float(np.median(np.abs(order_slopes))) <= float(
        gates["maximum_median_absolute_repeat_order_slope_pp_per_repeat"]
    )
    repeatability_component_passed = bool(scale_passed and order_passed)
    missing_components = list(config["identifiability"]["unavailable_components"])
    full_model_identified = len(missing_components) == 0
    decision: dict[str, object] = {
        "schema_version": "lifetwin.fastcharge_v10.snl_rpt_repeatability.v1",
        "experiment_id": str(config["experiment_id"]),
        "evidence_role": "private_post_outcome_measurement_development",
        "physical_cell_count": int(data["cell_id"].nunique()),
        "condition_cluster_count": int(data["condition_id"].nunique()),
        "visit_count": int(data[["cell_id", "visit_index"]].drop_duplicates().shape[0]),
        "repeat_measurement_count": len(data),
        "selected_noise_model": {
            "model_id": model.model_id,
            "distribution": model.distribution,
            "degrees_of_freedom": model.degrees_of_freedom,
            "scale_pp": model.scale_pp,
        },
        "repeat_order": {
            "visit_slope_count": len(order_slopes),
            "median_absolute_slope_pp_per_repeat": float(
                np.median(np.abs(order_slopes))
            ),
            "p95_absolute_slope_pp_per_repeat": float(
                np.quantile(np.abs(order_slopes), 0.95)
            ),
        },
        "repeatability_component_gates": {
            "scale_passed": bool(scale_passed),
            "repeat_order_passed": bool(order_passed),
            "passed": repeatability_component_passed,
        },
        "identified_components": list(
            config["identifiability"]["available_components"]
        ),
        "missing_components": missing_components,
        "full_measurement_model_identified": full_model_identified,
        "eligible_for_full_v9_qualification": bool(
            repeatability_component_passed and full_model_identified
        ),
        "next_action": (
            "retire_v7_dynamic_update_due_to_repeatability_failure"
            if not repeatability_component_passed
            else "block_before_blind_test_pending_reference_and_bridge_records"
        ),
        "future_outcomes_used_for_noise_estimation": False,
        "target_accuracy_evidence_created": False,
        "public_aggregate_release_permitted": False,
        "hashes": {
            "repeat_input_canonical_sha256": canonical_frame_sha256(
                frame, RPT_REPEAT_COLUMNS
            ),
            "residual_table_sha256": canonical_frame_sha256(
                residuals, RESIDUAL_COLUMNS
            ),
            "candidate_scores_sha256": canonical_frame_sha256(
                scores, CANDIDATE_SCORE_COLUMNS
            ),
            "condition_scales_sha256": canonical_frame_sha256(
                condition_scales, CONDITION_SCALE_COLUMNS
            ),
        },
        "claim_boundary": list(config["claim_boundaries"]),
    }
    decision["decision_content_sha256"] = canonical_json_sha256(decision)
    return residuals, scores, condition_scales, model, decision


__all__ = [
    "CANDIDATE_SCORE_COLUMNS",
    "CONDITION_SCALE_COLUMNS",
    "RESIDUAL_COLUMNS",
    "RepeatNoiseCandidate",
    "RepeatNoiseModel",
    "characterize_snl_rpt_repeatability",
    "crossfit_candidate_scores",
    "repeat_residuals",
    "validate_repeat_measurements",
]
