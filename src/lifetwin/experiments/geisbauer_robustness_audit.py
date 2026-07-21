from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import itertools
import json
import math

import numpy as np
import pandas as pd

from lifetwin.data.geisbauer_calendar import (
    GEISBAUER_CALENDAR_DATASET_ID,
    GEISBAUER_CALENDAR_EVIDENCE_ROLE,
    GEISBAUER_CALENDAR_OBSERVATIONS_SHA256,
    GEISBAUER_CALENDAR_STATISTICAL_UNIT,
    geisbauer_calendar_observations_sha256,
)
from lifetwin.experiments.geisbauer_external_stress import (
    EXPERIMENT_ID as BASE_EXPERIMENT_ID,
    GATED_HIERARCHICAL_ACTIVATION_METHOD,
    HIERARCHICAL_POWER_METHOD,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARATOR,
    geisbauer_external_prediction_sha256,
    run_geisbauer_external_stress,
)


AUDIT_ID = "geisbauer_lfp_calendar_external_robustness_audit_v1"
DESIGN_STATUS = "retrospective_audit_designed_after_v011_outcome_review"
INFERENCE_STATUS = "exploratory_nominal_diagnostics_not_confirmatory_inference"
PRIMARY_METRIC = "trajectory_iae_pp"
POINT_METRIC = "absolute_error_pp"
NUMERICAL_ZERO_TOLERANCE_PP = 1e-12
POST_HOC_EQUIVALENCE_MARGINS_PP = (0.0, 0.01, 0.05, 0.1)
MAXIMUM_EXACT_SIGN_FLIP_UNITS = 20
EXPECTED_PROHIBITED_CLAIMS = (
    "outcome_blind_external_validation",
    "independent_long_term_validation",
    "confirmatory_p_value",
    "activation_mechanism_confirmation",
    "formal_uncertainty_calibration",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

CELL_PAIRED_DELTA_COLUMNS = [
    "cell_id",
    "target_condition_id",
    "source_cell_number",
    "storage_soc_fraction",
    "candidate_method",
    "comparator_method",
    "candidate_trajectory_iae_pp",
    "comparator_trajectory_iae_pp",
    "paired_delta_trajectory_iae_pp",
    "candidate_final_absolute_error_pp",
    "comparator_final_absolute_error_pp",
    "paired_delta_final_absolute_error_pp",
    "candidate_outcome",
]

CELL_DAY_DELTA_COLUMNS = [
    "cell_id",
    "target_condition_id",
    "source_cell_number",
    "storage_soc_fraction",
    "target_elapsed_days",
    "candidate_method",
    "comparator_method",
    "true_capacity_retention_pct",
    "candidate_predicted_capacity_retention_pct",
    "comparator_predicted_capacity_retention_pct",
    "candidate_absolute_error_pp",
    "comparator_absolute_error_pp",
    "paired_delta_absolute_error_pp",
    "candidate_outcome",
]

STRATUM_DIAGNOSTIC_COLUMNS = [
    "scope_type",
    "scope",
    "metric",
    "physical_cell_count",
    "paired_observation_count",
    "candidate_error_mean_pp",
    "comparator_error_mean_pp",
    "mean_paired_delta_pp",
    "median_paired_delta_pp",
    "paired_delta_std_pp",
    "minimum_paired_delta_pp",
    "maximum_paired_delta_pp",
    "candidate_better_count",
    "candidate_worse_count",
    "candidate_numerical_zero_count",
    "numerical_zero_tolerance_pp",
    "exact_sign_test_nonzero_cell_count",
    "exact_sign_test_two_sided_p",
    "exact_mean_sign_flip_nonzero_cell_count",
    "exact_mean_sign_flip_permutation_count",
    "exact_mean_sign_flip_observed_mean_delta_pp",
    "exact_mean_sign_flip_extreme_threshold_pp",
    "exact_mean_sign_flip_inclusive_comparison_epsilon",
    "exact_mean_sign_flip_two_sided_p",
    "multiplicity_adjusted",
    "inference_status",
]

LEAVE_ONE_CELL_OUT_COLUMNS = [
    "omitted_cell_id",
    "omitted_target_condition_id",
    "omitted_storage_soc_fraction",
    "remaining_physical_cell_count",
    "full_mean_paired_delta_iae_pp",
    "leave_one_out_mean_paired_delta_iae_pp",
    "change_from_full_mean_pp",
    "leave_one_out_direction",
    "direction_flipped_from_full_sample",
]


def default_geisbauer_robustness_audit_protocol() -> dict[str, object]:
    return {
        "audit_id": AUDIT_ID,
        "design_status": DESIGN_STATUS,
        "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
        "base_experiment_id": BASE_EXPERIMENT_ID,
        "target_dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
        "statistical_unit": GEISBAUER_CALENDAR_STATISTICAL_UNIT,
        "candidate_method": PRIMARY_CANDIDATE,
        "comparator_method": PRIMARY_COMPARATOR,
        "primary_metric": PRIMARY_METRIC,
        "point_metric": POINT_METRIC,
        "numerical_zero_tolerance_pp": NUMERICAL_ZERO_TOLERANCE_PP,
        "strata": [
            "all_physical_cells",
            "storage_soc_fraction",
            "target_elapsed_days",
            "storage_soc_fraction_by_target_elapsed_days",
        ],
        "exact_diagnostics": {
            "sign_test": "two_sided_exact_binomial_nonzero_cell_differences",
            "mean_sign_flip": "two_sided_exhaustive_cell_level_sign_flip",
            "maximum_exact_units": MAXIMUM_EXACT_SIGN_FLIP_UNITS,
            "mean_sign_flip_extreme_threshold": "abs_observed_mean_delta",
            "inclusive_extreme_comparison": (
                "machine_epsilon_scaled_floating_point_inclusive"
            ),
            "numerical_zero_tolerance_role": ("exclude_numerically_zero_deltas_only"),
            "multiplicity_adjusted": False,
            "inference_status": INFERENCE_STATUS,
        },
        "sensitivity": {
            "leave_one_physical_cell_out": {
                "enabled": True,
                "expected_scenario_count": 15,
                "remaining_physical_cell_count": 14,
                "scenarios_are_highly_overlapping": True,
                "scenarios_are_independent_replications": False,
            },
            "post_hoc_equivalence_margin_sensitivity": {
                "margins_pp": list(POST_HOC_EQUIVALENCE_MARGINS_PP),
                "classification": (
                    "better_below_negative_margin_worse_above_positive_margin_"
                    "equivalent_within_inclusive_margin"
                ),
                "exploratory_only": True,
                "engineering_acceptance_gate": False,
            },
            "numerical_direction_definition": (
                "paired_delta_outside_numerical_zero_tolerance"
            ),
        },
        "claim_boundaries": {
            "outcomes_reviewed_before_audit_design": True,
            "target_horizon_days": 120,
            "target_temperature_c": 60.0,
            "long_term_validation_eligible": False,
            "confirmatory_inference_allowed": False,
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_geisbauer_robustness_audit_protocol(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(protocol, Mapping):
        raise ValueError("Geisbauer robustness audit protocol must be an object")
    parsed = dict(protocol)
    expected = default_geisbauer_robustness_audit_protocol()
    if _canonical_json_sha256(parsed) != _canonical_json_sha256(expected):
        raise ValueError("Geisbauer robustness audit protocol changed")
    return deepcopy(parsed)


def _finite_deltas(values: Sequence[float] | np.ndarray) -> np.ndarray:
    deltas = np.asarray(values, dtype=float)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("Paired deltas must be a non-empty one-dimensional array")
    if not np.isfinite(deltas).all():
        raise ValueError("Paired deltas must be finite")
    return deltas


def exact_two_sided_sign_test(
    values: Sequence[float] | np.ndarray,
    *,
    numerical_zero_tolerance: float = NUMERICAL_ZERO_TOLERANCE_PP,
) -> dict[str, int | float]:
    """Return the exact paired sign diagnostic after removing numerical zeros."""
    deltas = _finite_deltas(values)
    if not np.isfinite(numerical_zero_tolerance) or numerical_zero_tolerance < 0.0:
        raise ValueError("numerical_zero_tolerance must be finite and non-negative")
    negative = int(np.sum(deltas < -numerical_zero_tolerance))
    positive = int(np.sum(deltas > numerical_zero_tolerance))
    numerical_zero = int(len(deltas) - negative - positive)
    nonzero = negative + positive
    if nonzero == 0:
        p_value = 1.0
    else:
        smaller = min(negative, positive)
        lower_tail = sum(
            math.comb(nonzero, successes) for successes in range(smaller + 1)
        ) / (2**nonzero)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "negative_count": negative,
        "positive_count": positive,
        "numerical_zero_count": numerical_zero,
        "nonzero_count": nonzero,
        "two_sided_p": float(p_value),
    }


def exact_two_sided_mean_sign_flip_test(
    values: Sequence[float] | np.ndarray,
    *,
    numerical_zero_tolerance: float = NUMERICAL_ZERO_TOLERANCE_PP,
    maximum_exact_units: int = MAXIMUM_EXACT_SIGN_FLIP_UNITS,
) -> dict[str, int | float]:
    """Exhaustively sign-flip cell deltas; this is an exploratory diagnostic."""
    deltas = _finite_deltas(values)
    if not np.isfinite(numerical_zero_tolerance) or numerical_zero_tolerance < 0.0:
        raise ValueError("numerical_zero_tolerance must be finite and non-negative")
    nonzero = deltas[np.abs(deltas) > numerical_zero_tolerance]
    count = len(nonzero)
    if count > maximum_exact_units:
        raise ValueError(
            "Exact sign-flip unit count exceeds the frozen exhaustive limit"
        )
    if count == 0:
        return {
            "nonzero_count": 0,
            "permutation_count": 1,
            "observed_mean_delta_pp": 0.0,
            "extreme_threshold_pp": 0.0,
            "inclusive_comparison_machine_epsilon": float(np.finfo(float).eps),
            "two_sided_p": 1.0,
        }
    observed = float(nonzero.mean())
    threshold = abs(observed)
    machine_epsilon = float(np.finfo(float).eps)
    extreme = 0
    permutation_count = 2**count
    for signs in itertools.product((-1.0, 1.0), repeat=count):
        permuted_mean = float(np.mean(nonzero * np.asarray(signs)))
        permuted_magnitude = abs(permuted_mean)
        inclusive_epsilon = machine_epsilon * max(
            permuted_magnitude,
            threshold,
            float(np.finfo(float).tiny),
        )
        if (
            permuted_magnitude >= threshold
            or threshold - permuted_magnitude <= inclusive_epsilon
        ):
            extreme += 1
    return {
        "nonzero_count": count,
        "permutation_count": permutation_count,
        "observed_mean_delta_pp": observed,
        "extreme_threshold_pp": threshold,
        "inclusive_comparison_machine_epsilon": machine_epsilon,
        "two_sided_p": float(extreme / permutation_count),
    }


def _numerical_direction(delta: float, numerical_zero_tolerance: float) -> str:
    if delta < -numerical_zero_tolerance:
        return "candidate_better"
    if delta > numerical_zero_tolerance:
        return "candidate_worse_negative_transfer"
    return "numerical_zero_delta"


def _paired_cell_deltas(
    cell_metrics: pd.DataFrame,
    *,
    numerical_zero_tolerance: float,
) -> pd.DataFrame:
    identity = [
        "cell_id",
        "target_condition_id",
        "source_cell_number",
        "storage_soc_fraction",
    ]
    metric_columns = identity + [
        "trajectory_iae_pp",
        "final_absolute_error_pp",
    ]
    candidate = cell_metrics.loc[
        cell_metrics["method"] == PRIMARY_CANDIDATE,
        metric_columns,
    ].rename(
        columns={
            "trajectory_iae_pp": "candidate_trajectory_iae_pp",
            "final_absolute_error_pp": "candidate_final_absolute_error_pp",
        }
    )
    comparator = cell_metrics.loc[
        cell_metrics["method"] == PRIMARY_COMPARATOR,
        metric_columns,
    ].rename(
        columns={
            "trajectory_iae_pp": "comparator_trajectory_iae_pp",
            "final_absolute_error_pp": "comparator_final_absolute_error_pp",
        }
    )
    paired = candidate.merge(
        comparator,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != 15 or paired["cell_id"].nunique() != 15:
        raise ValueError("Robustness audit requires exactly 15 paired physical cells")
    paired["paired_delta_trajectory_iae_pp"] = (
        paired["candidate_trajectory_iae_pp"] - paired["comparator_trajectory_iae_pp"]
    )
    paired["paired_delta_final_absolute_error_pp"] = (
        paired["candidate_final_absolute_error_pp"]
        - paired["comparator_final_absolute_error_pp"]
    )
    paired["candidate_method"] = PRIMARY_CANDIDATE
    paired["comparator_method"] = PRIMARY_COMPARATOR
    paired["candidate_outcome"] = paired["paired_delta_trajectory_iae_pp"].map(
        lambda delta: _numerical_direction(float(delta), numerical_zero_tolerance)
    )
    return (
        paired[CELL_PAIRED_DELTA_COLUMNS]
        .sort_values("cell_id", kind="stable")
        .reset_index(drop=True)
    )


def _paired_cell_day_deltas(
    predictions: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    numerical_zero_tolerance: float,
) -> pd.DataFrame:
    identity = [
        "cell_id",
        "target_condition_id",
        "source_cell_number",
        "storage_soc_fraction",
        "target_elapsed_days",
    ]
    columns = identity + ["predicted_capacity_retention_pct"]
    candidate = predictions.loc[
        predictions["method"] == PRIMARY_CANDIDATE,
        columns,
    ].rename(
        columns={
            "predicted_capacity_retention_pct": (
                "candidate_predicted_capacity_retention_pct"
            )
        }
    )
    comparator = predictions.loc[
        predictions["method"] == PRIMARY_COMPARATOR,
        columns,
    ].rename(
        columns={
            "predicted_capacity_retention_pct": (
                "comparator_predicted_capacity_retention_pct"
            )
        }
    )
    paired = candidate.merge(
        comparator,
        on=identity,
        how="inner",
        validate="one_to_one",
    )
    truth = target_observations[
        ["cell_id", "elapsed_days", "capacity_retention_pct"]
    ].rename(
        columns={
            "elapsed_days": "target_elapsed_days",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    paired = paired.merge(
        truth,
        on=["cell_id", "target_elapsed_days"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (paired.pop("_merge") != "both").any() or len(paired) != 30:
        raise ValueError("Every cell-day prediction must match target truth")
    paired["candidate_absolute_error_pp"] = np.abs(
        paired["candidate_predicted_capacity_retention_pct"]
        - paired["true_capacity_retention_pct"]
    )
    paired["comparator_absolute_error_pp"] = np.abs(
        paired["comparator_predicted_capacity_retention_pct"]
        - paired["true_capacity_retention_pct"]
    )
    paired["paired_delta_absolute_error_pp"] = (
        paired["candidate_absolute_error_pp"] - paired["comparator_absolute_error_pp"]
    )
    paired["candidate_method"] = PRIMARY_CANDIDATE
    paired["comparator_method"] = PRIMARY_COMPARATOR
    paired["candidate_outcome"] = paired["paired_delta_absolute_error_pp"].map(
        lambda delta: _numerical_direction(float(delta), numerical_zero_tolerance)
    )
    return (
        paired[CELL_DAY_DELTA_COLUMNS]
        .sort_values(["cell_id", "target_elapsed_days"], kind="stable")
        .reset_index(drop=True)
    )


def _diagnostic_row(
    *,
    scope_type: str,
    scope: str,
    metric: str,
    cell_ids: pd.Series,
    candidate_error: pd.Series,
    comparator_error: pd.Series,
    deltas: pd.Series,
    numerical_zero_tolerance: float,
    maximum_exact_units: int,
) -> dict[str, object]:
    delta = deltas.to_numpy(dtype=float)
    if cell_ids.nunique() != len(delta):
        raise ValueError("Every robustness stratum must contain one row per cell")
    sign = exact_two_sided_sign_test(
        delta,
        numerical_zero_tolerance=numerical_zero_tolerance,
    )
    flip = exact_two_sided_mean_sign_flip_test(
        delta,
        numerical_zero_tolerance=numerical_zero_tolerance,
        maximum_exact_units=maximum_exact_units,
    )
    return {
        "scope_type": scope_type,
        "scope": scope,
        "metric": metric,
        "physical_cell_count": int(cell_ids.nunique()),
        "paired_observation_count": len(delta),
        "candidate_error_mean_pp": float(candidate_error.mean()),
        "comparator_error_mean_pp": float(comparator_error.mean()),
        "mean_paired_delta_pp": float(delta.mean()),
        "median_paired_delta_pp": float(np.median(delta)),
        "paired_delta_std_pp": float(np.std(delta, ddof=1)),
        "minimum_paired_delta_pp": float(delta.min()),
        "maximum_paired_delta_pp": float(delta.max()),
        "candidate_better_count": sign["negative_count"],
        "candidate_worse_count": sign["positive_count"],
        "candidate_numerical_zero_count": sign["numerical_zero_count"],
        "numerical_zero_tolerance_pp": numerical_zero_tolerance,
        "exact_sign_test_nonzero_cell_count": sign["nonzero_count"],
        "exact_sign_test_two_sided_p": sign["two_sided_p"],
        "exact_mean_sign_flip_nonzero_cell_count": flip["nonzero_count"],
        "exact_mean_sign_flip_permutation_count": flip["permutation_count"],
        "exact_mean_sign_flip_observed_mean_delta_pp": flip["observed_mean_delta_pp"],
        "exact_mean_sign_flip_extreme_threshold_pp": flip["extreme_threshold_pp"],
        "exact_mean_sign_flip_inclusive_comparison_epsilon": flip[
            "inclusive_comparison_machine_epsilon"
        ],
        "exact_mean_sign_flip_two_sided_p": flip["two_sided_p"],
        "multiplicity_adjusted": False,
        "inference_status": INFERENCE_STATUS,
    }


def _stratum_diagnostics(
    cell_deltas: pd.DataFrame,
    cell_day_deltas: pd.DataFrame,
    *,
    numerical_zero_tolerance: float,
    maximum_exact_units: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        _diagnostic_row(
            scope_type="all_physical_cells",
            scope="all_cells",
            metric=PRIMARY_METRIC,
            cell_ids=cell_deltas["cell_id"],
            candidate_error=cell_deltas["candidate_trajectory_iae_pp"],
            comparator_error=cell_deltas["comparator_trajectory_iae_pp"],
            deltas=cell_deltas["paired_delta_trajectory_iae_pp"],
            numerical_zero_tolerance=numerical_zero_tolerance,
            maximum_exact_units=maximum_exact_units,
        )
    )
    for soc, group in cell_deltas.groupby("storage_soc_fraction", sort=True):
        rows.append(
            _diagnostic_row(
                scope_type="storage_soc_fraction",
                scope=f"soc_{float(soc):.1f}",
                metric=PRIMARY_METRIC,
                cell_ids=group["cell_id"],
                candidate_error=group["candidate_trajectory_iae_pp"],
                comparator_error=group["comparator_trajectory_iae_pp"],
                deltas=group["paired_delta_trajectory_iae_pp"],
                numerical_zero_tolerance=numerical_zero_tolerance,
                maximum_exact_units=maximum_exact_units,
            )
        )
    for elapsed, group in cell_day_deltas.groupby("target_elapsed_days", sort=True):
        rows.append(
            _diagnostic_row(
                scope_type="target_elapsed_days",
                scope=f"day_{int(elapsed)}",
                metric=POINT_METRIC,
                cell_ids=group["cell_id"],
                candidate_error=group["candidate_absolute_error_pp"],
                comparator_error=group["comparator_absolute_error_pp"],
                deltas=group["paired_delta_absolute_error_pp"],
                numerical_zero_tolerance=numerical_zero_tolerance,
                maximum_exact_units=maximum_exact_units,
            )
        )
    grouped = cell_day_deltas.groupby(
        ["storage_soc_fraction", "target_elapsed_days"], sort=True
    )
    for (soc, elapsed), group in grouped:
        rows.append(
            _diagnostic_row(
                scope_type="storage_soc_fraction_by_target_elapsed_days",
                scope=f"soc_{float(soc):.1f}_day_{int(elapsed)}",
                metric=POINT_METRIC,
                cell_ids=group["cell_id"],
                candidate_error=group["candidate_absolute_error_pp"],
                comparator_error=group["comparator_absolute_error_pp"],
                deltas=group["paired_delta_absolute_error_pp"],
                numerical_zero_tolerance=numerical_zero_tolerance,
                maximum_exact_units=maximum_exact_units,
            )
        )
    return pd.DataFrame(rows, columns=STRATUM_DIAGNOSTIC_COLUMNS)


def _leave_one_cell_out(
    cell_deltas: pd.DataFrame,
    *,
    numerical_zero_tolerance: float,
) -> pd.DataFrame:
    full_mean = float(cell_deltas["paired_delta_trajectory_iae_pp"].mean())
    full_direction = _numerical_direction(
        full_mean,
        numerical_zero_tolerance,
    )
    rows: list[dict[str, object]] = []
    for omitted in cell_deltas.itertuples(index=False):
        retained = cell_deltas.loc[cell_deltas["cell_id"] != omitted.cell_id]
        mean_delta = float(retained["paired_delta_trajectory_iae_pp"].mean())
        leave_one_out_direction = _numerical_direction(
            mean_delta,
            numerical_zero_tolerance,
        )
        rows.append(
            {
                "omitted_cell_id": str(omitted.cell_id),
                "omitted_target_condition_id": str(omitted.target_condition_id),
                "omitted_storage_soc_fraction": float(omitted.storage_soc_fraction),
                "remaining_physical_cell_count": len(retained),
                "full_mean_paired_delta_iae_pp": full_mean,
                "leave_one_out_mean_paired_delta_iae_pp": mean_delta,
                "change_from_full_mean_pp": mean_delta - full_mean,
                "leave_one_out_direction": leave_one_out_direction,
                "direction_flipped_from_full_sample": (
                    leave_one_out_direction != full_direction
                ),
            }
        )
    return (
        pd.DataFrame(rows, columns=LEAVE_ONE_CELL_OUT_COLUMNS)
        .sort_values("omitted_cell_id", kind="stable")
        .reset_index(drop=True)
    )


def _post_hoc_equivalence_margin_sensitivity(
    cell_deltas: pd.DataFrame,
    *,
    margins_pp: Sequence[float],
    numerical_zero_tolerance: float,
) -> list[dict[str, object]]:
    deltas = cell_deltas["paired_delta_trajectory_iae_pp"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for raw_margin in margins_pp:
        margin = float(raw_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("Equivalence margins must be finite and non-negative")
        classification_boundary = max(margin, numerical_zero_tolerance)
        better = int(np.sum(deltas < -classification_boundary))
        worse = int(np.sum(deltas > classification_boundary))
        equivalent = int(len(deltas) - better - worse)
        rows.append(
            {
                "equivalence_margin_pp": margin,
                "candidate_better_cell_count": better,
                "candidate_worse_cell_count": worse,
                "equivalent_cell_count": equivalent,
            }
        )
    return rows


def run_geisbauer_robustness_audit(
    source_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    external_protocol: Mapping[str, object],
    audit_protocol: Mapping[str, object],
) -> tuple[
    dict[str, object],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    parsed = validate_geisbauer_robustness_audit_protocol(audit_protocol)
    base_result, predictions, cell_metrics, _, _ = run_geisbauer_external_stress(
        source_observations,
        target_observations,
        protocol=external_protocol,
    )
    if geisbauer_calendar_observations_sha256(target_observations) != (
        GEISBAUER_CALENDAR_OBSERVATIONS_SHA256
    ):
        raise ValueError("Geisbauer target outcome snapshot mismatch")
    target_days = sorted(target_observations["elapsed_days"].unique().tolist())
    if target_days != [0.0, 39.0, 59.0, 84.0, 120.0]:
        raise ValueError("Geisbauer target day support changed")
    if set(target_observations["temperature_c"].astype(float)) != {60.0}:
        raise ValueError("Geisbauer target temperature support changed")

    numerical_zero_tolerance = float(parsed["numerical_zero_tolerance_pp"])
    exact = parsed["exact_diagnostics"]
    maximum_exact_units = int(exact["maximum_exact_units"])
    cell_deltas = _paired_cell_deltas(
        cell_metrics,
        numerical_zero_tolerance=numerical_zero_tolerance,
    )
    cell_day_deltas = _paired_cell_day_deltas(
        predictions,
        target_observations,
        numerical_zero_tolerance=numerical_zero_tolerance,
    )
    strata = _stratum_diagnostics(
        cell_deltas,
        cell_day_deltas,
        numerical_zero_tolerance=numerical_zero_tolerance,
        maximum_exact_units=maximum_exact_units,
    )
    leave_one_out = _leave_one_cell_out(
        cell_deltas,
        numerical_zero_tolerance=numerical_zero_tolerance,
    )
    sensitivity_protocol = parsed["sensitivity"]
    equivalence_protocol = sensitivity_protocol[
        "post_hoc_equivalence_margin_sensitivity"
    ]
    equivalence_sensitivity = _post_hoc_equivalence_margin_sensitivity(
        cell_deltas,
        margins_pp=equivalence_protocol["margins_pp"],
        numerical_zero_tolerance=numerical_zero_tolerance,
    )

    candidate_predictions = predictions.loc[
        predictions["method"] == GATED_HIERARCHICAL_ACTIVATION_METHOD
    ].sort_values(["cell_id", "target_elapsed_days"], kind="stable")
    fallback_predictions = predictions.loc[
        predictions["method"] == HIERARCHICAL_POWER_METHOD
    ].sort_values(["cell_id", "target_elapsed_days"], kind="stable")
    fallback_exact = bool(
        np.array_equal(
            candidate_predictions["predicted_capacity_retention_pct"].to_numpy(
                dtype=float
            ),
            fallback_predictions["predicted_capacity_retention_pct"].to_numpy(
                dtype=float
            ),
        )
    )
    if not fallback_exact:
        raise ValueError("Candidate no longer exactly equals its power fallback")

    overall = strata.loc[strata["scope"] == "all_cells"].iloc[0]
    soc_rows = strata.loc[strata["scope_type"] == "storage_soc_fraction"].copy()
    day_rows = strata.loc[strata["scope_type"] == "target_elapsed_days"].copy()
    loo_values = leave_one_out["leave_one_out_mean_paired_delta_iae_pp"].to_numpy(
        dtype=float
    )
    maximum_influence_index = leave_one_out["change_from_full_mean_pp"].abs().idxmax()
    maximum_influence = leave_one_out.loc[maximum_influence_index]
    result: dict[str, object] = {
        "status": "retrospective_external_robustness_audit_complete",
        "execution_status": "completed",
        "audit_id": AUDIT_ID,
        "design_status": DESIGN_STATUS,
        "inference_status": INFERENCE_STATUS,
        "config_sha256": _canonical_json_sha256(parsed),
        "numerical_zero_tolerance_pp": numerical_zero_tolerance,
        "scope": {
            "dataset_id": GEISBAUER_CALENDAR_DATASET_ID,
            "evidence_role": GEISBAUER_CALENDAR_EVIDENCE_ROLE,
            "statistical_unit": GEISBAUER_CALENDAR_STATISTICAL_UNIT,
            "physical_cell_count": 15,
            "storage_temperature_c": 60.0,
            "maximum_observed_days": 120.0,
            "target_scoring_days": [84, 120],
            "outcomes_reviewed_before_audit_design": True,
        },
        "route_reality": {
            "candidate_method": PRIMARY_CANDIDATE,
            "comparator_method": PRIMARY_COMPARATOR,
            "candidate_exactly_equals_hierarchical_power_fallback": fallback_exact,
            "activation_gate_ready_physical_cell_count": int(
                base_result["mechanism_gate"]["gate_ready_physical_cell_count"]
            ),
            "activation_specialist_tested": bool(
                base_result["mechanism_gate"]["activation_mechanism_tested"]
            ),
        },
        "overall_paired_diagnostic": {
            "primary_metric": PRIMARY_METRIC,
            "candidate_error_mean_pp": float(overall["candidate_error_mean_pp"]),
            "comparator_error_mean_pp": float(overall["comparator_error_mean_pp"]),
            "mean_paired_delta_pp": float(overall["mean_paired_delta_pp"]),
            "median_paired_delta_pp": float(overall["median_paired_delta_pp"]),
            "candidate_better_cell_count": int(overall["candidate_better_count"]),
            "candidate_worse_cell_count": int(overall["candidate_worse_count"]),
            "candidate_numerical_zero_cell_count": int(
                overall["candidate_numerical_zero_count"]
            ),
            "exact_sign_test_nonzero_cell_count": int(
                overall["exact_sign_test_nonzero_cell_count"]
            ),
            "exact_sign_test_two_sided_p": float(
                overall["exact_sign_test_two_sided_p"]
            ),
            "exact_mean_sign_flip_nonzero_cell_count": int(
                overall["exact_mean_sign_flip_nonzero_cell_count"]
            ),
            "exact_mean_sign_flip_permutation_count": int(
                overall["exact_mean_sign_flip_permutation_count"]
            ),
            "exact_mean_sign_flip_observed_mean_delta_pp": float(
                overall["exact_mean_sign_flip_observed_mean_delta_pp"]
            ),
            "exact_mean_sign_flip_extreme_threshold_pp": float(
                overall["exact_mean_sign_flip_extreme_threshold_pp"]
            ),
            "exact_mean_sign_flip_inclusive_comparison_epsilon": float(
                overall["exact_mean_sign_flip_inclusive_comparison_epsilon"]
            ),
            "exact_mean_sign_flip_two_sided_p": float(
                overall["exact_mean_sign_flip_two_sided_p"]
            ),
            "nominal_diagnostics_are_confirmatory": False,
        },
        "leave_one_cell_out": {
            "scenario_count": len(leave_one_out),
            "remaining_physical_cell_count_per_scenario": 14,
            "scenarios_are_highly_overlapping": True,
            "scenarios_are_independent_replications": False,
            "minimum_mean_paired_delta_pp": float(loo_values.min()),
            "maximum_mean_paired_delta_pp": float(loo_values.max()),
            "candidate_better_direction_count": int(
                np.sum(loo_values < -numerical_zero_tolerance)
            ),
            "candidate_worse_direction_count": int(
                np.sum(loo_values > numerical_zero_tolerance)
            ),
            "numerical_zero_direction_count": int(
                np.sum(np.abs(loo_values) <= numerical_zero_tolerance)
            ),
            "direction_flip_scenario_count": int(
                leave_one_out["direction_flipped_from_full_sample"].sum()
            ),
            "maximum_absolute_influence_cell_id": str(
                maximum_influence["omitted_cell_id"]
            ),
            "maximum_absolute_change_from_full_mean_pp": abs(
                float(maximum_influence["change_from_full_mean_pp"])
            ),
        },
        "post_hoc_equivalence_margin_sensitivity": {
            "metric": PRIMARY_METRIC,
            "unit": "percentage_points",
            "numerical_zero_tolerance_pp": numerical_zero_tolerance,
            "margins": equivalence_sensitivity,
            "status": "exploratory_post_hoc_sensitivity",
            "engineering_acceptance_gate": False,
            "interpretation": (
                "These margins are exploratory descriptive reclassifications, "
                "not engineering acceptance thresholds or model gates."
            ),
        },
        "negative_transfer_diagnosis": {
            "aggregate_mean_negative_transfer_observed": bool(
                float(overall["mean_paired_delta_pp"]) > numerical_zero_tolerance
            ),
            "physical_cells_with_negative_transfer": int(
                overall["candidate_worse_count"]
            ),
            "soc_strata_mean_delta_pp": {
                str(row.scope): float(row.mean_paired_delta_pp)
                for row in soc_rows.itertuples(index=False)
            },
            "soc_strata_cell_outcomes": {
                str(row.scope): {
                    "mean_paired_delta_pp": float(row.mean_paired_delta_pp),
                    "median_paired_delta_pp": float(row.median_paired_delta_pp),
                    "candidate_better_cell_count": int(row.candidate_better_count),
                    "candidate_worse_cell_count": int(row.candidate_worse_count),
                    "candidate_numerical_zero_cell_count": int(
                        row.candidate_numerical_zero_count
                    ),
                    "exact_sign_test_two_sided_p": float(
                        row.exact_sign_test_two_sided_p
                    ),
                }
                for row in soc_rows.itertuples(index=False)
            },
            "day_strata_mean_delta_pp": {
                str(row.scope): float(row.mean_paired_delta_pp)
                for row in day_rows.itertuples(index=False)
            },
            "soc_strata_with_candidate_better_mean": [
                str(row.scope)
                for row in soc_rows.itertuples(index=False)
                if float(row.mean_paired_delta_pp) < -numerical_zero_tolerance
            ],
            "soc_strata_with_negative_transfer_mean": [
                str(row.scope)
                for row in soc_rows.itertuples(index=False)
                if float(row.mean_paired_delta_pp) > numerical_zero_tolerance
            ],
            "mean_median_direction_conflict": bool(
                float(overall["mean_paired_delta_pp"])
                * float(overall["median_paired_delta_pp"])
                < 0.0
            ),
            "soc_strata_with_mean_median_direction_conflict": [
                str(row.scope)
                for row in soc_rows.itertuples(index=False)
                if float(row.mean_paired_delta_pp) * float(row.median_paired_delta_pp)
                < 0.0
            ],
            "diagnostic_interpretation": (
                "Observed transfer harm belongs to the hierarchical-power fallback "
                "route; the activation specialist was never eligible. Mean, median, "
                "cell-win counts, SOC strata, and leave-one-cell-out results must be "
                "read together because the small cohort is heterogeneous."
            ),
        },
        "claim_boundary": {
            "model_validation_status": "not_confirmed",
            "confirmatory_inference_allowed": False,
            "independent_long_term_validation_claim_allowed": False,
            "reason": (
                "This audit was designed after review of the v0.11 outcomes and "
                "uses 15 cells stored at 60 C for only 120 days. Exact tests are "
                "unadjusted exploratory diagnostics whose cell-exchangeability or "
                "paired-difference symmetry assumptions cannot be established from "
                "this small designed cohort."
            ),
        },
        "base_evidence": {
            "base_experiment_id": base_result["experiment_id"],
            "base_design_status": base_result["design_status"],
            "label_free_prediction_sha256": (
                geisbauer_external_prediction_sha256(predictions)
            ),
            "target_outcome_sha256": (
                geisbauer_calendar_observations_sha256(target_observations)
            ),
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }
    return result, cell_deltas, cell_day_deltas, strata, leave_one_out
