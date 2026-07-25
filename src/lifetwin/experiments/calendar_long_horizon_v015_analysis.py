from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from lifetwin.experiments.calendar_long_horizon_v015_protocol import (
    CORE_FAMILY_IDS,
    PLACEBO_FIELDS,
    REAL_OPERATING_FIELDS,
    OperatingCovariates,
    ValidatedV015Protocol,
    evaluate_intrinsic_pair_retention,
    evaluate_stress_plan_pair_retention,
    truth_is_admissible,
)


class V015AnalysisError(ValueError):
    """Raised when a frozen V0.15 estimand cannot be reconstructed."""


class V015InconclusiveError(V015AnalysisError):
    """Raised when a frozen estimand is unavailable for a declared count reason."""


TEST_ISSUE_COUNT = 950
AUDIT_ISSUE_COUNT = 475
TEST_MINIMUM_COMMON_POOL = 1805
AUDIT_MINIMUM_COMMON_POOL = 903
TEST_MINIMUM_CATASTROPHIC = 60
AUDIT_MINIMUM_CATASTROPHIC = 30
RANDOM_RANKING_COUNT = 10_000
BOOTSTRAP_RESAMPLES = 5_000
STRESS_PERMUTATIONS = 10_000
RANDOM_ROOT = 202607230110
BOOTSTRAP_ROOT = 202607230111
STRESS_PERMUTATION_ROOT = 202607230112
CORE_FAMILIES = (
    "single_power",
    "dual_power",
    "saturating_plus_slow",
    "early_activation_plus_power",
    "late_knee",
    "linear_drift_plus_power",
)
NOVEL_FAMILIES = ("smooth_broken_power", "saturating_logistic_knee")
TEST_FAMILIES = CORE_FAMILIES + NOVEL_FAMILIES
FORECAST_DAYS = (
    1095.75,
    1461.0,
    1826.25,
    2556.75,
    3652.5,
    5478.75,
    7305.0,
    9131.25,
)
RISK_SCORE_IDS = (
    "prefix_only",
    "visible_stress",
    "placebo_8",
    "arm_a_plus_s_plan",
    "strongest_single_feature",
    "planned_stress_only",
    "prefix_rmse_only",
    "v1_max_envelope_only",
    "center_sqrt_abs_difference_only",
)
REQUIRED_GATE_IDS = (
    "test_common_pool_minimum_counts",
    "audit_common_pool_minimum_counts",
    "visible_stress_catastrophic_risk_reduction",
    "visible_stress_increment_over_prefix_only",
    "core_test_simultaneous_trajectory_coverage",
    "intrinsic_pair_simultaneous_both_future_coverage",
    "issued_center_trajectory_iae_noninferiority",
    *(f"test_family_{family}_nonnegative_risk_reduction" for family in TEST_FAMILIES),
    "test_novel_positive_risk_reduction",
    "test_novel_nonnegative_increment",
    "placebo_point_negative_control",
    "placebo_interval_negative_control",
    "stress_permutation_negative_control",
    "intrinsic_output_invariance",
    "stress_plan_arm_a_invariance",
    "audit_visible_stress_positive_risk_reduction",
    "audit_visible_stress_positive_increment",
    "audit_issued_center_iae_noninferiority",
    "audit_novel_nonnegative_risk_reduction",
    "audit_late_knee_nonnegative_risk_reduction",
)


@dataclass(frozen=True)
class RiskReduction:
    issued_count: int
    issued_catastrophic_rate: float
    random_expected_catastrophic_rate: float
    relative_risk_reduction: float


@dataclass(frozen=True)
class CoverageSummary:
    n: int
    covered: int
    coverage: float
    one_sided_95_lower: float
    median_max_width_pp: float
    percentile_95_max_width_pp: float


@dataclass(frozen=True)
class StressPairSummary:
    pair_count: int
    arm_a_exact_tie_count: int
    arm_b_correct_order_count: int
    arm_b_correct_order_fraction: float
    arm_b_two_sided_95_lower: float
    arm_b_two_sided_95_upper: float


@dataclass(frozen=True)
class SubsetRiskSummary:
    subset_id: str
    source_count: int
    eligible_count: int
    catastrophic_count: int
    issued_count: int
    prefix_only: RiskReduction
    visible_stress: RiskReduction
    visible_minus_prefix_increment: float


@dataclass(frozen=True)
class MeanBaselineSelection:
    selected_model_id: str
    calibration_cluster_count: int
    mean_trajectory_iae_pp: float


@dataclass(frozen=True)
class StressPermutationSummary:
    permutation_count: int
    observed_visible_minus_prefix_increment: float
    strictly_lower_count: int
    strictly_lower_fraction: float
    gate_passed: bool


@dataclass(frozen=True)
class GateEvaluation:
    gate_id: str
    state: str
    estimate: float | bool | None
    threshold: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.state not in {"pass", "fail", "inconclusive"}:
            raise ValueError(f"Invalid gate state: {self.state}")


def _require_columns(
    frame: pd.DataFrame, required: Iterable[str], *, context: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise V015AnalysisError(f"{context} missing columns: {missing}")


def _strict_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise V015AnalysisError(f"{context} must be boolean")
    return bool(value)


def _strict_bool_series(series: pd.Series, *, context: str) -> pd.Series:
    return series.map(lambda value: _strict_bool(value, context=context)).astype(bool)


def _finite_vector(values: Sequence[object], *, context: str) -> np.ndarray:
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if not np.isfinite(result).all():
        raise V015AnalysisError(f"{context} must be finite")
    return result


def _sha256_text(value: object, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise V015AnalysisError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _float64_bytes(value: object, *, context: str) -> bytes:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise V015AnalysisError(f"{context} must be a finite float64 value") from exc
    if not math.isfinite(numeric):
        raise V015AnalysisError(f"{context} must be a finite float64 value")
    return struct.pack(">d", numeric)


def _float64_array_bytes(
    values: Sequence[object] | pd.Series,
    *,
    context: str,
) -> bytes:
    numeric = _finite_vector(values, context=context)
    return np.asarray(numeric, dtype=">f8").tobytes()


def _require_bitwise_equal_numeric(
    left: Sequence[object] | pd.Series,
    right: Sequence[object] | pd.Series,
    *,
    context: str,
) -> None:
    if len(left) != len(right) or _float64_array_bytes(
        left, context=f"{context}/left"
    ) != _float64_array_bytes(right, context=f"{context}/right"):
        raise V015AnalysisError(f"{context} differs bitwise")


def _canonical_scalar_equal(left: object, right: object) -> bool:
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    if isinstance(left, (bool, np.bool_)) or isinstance(right, (bool, np.bool_)):
        return (
            isinstance(left, (bool, np.bool_))
            and isinstance(right, (bool, np.bool_))
            and bool(left) is bool(right)
        )
    if isinstance(left, (int, float, np.integer, np.floating)) or isinstance(
        right, (int, float, np.integer, np.floating)
    ):
        try:
            return _float64_bytes(left, context="canonical scalar") == _float64_bytes(
                right, context="canonical scalar"
            )
        except V015AnalysisError:
            return False
    return type(left) is type(right) and left == right


def _require_canonical_rows_equal(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    columns: Sequence[str],
    context: str,
) -> None:
    if len(left) != len(right):
        raise V015AnalysisError(f"{context} row counts differ")
    for column in columns:
        for left_value, right_value in zip(
            left[column].tolist(), right[column].tolist(), strict=True
        ):
            if not _canonical_scalar_equal(left_value, right_value):
                raise V015AnalysisError(f"{context} differs canonically in {column}")


def clopper_pearson_lower(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise V015AnalysisError("Invalid binomial counts")
    if successes == 0:
        return 0.0
    return float(beta.ppf(0.05, successes, trials - successes + 1))


def clopper_pearson_two_sided(successes: int, trials: int) -> tuple[float, float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise V015AnalysisError("Invalid binomial counts")
    lower = (
        0.0
        if successes == 0
        else float(beta.ppf(0.025, successes, trials - successes + 1))
    )
    upper = (
        1.0
        if successes == trials
        else float(beta.ppf(0.975, successes + 1, trials - successes))
    )
    return lower, upper


def score_trajectory_table(
    prediction_bundle: pd.DataFrame,
    truth_pack: pd.DataFrame,
    risk_bundle: pd.DataFrame,
    decision_bundle: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct point and trajectory scores from committed bundles."""
    prediction_columns = {
        "protocol_id",
        "partition",
        "cluster_id",
        "forecast_day",
        "center_forecast_pct",
        "sqrt_time_forecast_pct",
        "bounded_power_forecast_pct",
        "base_interval_lower_pct",
        "base_interval_upper_pct",
        "calibrated_interval_lower_pct",
        "calibrated_interval_upper_pct",
        "canonical_prefix_content_sha256",
    }
    truth_columns = {
        "protocol_id",
        "partition",
        "cluster_id",
        "truth_family",
        "forecast_day",
        "latent_retention_pct",
        "noisy_retention_pct",
    }
    risk_columns = {
        "protocol_id",
        "partition",
        "cluster_id",
        "score_id",
        "raw_risk_score",
        "calibrated_catastrophic_probability",
        "canonical_predictor_content_sha256",
    }
    decision_columns = {
        "protocol_id",
        "partition",
        "cluster_id",
        "arm",
        "hard_eligible",
        "issued",
        "issuance_rank",
        "raw_risk_score",
        "canonical_predictor_content_sha256",
    }
    _require_columns(prediction_bundle, prediction_columns, context="Predictions")
    _require_columns(truth_pack, truth_columns, context="Truth")
    _require_columns(risk_bundle, risk_columns, context="Risk")
    _require_columns(decision_bundle, decision_columns, context="Decisions")

    point_key = ["protocol_id", "partition", "cluster_id", "forecast_day"]
    if prediction_bundle.duplicated(point_key).any():
        raise V015AnalysisError("Prediction coordinates are not unique")
    if truth_pack.duplicated(point_key).any():
        raise V015AnalysisError("Truth coordinates are not unique")
    points = prediction_bundle.merge(
        truth_pack.loc[:, sorted(truth_columns)],
        on=point_key,
        how="inner",
        validate="one_to_one",
    )
    if len(points) != len(prediction_bundle) or len(points) != len(truth_pack):
        raise V015AnalysisError("Prediction/truth coordinates differ")
    required_finite_columns = (
        "center_forecast_pct",
        "sqrt_time_forecast_pct",
        "bounded_power_forecast_pct",
        "latent_retention_pct",
        "noisy_retention_pct",
    )
    interval_columns = (
        "base_interval_lower_pct",
        "base_interval_upper_pct",
        "calibrated_interval_lower_pct",
        "calibrated_interval_upper_pct",
    )
    for column in (*required_finite_columns, *interval_columns):
        points[column] = pd.to_numeric(points[column], errors="coerce")
    if not np.isfinite(points.loc[:, required_finite_columns].to_numpy(float)).all():
        raise V015AnalysisError(
            "Center, comparator, and truth point inputs must all be finite"
        )
    interval_values = points.loc[:, interval_columns].to_numpy(float)
    if np.isinf(interval_values).any():
        raise V015AnalysisError("Prediction intervals cannot contain infinity")
    interval_missing_by_cluster: dict[tuple[str, str, str], bool] = {}
    for key, group in points.groupby(
        ["protocol_id", "partition", "cluster_id"], sort=False
    ):
        missing = group.loc[:, interval_columns].isna().to_numpy()
        if missing.all():
            interval_missing_by_cluster[key] = True
        elif missing.any():
            raise V015AnalysisError(
                "A prediction trajectory contains partially missing intervals"
            )
        else:
            interval_missing_by_cluster[key] = False
    finite_interval_rows = ~points[list(interval_columns)].isna().any(axis=1)
    if (
        points.loc[finite_interval_rows, "base_interval_lower_pct"]
        > points.loc[finite_interval_rows, "base_interval_upper_pct"]
    ).any() or (
        points.loc[finite_interval_rows, "calibrated_interval_lower_pct"]
        > points.loc[finite_interval_rows, "calibrated_interval_upper_pct"]
    ).any():
        raise V015AnalysisError("Prediction interval bounds are reversed")
    points["center_absolute_error_pp"] = np.abs(
        points["center_forecast_pct"] - points["latent_retention_pct"]
    )
    points["sqrt_absolute_error_pp"] = np.abs(
        points["sqrt_time_forecast_pct"] - points["latent_retention_pct"]
    )
    points["bounded_power_absolute_error_pp"] = np.abs(
        points["bounded_power_forecast_pct"] - points["latent_retention_pct"]
    )
    points["interval_covers_truth"] = (
        (points["calibrated_interval_lower_pct"] <= points["latent_retention_pct"])
        & (points["latent_retention_pct"] <= points["calibrated_interval_upper_pct"])
    ).astype("boolean")
    points.loc[~finite_interval_rows, "interval_covers_truth"] = pd.NA
    points["interval_width_pp"] = (
        points["calibrated_interval_upper_pct"]
        - points["calibrated_interval_lower_pct"]
    )
    points["base_interval_covers_truth"] = (
        (points["base_interval_lower_pct"] <= points["latent_retention_pct"])
        & (points["latent_retention_pct"] <= points["base_interval_upper_pct"])
    ).astype("boolean")
    points.loc[~finite_interval_rows, "base_interval_covers_truth"] = pd.NA
    points["base_interval_width_pp"] = (
        points["base_interval_upper_pct"] - points["base_interval_lower_pct"]
    )

    cluster_key = ["protocol_id", "partition", "cluster_id"]
    if risk_bundle.duplicated(cluster_key + ["score_id"]).any():
        raise V015AnalysisError("Risk coordinates are not unique")
    family_counts = risk_bundle.loc[:, cluster_key].copy()
    if "successful_structure_family_count" in risk_bundle.columns:
        family_counts["successful_structure_family_count"] = risk_bundle[
            "successful_structure_family_count"
        ]
    else:
        family_counts["successful_structure_family_count"] = 1
    family_counts["successful_structure_family_count"] = pd.to_numeric(
        family_counts["successful_structure_family_count"], errors="coerce"
    )
    for key, group in family_counts.groupby(cluster_key, sort=False):
        values = group["successful_structure_family_count"].to_numpy(float)
        if (
            len(values) != len(RISK_SCORE_IDS)
            or not np.isfinite(values).all()
            or not np.all(values == values[0])
            or values[0] != math.floor(values[0])
            or not 0 <= values[0] <= len(CORE_FAMILIES) + 1
        ):
            raise V015AnalysisError(
                "Successful-family count is invalid or differs across scores"
            )
        missing = interval_missing_by_cluster.get(tuple(str(item) for item in key))
        if missing is None:
            raise V015AnalysisError("Risk and prediction cluster coordinates differ")
        if missing != (values[0] == 0):
            raise V015AnalysisError(
                "Missing intervals do not match zero structural support"
            )
    risk_wide = risk_bundle.pivot(
        index=cluster_key,
        columns="score_id",
        values=[
            "raw_risk_score",
            "calibrated_catastrophic_probability",
            "canonical_predictor_content_sha256",
        ],
    )
    required_scores = set(RISK_SCORE_IDS)
    if any(
        not required_scores.issubset(set(risk_wide[field].columns))
        for field in (
            "raw_risk_score",
            "calibrated_catastrophic_probability",
            "canonical_predictor_content_sha256",
        )
    ):
        raise V015AnalysisError("Primary or placebo risk score is absent")
    risk_wide.columns = [
        (
            f"risk_{score_id}"
            if field == "raw_risk_score"
            else (
                f"calibrated_risk_{score_id}"
                if field == "calibrated_catastrophic_probability"
                else f"risk_hash_{score_id}"
            )
        )
        for field, score_id in risk_wide.columns.to_flat_index()
    ]
    risk_wide = risk_wide.reset_index()
    if decision_bundle.duplicated(cluster_key + ["arm"]).any():
        raise V015AnalysisError("Decision coordinates are not unique")
    decisions = decision_bundle.pivot(
        index=cluster_key,
        columns="arm",
        values=[
            "hard_eligible",
            "issued",
            "issuance_rank",
            "raw_risk_score",
            "canonical_predictor_content_sha256",
        ],
    )
    for arm in ("prefix_only", "visible_stress"):
        for field in (
            "hard_eligible",
            "issued",
            "issuance_rank",
            "raw_risk_score",
            "canonical_predictor_content_sha256",
        ):
            if (field, arm) not in decisions.columns:
                raise V015AnalysisError(f"Decision field absent: {field}/{arm}")
    decisions.columns = [
        f"{field}_{arm}" for field, arm in decisions.columns.to_flat_index()
    ]
    decisions = decisions.reset_index()

    trajectory_rows: list[dict[str, object]] = []
    expected_days: tuple[float, ...] | None = None
    for key, group in points.groupby(cluster_key, sort=True):
        ordered = group.sort_values("forecast_day", kind="stable")
        days = _finite_vector(ordered["forecast_day"], context="Forecast days")
        if len(days) != 8 or np.any(np.diff(days) <= 0):
            raise V015AnalysisError("Every trajectory must have eight ordered days")
        current_days = tuple(float(value) for value in days)
        if current_days != FORECAST_DAYS:
            raise V015AnalysisError("A trajectory does not use the frozen grid")
        if expected_days is None:
            expected_days = current_days
        elif current_days != expected_days:
            raise V015AnalysisError("Forecast grids differ across clusters")
        for column in ("truth_family", "canonical_prefix_content_sha256"):
            if ordered[column].nunique(dropna=False) != 1:
                raise V015AnalysisError(f"Cluster metadata changed: {column}")
        center_error = ordered["center_absolute_error_pp"].to_numpy(float)
        sqrt_error = ordered["sqrt_absolute_error_pp"].to_numpy(float)
        bounded_error = ordered["bounded_power_absolute_error_pp"].to_numpy(float)
        intervals_missing = bool(
            ordered[list(interval_columns)].isna().to_numpy().all()
        )
        trajectory_rows.append(
            {
                "protocol_id": key[0],
                "partition": key[1],
                "cluster_id": key[2],
                "truth_family": str(ordered["truth_family"].iloc[0]),
                "canonical_prefix_content_sha256": str(
                    ordered["canonical_prefix_content_sha256"].iloc[0]
                ),
                "center_endpoint_absolute_error_pp": float(center_error[-1]),
                "center_trajectory_iae_pp": float(
                    np.trapezoid(center_error, days) / (days[-1] - days[0])
                ),
                "sqrt_trajectory_iae_pp": float(
                    np.trapezoid(sqrt_error, days) / (days[-1] - days[0])
                ),
                "bounded_power_trajectory_iae_pp": float(
                    np.trapezoid(bounded_error, days) / (days[-1] - days[0])
                ),
                "catastrophic": bool(center_error[-1] >= 5.0),
                "simultaneous_interval_covered": (
                    pd.NA
                    if intervals_missing
                    else bool(ordered["interval_covers_truth"].all())
                ),
                "max_interval_width_pp": (
                    math.nan
                    if intervals_missing
                    else float(ordered["interval_width_pp"].max())
                ),
                "base_simultaneous_interval_covered": (
                    pd.NA
                    if intervals_missing
                    else bool(ordered["base_interval_covers_truth"].all())
                ),
                "base_max_interval_width_pp": (
                    math.nan
                    if intervals_missing
                    else float(ordered["base_interval_width_pp"].max())
                ),
            }
        )
    trajectories = pd.DataFrame(trajectory_rows)
    trajectories = trajectories.merge(
        risk_wide, on=cluster_key, how="left", validate="one_to_one"
    ).merge(decisions, on=cluster_key, how="left", validate="one_to_one")
    if trajectories.isna().any().any():
        nullable_risk_columns = {
            *(f"risk_{score_id}" for score_id in RISK_SCORE_IDS),
            *(f"calibrated_risk_{score_id}" for score_id in RISK_SCORE_IDS),
            "raw_risk_score_prefix_only",
            "raw_risk_score_visible_stress",
            "issuance_rank_prefix_only",
            "issuance_rank_visible_stress",
            "simultaneous_interval_covered",
            "max_interval_width_pp",
            "base_simultaneous_interval_covered",
            "base_max_interval_width_pp",
        }
        bad = [
            column
            for column in trajectories.columns
            if column not in nullable_risk_columns and trajectories[column].isna().any()
        ]
        if bad:
            raise V015AnalysisError(f"Trajectory metadata missing: {bad}")

    verified_risk_hashes: dict[str, list[str]] = {}
    for score_id in RISK_SCORE_IDS:
        verified_risk_hashes[score_id] = [
            _sha256_text(value, context=f"risk hash/{score_id}")
            for value in trajectories[f"risk_hash_{score_id}"]
        ]

    for arm in ("prefix_only", "visible_stress"):
        eligible = trajectories[f"hard_eligible_{arm}"].map(
            lambda value: _strict_bool(value, context=f"eligible/{arm}")
        )
        issued = trajectories[f"issued_{arm}"].map(
            lambda value: _strict_bool(value, context=f"issued/{arm}")
        )
        trajectories[f"hard_eligible_{arm}"] = eligible
        trajectories[f"issued_{arm}"] = issued
        if (issued & ~eligible).any():
            raise V015AnalysisError("An ineligible trajectory was issued")

        rank_column = f"issuance_rank_{arm}"
        rank_present = trajectories[rank_column].notna()
        if ((~eligible) & rank_present).any():
            raise V015AnalysisError(
                f"An ineligible trajectory has an issuance rank for {arm}"
            )
        if (issued & ~rank_present).any():
            raise V015AnalysisError(
                f"An issued trajectory lacks an issuance rank for {arm}"
            )
        if rank_present.any():
            ranks = _finite_vector(
                trajectories.loc[rank_present, rank_column],
                context=f"issuance rank/{arm}",
            )
            if np.any(ranks < 1.0) or np.any(ranks != np.floor(ranks)):
                raise V015AnalysisError(
                    f"Issuance ranks must be positive integers for {arm}"
                )

        risk_values = _finite_vector(
            trajectories.loc[eligible, f"risk_{arm}"],
            context=f"risk/{arm}",
        )
        decision_values = _finite_vector(
            trajectories.loc[eligible, f"raw_risk_score_{arm}"],
            context=f"decision risk/{arm}",
        )
        if not np.array_equal(risk_values, decision_values):
            raise V015AnalysisError(
                f"Decision risk does not match risk bundle for {arm}"
            )
        calibrated = _finite_vector(
            trajectories.loc[eligible, f"calibrated_risk_{arm}"],
            context=f"calibrated risk/{arm}",
        )
        if np.any((calibrated < 0.0) | (calibrated > 1.0)):
            raise V015AnalysisError(f"Calibrated risk must be a probability for {arm}")
        decision_hashes = [
            _sha256_text(value, context=f"decision hash/{arm}")
            for value in trajectories[f"canonical_predictor_content_sha256_{arm}"]
        ]
        if verified_risk_hashes[arm] != decision_hashes:
            raise V015AnalysisError(
                f"Decision content hash does not match risk bundle for {arm}"
            )

    if not trajectories["hard_eligible_prefix_only"].equals(
        trajectories["hard_eligible_visible_stress"]
    ):
        raise V015AnalysisError("Primary arms do not share one eligibility pool")
    common_eligible = trajectories["hard_eligible_visible_stress"]
    for score_id in RISK_SCORE_IDS:
        _finite_vector(
            trajectories.loc[common_eligible, f"risk_{score_id}"],
            context=f"risk/{score_id}",
        )

    prediction_hashes = [
        _sha256_text(value, context="prediction prefix content hash")
        for value in trajectories["canonical_prefix_content_sha256"]
    ]
    if prediction_hashes != verified_risk_hashes["prefix_only"]:
        raise V015AnalysisError(
            "Prediction prefix hash does not match the Arm-A risk content hash"
        )
    trajectories = trajectories.drop(
        columns=[f"calibrated_risk_{score_id}" for score_id in RISK_SCORE_IDS]
    )
    return points.sort_values(point_key).reset_index(drop=True), trajectories


def risk_reduction(
    trajectories: pd.DataFrame,
    *,
    issued_column: str,
    eligibility_column: str = "hard_eligible_visible_stress",
    expected_issue_count: int | None = None,
) -> RiskReduction:
    _require_columns(
        trajectories,
        {issued_column, eligibility_column, "catastrophic"},
        context="Risk-reduction table",
    )
    eligible = _strict_bool_series(
        trajectories[eligibility_column], context=eligibility_column
    )
    issued = _strict_bool_series(trajectories[issued_column], context=issued_column)
    if (issued & ~eligible).any():
        raise V015AnalysisError("Issued set is not a subset of eligible pool")
    issue_count = int(issued.sum())
    if expected_issue_count is not None and issue_count != expected_issue_count:
        raise V015InconclusiveError(
            f"Expected {expected_issue_count} issued rows, observed {issue_count}"
        )
    if issue_count < 1 or not eligible.any():
        raise V015InconclusiveError(
            "Risk reduction needs nonzero issued and eligible counts"
        )
    catastrophes = _strict_bool_series(
        trajectories["catastrophic"], context="catastrophic"
    )
    issued_rate = float(catastrophes[issued].mean())
    random_rate = float(catastrophes[eligible].mean())
    if random_rate <= 0.0 or not math.isfinite(random_rate):
        raise V015InconclusiveError(
            "Eligible-pool catastrophic prevalence is zero or nonfinite"
        )
    reduction = 1.0 - issued_rate / random_rate
    return RiskReduction(issue_count, issued_rate, random_rate, reduction)


def risk_reduction_against_random_rankings(
    trajectories: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    issued_column: str,
    eligibility_column: str = "hard_eligible_visible_stress",
    expected_issue_count: int | None = None,
) -> RiskReduction:
    """Use the frozen mean same-count random-ranking risk for the point estimand."""
    _require_columns(
        random_rankings,
        {"issued_count", "issued_catastrophic_rate"},
        context="Random-ranking metrics",
    )
    if len(random_rankings) != RANDOM_RANKING_COUNT:
        raise V015AnalysisError("All 10000 random rankings are required")
    counts = pd.to_numeric(random_rankings["issued_count"], errors="coerce").to_numpy(
        float
    )
    rates = _finite_vector(
        random_rankings["issued_catastrophic_rate"],
        context="Random-ranking catastrophic rates",
    )
    if not np.all(counts == counts[0]) or np.any((rates < 0.0) | (rates > 1.0)):
        raise V015AnalysisError("Random-ranking metrics are inconsistent")
    frozen_count = int(counts[0])
    if expected_issue_count is not None and frozen_count != expected_issue_count:
        raise V015AnalysisError("Random rankings use the wrong issuance count")
    point = risk_reduction(
        trajectories,
        issued_column=issued_column,
        eligibility_column=eligibility_column,
        expected_issue_count=frozen_count,
    )
    random_mean = float(np.mean(rates))
    if random_mean <= 0.0:
        raise V015InconclusiveError("Mean random-ranking risk is zero")
    return RiskReduction(
        issued_count=point.issued_count,
        issued_catastrophic_rate=point.issued_catastrophic_rate,
        random_expected_catastrophic_rate=random_mean,
        relative_risk_reduction=(1.0 - point.issued_catastrophic_rate / random_mean),
    )


def rank_policy(
    frame: pd.DataFrame,
    *,
    protocol_id: str,
    arm: str,
    score_column: str,
    predictor_hash_column: str,
    issue_count: int,
    eligibility_column: str = "hard_eligible_visible_stress",
) -> pd.Series:
    """Return the frozen lowest-danger issuance mask on the common pool."""
    _require_columns(
        frame,
        {score_column, predictor_hash_column, eligibility_column},
        context=f"Ranking/{arm}",
    )
    if issue_count < 1:
        raise V015InconclusiveError("Ranking issuance count is zero")
    eligible = _strict_bool_series(
        frame[eligibility_column], context=eligibility_column
    )
    if int(eligible.sum()) < issue_count:
        raise V015InconclusiveError("Ranking has too few eligible rows")
    working = frame.loc[eligible, [score_column, predictor_hash_column]].copy()
    working["_validated_score"] = _finite_vector(
        working[score_column], context=f"Risk score/{arm}"
    )
    verified_hashes = [
        _sha256_text(value, context=f"Predictor content/{arm}")
        for value in working[predictor_hash_column]
    ]
    if len(set(verified_hashes)) != len(verified_hashes):
        raise V015AnalysisError(f"Ordinary predictor content is duplicated for {arm}")
    working["_tie_hash"] = [
        _arm_tie_digest(protocol_id, arm, value) for value in verified_hashes
    ]
    selected_index = working.sort_values(
        ["_validated_score", "_tie_hash"], kind="stable"
    ).index[:issue_count]
    result = pd.Series(False, index=frame.index, dtype=bool)
    result.loc[selected_index] = True
    return result


def _ranked_policy_risk(
    frame: pd.DataFrame,
    *,
    protocol_id: str,
    score_id: str,
    issue_count: int,
) -> RiskReduction:
    if issue_count < 1:
        raise V015InconclusiveError("Ranked policy issuance count is zero")
    required = {
        "catastrophic",
        "hard_eligible_visible_stress",
        f"risk_{score_id}",
        f"risk_hash_{score_id}",
    }
    _require_columns(frame, required, context=f"Ranked policy/{score_id}")
    issued = rank_policy(
        frame,
        protocol_id=protocol_id,
        arm=score_id,
        score_column=f"risk_{score_id}",
        predictor_hash_column=f"risk_hash_{score_id}",
        issue_count=issue_count,
    )
    eligible_mask = _strict_bool_series(
        frame["hard_eligible_visible_stress"],
        context="hard_eligible_visible_stress",
    )
    eligible = frame.loc[eligible_mask]
    catastrophe = _strict_bool_series(eligible["catastrophic"], context="catastrophic")
    random_rate = float(catastrophe.mean())
    if random_rate <= 0.0:
        raise V015InconclusiveError(
            "Subset eligible-pool catastrophic prevalence is zero"
        )
    issued_rate = float(
        _strict_bool_series(
            frame.loc[issued, "catastrophic"],
            context="issued catastrophic",
        ).mean()
    )
    return RiskReduction(
        issued_count=issue_count,
        issued_catastrophic_rate=issued_rate,
        random_expected_catastrophic_rate=random_rate,
        relative_risk_reduction=1.0 - issued_rate / random_rate,
    )


def subset_risk_summary(
    frame: pd.DataFrame,
    *,
    subset_id: str,
    protocol_id: str,
    issue_count: int | None = None,
) -> SubsetRiskSummary:
    """Independently rerank both arms on one frozen subset."""
    _require_columns(
        frame,
        {
            "catastrophic",
            "hard_eligible_visible_stress",
            "risk_prefix_only",
            "risk_visible_stress",
            "risk_hash_prefix_only",
            "risk_hash_visible_stress",
        },
        context=f"Subset/{subset_id}",
    )
    if frame.empty:
        raise V015InconclusiveError(f"Subset is empty: {subset_id}")
    eligible = _strict_bool_series(
        frame["hard_eligible_visible_stress"],
        context="hard_eligible_visible_stress",
    )
    eligible_count = int(eligible.sum())
    frozen_issue_count = (
        eligible_count // 2 if issue_count is None else int(issue_count)
    )
    if frozen_issue_count < 1:
        raise V015InconclusiveError("Subset issuance count is zero")
    catastrophic_count = int(
        _strict_bool_series(
            frame.loc[eligible, "catastrophic"],
            context="subset catastrophic",
        ).sum()
    )
    prefix = _ranked_policy_risk(
        frame,
        protocol_id=protocol_id,
        score_id="prefix_only",
        issue_count=frozen_issue_count,
    )
    visible = _ranked_policy_risk(
        frame,
        protocol_id=protocol_id,
        score_id="visible_stress",
        issue_count=frozen_issue_count,
    )
    return SubsetRiskSummary(
        subset_id=subset_id,
        source_count=len(frame),
        eligible_count=eligible_count,
        catastrophic_count=catastrophic_count,
        issued_count=frozen_issue_count,
        prefix_only=prefix,
        visible_stress=visible,
        visible_minus_prefix_increment=(
            visible.relative_risk_reduction - prefix.relative_risk_reduction
        ),
    )


def subset_risk_record(summary: SubsetRiskSummary) -> dict[str, object]:
    return {
        "subset_id": summary.subset_id,
        "source_count": summary.source_count,
        "eligible_count": summary.eligible_count,
        "catastrophic_count": summary.catastrophic_count,
        "issued_count": summary.issued_count,
        "prefix_only_issued_catastrophic_rate": (
            summary.prefix_only.issued_catastrophic_rate
        ),
        "prefix_only_risk_reduction": (summary.prefix_only.relative_risk_reduction),
        "visible_stress_issued_catastrophic_rate": (
            summary.visible_stress.issued_catastrophic_rate
        ),
        "visible_stress_risk_reduction": (
            summary.visible_stress.relative_risk_reduction
        ),
        "visible_minus_prefix_increment": (summary.visible_minus_prefix_increment),
    }


def _gate(
    gate_id: str,
    *,
    passed: bool,
    estimate: float | bool,
    threshold: str,
) -> GateEvaluation:
    return GateEvaluation(
        gate_id=gate_id,
        state="pass" if passed else "fail",
        estimate=estimate,
        threshold=threshold,
    )


def _inconclusive_gate(
    gate_id: str,
    *,
    threshold: str,
    reasons: Sequence[str],
) -> GateEvaluation:
    return GateEvaluation(
        gate_id=gate_id,
        state="inconclusive",
        estimate=None,
        threshold=threshold,
        reasons=tuple(str(reason) for reason in reasons),
    )


def _availability_counts(
    frame: pd.DataFrame,
) -> tuple[int, int]:
    _require_columns(
        frame,
        {"hard_eligible_visible_stress", "catastrophic"},
        context="Subset availability",
    )
    eligible = _strict_bool_series(
        frame["hard_eligible_visible_stress"],
        context="subset eligibility",
    )
    catastrophes = _strict_bool_series(
        frame.loc[eligible, "catastrophic"],
        context="subset catastrophic",
    )
    return int(eligible.sum()), int(catastrophes.sum())


def common_pool_availability_reasons(
    trajectories: pd.DataFrame,
) -> tuple[str, ...]:
    """Return every frozen ordinary-cohort minimum-count shortfall."""

    _require_columns(
        trajectories,
        {"partition", "hard_eligible_visible_stress", "catastrophic"},
        context="Common-pool availability",
    )
    specifications = (
        (
            "test",
            1900,
            TEST_MINIMUM_COMMON_POOL,
            TEST_MINIMUM_CATASTROPHIC,
        ),
        (
            "audit",
            950,
            AUDIT_MINIMUM_COMMON_POOL,
            AUDIT_MINIMUM_CATASTROPHIC,
        ),
    )
    reasons: list[str] = []
    for partition, expected, eligible_minimum, catastrophic_minimum in specifications:
        subset = trajectories.loc[trajectories["partition"].eq(partition)]
        eligible_count, catastrophic_count = _availability_counts(subset)
        if len(subset) != expected:
            reasons.append(
                f"{partition}_source_count={len(subset)} expected={expected}"
            )
        if eligible_count < eligible_minimum:
            reasons.append(
                f"{partition}_common_eligible_count={eligible_count} "
                f"minimum={eligible_minimum}"
            )
        if catastrophic_count < catastrophic_minimum:
            reasons.append(
                f"{partition}_common_catastrophic_count={catastrophic_count} "
                f"minimum={catastrophic_minimum}"
            )
    return tuple(reasons)


def common_pool_gate_evaluations(
    trajectories: pd.DataFrame,
) -> tuple[GateEvaluation, GateEvaluation]:
    """Record the two frozen global common-pool count gates."""

    _require_columns(
        trajectories,
        {"partition", "hard_eligible_visible_stress", "catastrophic"},
        context="Common-pool gates",
    )
    specifications = (
        (
            "test",
            1900,
            TEST_MINIMUM_COMMON_POOL,
            TEST_MINIMUM_CATASTROPHIC,
        ),
        (
            "audit",
            950,
            AUDIT_MINIMUM_COMMON_POOL,
            AUDIT_MINIMUM_CATASTROPHIC,
        ),
    )
    evaluations: list[GateEvaluation] = []
    for partition, expected, eligible_minimum, catastrophic_minimum in specifications:
        subset = trajectories.loc[trajectories["partition"].eq(partition)]
        eligible_count, catastrophic_count = _availability_counts(subset)
        reasons: list[str] = []
        if len(subset) != expected:
            reasons.append(f"source_count={len(subset)} expected={expected}")
        if eligible_count < eligible_minimum:
            reasons.append(
                f"eligible_count={eligible_count} minimum={eligible_minimum}"
            )
        if catastrophic_count < catastrophic_minimum:
            reasons.append(
                "catastrophic_count="
                f"{catastrophic_count} minimum={catastrophic_minimum}"
            )
        gate_id = f"{partition}_common_pool_minimum_counts"
        threshold = (
            f"source_count={expected}; eligible_count>={eligible_minimum}; "
            f"catastrophic_count>={catastrophic_minimum}"
        )
        evaluations.append(
            _inconclusive_gate(
                gate_id,
                threshold=threshold,
                reasons=reasons,
            )
            if reasons
            else _gate(
                gate_id,
                passed=True,
                estimate=True,
                threshold=threshold,
            )
        )
    return evaluations[0], evaluations[1]


def evaluate_test_safety_gates(
    trajectories: pd.DataFrame,
    *,
    protocol_id: str,
) -> tuple[pd.DataFrame, tuple[GateEvaluation, ...]]:
    """Evaluate all eight family gates and the combined novel gates."""
    _require_columns(
        trajectories,
        {"partition", "truth_family"},
        context="Test safety table",
    )
    test = trajectories.loc[trajectories["partition"].eq("test")].copy()
    expected_counts = {
        **{family: 250 for family in CORE_FAMILIES},
        **{family: 200 for family in NOVEL_FAMILIES},
    }
    records: list[dict[str, object]] = []
    gates: list[GateEvaluation] = []
    for family in TEST_FAMILIES:
        subset = test.loc[test["truth_family"].eq(family)].copy()
        eligible_count, catastrophic_count = _availability_counts(subset)
        reasons: list[str] = []
        expected = expected_counts[family]
        if len(subset) != expected:
            reasons.append(f"source_count={len(subset)} expected={expected}")
        minimum_eligible = math.ceil(0.90 * expected)
        if eligible_count < minimum_eligible:
            reasons.append(
                f"eligible_count={eligible_count} minimum={minimum_eligible}"
            )
        if catastrophic_count < 30:
            reasons.append(f"catastrophic_count={catastrophic_count} minimum=30")
        gate_id = f"test_family_{family}_nonnegative_risk_reduction"
        if reasons:
            records.append(
                {
                    "subset_id": f"test_family_{family}",
                    "source_count": len(subset),
                    "eligible_count": eligible_count,
                    "catastrophic_count": catastrophic_count,
                    "issued_count": eligible_count // 2,
                    "prefix_only_risk_reduction": math.nan,
                    "visible_stress_risk_reduction": math.nan,
                    "visible_minus_prefix_increment": math.nan,
                }
            )
            gates.append(
                _inconclusive_gate(
                    gate_id,
                    threshold="visible_stress_risk_reduction >= 0",
                    reasons=reasons,
                )
            )
            continue
        summary = subset_risk_summary(
            subset,
            subset_id=f"test_family_{family}",
            protocol_id=protocol_id,
        )
        records.append(subset_risk_record(summary))
        gates.append(
            _gate(
                gate_id,
                passed=summary.visible_stress.relative_risk_reduction >= 0.0,
                estimate=summary.visible_stress.relative_risk_reduction,
                threshold=">= 0",
            )
        )

    novel = test.loc[test["truth_family"].isin(NOVEL_FAMILIES)].copy()
    novel_eligible, novel_catastrophic = _availability_counts(novel)
    novel_reasons: list[str] = []
    if len(novel) != 400:
        novel_reasons.append(f"source_count={len(novel)} expected=400")
    if novel_eligible < 360:
        novel_reasons.append(f"eligible_count={novel_eligible} minimum=360")
    if novel_catastrophic < 30:
        novel_reasons.append(f"catastrophic_count={novel_catastrophic} minimum=30")
    novel_gate_ids = (
        (
            "test_novel_positive_risk_reduction",
            "visible_stress_risk_reduction > 0",
        ),
        (
            "test_novel_nonnegative_increment",
            "visible_minus_prefix_increment >= 0",
        ),
    )
    if novel_reasons:
        records.append(
            {
                "subset_id": "test_novel_combined",
                "source_count": len(novel),
                "eligible_count": novel_eligible,
                "catastrophic_count": novel_catastrophic,
                "issued_count": novel_eligible // 2,
                "prefix_only_risk_reduction": math.nan,
                "visible_stress_risk_reduction": math.nan,
                "visible_minus_prefix_increment": math.nan,
            }
        )
        gates.extend(
            _inconclusive_gate(gate_id, threshold=threshold, reasons=novel_reasons)
            for gate_id, threshold in novel_gate_ids
        )
    else:
        novel_summary = subset_risk_summary(
            novel,
            subset_id="test_novel_combined",
            protocol_id=protocol_id,
        )
        records.append(subset_risk_record(novel_summary))
        gates.extend(
            (
                _gate(
                    "test_novel_positive_risk_reduction",
                    passed=(novel_summary.visible_stress.relative_risk_reduction > 0.0),
                    estimate=(novel_summary.visible_stress.relative_risk_reduction),
                    threshold="> 0",
                ),
                _gate(
                    "test_novel_nonnegative_increment",
                    passed=(novel_summary.visible_minus_prefix_increment >= 0.0),
                    estimate=novel_summary.visible_minus_prefix_increment,
                    threshold=">= 0",
                ),
            )
        )
    return pd.DataFrame(records), tuple(gates)


def evaluate_audit_directional_gates(
    trajectories: pd.DataFrame,
    *,
    protocol_id: str,
    issued_center_minus_baseline_iae_pp: float | None,
) -> tuple[pd.DataFrame, tuple[GateEvaluation, ...]]:
    """Evaluate the frozen untuned audit directional gates."""
    _require_columns(
        trajectories,
        {"partition", "truth_family"},
        context="Audit safety table",
    )
    audit = trajectories.loc[trajectories["partition"].eq("audit")].copy()
    definitions = (
        (
            "audit_overall",
            audit,
            950,
            903,
            30,
            475,
        ),
        (
            "audit_novel",
            audit.loc[audit["truth_family"].isin(NOVEL_FAMILIES)].copy(),
            200,
            180,
            20,
            None,
        ),
        (
            "audit_late_knee",
            audit.loc[audit["truth_family"].eq("late_knee")].copy(),
            200,
            180,
            20,
            None,
        ),
    )
    summaries: dict[str, SubsetRiskSummary] = {}
    records: list[dict[str, object]] = []
    availability: dict[str, tuple[str, ...]] = {}
    for (
        subset_id,
        subset,
        source_min,
        eligible_min,
        cat_min,
        fixed_issue,
    ) in definitions:
        eligible_count, catastrophic_count = _availability_counts(subset)
        reasons: list[str] = []
        if len(subset) != source_min:
            reasons.append(f"source_count={len(subset)} expected={source_min}")
        if eligible_count < eligible_min:
            reasons.append(f"eligible_count={eligible_count} minimum={eligible_min}")
        if catastrophic_count < cat_min:
            reasons.append(f"catastrophic_count={catastrophic_count} minimum={cat_min}")
        availability[subset_id] = tuple(reasons)
        if reasons:
            records.append(
                {
                    "subset_id": subset_id,
                    "source_count": len(subset),
                    "eligible_count": eligible_count,
                    "catastrophic_count": catastrophic_count,
                    "issued_count": (
                        eligible_count // 2 if fixed_issue is None else fixed_issue
                    ),
                    "prefix_only_risk_reduction": math.nan,
                    "visible_stress_risk_reduction": math.nan,
                    "visible_minus_prefix_increment": math.nan,
                }
            )
            continue
        summary = subset_risk_summary(
            subset,
            subset_id=subset_id,
            protocol_id=protocol_id,
            issue_count=fixed_issue,
        )
        summaries[subset_id] = summary
        records.append(subset_risk_record(summary))

    gates: list[GateEvaluation] = []
    overall_specs = (
        (
            "audit_visible_stress_positive_risk_reduction",
            "visible",
            "> 0",
        ),
        (
            "audit_visible_stress_positive_increment",
            "increment",
            "> 0",
        ),
    )
    if availability["audit_overall"]:
        gates.extend(
            _inconclusive_gate(
                gate_id,
                threshold=threshold,
                reasons=availability["audit_overall"],
            )
            for gate_id, _, threshold in overall_specs
        )
    else:
        overall = summaries["audit_overall"]
        overall_values = {
            "visible": overall.visible_stress.relative_risk_reduction,
            "increment": overall.visible_minus_prefix_increment,
        }
        gates.extend(
            _gate(
                gate_id,
                passed=overall_values[field] > 0.0,
                estimate=overall_values[field],
                threshold=threshold,
            )
            for gate_id, field, threshold in overall_specs
        )

    if issued_center_minus_baseline_iae_pp is None or not math.isfinite(
        issued_center_minus_baseline_iae_pp
    ):
        gates.append(
            _inconclusive_gate(
                "audit_issued_center_iae_noninferiority",
                threshold="<= 0.10 pp",
                reasons=("audit issued IAE comparison unavailable",),
            )
        )
    else:
        gates.append(
            _gate(
                "audit_issued_center_iae_noninferiority",
                passed=issued_center_minus_baseline_iae_pp <= 0.10,
                estimate=issued_center_minus_baseline_iae_pp,
                threshold="<= 0.10 pp",
            )
        )

    for subset_id, gate_id in (
        ("audit_novel", "audit_novel_nonnegative_risk_reduction"),
        ("audit_late_knee", "audit_late_knee_nonnegative_risk_reduction"),
    ):
        if availability[subset_id]:
            gates.append(
                _inconclusive_gate(
                    gate_id,
                    threshold=">= 0",
                    reasons=availability[subset_id],
                )
            )
        else:
            value = summaries[subset_id].visible_stress.relative_risk_reduction
            gates.append(
                _gate(
                    gate_id,
                    passed=value >= 0.0,
                    estimate=value,
                    threshold=">= 0",
                )
            )
    return pd.DataFrame(records), tuple(gates)


def select_strongest_mean_baseline(
    calibration_metrics: pd.DataFrame,
    *,
    expected_clusters: int = 900,
) -> tuple[MeanBaselineSelection, pd.DataFrame]:
    """Select the lowest complete calibration mean IAE with lexical ties."""
    required = {
        "cluster_id",
        "model_id",
        "trajectory_iae_pp",
        "finite_forecast",
    }
    _require_columns(
        calibration_metrics, required, context="Calibration baseline metrics"
    )
    baseline_ids = (
        "target_prefix_persistence",
        "target_prefix_sqrt_time",
        "target_prefix_bounded_power_law",
    )
    rows: list[dict[str, object]] = []
    for model_id in baseline_ids:
        subset = calibration_metrics.loc[calibration_metrics["model_id"].eq(model_id)]
        complete = bool(
            len(subset) == expected_clusters
            and not subset["cluster_id"].duplicated().any()
            and subset["finite_forecast"]
            .map(
                lambda value: _strict_bool(value, context=f"finite baseline/{model_id}")
            )
            .all()
        )
        mean_iae = math.nan
        if complete:
            values = _finite_vector(
                subset["trajectory_iae_pp"],
                context=f"baseline IAE/{model_id}",
            )
            if np.any(values < 0.0):
                raise V015AnalysisError("Baseline IAE cannot be negative")
            mean_iae = float(np.mean(values))
        rows.append(
            {
                "model_id": model_id,
                "cluster_count": len(subset),
                "complete": complete,
                "mean_trajectory_iae_pp": mean_iae,
            }
        )
    table = pd.DataFrame(rows)
    if not table["complete"].all():
        raise V015AnalysisError("Every calibration baseline must have 900 rows")
    ordered = table.sort_values(["mean_trajectory_iae_pp", "model_id"], kind="stable")
    winner = ordered.iloc[0]
    return (
        MeanBaselineSelection(
            selected_model_id=str(winner["model_id"]),
            calibration_cluster_count=expected_clusters,
            mean_trajectory_iae_pp=float(winner["mean_trajectory_iae_pp"]),
        ),
        table,
    )


def issued_center_minus_baseline_iae(
    trajectories: pd.DataFrame,
    *,
    issued_column: str,
    baseline_iae_column: str,
    expected_issue_count: int,
) -> float:
    _require_columns(
        trajectories,
        {
            issued_column,
            "center_trajectory_iae_pp",
            baseline_iae_column,
        },
        context="Issued IAE comparison",
    )
    issued = _strict_bool_series(trajectories[issued_column], context=issued_column)
    if int(issued.sum()) != expected_issue_count:
        raise V015AnalysisError("Issued IAE comparison has the wrong denominator")
    center = _finite_vector(
        trajectories.loc[issued, "center_trajectory_iae_pp"],
        context="Issued center IAE",
    )
    baseline = _finite_vector(
        trajectories.loc[issued, baseline_iae_column],
        context="Issued baseline IAE",
    )
    if np.any(center < 0.0) or np.any(baseline < 0.0):
        raise V015AnalysisError("Trajectory IAE cannot be negative")
    return float(np.mean(center) - np.mean(baseline))


def _random_digest(index: int, content_hash: str) -> str:
    material = f"{RANDOM_ROOT}|{index}|{content_hash}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def deterministic_random_rankings(
    trajectories: pd.DataFrame,
    *,
    issue_count: int,
    rankings: int = RANDOM_RANKING_COUNT,
) -> pd.DataFrame:
    _require_columns(
        trajectories,
        {
            "canonical_prefix_content_sha256",
            "hard_eligible_visible_stress",
            "catastrophic",
        },
        context="Random-ranking table",
    )
    eligible = trajectories.loc[
        _strict_bool_series(
            trajectories["hard_eligible_visible_stress"],
            context="hard_eligible_visible_stress",
        )
    ].copy()
    if len(eligible) < issue_count:
        raise V015InconclusiveError("Random ranking has too few eligible rows")
    if eligible["canonical_prefix_content_sha256"].duplicated().any():
        raise V015AnalysisError("Ordinary random-policy content must be unique")
    hashes = np.asarray(
        [
            _sha256_text(value, context="random-policy content hash")
            for value in eligible["canonical_prefix_content_sha256"]
        ],
        dtype="U64",
    )
    catastrophe = _strict_bool_series(
        eligible["catastrophic"], context="catastrophic"
    ).to_numpy()
    expected = float(catastrophe.mean())
    if expected <= 0.0:
        raise V015InconclusiveError(
            "Random-ranking eligible-pool catastrophic prevalence is zero"
        )
    records: list[dict[str, object]] = []
    for index in range(rankings):
        order = np.argsort(
            np.asarray(
                [_random_digest(index, value) for value in hashes],
                dtype="U64",
            ),
            kind="stable",
        )
        rate = float(catastrophe[order[:issue_count]].mean())
        records.append(
            {
                "ranking_index": index,
                "issued_count": issue_count,
                "issued_catastrophic_rate": rate,
                "analytic_random_expected_rate": expected,
                "relative_risk_reduction": 1.0 - rate / expected,
            }
        )
    return pd.DataFrame(records)


def evaluate_policy_rankings(
    trajectories: pd.DataFrame,
    random_rankings: pd.DataFrame,
    *,
    protocol_id: str,
    issue_count: int,
) -> pd.DataFrame:
    """Report every frozen head/comparator against one random baseline."""
    _require_columns(
        trajectories,
        {
            "catastrophic",
            "hard_eligible_visible_stress",
            *(f"risk_{score_id}" for score_id in RISK_SCORE_IDS),
            *(f"risk_hash_{score_id}" for score_id in RISK_SCORE_IDS),
        },
        context="Policy comparison",
    )
    _require_columns(
        random_rankings,
        {"issued_count", "issued_catastrophic_rate"},
        context="Policy random baseline",
    )
    if len(random_rankings) != RANDOM_RANKING_COUNT:
        raise V015AnalysisError("Policy comparison requires 10000 random rankings")
    counts = _finite_vector(
        random_rankings["issued_count"], context="Random issuance counts"
    )
    if not np.all(counts == issue_count):
        raise V015AnalysisError("Random rankings use a different issuance count")
    random_rates = _finite_vector(
        random_rankings["issued_catastrophic_rate"],
        context="Random issued risks",
    )
    random_mean = float(np.mean(random_rates))
    if random_mean <= 0.0:
        raise V015AnalysisError("Policy random baseline is zero")
    eligible = _strict_bool_series(
        trajectories["hard_eligible_visible_stress"],
        context="policy eligibility",
    )
    catastrophic = _strict_bool_series(
        trajectories["catastrophic"], context="policy catastrophic"
    )
    rows: list[dict[str, object]] = []
    for score_id in RISK_SCORE_IDS:
        issued = rank_policy(
            trajectories,
            protocol_id=protocol_id,
            arm=score_id,
            score_column=f"risk_{score_id}",
            predictor_hash_column=f"risk_hash_{score_id}",
            issue_count=issue_count,
        )
        issued_rate = float(catastrophic[issued].mean())
        rows.append(
            {
                "score_id": score_id,
                "source_count": len(trajectories),
                "eligible_count": int(eligible.sum()),
                "issued_count": int(issued.sum()),
                "source_coverage": int(issued.sum()) / len(trajectories),
                "eligible_coverage": int(issued.sum()) / int(eligible.sum()),
                "issued_catastrophic_rate": issued_rate,
                "mean_random_issued_catastrophic_rate": random_mean,
                "relative_risk_reduction": 1.0 - issued_rate / random_mean,
            }
        )
    return pd.DataFrame(rows)


def coverage_summary(frame: pd.DataFrame) -> CoverageSummary:
    _require_columns(
        frame,
        {"simultaneous_interval_covered", "max_interval_width_pp"},
        context="Coverage table",
    )
    if frame.empty:
        raise V015AnalysisError("Coverage table is empty")
    covered = _strict_bool_series(
        frame["simultaneous_interval_covered"],
        context="simultaneous_interval_covered",
    )
    widths = pd.to_numeric(frame["max_interval_width_pp"], errors="coerce").to_numpy(
        float
    )
    if not np.isfinite(widths).all() or np.any(widths < 0):
        raise V015AnalysisError("Coverage widths must be finite and nonnegative")
    successes = int(covered.sum())
    trials = len(frame)
    return CoverageSummary(
        n=trials,
        covered=successes,
        coverage=successes / trials,
        one_sided_95_lower=clopper_pearson_lower(successes, trials),
        median_max_width_pp=float(np.quantile(widths, 0.5, method="linear")),
        percentile_95_max_width_pp=float(np.quantile(widths, 0.95, method="linear")),
    )


def core_test_coverage_summary(
    trajectories: pd.DataFrame,
    *,
    calibrated: bool = True,
) -> CoverageSummary:
    _require_columns(
        trajectories,
        {
            "partition",
            "truth_family",
            (
                "simultaneous_interval_covered"
                if calibrated
                else "base_simultaneous_interval_covered"
            ),
            ("max_interval_width_pp" if calibrated else "base_max_interval_width_pp"),
        },
        context="Core-test coverage",
    )
    core = trajectories.loc[
        trajectories["partition"].eq("test")
        & trajectories["truth_family"].isin(CORE_FAMILIES)
    ].copy()
    counts = core.groupby("truth_family", sort=False).size().to_dict()
    if len(core) != 1500 or counts != {family: 250 for family in CORE_FAMILIES}:
        raise V015AnalysisError(
            "Core-test coverage requires exactly 250 rows from each core family"
        )
    if not calibrated:
        core = core.rename(
            columns={
                "base_simultaneous_interval_covered": ("simultaneous_interval_covered"),
                "base_max_interval_width_pp": "max_interval_width_pp",
            }
        )
    return coverage_summary(core)


def _validated_pair_members(
    pair_mapping: pd.DataFrame,
    *,
    context: str,
) -> list[tuple[str, str, str]]:
    required = {"pair_id", "left_cluster_id", "right_cluster_id"}
    _require_columns(pair_mapping, required, context=context)
    if len(pair_mapping) != 250 or pair_mapping["pair_id"].duplicated().any():
        raise V015AnalysisError(f"{context} must contain 250 unique pairs")
    members: list[str] = []
    pairs: list[tuple[str, str, str]] = []
    for row in pair_mapping.itertuples(index=False):
        pair_id = str(row.pair_id)
        left = str(row.left_cluster_id)
        right = str(row.right_cluster_id)
        if left == right:
            raise V015AnalysisError(f"{context} contains a self-pair")
        members.extend((left, right))
        pairs.append((pair_id, left, right))
    if len(set(members)) != 500:
        raise V015AnalysisError(f"{context} reuses a pair member")
    return pairs


def _pair_mapping_row(
    pair_mapping: pd.DataFrame,
    pair_id: str,
    *,
    context: str,
) -> pd.Series:
    rows = pair_mapping.loc[pair_mapping["pair_id"].eq(pair_id)]
    if len(rows) != 1:
        raise V015AnalysisError(f"{context} pair ID is not unique")
    return rows.iloc[0]


def _pair_prefix_rows(
    prefix_pack: pd.DataFrame,
    cluster_id: str,
    protocol: ValidatedV015Protocol,
    *,
    context: str,
) -> pd.DataFrame:
    _require_columns(
        prefix_pack,
        {"cluster_id", "prefix_day", "observed_retention_pct"},
        context=context,
    )
    rows = prefix_pack.loc[prefix_pack["cluster_id"].eq(cluster_id)].sort_values(
        "prefix_day", kind="stable"
    )
    if len(rows) != len(protocol.prefix_days):
        raise V015AnalysisError(f"{context} has an incomplete prefix")
    _require_bitwise_equal_numeric(
        rows["prefix_day"],
        pd.Series(protocol.prefix_days),
        context=f"{context} prefix coordinates",
    )
    return rows


def _pair_operating_row(
    operating_pack: pd.DataFrame,
    cluster_id: str,
    *,
    context: str,
) -> pd.Series:
    required = {"cluster_id", *REAL_OPERATING_FIELDS, *PLACEBO_FIELDS}
    _require_columns(operating_pack, required, context=context)
    rows = operating_pack.loc[operating_pack["cluster_id"].eq(cluster_id)]
    if len(rows) != 1:
        raise V015AnalysisError(f"{context} must have one operating row")
    for field in (*REAL_OPERATING_FIELDS, *PLACEBO_FIELDS):
        _float64_bytes(rows.iloc[0][field], context=f"{context}/{field}")
    return rows.iloc[0]


def _operating_covariates_from_row(row: pd.Series) -> OperatingCovariates:
    values = tuple(float(row[field]) for field in REAL_OPERATING_FIELDS)
    placebo = tuple(float(row[field]) for field in PLACEBO_FIELDS)
    return OperatingCovariates(*values, placebo_controls=placebo)


def _require_closed_support(
    value: object,
    lower: float,
    upper: float,
    *,
    context: str,
) -> float:
    numeric = float(value)
    _float64_bytes(numeric, context=context)
    if numeric < lower or numeric > upper:
        raise V015AnalysisError(
            f"{context}={numeric} is outside frozen support [{lower}, {upper}]"
        )
    return numeric


def _validate_operating_support(
    row: pd.Series,
    protocol: ValidatedV015Protocol,
    *,
    context: str,
) -> None:
    support = protocol.support_map()
    for field in REAL_OPERATING_FIELDS:
        lower, upper = support[field]
        _require_closed_support(
            row[field],
            lower,
            upper,
            context=f"{context}/{field}",
        )
    for field in PLACEBO_FIELDS:
        _require_closed_support(
            row[field],
            -1.0,
            1.0,
            context=f"{context}/{field}",
        )


def _validate_family_parameter_support(
    protocol: ValidatedV015Protocol,
    family_id: str,
    parameters: Mapping[str, float],
    *,
    context: str,
) -> None:
    try:
        definitions = protocol.family_map()[family_id].parameters
    except KeyError as exc:
        raise V015AnalysisError(f"{context} has an unknown truth family") from exc
    expected = {definition.parameter_name for definition in definitions}
    if set(parameters) != expected:
        raise V015AnalysisError(f"{context} truth parameters changed")
    for definition in definitions:
        _require_closed_support(
            parameters[definition.parameter_name],
            definition.minimum,
            definition.maximum,
            context=f"{context}/{definition.parameter_name}",
        )


def _validate_matched_gamma(value: float, *, context: str) -> None:
    _require_closed_support(value, 0.05, 0.25, context=f"{context}/gamma")


def _decode_truth_parameters(value: object, *, context: str) -> dict[str, float]:
    if not isinstance(value, str):
        raise V015AnalysisError(f"{context} parameters must be canonical JSON text")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise V015AnalysisError(f"{context} parameters are invalid JSON") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise V015AnalysisError(f"{context} parameters must be a nonempty object")
    result: dict[str, float] = {}
    for key, item in decoded.items():
        if not isinstance(key, str) or isinstance(item, bool):
            raise V015AnalysisError(f"{context} parameters have invalid types")
        result[key] = float(item)
        _float64_bytes(result[key], context=f"{context}/{key}")
    return result


def _pair_truth_rows(
    truth_pack: pd.DataFrame,
    cluster_id: str,
    protocol: ValidatedV015Protocol,
    *,
    context: str,
) -> tuple[pd.DataFrame, str, dict[str, float], float]:
    required = {
        "cluster_id",
        "truth_family",
        "truth_parameters_json",
        "gamma",
        "forecast_day",
        "latent_retention_pct",
        "noisy_retention_pct",
    }
    _require_columns(truth_pack, required, context=context)
    rows = truth_pack.loc[truth_pack["cluster_id"].eq(cluster_id)].sort_values(
        "forecast_day", kind="stable"
    )
    if len(rows) != len(protocol.forecast_days):
        raise V015AnalysisError(f"{context} has an incomplete future")
    _require_bitwise_equal_numeric(
        rows["forecast_day"],
        pd.Series(protocol.forecast_days),
        context=f"{context} forecast coordinates",
    )
    for column in ("truth_family", "truth_parameters_json"):
        if rows[column].nunique(dropna=False) != 1:
            raise V015AnalysisError(f"{context} {column} changes across horizons")
    gamma_bits = {
        _float64_bytes(value, context=f"{context}/gamma")
        for value in rows["gamma"].tolist()
    }
    if len(gamma_bits) != 1:
        raise V015AnalysisError(f"{context} gamma changes across horizons")
    _finite_vector(rows["latent_retention_pct"], context=f"{context} latent truth")
    _finite_vector(rows["noisy_retention_pct"], context=f"{context} noisy truth")
    parameters = _decode_truth_parameters(
        rows["truth_parameters_json"].iloc[0],
        context=context,
    )
    return (
        rows,
        str(rows["truth_family"].iloc[0]),
        parameters,
        float(rows["gamma"].iloc[0]),
    )


def _require_shared_measurement_noise(
    left_truth: pd.DataFrame,
    right_truth: pd.DataFrame,
    *,
    context: str,
) -> None:
    left_error = left_truth["noisy_retention_pct"].to_numpy(float) - left_truth[
        "latent_retention_pct"
    ].to_numpy(float)
    right_error = right_truth["noisy_retention_pct"].to_numpy(float) - right_truth[
        "latent_retention_pct"
    ].to_numpy(float)
    if not np.allclose(left_error, right_error, rtol=0.0, atol=2e-14):
        raise V015AnalysisError(f"{context} does not share measurement noise")


def _require_zero_mapping_prefix(row: pd.Series, *, context: str) -> None:
    positive_zero = struct.pack(">d", 0.0)
    for field in (
        "latent_prefix_rmse_pp",
        "latent_prefix_max_abs_difference_pp",
    ):
        if _float64_bytes(row[field], context=f"{context}/{field}") != positive_zero:
            raise V015AnalysisError(
                f"{context} does not declare an exactly equal latent prefix"
            )


def validate_intrinsic_pair_construction(
    prefix_pack: pd.DataFrame,
    operating_pack: pd.DataFrame,
    truth_pack: pd.DataFrame,
    pair_mapping: pd.DataFrame,
    protocol: ValidatedV015Protocol,
) -> None:
    """Reconstruct every intrinsic pair before it enters the fixed denominator."""

    pairs = _validated_pair_members(pair_mapping, context="Intrinsic pair mapping")
    required_mapping = {
        "pair_id",
        "construction_family",
        "left_side_code",
        "right_side_code",
        "latent_prefix_rmse_pp",
        "latent_prefix_max_abs_difference_pp",
        "truth_separation_25y_pp",
    }
    _require_columns(
        pair_mapping, required_mapping, context="Intrinsic pair construction"
    )
    mechanism_counts = pair_mapping["construction_family"].value_counts().to_dict()
    if mechanism_counts != {
        "piecewise_linear_knee": 125,
        "compact_smoothstep": 125,
    }:
        raise V015AnalysisError(
            "Intrinsic construction requires exactly 125 pairs per mechanism"
        )
    for pair_id, left_id, right_id in pairs:
        context = f"Intrinsic pair {pair_id}"
        mapping = _pair_mapping_row(pair_mapping, pair_id, context=context)
        mechanism = str(mapping["construction_family"])
        if mechanism not in {"piecewise_linear_knee", "compact_smoothstep"}:
            raise V015AnalysisError(f"{context} has an invalid mechanism")
        if (
            mapping["left_side_code"] != "smooth_reference"
            or mapping["right_side_code"] != mechanism
        ):
            raise V015AnalysisError(f"{context} side labels changed")
        _require_zero_mapping_prefix(mapping, context=context)

        left_prefix = _pair_prefix_rows(
            prefix_pack, left_id, protocol, context=f"{context}/left"
        )
        right_prefix = _pair_prefix_rows(
            prefix_pack, right_id, protocol, context=f"{context}/right"
        )
        _require_bitwise_equal_numeric(
            left_prefix["observed_retention_pct"],
            right_prefix["observed_retention_pct"],
            context=f"{context} observed prefix",
        )
        left_operating = _pair_operating_row(
            operating_pack, left_id, context=f"{context}/left"
        )
        right_operating = _pair_operating_row(
            operating_pack, right_id, context=f"{context}/right"
        )
        _validate_operating_support(
            left_operating,
            protocol,
            context=f"{context}/left",
        )
        _validate_operating_support(
            right_operating,
            protocol,
            context=f"{context}/right",
        )
        _require_canonical_rows_equal(
            left_operating.to_frame().T,
            right_operating.to_frame().T,
            columns=(*REAL_OPERATING_FIELDS, *PLACEBO_FIELDS),
            context=f"{context} operating covariates",
        )

        left_truth, left_family, left_parameters, left_gamma = _pair_truth_rows(
            truth_pack, left_id, protocol, context=f"{context}/left"
        )
        right_truth, right_family, right_parameters, right_gamma = _pair_truth_rows(
            truth_pack, right_id, protocol, context=f"{context}/right"
        )
        expected_right_family = f"intrinsic_{mechanism}"
        if (
            left_family != "intrinsic_single_power"
            or right_family != expected_right_family
        ):
            raise V015AnalysisError(f"{context} truth family labels changed")
        if set(left_parameters) != {"a", "b"}:
            raise V015AnalysisError(f"{context} base parameters changed")
        for field, bounds in {
            "a": (0.2, 0.8),
            "b": (0.35, 0.70),
        }.items():
            _require_closed_support(
                left_parameters[field],
                *bounds,
                context=f"{context}/{field}",
            )
        mechanism_parameters = {
            key: value
            for key, value in right_parameters.items()
            if key not in {"a", "b"}
        }
        mechanism_support = (
            {
                "k_pp_per_day": (0.0015, 0.0025),
                "t_knee_days": (1461.0, 2922.0),
            }
            if mechanism == "piecewise_linear_knee"
            else {
                "amplitude_pp": (6.5, 7.0),
                "t_start_days": (1095.75, 3652.5),
                "duration_days": (365.25, 1826.25),
            }
        )
        if set(mechanism_parameters) != set(mechanism_support):
            raise V015AnalysisError(f"{context} mechanism parameters changed")
        for field, bounds in mechanism_support.items():
            _require_closed_support(
                mechanism_parameters[field],
                *bounds,
                context=f"{context}/{field}",
            )
        for field in ("a", "b"):
            if field not in right_parameters or _float64_bytes(
                left_parameters[field], context=f"{context}/{field}/left"
            ) != _float64_bytes(
                right_parameters[field], context=f"{context}/{field}/right"
            ):
                raise V015AnalysisError(f"{context} does not share base parameters")
        if _float64_bytes(
            left_gamma, context=f"{context}/gamma/left"
        ) != _float64_bytes(right_gamma, context=f"{context}/gamma/right"):
            raise V015AnalysisError(f"{context} does not share gamma")
        _validate_matched_gamma(left_gamma, context=context)
        expected_left, expected_right = evaluate_intrinsic_pair_retention(
            left_parameters,
            _operating_covariates_from_row(left_operating),
            left_gamma,
            protocol.combined_days,
            mechanism=mechanism,
            mechanism_parameters=mechanism_parameters,
            time_scale_days=protocol.time_scale_days,
        )
        prefix_count = len(protocol.prefix_days)
        _require_bitwise_equal_numeric(
            expected_left[:prefix_count],
            expected_right[:prefix_count],
            context=f"{context} reconstructed latent prefix",
        )
        _require_bitwise_equal_numeric(
            left_truth["latent_retention_pct"],
            expected_left[prefix_count:],
            context=f"{context} left latent truth",
        )
        _require_bitwise_equal_numeric(
            right_truth["latent_retention_pct"],
            expected_right[prefix_count:],
            context=f"{context} right latent truth",
        )
        _require_shared_measurement_noise(left_truth, right_truth, context=context)
        for side, curve in (("left", expected_left), ("right", expected_right)):
            if (
                not np.isfinite(curve).all()
                or _float64_bytes(curve[0], context=f"{context}/{side}/day_zero")
                != _float64_bytes(100.0, context=f"{context}/{side}/expected_day_zero")
                or float(np.min(curve)) < 50.0
                or float(np.max(curve)) > 105.0
            ):
                raise V015AnalysisError(
                    f"{context} {side} curve violates matched admissibility"
                )
        separation = float(abs(expected_left[-1] - expected_right[-1]))
        if separation < 5.0:
            raise V015AnalysisError(
                f"{context} 25-year latent separation is below 5 pp"
            )
        if _float64_bytes(
            mapping["truth_separation_25y_pp"],
            context=f"{context}/declared separation",
        ) != _float64_bytes(separation, context=f"{context}/recomputed separation"):
            raise V015AnalysisError(f"{context} separation metadata is false")


def validate_stress_plan_pair_construction(
    prefix_pack: pd.DataFrame,
    operating_pack: pd.DataFrame,
    truth_pack: pd.DataFrame,
    pair_mapping: pd.DataFrame,
    protocol: ValidatedV015Protocol,
) -> None:
    """Verify that every stress-plan pair is a clean low/high plan contrast."""

    pairs = _validated_pair_members(pair_mapping, context="Stress-plan pair mapping")
    required_mapping = {
        "pair_id",
        "construction_family",
        "left_side_code",
        "right_side_code",
        "latent_prefix_rmse_pp",
        "latent_prefix_max_abs_difference_pp",
        "truth_separation_25y_pp",
    }
    _require_columns(
        pair_mapping, required_mapping, context="Stress-plan pair construction"
    )
    family_counts = pair_mapping["construction_family"].value_counts().to_dict()
    expected_family_counts = {
        family: 42 if index < 4 else 41 for index, family in enumerate(CORE_FAMILY_IDS)
    }
    if family_counts != expected_family_counts:
        raise V015AnalysisError("Stress-plan construction-family counts changed")
    support = protocol.support_map()
    for pair_id, low_id, high_id in pairs:
        context = f"Stress-plan pair {pair_id}"
        mapping = _pair_mapping_row(pair_mapping, pair_id, context=context)
        family = str(mapping["construction_family"])
        if (
            mapping["left_side_code"] != "low_plan"
            or mapping["right_side_code"] != "high_plan"
        ):
            raise V015AnalysisError(f"{context} side labels changed")
        _require_zero_mapping_prefix(mapping, context=context)
        low_prefix = _pair_prefix_rows(
            prefix_pack, low_id, protocol, context=f"{context}/low"
        )
        high_prefix = _pair_prefix_rows(
            prefix_pack, high_id, protocol, context=f"{context}/high"
        )
        _require_bitwise_equal_numeric(
            low_prefix["observed_retention_pct"],
            high_prefix["observed_retention_pct"],
            context=f"{context} observed prefix",
        )

        low_operating = _pair_operating_row(
            operating_pack, low_id, context=f"{context}/low"
        )
        high_operating = _pair_operating_row(
            operating_pack, high_id, context=f"{context}/high"
        )
        _validate_operating_support(
            low_operating,
            protocol,
            context=f"{context}/low",
        )
        _validate_operating_support(
            high_operating,
            protocol,
            context=f"{context}/high",
        )
        _require_canonical_rows_equal(
            low_operating.to_frame().T,
            high_operating.to_frame().T,
            columns=(*REAL_OPERATING_FIELDS[:4], *PLACEBO_FIELDS),
            context=f"{context} shared past/placebo",
        )
        for field in REAL_OPERATING_FIELDS[4:]:
            lower, upper = support[field]
            midpoint = (lower + upper) / 2.0
            low_value = float(low_operating[field])
            high_value = float(high_operating[field])
            if not lower <= low_value < midpoint:
                raise V015AnalysisError(
                    f"{context} low {field} is outside the lower support half"
                )
            if not midpoint <= high_value <= upper:
                raise V015AnalysisError(
                    f"{context} high {field} is outside the upper support half"
                )
            if not low_value < high_value:
                raise V015AnalysisError(
                    f"{context} planned {field} is not a strict contrast"
                )

        low_truth, low_family, low_parameters, low_gamma = _pair_truth_rows(
            truth_pack, low_id, protocol, context=f"{context}/low"
        )
        high_truth, high_family, high_parameters, high_gamma = _pair_truth_rows(
            truth_pack, high_id, protocol, context=f"{context}/high"
        )
        if low_family != family or high_family != family:
            raise V015AnalysisError(f"{context} truth family does not match mapping")
        if low_parameters != high_parameters:
            raise V015AnalysisError(f"{context} does not share base parameters")
        _validate_family_parameter_support(
            protocol,
            family,
            low_parameters,
            context=context,
        )
        if _float64_bytes(low_gamma, context=f"{context}/gamma/low") != _float64_bytes(
            high_gamma, context=f"{context}/gamma/high"
        ):
            raise V015AnalysisError(f"{context} does not share gamma")
        _validate_matched_gamma(low_gamma, context=context)
        expected_low, expected_high = evaluate_stress_plan_pair_retention(
            family,
            low_parameters,
            _operating_covariates_from_row(low_operating),
            _operating_covariates_from_row(high_operating),
            low_gamma,
            protocol.combined_days,
            time_scale_days=protocol.time_scale_days,
        )
        prefix_count = len(protocol.prefix_days)
        _require_bitwise_equal_numeric(
            expected_low[:prefix_count],
            expected_high[:prefix_count],
            context=f"{context} reconstructed latent prefix",
        )
        _require_bitwise_equal_numeric(
            low_truth["latent_retention_pct"],
            expected_low[prefix_count:],
            context=f"{context} low latent truth",
        )
        _require_bitwise_equal_numeric(
            high_truth["latent_retention_pct"],
            expected_high[prefix_count:],
            context=f"{context} high latent truth",
        )
        _require_shared_measurement_noise(low_truth, high_truth, context=context)
        if not truth_is_admissible(
            protocol,
            family,
            expected_low,
        ) or not truth_is_admissible(
            protocol,
            family,
            expected_high,
        ):
            raise V015AnalysisError(f"{context} violates ordinary truth admissibility")
        separation = float(abs(expected_low[-1] - expected_high[-1]))
        if _float64_bytes(
            mapping["truth_separation_25y_pp"],
            context=f"{context}/declared separation",
        ) != _float64_bytes(separation, context=f"{context}/recomputed separation"):
            raise V015AnalysisError(f"{context} separation metadata is false")


def _exact_pair_point_outputs(
    points: pd.DataFrame,
    left_id: str,
    right_id: str,
    *,
    context: str,
) -> None:
    required = {
        "cluster_id",
        "forecast_day",
        "center_forecast_pct",
        "sqrt_time_forecast_pct",
        "bounded_power_forecast_pct",
        "base_interval_lower_pct",
        "base_interval_upper_pct",
        "calibrated_interval_lower_pct",
        "calibrated_interval_upper_pct",
        "canonical_prefix_content_sha256",
    }
    _require_columns(points, required, context=context)
    compare_columns = (
        "forecast_day",
        "center_forecast_pct",
        "sqrt_time_forecast_pct",
        "bounded_power_forecast_pct",
        "base_interval_lower_pct",
        "base_interval_upper_pct",
        "calibrated_interval_lower_pct",
        "calibrated_interval_upper_pct",
    )
    left = points.loc[points["cluster_id"].eq(left_id)].sort_values(
        "forecast_day", kind="stable"
    )
    right = points.loc[points["cluster_id"].eq(right_id)].sort_values(
        "forecast_day", kind="stable"
    )
    if len(left) != 8 or len(right) != 8:
        raise V015AnalysisError(f"{context} has an incomplete pair trajectory")
    for column in compare_columns:
        _require_bitwise_equal_numeric(
            left[column],
            right[column],
            context=f"{context} point output/{column}",
        )
    left_hash = _sha256_text(
        left["canonical_prefix_content_sha256"].iloc[0],
        context=f"{context} left prefix hash",
    )
    right_hash = _sha256_text(
        right["canonical_prefix_content_sha256"].iloc[0],
        context=f"{context} right prefix hash",
    )
    if left_hash != right_hash:
        raise V015AnalysisError(f"{context} prefix content differs within a pair")


def validate_intrinsic_output_invariance(
    points: pd.DataFrame,
    trajectories: pd.DataFrame,
    pair_mapping: pd.DataFrame,
) -> None:
    """Fail closed if identical M0 predictor content changes any output."""
    pairs = _validated_pair_members(pair_mapping, context="Intrinsic pair mapping")
    required = {
        "cluster_id",
        *(f"risk_{score_id}" for score_id in RISK_SCORE_IDS),
        *(f"risk_hash_{score_id}" for score_id in RISK_SCORE_IDS),
        "hard_eligible_prefix_only",
        "hard_eligible_visible_stress",
        "issued_prefix_only",
        "issued_visible_stress",
        "issuance_rank_prefix_only",
        "issuance_rank_visible_stress",
    }
    _require_columns(trajectories, required, context="Intrinsic deterministic outputs")
    indexed = trajectories.set_index("cluster_id")
    if not indexed.index.is_unique:
        raise V015AnalysisError("Intrinsic outputs contain duplicate cluster IDs")
    numeric_columns = [*(f"risk_{score_id}" for score_id in RISK_SCORE_IDS)]
    hash_columns = [*(f"risk_hash_{score_id}" for score_id in RISK_SCORE_IDS)]
    exact_columns = [
        "hard_eligible_prefix_only",
        "hard_eligible_visible_stress",
        "issued_prefix_only",
        "issued_visible_stress",
        "issuance_rank_prefix_only",
        "issuance_rank_visible_stress",
    ]
    for _, left_id, right_id in pairs:
        if left_id not in indexed.index or right_id not in indexed.index:
            raise V015AnalysisError("Intrinsic pair member is absent")
        _exact_pair_point_outputs(
            points,
            left_id,
            right_id,
            context="Intrinsic invariance",
        )
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        for column in numeric_columns:
            if not _canonical_scalar_equal(left[column], right[column]):
                raise V015AnalysisError(
                    f"Intrinsic deterministic output differs bitwise in {column}"
                )
        for column in hash_columns:
            if _sha256_text(
                left[column], context=f"Intrinsic left {column}"
            ) != _sha256_text(right[column], context=f"Intrinsic right {column}"):
                raise V015AnalysisError(
                    f"Intrinsic deterministic content differs in {column}"
                )
        for column in exact_columns:
            if not _canonical_scalar_equal(left[column], right[column]):
                raise V015AnalysisError(
                    f"Intrinsic deterministic output differs in {column}"
                )


def validate_stress_plan_arm_a_invariance(
    points: pd.DataFrame,
    trajectories: pd.DataFrame,
    pair_mapping: pd.DataFrame,
) -> None:
    """Fail closed if plan-only changes leak into any Arm-A output."""
    pairs = _validated_pair_members(pair_mapping, context="Stress-plan pair mapping")
    arm_a_score_ids = (
        "prefix_only",
        "placebo_8",
        "strongest_single_feature",
        "prefix_rmse_only",
        "v1_max_envelope_only",
        "center_sqrt_abs_difference_only",
    )
    required = {
        "cluster_id",
        *(f"risk_{score_id}" for score_id in arm_a_score_ids),
        *(f"risk_hash_{score_id}" for score_id in arm_a_score_ids),
        "hard_eligible_prefix_only",
        "issued_prefix_only",
        "issuance_rank_prefix_only",
    }
    _require_columns(trajectories, required, context="Stress-plan Arm-A outputs")
    indexed = trajectories.set_index("cluster_id")
    if not indexed.index.is_unique:
        raise V015AnalysisError("Stress-plan outputs contain duplicate cluster IDs")
    for _, left_id, right_id in pairs:
        if left_id not in indexed.index or right_id not in indexed.index:
            raise V015AnalysisError("Stress-plan pair member is absent")
        _exact_pair_point_outputs(
            points,
            left_id,
            right_id,
            context="Stress-plan Arm-A invariance",
        )
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        for score_id in arm_a_score_ids:
            risk_column = f"risk_{score_id}"
            if not _canonical_scalar_equal(left[risk_column], right[risk_column]):
                raise V015AnalysisError(
                    f"Stress-plan Arm-A output differs bitwise in {risk_column}"
                )
            hash_column = f"risk_hash_{score_id}"
            if _sha256_text(
                left[hash_column], context=f"Stress-plan left {hash_column}"
            ) != _sha256_text(
                right[hash_column], context=f"Stress-plan right {hash_column}"
            ):
                raise V015AnalysisError(
                    f"Stress-plan Arm-A content differs in {hash_column}"
                )
        for column in (
            "hard_eligible_prefix_only",
            "issued_prefix_only",
            "issuance_rank_prefix_only",
        ):
            if not _canonical_scalar_equal(left[column], right[column]):
                raise V015AnalysisError(f"Stress-plan Arm-A output differs in {column}")


def evaluate_intrinsic_pairs(
    trajectories: pd.DataFrame, pair_mapping: pd.DataFrame
) -> tuple[pd.DataFrame, CoverageSummary | None]:
    required_trajectory = {
        "cluster_id",
        "simultaneous_interval_covered",
        "max_interval_width_pp",
        "risk_prefix_only",
        "risk_visible_stress",
    }
    _require_columns(
        trajectories, required_trajectory, context="Intrinsic trajectories"
    )
    pairs = _validated_pair_members(pair_mapping, context="Intrinsic pair mapping")
    indexed = trajectories.set_index("cluster_id")
    if not indexed.index.is_unique:
        raise V015AnalysisError("Intrinsic trajectories contain duplicate cluster IDs")
    records: list[dict[str, object]] = []
    for pair_id, left_id, right_id in pairs:
        if left_id not in indexed.index or right_id not in indexed.index:
            raise V015AnalysisError("Intrinsic pair member is absent")
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        arm_a_equal = bool(
            float(left["risk_prefix_only"]) == float(right["risk_prefix_only"])
        )
        arm_b_equal = bool(
            float(left["risk_visible_stress"]) == float(right["risk_visible_stress"])
        )
        left_width = float(left["max_interval_width_pp"])
        right_width = float(right["max_interval_width_pp"])
        width_equal = bool(
            left_width == right_width
            or (math.isnan(left_width) and math.isnan(right_width))
        )
        if not (arm_a_equal and arm_b_equal and width_equal):
            raise V015AnalysisError(
                "Identical-input intrinsic pair produced unequal outputs"
            )
        left_coverage = left["simultaneous_interval_covered"]
        right_coverage = right["simultaneous_interval_covered"]
        missing_coverage = pd.isna(left_coverage) or pd.isna(right_coverage)
        both_covered: object
        if missing_coverage:
            if not (pd.isna(left_coverage) and pd.isna(right_coverage)):
                raise V015AnalysisError(
                    "Intrinsic pair has partially missing interval coverage"
                )
            both_covered = pd.NA
        else:
            both_covered = bool(
                _strict_bool(
                    left_coverage,
                    context="intrinsic left coverage",
                )
                and _strict_bool(
                    right_coverage,
                    context="intrinsic right coverage",
                )
            )
        records.append(
            {
                "pair_id": pair_id,
                "left_cluster_id": left_id,
                "right_cluster_id": right_id,
                "arm_a_exact_equal": arm_a_equal,
                "arm_b_exact_equal": arm_b_equal,
                "interval_width_exact_equal": width_equal,
                "both_futures_simultaneously_covered": both_covered,
                "max_interval_width_pp": left_width,
            }
        )
    scores = pd.DataFrame(records)
    summary_input = scores.rename(
        columns={
            "both_futures_simultaneously_covered": ("simultaneous_interval_covered")
        }
    )
    if (
        summary_input[["simultaneous_interval_covered", "max_interval_width_pp"]]
        .isna()
        .any()
        .any()
    ):
        return scores, None
    return scores, coverage_summary(summary_input)


def evaluate_stress_plan_pairs(
    trajectories: pd.DataFrame, pair_mapping: pd.DataFrame
) -> tuple[pd.DataFrame, StressPairSummary]:
    required_trajectory = {
        "cluster_id",
        "center_endpoint_absolute_error_pp",
        "risk_prefix_only",
        "risk_visible_stress",
    }
    _require_columns(
        trajectories, required_trajectory, context="Stress-plan trajectories"
    )
    pairs = _validated_pair_members(pair_mapping, context="Stress-plan mapping")
    indexed = trajectories.set_index("cluster_id")
    if not indexed.index.is_unique:
        raise V015AnalysisError(
            "Stress-plan trajectories contain duplicate cluster IDs"
        )
    records: list[dict[str, object]] = []
    for pair_id, left_id, right_id in pairs:
        if left_id not in indexed.index or right_id not in indexed.index:
            raise V015AnalysisError("Stress-plan pair member is absent")
        left = indexed.loc[left_id]
        right = indexed.loc[right_id]
        arm_a_tie = bool(
            float(left["risk_prefix_only"]) == float(right["risk_prefix_only"])
        )
        if not arm_a_tie:
            raise V015AnalysisError("Arm A changed across an identical prefix")
        error_delta = float(left["center_endpoint_absolute_error_pp"]) - float(
            right["center_endpoint_absolute_error_pp"]
        )
        risk_delta = float(left["risk_visible_stress"]) - float(
            right["risk_visible_stress"]
        )
        correct = bool(
            error_delta != 0.0 and risk_delta != 0.0 and error_delta * risk_delta > 0.0
        )
        records.append(
            {
                "pair_id": pair_id,
                "left_cluster_id": left_id,
                "right_cluster_id": right_id,
                "arm_a_exact_tie": arm_a_tie,
                "center_endpoint_error_delta_pp": error_delta,
                "arm_b_risk_delta": risk_delta,
                "arm_b_correct_error_order": correct,
            }
        )
    scores = pd.DataFrame(records)
    successes = int(scores["arm_b_correct_error_order"].sum())
    lower, upper = clopper_pearson_two_sided(successes, len(scores))
    summary = StressPairSummary(
        pair_count=len(scores),
        arm_a_exact_tie_count=int(scores["arm_a_exact_tie"].sum()),
        arm_b_correct_order_count=successes,
        arm_b_correct_order_fraction=successes / len(scores),
        arm_b_two_sided_95_lower=lower,
        arm_b_two_sided_95_upper=upper,
    )
    return scores, summary


def _bootstrap_seed(replicate: int, family: str, protocol_id: str) -> int:
    material = (
        f"{protocol_id}|{BOOTSTRAP_ROOT}|bootstrap|{replicate}|{family}"
    ).encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:16], 16) % (2**63 - 1)


def _arm_tie_digest(protocol_id: str, arm: str, content_hash: object) -> str:
    verified = _sha256_text(content_hash, context=f"{arm} content hash")
    return hashlib.sha256(f"{protocol_id}|{arm}|{verified}".encode("ascii")).hexdigest()


def bootstrap_risk_reductions(
    trajectories: pd.DataFrame,
    *,
    protocol_id: str,
    issue_count: int = TEST_ISSUE_COUNT,
    resamples: int = BOOTSTRAP_RESAMPLES,
    families: Sequence[str] = TEST_FAMILIES,
) -> pd.DataFrame:
    required = {
        "truth_family",
        "canonical_prefix_content_sha256",
        "catastrophic",
        "hard_eligible_visible_stress",
        "risk_prefix_only",
        "risk_visible_stress",
        "risk_placebo_8",
        "risk_hash_prefix_only",
        "risk_hash_visible_stress",
        "risk_hash_placebo_8",
    }
    _require_columns(trajectories, required, context="Bootstrap table")
    family_frames: dict[str, pd.DataFrame] = {}
    for family in families:
        subset = trajectories.loc[trajectories["truth_family"].eq(family)].copy()
        if subset.empty:
            raise V015AnalysisError(f"Bootstrap family absent: {family}")
        subset = subset.sort_values(
            "canonical_prefix_content_sha256", kind="stable"
        ).reset_index(drop=True)
        if subset["canonical_prefix_content_sha256"].duplicated().any():
            raise V015AnalysisError(
                f"Bootstrap ordinary content is duplicated in {family}"
            )
        for score_id in ("prefix_only", "visible_stress", "placebo_8"):
            subset[f"_tie_{score_id}"] = [
                _arm_tie_digest(protocol_id, score_id, value)
                for value in subset[f"risk_hash_{score_id}"]
            ]
        family_frames[family] = subset

    rows: list[dict[str, object]] = []
    for replicate in range(resamples):
        sampled: list[pd.DataFrame] = []
        for family_index, family in enumerate(families):
            source = family_frames[family]
            rng = np.random.Generator(
                np.random.PCG64DXSM(_bootstrap_seed(replicate, family, protocol_id))
            )
            indices = rng.integers(
                0, len(source), size=len(source), endpoint=False, dtype=np.int64
            )
            draw = source.iloc[indices].copy()
            draw["_family_index"] = family_index
            draw["_occurrence_ordinal"] = np.arange(len(source), dtype=np.int64)
            sampled.append(draw)
        boot = pd.concat(sampled, ignore_index=True)
        eligible = _strict_bool_series(
            boot["hard_eligible_visible_stress"],
            context="bootstrap eligibility",
        )
        defined = bool(eligible.sum() >= issue_count)
        random_rate = float(
            _strict_bool_series(
                boot.loc[eligible, "catastrophic"],
                context="bootstrap catastrophic",
            ).mean()
        )
        defined = defined and math.isfinite(random_rate) and random_rate > 0.0
        record: dict[str, object] = {
            "replicate_index": replicate,
            "defined": defined,
            "eligible_count": int(eligible.sum()),
            "random_expected_catastrophic_rate": random_rate,
        }
        if not defined:
            record.update(
                {
                    "prefix_only_risk_reduction": math.nan,
                    "visible_stress_risk_reduction": math.nan,
                    "visible_minus_prefix_increment": math.nan,
                    "placebo_minus_prefix_increment": math.nan,
                }
            )
            rows.append(record)
            continue
        reductions: dict[str, float] = {}
        for score_id in ("prefix_only", "visible_stress", "placebo_8"):
            score_column = f"risk_{score_id}"
            tie_column = f"_tie_{score_id}"
            ranked = boot.loc[eligible].sort_values(
                [
                    score_column,
                    tie_column,
                    "_family_index",
                    "_occurrence_ordinal",
                ],
                kind="stable",
            )
            rate = float(
                _strict_bool_series(
                    ranked.iloc[:issue_count]["catastrophic"],
                    context="bootstrap issued catastrophic",
                ).mean()
            )
            reductions[score_id] = 1.0 - rate / random_rate
        record.update(
            {
                "prefix_only_risk_reduction": reductions["prefix_only"],
                "visible_stress_risk_reduction": reductions["visible_stress"],
                "visible_minus_prefix_increment": (
                    reductions["visible_stress"] - reductions["prefix_only"]
                ),
                "placebo_minus_prefix_increment": (
                    reductions["placebo_8"] - reductions["prefix_only"]
                ),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def bootstrap_gate_summary(replicates: pd.DataFrame) -> Mapping[str, float]:
    required = {
        "defined",
        "visible_stress_risk_reduction",
        "visible_minus_prefix_increment",
        "placebo_minus_prefix_increment",
    }
    _require_columns(replicates, required, context="Bootstrap replicates")
    if len(replicates) != BOOTSTRAP_RESAMPLES:
        raise V015AnalysisError("Every one of 5000 replicates is required")
    if not _strict_bool_series(
        replicates["defined"], context="bootstrap defined"
    ).all():
        raise V015AnalysisError("An undefined bootstrap makes gates inconclusive")
    visible = _finite_vector(
        replicates["visible_stress_risk_reduction"], context="Visible bootstrap"
    )
    increment = _finite_vector(
        replicates["visible_minus_prefix_increment"], context="Increment bootstrap"
    )
    placebo = _finite_vector(
        replicates["placebo_minus_prefix_increment"], context="Placebo bootstrap"
    )
    return {
        "visible_one_sided_95_lower": float(
            np.quantile(visible, 0.05, method="linear")
        ),
        "increment_one_sided_95_lower": float(
            np.quantile(increment, 0.05, method="linear")
        ),
        "placebo_two_sided_95_lower": float(
            np.quantile(placebo, 0.025, method="linear")
        ),
        "placebo_two_sided_95_upper": float(
            np.quantile(placebo, 0.975, method="linear")
        ),
    }


def _permutation_digest(
    permutation_index: int,
    family: str,
    random_policy_hash: object,
) -> str:
    verified = _sha256_text(
        random_policy_hash, context="permutation random-policy hash"
    )
    material = (
        f"{STRESS_PERMUTATION_ROOT}|{permutation_index}|{family}|{verified}"
    ).encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _aligned_operating_matrix(
    frame: pd.DataFrame,
    operating_pack: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    key = ["partition", "cluster_id"]
    _require_columns(frame, set(key), context="Feature table")
    _require_columns(
        operating_pack,
        {*key, *REAL_OPERATING_FIELDS},
        context="Operating pack",
    )
    if frame.duplicated(key).any() or operating_pack.duplicated(key).any():
        raise V015AnalysisError("Permutation cluster coordinates are duplicated")
    operating = frame.loc[:, key].merge(
        operating_pack,
        on=key,
        how="left",
        validate="one_to_one",
        suffixes=("_feature", ""),
    )
    if len(operating) != len(frame):
        raise V015AnalysisError("Permutation operating rows are incomplete")
    values = (
        operating.loc[:, REAL_OPERATING_FIELDS]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    if not np.isfinite(values).all():
        raise V015AnalysisError("Permutation operating block is nonfinite")
    return values, operating


def _permuted_boundary_tie_hashes(
    *,
    frame: pd.DataFrame,
    assigned_operating: np.ndarray,
    tie_positions: np.ndarray,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    operating_pack: pd.DataFrame,
    protocol_id: str,
) -> list[str]:
    from lifetwin.experiments.calendar_long_horizon_v015_io import (
        predictor_content_hashes,
    )

    key = ["partition", "cluster_id"]
    operating_index = operating_pack.set_index(key)
    if not operating_index.index.is_unique:
        raise V015AnalysisError("Operating pack contains duplicate cluster keys")
    hashes: list[str] = []
    for position in tie_positions:
        row = frame.iloc[int(position)]
        partition = str(row["partition"])
        cluster_id = str(row["cluster_id"])
        prefix = prefix_pack.loc[
            prefix_pack["partition"].eq(partition)
            & prefix_pack["cluster_id"].eq(cluster_id)
        ]
        coordinates = forecast_coordinates.loc[
            forecast_coordinates["partition"].eq(partition)
            & forecast_coordinates["cluster_id"].eq(cluster_id)
        ]
        original = operating_index.loc[(partition, cluster_id)].to_dict()
        for field, value in zip(
            REAL_OPERATING_FIELDS,
            assigned_operating[int(position)],
            strict=True,
        ):
            original[field] = float(value)
        content = predictor_content_hashes(prefix, coordinates, original)
        hashes.append(_arm_tie_digest(protocol_id, "visible_stress", content.arm_b))
    return hashes


def stress_permutation_metrics(
    trajectories_and_features: pd.DataFrame,
    operating_pack: pd.DataFrame,
    prefix_pack: pd.DataFrame,
    forecast_coordinates: pd.DataFrame,
    *,
    visible_stress_state: Any,
    protocol_id: str,
    random_expected_catastrophic_rate: float,
    observed_prefix_only_risk_reduction: float,
    issue_count: int = TEST_ISSUE_COUNT,
    permutations: int = STRESS_PERMUTATIONS,
) -> pd.DataFrame:
    """Jointly permute the eight-field stress block within frozen test families."""
    prefix_feature_names = tuple(visible_stress_state.feature_names)[
        : -len(REAL_OPERATING_FIELDS)
    ]
    expected_feature_names = prefix_feature_names + REAL_OPERATING_FIELDS
    if tuple(visible_stress_state.feature_names) != expected_feature_names:
        raise V015AnalysisError(
            "Visible-stress state does not end in the frozen operating fields"
        )
    required = {
        "partition",
        "cluster_id",
        "truth_family",
        "canonical_prefix_content_sha256",
        "hard_eligible_visible_stress",
        "catastrophic",
        *prefix_feature_names,
    }
    _require_columns(trajectories_and_features, required, context="Stress permutation")
    frame = trajectories_and_features.loc[
        trajectories_and_features["partition"].eq("test")
    ].copy()
    if frame.empty or frame["cluster_id"].duplicated().any():
        raise V015AnalysisError("Stress permutation test pool is invalid")
    if permutations < 1 or issue_count < 1:
        raise V015AnalysisError("Stress permutation counts must be positive")
    if (
        not math.isfinite(random_expected_catastrophic_rate)
        or random_expected_catastrophic_rate <= 0.0
        or not math.isfinite(observed_prefix_only_risk_reduction)
    ):
        raise V015AnalysisError("Stress permutation reference metrics are invalid")

    random_hashes = [
        _sha256_text(value, context="permutation random-policy hash")
        for value in frame["canonical_prefix_content_sha256"]
    ]
    if len(set(random_hashes)) != len(random_hashes):
        raise V015AnalysisError("Permutation ordinary content is duplicated")
    frame["_random_hash"] = random_hashes
    eligible = _strict_bool_series(
        frame["hard_eligible_visible_stress"],
        context="permutation eligibility",
    ).to_numpy()
    if int(eligible.sum()) < issue_count:
        raise V015InconclusiveError("Permutation pool has too few eligible rows")
    catastrophic = _strict_bool_series(
        frame["catastrophic"], context="permutation catastrophic"
    ).to_numpy()
    prefix_features = (
        frame.loc[:, prefix_feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(float)
    )
    if not np.isfinite(prefix_features).all():
        raise V015AnalysisError("Permutation prefix features are nonfinite")
    operating_values, _ = _aligned_operating_matrix(frame, operating_pack)
    family_positions: dict[str, np.ndarray] = {}
    for family in TEST_FAMILIES:
        positions = np.flatnonzero(frame["truth_family"].eq(family).to_numpy())
        if not len(positions):
            raise V015AnalysisError(f"Permutation family is absent: {family}")
        recipient_order = sorted(
            positions, key=lambda position: random_hashes[int(position)]
        )
        family_positions[family] = np.asarray(recipient_order, dtype=np.int64)

    records: list[dict[str, object]] = []
    for permutation_index in range(permutations):
        assigned = np.empty_like(operating_values)
        for family in TEST_FAMILIES:
            recipients = family_positions[family]
            donors = sorted(
                recipients,
                key=lambda position: _permutation_digest(
                    permutation_index,
                    family,
                    random_hashes[int(position)],
                ),
            )
            assigned[recipients] = operating_values[np.asarray(donors, dtype=np.int64)]
        feature_matrix = np.concatenate((prefix_features, assigned), axis=1)
        scores = np.asarray(
            visible_stress_state.decision_function(feature_matrix), dtype=float
        )
        if scores.shape != (len(frame),) or not np.isfinite(scores).all():
            raise V015AnalysisError("Permuted visible-stress score is invalid")
        eligible_positions = np.flatnonzero(eligible)
        eligible_scores = scores[eligible_positions]
        cutoff = float(np.partition(eligible_scores, issue_count - 1)[issue_count - 1])
        below = eligible_positions[eligible_scores < cutoff]
        boundary = eligible_positions[eligible_scores == cutoff]
        needed = issue_count - len(below)
        if needed < 1 or needed > len(boundary):
            raise V015AnalysisError("Permutation cutoff reconstruction failed")
        if len(boundary) > needed:
            tie_hashes = _permuted_boundary_tie_hashes(
                frame=frame,
                assigned_operating=assigned,
                tie_positions=boundary,
                prefix_pack=prefix_pack,
                forecast_coordinates=forecast_coordinates,
                operating_pack=operating_pack,
                protocol_id=protocol_id,
            )
            chosen_order = np.argsort(
                np.asarray(tie_hashes, dtype="U64"), kind="stable"
            )
            boundary = boundary[chosen_order[:needed]]
        selected = np.concatenate((below, boundary[:needed]))
        issued_rate = float(catastrophic[selected].mean())
        visible_reduction = 1.0 - issued_rate / random_expected_catastrophic_rate
        records.append(
            {
                "permutation_index": permutation_index,
                "issued_count": len(selected),
                "issued_catastrophic_rate": issued_rate,
                "visible_stress_risk_reduction": visible_reduction,
                "visible_minus_prefix_increment": (
                    visible_reduction - observed_prefix_only_risk_reduction
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_stress_permutations(
    metrics: pd.DataFrame,
    *,
    observed_visible_minus_prefix_increment: float,
) -> StressPermutationSummary:
    _require_columns(
        metrics,
        {"visible_minus_prefix_increment"},
        context="Stress permutation metrics",
    )
    if len(metrics) != STRESS_PERMUTATIONS:
        raise V015AnalysisError("All 10000 stress permutations are required")
    values = _finite_vector(
        metrics["visible_minus_prefix_increment"],
        context="Stress permutation increments",
    )
    if not math.isfinite(observed_visible_minus_prefix_increment):
        raise V015AnalysisError("Observed stress increment is nonfinite")
    lower_count = int(np.sum(values < observed_visible_minus_prefix_increment))
    return StressPermutationSummary(
        permutation_count=len(values),
        observed_visible_minus_prefix_increment=(
            observed_visible_minus_prefix_increment
        ),
        strictly_lower_count=lower_count,
        strictly_lower_fraction=lower_count / len(values),
        gate_passed=lower_count >= 9900,
    )


def placebo_negative_control_gates(
    *,
    placebo_minus_prefix_increment: float,
    bootstrap_summary: Mapping[str, float],
) -> tuple[GateEvaluation, GateEvaluation]:
    if not math.isfinite(placebo_minus_prefix_increment):
        raise V015AnalysisError("Placebo point increment is nonfinite")
    lower = float(bootstrap_summary["placebo_two_sided_95_lower"])
    upper = float(bootstrap_summary["placebo_two_sided_95_upper"])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise V015AnalysisError("Placebo bootstrap interval is invalid")
    return (
        _gate(
            "placebo_point_negative_control",
            passed=abs(placebo_minus_prefix_increment) < 0.05,
            estimate=placebo_minus_prefix_increment,
            threshold="abs(increment) < 0.05",
        ),
        _gate(
            "placebo_interval_negative_control",
            passed=lower <= 0.0 <= upper,
            estimate=bool(lower <= 0.0 <= upper),
            threshold="two-sided 95% interval contains 0",
        ),
    )


def primary_gate_flags(
    *,
    visible_reduction: float,
    increment: float,
    bootstrap_summary: Mapping[str, float],
    core_coverage: CoverageSummary,
    intrinsic_coverage: CoverageSummary,
    issued_center_minus_baseline_iae_pp: float,
) -> Mapping[str, bool]:
    values = (
        visible_reduction,
        increment,
        issued_center_minus_baseline_iae_pp,
        float(bootstrap_summary["visible_one_sided_95_lower"]),
        float(bootstrap_summary["increment_one_sided_95_lower"]),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise V015AnalysisError("Primary gates require finite values")
    return {
        "visible_stress_catastrophic_risk_reduction": bool(
            visible_reduction >= 0.30
            and float(bootstrap_summary["visible_one_sided_95_lower"]) > 0.0
        ),
        "visible_stress_increment_over_prefix_only": bool(
            increment >= 0.10
            and float(bootstrap_summary["increment_one_sided_95_lower"]) > 0.0
        ),
        "core_test_simultaneous_trajectory_coverage": bool(
            core_coverage.n == 1500
            and core_coverage.coverage >= 0.85
            and core_coverage.one_sided_95_lower > 0.83
            and core_coverage.median_max_width_pp <= 40.0
            and core_coverage.percentile_95_max_width_pp <= 50.0
        ),
        "intrinsic_pair_simultaneous_both_future_coverage": bool(
            intrinsic_coverage.n == 250
            and intrinsic_coverage.coverage >= 0.80
            and intrinsic_coverage.one_sided_95_lower > 0.75
            and intrinsic_coverage.median_max_width_pp <= 40.0
            and intrinsic_coverage.percentile_95_max_width_pp <= 50.0
        ),
        "issued_center_trajectory_iae_noninferiority": bool(
            issued_center_minus_baseline_iae_pp <= 0.10
        ),
    }


def primary_gate_evaluations(
    *,
    visible_reduction: float,
    increment: float,
    bootstrap_summary: Mapping[str, float],
    core_coverage: CoverageSummary,
    intrinsic_coverage: CoverageSummary,
    issued_center_minus_baseline_iae_pp: float,
) -> tuple[GateEvaluation, ...]:
    flags = primary_gate_flags(
        visible_reduction=visible_reduction,
        increment=increment,
        bootstrap_summary=bootstrap_summary,
        core_coverage=core_coverage,
        intrinsic_coverage=intrinsic_coverage,
        issued_center_minus_baseline_iae_pp=(issued_center_minus_baseline_iae_pp),
    )
    values: dict[str, float] = {
        "visible_stress_catastrophic_risk_reduction": visible_reduction,
        "visible_stress_increment_over_prefix_only": increment,
        "core_test_simultaneous_trajectory_coverage": core_coverage.coverage,
        "intrinsic_pair_simultaneous_both_future_coverage": (
            intrinsic_coverage.coverage
        ),
        "issued_center_trajectory_iae_noninferiority": (
            issued_center_minus_baseline_iae_pp
        ),
    }
    thresholds = {
        "visible_stress_catastrophic_risk_reduction": (
            "estimate >= 0.30 and bootstrap one-sided lower > 0"
        ),
        "visible_stress_increment_over_prefix_only": (
            "estimate >= 0.10 and paired-bootstrap one-sided lower > 0"
        ),
        "core_test_simultaneous_trajectory_coverage": (
            "coverage >= 0.85, CP lower > 0.83, median width <= 40, p95 width <= 50"
        ),
        "intrinsic_pair_simultaneous_both_future_coverage": (
            "coverage >= 0.80, CP lower > 0.75, median width <= 40, p95 width <= 50"
        ),
        "issued_center_trajectory_iae_noninferiority": "<= +0.10 pp",
    }
    return tuple(
        _gate(
            gate_id,
            passed=passed,
            estimate=values[gate_id],
            threshold=thresholds[gate_id],
        )
        for gate_id, passed in flags.items()
    )


def resolve_result_status(
    evaluations: Sequence[GateEvaluation],
    *,
    void_reasons: Sequence[str] = (),
    external_inconclusive_reasons: Sequence[str] = (),
    required_gate_ids: Sequence[str] = REQUIRED_GATE_IDS,
) -> dict[str, object]:
    """Resolve V2 status without allowing missing metrics to hide a failed gate."""
    required = tuple(required_gate_ids)
    if len(set(required)) != len(required):
        raise V015AnalysisError("Required gate registry contains duplicates")
    by_id: dict[str, GateEvaluation] = {}
    for evaluation in evaluations:
        if evaluation.gate_id in by_id:
            raise V015AnalysisError(f"Gate was evaluated twice: {evaluation.gate_id}")
        by_id[evaluation.gate_id] = evaluation
    missing = [gate_id for gate_id in required if gate_id not in by_id]
    extra = sorted(set(by_id) - set(required))
    if extra:
        raise V015AnalysisError(f"Unexpected required-gate evaluations: {extra}")

    void = tuple(str(reason) for reason in void_reasons if str(reason))
    inconclusive = [
        str(reason) for reason in external_inconclusive_reasons if str(reason)
    ]
    inconclusive.extend(f"missing_gate:{gate_id}" for gate_id in missing)
    inconclusive.extend(
        f"{gate_id}:{reason}"
        for gate_id in required
        if gate_id in by_id and by_id[gate_id].state == "inconclusive"
        for reason in (by_id[gate_id].reasons or ("unavailable",))
    )
    failed = [
        gate_id
        for gate_id in required
        if gate_id in by_id and by_id[gate_id].state == "fail"
    ]
    passed = [
        gate_id
        for gate_id in required
        if gate_id in by_id and by_id[gate_id].state == "pass"
    ]
    if void:
        status = "void"
    elif failed:
        status = "failure"
    elif inconclusive:
        status = "inconclusive_not_success"
    else:
        status = "success"
    return {
        "status": status,
        "status_resolution_convention": (
            "void > failure > inconclusive_not_success > success"
        ),
        "void_reasons": list(void),
        "inconclusive_reasons": inconclusive,
        "failed_gate_ids": failed,
        "passed_gate_ids": passed,
        "required_gate_ids": list(required),
        "gate_evaluations": [
            {
                "gate_id": by_id[gate_id].gate_id,
                "state": by_id[gate_id].state,
                "estimate": by_id[gate_id].estimate,
                "threshold": by_id[gate_id].threshold,
                "reasons": list(by_id[gate_id].reasons),
            }
            for gate_id in required
            if gate_id in by_id
        ],
    }
