from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lifetwin.experiments.calendar_long_horizon_synthetic import (
    FrozenScoreResult,
    MatchedPairAuditResult,
    SyntheticProtocolError,
    TRUTH_FAMILY_IDS,
    ValidatedSyntheticProtocol,
)


MODEL_FORECAST_COLUMNS = {
    "candidate": "candidate_point_forecast_pct",
    "target_prefix_persistence": "persistence_forecast_pct",
    "target_prefix_sqrt_time": "sqrt_time_forecast_pct",
    "target_prefix_bounded_power_law": "bounded_power_forecast_pct",
}
RANDOM_REJECTION_SEED = 2026072206
BOOTSTRAP_SEED = 2026072207
RANDOM_RANKING_COUNT = 10_000
BOOTSTRAP_RESAMPLES = 5_000
TEST_ISSUE_COUNT = 500
AUDIT_ISSUE_COUNT = 250
SECONDARY_ISSUANCE_FRACTIONS = (0.25, 0.50, 0.75)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SyntheticAnalysisResult:
    report: dict[str, Any]
    model_metrics: pd.DataFrame
    family_metrics: pd.DataFrame
    forecast_day_metrics: pd.DataFrame
    risk_coverage: pd.DataFrame
    rejection_policy_metrics: pd.DataFrame
    member_fit_metrics: pd.DataFrame
    noise_sensitivity_metrics: pd.DataFrame
    random_rejection: pd.DataFrame
    bootstrap: pd.DataFrame


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SyntheticProtocolError(f"{context} is missing columns: {missing}")


def _strict_bool(value: object, *, context: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise SyntheticProtocolError(f"{context} must be boolean")
    return bool(value)


def _nonnegative_integer(value: object, *, context: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SyntheticProtocolError(f"{context} must be a non-negative integer")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or not numeric.is_integer():
        raise SyntheticProtocolError(f"{context} must be a non-negative integer")
    return int(numeric)


def _sha256_text(value: object, *, context: str) -> str:
    text = str(value)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise SyntheticProtocolError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _finite_or_none(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(bool(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if value is None or value is pd.NA:
                record[str(key)] = None
            elif isinstance(value, (bool, np.bool_)):
                record[str(key)] = bool(value)
            elif isinstance(value, (int, np.integer)):
                record[str(key)] = int(value)
            elif isinstance(value, (float, np.floating)):
                record[str(key)] = _finite_or_none(value)
            else:
                record[str(key)] = str(value)
        records.append(record)
    return records


def _assert_json_finite(report: dict[str, Any]) -> None:
    try:
        json.dumps(report, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SyntheticProtocolError(
            "Synthetic analysis report contains a non-JSON or non-finite value"
        ) from exc


def _prefix_tie_digest(protocol_id: str, prefix_hash: str) -> str:
    return hashlib.sha256(f"{protocol_id}|{prefix_hash}".encode("utf-8")).hexdigest()


def _validate_distinct_prefix_hashes(frame: pd.DataFrame, *, context: str) -> None:
    hashes = [
        _sha256_text(value, context=f"{context} canonical prefix hash")
        for value in frame["canonical_prefix_content_sha256"]
    ]
    if len(hashes) != len(set(hashes)):
        raise SyntheticProtocolError(
            f"{context} contains a canonical prefix hash collision"
        )


def build_model_metrics(point_scores: pd.DataFrame) -> pd.DataFrame:
    """Reduce repeated future points to independent trajectory-level metrics."""
    required = {
        "partition",
        "cluster_id",
        "truth_family",
        "forecast_day",
        "canonical_prefix_content_sha256",
        "hard_eligible",
        "primary_issued",
        "credible_structure_family_count",
        "fit_failure_count",
        "best_prefix_rmse_pp",
        "disagreement_score_pp",
        "latent_retention_pct",
        "noisy_retention_pct",
        *MODEL_FORECAST_COLUMNS.values(),
    }
    _require_columns(point_scores, required, context="Point scores")
    if point_scores.empty:
        raise SyntheticProtocolError("Point scores cannot be empty")
    if point_scores.duplicated(["partition", "cluster_id", "forecast_day"]).any():
        raise SyntheticProtocolError("Point-score coordinates must be unique")

    records: list[dict[str, Any]] = []
    metadata_columns = (
        "truth_family",
        "canonical_prefix_content_sha256",
        "hard_eligible",
        "primary_issued",
        "credible_structure_family_count",
        "fit_failure_count",
        "best_prefix_rmse_pp",
        "disagreement_score_pp",
    )
    for (partition, cluster_id), group in point_scores.groupby(
        ["partition", "cluster_id"], sort=True
    ):
        ordered = group.sort_values("forecast_day", kind="stable")
        days = pd.to_numeric(ordered["forecast_day"], errors="coerce").to_numpy(
            dtype=float
        )
        truth = pd.to_numeric(
            ordered["latent_retention_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        noisy_truth = pd.to_numeric(
            ordered["noisy_retention_pct"], errors="coerce"
        ).to_numpy(dtype=float)
        if (
            len(days) < 2
            or not np.isfinite(days).all()
            or not np.isfinite(truth).all()
            or not np.isfinite(noisy_truth).all()
            or np.any(np.diff(days) <= 0.0)
        ):
            raise SyntheticProtocolError("Trajectory scoring coordinates are invalid")
        for column in metadata_columns:
            if ordered[column].nunique(dropna=False) != 1:
                raise SyntheticProtocolError(
                    f"Trajectory metadata changed within cluster: {column}"
                )
        prefix_hash = _sha256_text(
            ordered["canonical_prefix_content_sha256"].iloc[0],
            context="canonical_prefix_content_sha256",
        )
        eligible = _strict_bool(
            ordered["hard_eligible"].iloc[0], context="hard_eligible"
        )
        issued = _strict_bool(
            ordered["primary_issued"].iloc[0], context="primary_issued"
        )
        credible_count = _nonnegative_integer(
            ordered["credible_structure_family_count"].iloc[0],
            context="credible_structure_family_count",
        )
        failure_count = _nonnegative_integer(
            ordered["fit_failure_count"].iloc[0], context="fit_failure_count"
        )
        best_rmse = float(ordered["best_prefix_rmse_pp"].iloc[0])
        disagreement = float(ordered["disagreement_score_pp"].iloc[0])
        if best_rmse < 0.0 or math.isnan(best_rmse):
            raise SyntheticProtocolError("best_prefix_rmse_pp is invalid")
        if disagreement < 0.0 or math.isnan(disagreement):
            raise SyntheticProtocolError("disagreement_score_pp is invalid")
        if eligible and (
            credible_count < 2
            or not math.isfinite(disagreement)
            or not math.isfinite(best_rmse)
        ):
            raise SyntheticProtocolError("Hard eligibility metadata is inconsistent")

        common = {
            "partition": str(partition),
            "cluster_id": str(cluster_id),
            "truth_family": str(ordered["truth_family"].iloc[0]),
            "canonical_prefix_content_sha256": prefix_hash,
            "hard_eligible": eligible,
            "primary_issued": issued,
            "credible_structure_family_count": credible_count,
            "fit_failure_count": failure_count,
            "best_prefix_rmse_pp": best_rmse,
            "disagreement_score_pp": disagreement,
        }
        for model_id, forecast_column in MODEL_FORECAST_COLUMNS.items():
            forecast = pd.to_numeric(
                ordered[forecast_column], errors="coerce"
            ).to_numpy(dtype=float)
            finite = bool(np.isfinite(forecast).all())
            absolute = np.abs(forecast - truth) if finite else None
            noisy_absolute = np.abs(forecast - noisy_truth) if finite else None
            records.append(
                {
                    **common,
                    "model_id": model_id,
                    "finite_forecast": finite,
                    "endpoint_absolute_error_pp": (
                        float(absolute[-1]) if absolute is not None else math.nan
                    ),
                    "trajectory_iae_pp": (
                        float(np.trapezoid(absolute, days) / (days[-1] - days[0]))
                        if absolute is not None
                        else math.nan
                    ),
                    "noisy_endpoint_absolute_error_pp": (
                        float(noisy_absolute[-1])
                        if noisy_absolute is not None
                        else math.nan
                    ),
                    "noisy_trajectory_iae_pp": (
                        float(np.trapezoid(noisy_absolute, days) / (days[-1] - days[0]))
                        if noisy_absolute is not None
                        else math.nan
                    ),
                    "catastrophic_error": bool(
                        absolute is None or float(absolute[-1]) >= 5.0
                    ),
                    "noisy_catastrophic_error": bool(
                        noisy_absolute is None or float(noisy_absolute[-1]) >= 5.0
                    ),
                }
            )
    result = pd.DataFrame(records).sort_values(
        ["partition", "cluster_id", "model_id"], kind="stable"
    )
    return result.reset_index(drop=True)


def select_strongest_calibration_baseline(
    model_metrics: pd.DataFrame,
) -> tuple[str | None, pd.DataFrame]:
    """Freeze the calibration winner, or mark selection unavailable without dropping."""
    baseline_ids = sorted(set(MODEL_FORECAST_COLUMNS) - {"candidate"})
    calibration = model_metrics.loc[
        model_metrics["partition"].eq("calibration")
        & model_metrics["model_id"].isin(baseline_ids)
    ].copy()
    candidate_calibration = model_metrics.loc[
        model_metrics["partition"].eq("calibration")
        & model_metrics["model_id"].eq("candidate")
    ]
    expected_clusters = set(candidate_calibration["cluster_id"].astype(str))
    if not expected_clusters:
        raise SyntheticProtocolError("Calibration trajectories are required")

    records: list[dict[str, Any]] = []
    selection_evaluable = True
    for model_id in baseline_ids:
        rows = calibration.loc[calibration["model_id"].eq(model_id)]
        observed_clusters = set(rows["cluster_id"].astype(str))
        complete = bool(
            len(rows) == len(expected_clusters)
            and observed_clusters == expected_clusters
            and not rows["cluster_id"].duplicated().any()
        )
        finite_count = int(rows["finite_forecast"].sum()) if complete else 0
        unavailable_count = (
            int((~rows["finite_forecast"]).sum())
            if complete
            else len(expected_clusters)
        )
        fully_finite = bool(complete and finite_count == len(expected_clusters))
        mean_iae = (
            float(rows["trajectory_iae_pp"].to_numpy(dtype=float).mean())
            if fully_finite
            else math.nan
        )
        selection_evaluable = selection_evaluable and fully_finite
        records.append(
            {
                "model_id": model_id,
                "expected_calibration_cluster_count": len(expected_clusters),
                "observed_calibration_cluster_count": int(len(rows)),
                "finite_trajectory_count": finite_count,
                "unavailable_trajectory_iae_count": unavailable_count,
                "selection_eligible": fully_finite,
                "mean_trajectory_iae_pp": mean_iae,
            }
        )
    summary = pd.DataFrame(records)
    if not selection_evaluable:
        return None, summary
    ranked = summary.sort_values(
        ["mean_trajectory_iae_pp", "model_id"], kind="stable"
    ).reset_index(drop=True)
    return str(ranked.iloc[0]["model_id"]), ranked


def _candidate_trajectories(
    model_metrics: pd.DataFrame, partition: str
) -> pd.DataFrame:
    rows = model_metrics.loc[
        model_metrics["partition"].eq(partition)
        & model_metrics["model_id"].eq("candidate")
    ].copy()
    if rows.empty or rows["cluster_id"].duplicated().any():
        raise SyntheticProtocolError(
            f"Candidate trajectories for {partition} must be unique and non-empty"
        )
    _validate_distinct_prefix_hashes(rows, context=f"{partition} candidate pool")
    return rows.reset_index(drop=True)


def _random_rejection_distribution(
    eligible: pd.DataFrame,
    *,
    issue_count: int,
    ranking_count: int | None = None,
) -> pd.DataFrame:
    """Run full-SHA frozen rankings and fail on any exact digest collision."""
    count = RANDOM_RANKING_COUNT if ranking_count is None else int(ranking_count)
    if count <= 0:
        raise SyntheticProtocolError("Random ranking count must be positive")
    columns = (
        "ranking_index",
        "status",
        "issued_count",
        "catastrophic_count",
        "catastrophic_rate",
    )
    if len(eligible) < issue_count:
        return pd.DataFrame(
            [
                {
                    "ranking_index": ranking_index,
                    "status": "undefined_insufficient_eligible",
                    "issued_count": 0,
                    "catastrophic_count": None,
                    "catastrophic_rate": None,
                }
                for ranking_index in range(count)
            ],
            columns=columns,
        )
    _validate_distinct_prefix_hashes(eligible, context="Random-policy eligible pool")
    prefix_hashes = eligible["canonical_prefix_content_sha256"].astype(str).tolist()
    catastrophic = eligible["catastrophic_error"].to_numpy(dtype=bool)
    records: list[dict[str, Any]] = []
    for ranking_index in range(count):
        digests = [
            hashlib.sha256(
                f"{RANDOM_REJECTION_SEED}|{ranking_index}|{prefix_hash}".encode("ascii")
            ).digest()
            for prefix_hash in prefix_hashes
        ]
        if len(digests) != len(set(digests)):
            raise SyntheticProtocolError(
                "Full SHA-256 random-ranking collision prevents frozen comparison"
            )
        selected = sorted(range(len(digests)), key=digests.__getitem__)[:issue_count]
        selected_catastrophic = catastrophic[np.asarray(selected, dtype=int)]
        records.append(
            {
                "ranking_index": ranking_index,
                "status": "defined",
                "issued_count": issue_count,
                "catastrophic_count": int(selected_catastrophic.sum()),
                "catastrophic_rate": float(selected_catastrophic.mean()),
            }
        )
    return pd.DataFrame(records, columns=columns)


def _partition_policy_summary(
    candidate: pd.DataFrame,
    *,
    issue_count: int,
) -> dict[str, Any]:
    eligible = candidate.loc[candidate["hard_eligible"]].copy()
    issued = candidate.loc[candidate["primary_issued"]].copy()
    if not issued.empty and (
        not issued["hard_eligible"].all() or not issued["finite_forecast"].all()
    ):
        raise SyntheticProtocolError("Issued trajectories must be finite and eligible")
    enough_eligible = len(eligible) >= issue_count
    if enough_eligible and len(issued) != issue_count:
        raise SyntheticProtocolError(
            f"Frozen policy must issue exactly {issue_count} trajectories"
        )
    random_expected = (
        float(eligible["catastrophic_error"].mean()) if not eligible.empty else None
    )
    issued_risk = (
        float(issued["catastrophic_error"].mean())
        if enough_eligible and len(issued) == issue_count
        else None
    )
    reduction = (
        1.0 - issued_risk / random_expected
        if issued_risk is not None
        and random_expected is not None
        and random_expected > 0.0
        else None
    )
    return {
        "status": "evaluable" if issued_risk is not None else "insufficient_eligible",
        "cluster_count": int(len(candidate)),
        "hard_eligible_count": int(len(eligible)),
        "hard_eligible_catastrophic_count": int(eligible["catastrophic_error"].sum()),
        "target_issue_count": int(issue_count),
        "issued_count": int(len(issued)),
        "issued_catastrophic_count": (
            int(issued["catastrophic_error"].sum()) if issued_risk is not None else None
        ),
        "issued_catastrophic_rate": issued_risk,
        "analytic_random_expected_catastrophic_rate": random_expected,
        "analytic_risk_reduction_fraction": reduction,
    }


def _verify_frozen_disagreement_issuance(
    candidate: pd.DataFrame,
    *,
    issue_count: int,
    protocol_id: str,
) -> None:
    eligible = candidate.loc[candidate["hard_eligible"]].copy()
    if len(eligible) < issue_count:
        return
    ranked = _ranked_policy_rows(
        eligible, protocol_id=protocol_id, metric="disagreement_score_pp"
    )
    expected = set(ranked.iloc[:issue_count]["cluster_id"].astype(str))
    observed = set(candidate.loc[candidate["primary_issued"], "cluster_id"].astype(str))
    if observed != expected:
        raise SyntheticProtocolError(
            "Frozen issued set differs from disagreement ranking and tie rule"
        )


def _bootstrap_risk_reduction(
    test: pd.DataFrame,
    *,
    issue_count: int = TEST_ISSUE_COUNT,
    resamples: int | None = None,
) -> pd.DataFrame:
    count = BOOTSTRAP_RESAMPLES if resamples is None else int(resamples)
    if count <= 0:
        raise SyntheticProtocolError("Bootstrap resample count must be positive")
    _validate_distinct_prefix_hashes(test, context="Bootstrap test pool")
    observed_families = set(test["truth_family"].astype(str))
    expected_families = set(TRUTH_FAMILY_IDS)
    if observed_families != expected_families:
        missing = sorted(expected_families - observed_families)
        unknown = sorted(observed_families - expected_families)
        raise SyntheticProtocolError(
            "Bootstrap truth-family strata differ from frozen v1: "
            f"missing={missing}, unknown={unknown}"
        )
    family_indices = {
        family: np.flatnonzero(test["truth_family"].astype(str).eq(family).to_numpy())
        for family in TRUTH_FAMILY_IDS
    }
    if any(len(indices) == 0 for indices in family_indices.values()):
        raise SyntheticProtocolError("Bootstrap truth-family strata cannot be empty")
    rng = np.random.Generator(np.random.PCG64DXSM(BOOTSTRAP_SEED))
    records: list[dict[str, Any]] = []
    for replicate in range(count):
        drawn = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in family_indices.values()
            ]
        )
        sample = test.iloc[drawn].copy().reset_index(drop=True)
        sample["_occurrence"] = np.arange(len(sample), dtype=int)
        eligible = sample.loc[sample["hard_eligible"]].copy()
        if len(eligible) < issue_count:
            records.append(
                {
                    "replicate": replicate,
                    "status": "undefined_insufficient_eligible",
                    "hard_eligible_count": int(len(eligible)),
                    "random_expected_catastrophic_rate": None,
                    "issued_catastrophic_rate": None,
                    "risk_reduction_fraction": None,
                }
            )
            continue
        eligible = eligible.sort_values(
            [
                "disagreement_score_pp",
                "canonical_prefix_content_sha256",
                "_occurrence",
            ],
            kind="stable",
        )
        issued = eligible.iloc[:issue_count]
        random_expected = float(eligible["catastrophic_error"].mean())
        issued_risk = float(issued["catastrophic_error"].mean())
        if random_expected <= 0.0:
            records.append(
                {
                    "replicate": replicate,
                    "status": "undefined_zero_random_risk",
                    "hard_eligible_count": int(len(eligible)),
                    "random_expected_catastrophic_rate": random_expected,
                    "issued_catastrophic_rate": issued_risk,
                    "risk_reduction_fraction": None,
                }
            )
            continue
        records.append(
            {
                "replicate": replicate,
                "status": "defined",
                "hard_eligible_count": int(len(eligible)),
                "random_expected_catastrophic_rate": random_expected,
                "issued_catastrophic_rate": issued_risk,
                "risk_reduction_fraction": 1.0 - issued_risk / random_expected,
            }
        )
    return pd.DataFrame(records)


def _ranked_policy_rows(
    eligible: pd.DataFrame,
    *,
    protocol_id: str,
    metric: str,
) -> pd.DataFrame:
    ranked = eligible.copy()
    numeric = pd.to_numeric(ranked[metric], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SyntheticProtocolError(f"Eligible {metric} values must be finite")
    ranked["_ranking_metric"] = numeric
    ranked["_tie_digest"] = [
        _prefix_tie_digest(protocol_id, str(value))
        for value in ranked["canonical_prefix_content_sha256"]
    ]
    if ranked["_tie_digest"].duplicated().any():
        raise SyntheticProtocolError("Prefix-content tie-break digest collision")
    return ranked.sort_values(
        ["_ranking_metric", "_tie_digest"], kind="stable"
    ).reset_index(drop=True)


def _rejection_policy_metrics(test: pd.DataFrame, *, protocol_id: str) -> pd.DataFrame:
    eligible = test.loc[test["hard_eligible"]].copy()
    _validate_distinct_prefix_hashes(eligible, context="Rejection-policy eligible pool")
    total = len(test)
    random_expected = (
        float(eligible["catastrophic_error"].mean()) if not eligible.empty else None
    )
    policies: tuple[tuple[str, str], ...] = (
        ("structure_disagreement", "disagreement_score_pp"),
        ("prefix_fit_error", "best_prefix_rmse_pp"),
    )
    records: list[dict[str, Any]] = []
    for policy_id, metric in policies:
        ranked = (
            _ranked_policy_rows(eligible, protocol_id=protocol_id, metric=metric)
            if not eligible.empty
            else eligible
        )
        for fraction in SECONDARY_ISSUANCE_FRACTIONS:
            target = int(round(total * fraction))
            evaluable = len(ranked) >= target and target > 0
            selected = ranked.iloc[:target] if evaluable else ranked.iloc[:0]
            risk = float(selected["catastrophic_error"].mean()) if evaluable else None
            reduction = (
                1.0 - risk / random_expected
                if risk is not None
                and random_expected is not None
                and random_expected > 0.0
                else None
            )
            records.append(
                {
                    "policy_id": policy_id,
                    "issuance_fraction_of_all_test_clusters": fraction,
                    "target_issued_count": target,
                    "hard_eligible_count": int(len(eligible)),
                    "evaluable": evaluable,
                    "issued_count": int(len(selected)),
                    "catastrophic_count": (
                        int(selected["catastrophic_error"].sum()) if evaluable else None
                    ),
                    "catastrophic_rate": risk,
                    "analytic_random_expected_catastrophic_rate": random_expected,
                    "risk_reduction_vs_analytic_random_fraction": reduction,
                }
            )
    return pd.DataFrame(records)


def _risk_coverage_curve(candidate: pd.DataFrame, *, protocol_id: str) -> pd.DataFrame:
    eligible = candidate.loc[candidate["hard_eligible"]].copy()
    columns = (
        "issued_count",
        "coverage_fraction_of_all_test_clusters",
        "coverage_fraction_of_hard_eligible_clusters",
        "catastrophic_count",
        "catastrophic_rate",
        "mean_trajectory_iae_pp",
        "maximum_issued_disagreement_pp",
        "predeclared_25_50_75_marker",
    )
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    ranked = _ranked_policy_rows(
        eligible, protocol_id=protocol_id, metric="disagreement_score_pp"
    )
    catastrophic = ranked["catastrophic_error"].to_numpy(dtype=int)
    iae = ranked["trajectory_iae_pp"].to_numpy(dtype=float)
    disagreement = ranked["disagreement_score_pp"].to_numpy(dtype=float)
    if not np.isfinite(iae).all():
        raise SyntheticProtocolError("Eligible candidate IAE must be finite")
    cumulative_catastrophic = np.cumsum(catastrophic)
    cumulative_iae = np.cumsum(iae)
    total = len(candidate)
    eligible_count = len(ranked)
    marked_counts = {
        int(round(total * fraction)): fraction
        for fraction in SECONDARY_ISSUANCE_FRACTIONS
    }
    return pd.DataFrame(
        [
            {
                "issued_count": index,
                "coverage_fraction_of_all_test_clusters": index / total,
                "coverage_fraction_of_hard_eligible_clusters": index / eligible_count,
                "catastrophic_count": int(cumulative_catastrophic[index - 1]),
                "catastrophic_rate": float(cumulative_catastrophic[index - 1] / index),
                "mean_trajectory_iae_pp": float(cumulative_iae[index - 1] / index),
                "maximum_issued_disagreement_pp": float(disagreement[index - 1]),
                "predeclared_25_50_75_marker": marked_counts.get(index),
            }
            for index in range(1, eligible_count + 1)
        ],
        columns=columns,
    )


def _family_metrics(test: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    slices: list[tuple[str, pd.DataFrame]] = [("__all__", test)]
    slices.extend(
        (str(family), group)
        for family, group in test.groupby("truth_family", sort=True)
    )
    for family, rows in slices:
        finite = rows.loc[rows["finite_forecast"]]
        eligible = rows.loc[rows["hard_eligible"]]
        issued = rows.loc[rows["primary_issued"]]
        random_expected_risk = (
            float(eligible["catastrophic_error"].mean())
            if not eligible.empty
            else math.nan
        )
        issued_risk = (
            float(issued["catastrophic_error"].mean()) if not issued.empty else math.nan
        )
        risk_reduction = (
            1.0 - issued_risk / random_expected_risk
            if math.isfinite(issued_risk)
            and math.isfinite(random_expected_risk)
            and random_expected_risk > 0.0
            else math.nan
        )
        reversal_evaluable = bool(
            math.isfinite(issued_risk) and math.isfinite(random_expected_risk)
        )
        finite_disagreement = pd.to_numeric(
            rows.loc[
                np.isfinite(rows["disagreement_score_pp"]), "disagreement_score_pp"
            ],
            errors="coerce",
        )
        records.append(
            {
                "truth_family": family,
                "cluster_count": int(len(rows)),
                "finite_forecast_count": int(len(finite)),
                "unavailable_forecast_count": int(len(rows) - len(finite)),
                "hard_eligible_count": int(len(eligible)),
                "issued_count": int(len(issued)),
                "issued_catastrophic_count": int(issued["catastrophic_error"].sum()),
                "issued_catastrophic_rate": issued_risk,
                "hard_eligible_catastrophic_count": int(
                    eligible["catastrophic_error"].sum()
                ),
                "hard_eligible_catastrophic_rate": random_expected_risk,
                "analytic_random_expected_catastrophic_rate": random_expected_risk,
                "issued_vs_analytic_random_risk_reduction_fraction": risk_reduction,
                "family_specific_reversal_evaluable": reversal_evaluable,
                "family_specific_reversal": bool(
                    reversal_evaluable and issued_risk > random_expected_risk
                ),
                "endpoint_absolute_error_pp_median_among_finite": (
                    float(finite["endpoint_absolute_error_pp"].median())
                    if not finite.empty
                    else math.nan
                ),
                "trajectory_iae_pp_mean_among_finite": (
                    float(finite["trajectory_iae_pp"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "finite_disagreement_count": int(len(finite_disagreement)),
                "disagreement_score_pp_median_among_finite": (
                    float(finite_disagreement.median())
                    if not finite_disagreement.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _family_specific_reversal_records(
    family_metrics: pd.DataFrame, *, partition: str
) -> list[dict[str, Any]]:
    required = {
        "truth_family",
        "hard_eligible_count",
        "issued_count",
        "issued_catastrophic_count",
        "issued_catastrophic_rate",
        "analytic_random_expected_catastrophic_rate",
        "issued_vs_analytic_random_risk_reduction_fraction",
        "family_specific_reversal",
    }
    _require_columns(family_metrics, required, context="Truth-family metrics")
    reversals = family_metrics.loc[
        family_metrics["truth_family"].ne("__all__")
        & family_metrics["family_specific_reversal"].eq(True)  # noqa: E712
    ].copy()
    if reversals.empty:
        return []
    reversals.insert(0, "partition", partition)
    return _json_safe_records(
        reversals.loc[
            :,
            [
                "partition",
                "truth_family",
                "hard_eligible_count",
                "issued_count",
                "issued_catastrophic_count",
                "issued_catastrophic_rate",
                "analytic_random_expected_catastrophic_rate",
                "issued_vs_analytic_random_risk_reduction_fraction",
                "family_specific_reversal",
            ],
        ]
    )


def _forecast_day_metrics(point_scores: pd.DataFrame) -> pd.DataFrame:
    test = point_scores.loc[point_scores["partition"].eq("test")].copy()
    records: list[dict[str, Any]] = []
    for model_id, forecast_column in MODEL_FORECAST_COLUMNS.items():
        test["_forecast"] = pd.to_numeric(test[forecast_column], errors="coerce")
        test["_absolute_error"] = (
            test["_forecast"]
            - pd.to_numeric(test["latent_retention_pct"], errors="coerce")
        ).abs()
        test["_noisy_absolute_error"] = (
            test["_forecast"]
            - pd.to_numeric(test["noisy_retention_pct"], errors="coerce")
        ).abs()
        slices: list[tuple[str, pd.DataFrame]] = [("__all__", test)]
        slices.extend(
            (str(family), group)
            for family, group in test.groupby("truth_family", sort=True)
        )
        for family, rows in slices:
            for day, group in rows.groupby("forecast_day", sort=True):
                finite = group.loc[np.isfinite(group["_forecast"])]
                records.append(
                    {
                        "truth_family": family,
                        "model_id": model_id,
                        "forecast_day": float(day),
                        "cluster_count": int(len(group)),
                        "finite_forecast_count": int(len(finite)),
                        "unavailable_forecast_count": int(len(group) - len(finite)),
                        "mean_absolute_error_pp_among_finite": (
                            float(finite["_absolute_error"].mean())
                            if not finite.empty
                            else math.nan
                        ),
                        "median_absolute_error_pp_among_finite": (
                            float(finite["_absolute_error"].median())
                            if not finite.empty
                            else math.nan
                        ),
                        "p90_absolute_error_pp_among_finite": (
                            float(finite["_absolute_error"].quantile(0.90))
                            if not finite.empty
                            else math.nan
                        ),
                        "mean_noisy_absolute_error_pp_among_finite": (
                            float(finite["_noisy_absolute_error"].mean())
                            if not finite.empty
                            else math.nan
                        ),
                        "median_noisy_absolute_error_pp_among_finite": (
                            float(finite["_noisy_absolute_error"].median())
                            if not finite.empty
                            else math.nan
                        ),
                    }
                )
    return (
        pd.DataFrame(records)
        .sort_values(["truth_family", "model_id", "forecast_day"], kind="stable")
        .reset_index(drop=True)
    )


def _declared_candidate_variant_count(
    protocol: ValidatedSyntheticProtocol,
) -> int:
    specs = protocol.candidate_config()["structure_member_specs"]
    total = 0
    for spec in specs:
        fixed_grid = spec.get("fixed_grid")
        if fixed_grid is None:
            total += 1
            continue
        if not isinstance(fixed_grid, dict) or not fixed_grid:
            raise SyntheticProtocolError("Candidate fixed grid must be non-empty")
        grid_variant_count = 1
        for name, values in fixed_grid.items():
            if not isinstance(values, list) or not values:
                raise SyntheticProtocolError(
                    f"Candidate fixed grid {name} must contain values"
                )
            grid_variant_count *= len(values)
        total += grid_variant_count
    if total <= 0:
        raise SyntheticProtocolError("Candidate must declare at least one variant")
    return total


def _member_fit_metrics(
    metrics: pd.DataFrame, *, declared_variant_count: int
) -> pd.DataFrame:
    if declared_variant_count <= 0:
        raise SyntheticProtocolError(
            "Declared candidate variant count must be positive"
        )
    candidate = metrics.loc[metrics["model_id"].eq("candidate")].copy()
    if candidate.empty:
        raise SyntheticProtocolError("Candidate fit metrics cannot be empty")
    failure_counts = pd.to_numeric(
        candidate["fit_failure_count"], errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(failure_counts).all()
        or np.any(failure_counts < 0.0)
        or np.any(failure_counts != np.floor(failure_counts))
        or np.any(failure_counts > declared_variant_count)
    ):
        raise SyntheticProtocolError(
            "Candidate fit failure counts exceed the declared variant inventory"
        )
    candidate["fit_failure_status"] = np.where(
        candidate["fit_failure_count"].gt(0), "one_or_more_failures", "no_failures"
    )
    result = (
        candidate.groupby(
            [
                "partition",
                "truth_family",
                "credible_structure_family_count",
                "fit_failure_status",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            cluster_count=("cluster_id", "count"),
            total_failed_variant_count=("fit_failure_count", "sum"),
            hard_eligible_count=("hard_eligible", "sum"),
            issued_count=("primary_issued", "sum"),
        )
        .reset_index()
    )
    result["declared_variant_count_per_cluster"] = declared_variant_count
    result["total_declared_variant_fit_count"] = (
        result["cluster_count"] * declared_variant_count
    )
    result["failed_variant_fit_rate"] = (
        result["total_failed_variant_count"]
        / result["total_declared_variant_fit_count"]
    )
    result["cluster_with_any_fit_failure_count"] = np.where(
        result["fit_failure_status"].eq("one_or_more_failures"),
        result["cluster_count"],
        0,
    )
    result["cluster_with_any_fit_failure_rate"] = (
        result["cluster_with_any_fit_failure_count"] / result["cluster_count"]
    )
    return result


def _model_fit_failure_summary(
    metrics: pd.DataFrame, *, declared_variant_count: int
) -> list[dict[str, Any]]:
    candidate = metrics.loc[metrics["model_id"].eq("candidate")].copy()
    records: list[dict[str, Any]] = []
    slices: list[tuple[str, pd.DataFrame]] = [("__all__", candidate)]
    slices.extend(
        (str(partition), group)
        for partition, group in candidate.groupby("partition", sort=True)
    )
    for partition, rows in slices:
        failure_counts = rows["fit_failure_count"].to_numpy(dtype=int)
        cluster_count = int(len(rows))
        declared_fit_count = cluster_count * declared_variant_count
        failed_fit_count = int(failure_counts.sum())
        clusters_with_failure = int(np.count_nonzero(failure_counts))
        records.append(
            {
                "partition": partition,
                "cluster_count": cluster_count,
                "declared_variant_count_per_cluster": declared_variant_count,
                "total_declared_variant_fit_count": declared_fit_count,
                "failed_variant_fit_count": failed_fit_count,
                "failed_variant_fit_rate": (
                    float(failed_fit_count / declared_fit_count)
                    if declared_fit_count
                    else math.nan
                ),
                "cluster_with_any_fit_failure_count": clusters_with_failure,
                "cluster_with_any_fit_failure_rate": (
                    float(clusters_with_failure / cluster_count)
                    if cluster_count
                    else math.nan
                ),
            }
        )
    return records


def _matched_model_failure_summary(pair_scores: pd.DataFrame) -> dict[str, Any]:
    if pair_scores.empty:
        return {
            "evaluated_pair_row_count": 0,
            "evaluated_member_count": 0,
            "nonfinite_disagreement_member_count": 0,
            "model_failure_member_count": 0,
            "pair_with_any_nonfinite_disagreement_count": 0,
            "pair_with_any_model_failure_count": 0,
            "model_failure_definition": "nonfinite_disagreement_score",
        }
    required = {
        "left_disagreement_score_pp",
        "right_disagreement_score_pp",
    }
    _require_columns(pair_scores, required, context="Matched pair scores")
    left = pd.to_numeric(
        pair_scores["left_disagreement_score_pp"], errors="coerce"
    ).to_numpy(dtype=float)
    right = pd.to_numeric(
        pair_scores["right_disagreement_score_pp"], errors="coerce"
    ).to_numpy(dtype=float)
    left_failure = ~np.isfinite(left)
    right_failure = ~np.isfinite(right)
    failed_member_count = int(left_failure.sum() + right_failure.sum())
    failed_pair_count = int(np.count_nonzero(left_failure | right_failure))
    return {
        "evaluated_pair_row_count": int(len(pair_scores)),
        "evaluated_member_count": int(2 * len(pair_scores)),
        "nonfinite_disagreement_member_count": failed_member_count,
        "model_failure_member_count": failed_member_count,
        "pair_with_any_nonfinite_disagreement_count": failed_pair_count,
        "pair_with_any_model_failure_count": failed_pair_count,
        "model_failure_definition": "nonfinite_disagreement_score",
    }


def _noise_sensitivity_metrics(test: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    slices: list[tuple[str, pd.DataFrame]] = [("__all__", test)]
    slices.extend(
        (str(family), group)
        for family, group in test.groupby("truth_family", sort=True)
    )
    for family, rows in slices:
        finite = rows.loc[rows["finite_forecast"]]
        records.append(
            {
                "truth_family": family,
                "cluster_count": int(len(rows)),
                "finite_forecast_count": int(len(finite)),
                "latent_catastrophic_rate": (
                    float(finite["catastrophic_error"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "noisy_catastrophic_rate": (
                    float(finite["noisy_catastrophic_error"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "noisy_minus_latent_catastrophic_rate": (
                    float(
                        finite["noisy_catastrophic_error"].mean()
                        - finite["catastrophic_error"].mean()
                    )
                    if not finite.empty
                    else math.nan
                ),
                "latent_mean_trajectory_iae_pp": (
                    float(finite["trajectory_iae_pp"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "noisy_mean_trajectory_iae_pp": (
                    float(finite["noisy_trajectory_iae_pp"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "noisy_minus_latent_mean_trajectory_iae_pp": (
                    float(
                        finite["noisy_trajectory_iae_pp"].mean()
                        - finite["trajectory_iae_pp"].mean()
                    )
                    if not finite.empty
                    else math.nan
                ),
                "latent_mean_endpoint_absolute_error_pp": (
                    float(finite["endpoint_absolute_error_pp"].mean())
                    if not finite.empty
                    else math.nan
                ),
                "noisy_mean_endpoint_absolute_error_pp": (
                    float(finite["noisy_endpoint_absolute_error_pp"].mean())
                    if not finite.empty
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def _disagreement_auroc(test: pd.DataFrame) -> tuple[float | None, int]:
    pool = test.loc[
        test["hard_eligible"]
        & test["finite_forecast"]
        & np.isfinite(test["disagreement_score_pp"])
    ]
    if pool.empty or pool["catastrophic_error"].nunique() != 2:
        return None, int(len(pool))
    return (
        float(
            roc_auc_score(
                pool["catastrophic_error"].astype(int),
                pool["disagreement_score_pp"].astype(float),
            )
        ),
        int(len(pool)),
    )


def _baseline_iae_endpoint(
    metrics: pd.DataFrame,
    issued_test: pd.DataFrame,
    strongest_baseline: str | None,
) -> dict[str, Any]:
    if len(issued_test) != TEST_ISSUE_COUNT:
        return {
            "evaluable": False,
            "unavailable_reason": "candidate_issuance_count_not_500",
            "strongest_calibration_baseline": strongest_baseline,
            "candidate_issued_mean_trajectory_iae_pp": None,
            "baseline_on_same_issued_clusters_mean_trajectory_iae_pp": None,
            "candidate_minus_baseline_iae_pp": None,
        }
    if not issued_test["finite_forecast"].all():
        return {
            "evaluable": False,
            "unavailable_reason": "candidate_issued_trajectory_iae_unavailable",
            "strongest_calibration_baseline": strongest_baseline,
            "candidate_issued_mean_trajectory_iae_pp": None,
            "baseline_on_same_issued_clusters_mean_trajectory_iae_pp": None,
            "candidate_minus_baseline_iae_pp": None,
        }
    if strongest_baseline is None:
        return {
            "evaluable": False,
            "unavailable_reason": "calibration_baseline_selection_unavailable",
            "strongest_calibration_baseline": None,
            "candidate_issued_mean_trajectory_iae_pp": float(
                issued_test["trajectory_iae_pp"].mean()
            ),
            "baseline_on_same_issued_clusters_mean_trajectory_iae_pp": None,
            "candidate_minus_baseline_iae_pp": None,
        }
    issued_ids = set(issued_test["cluster_id"].astype(str))
    selected = metrics.loc[
        metrics["partition"].eq("test")
        & metrics["model_id"].eq(strongest_baseline)
        & metrics["cluster_id"].astype(str).isin(issued_ids)
    ]
    if (
        len(selected) != TEST_ISSUE_COUNT
        or selected["cluster_id"].duplicated().any()
        or not selected["finite_forecast"].all()
    ):
        return {
            "evaluable": False,
            "unavailable_reason": "issued_baseline_trajectory_iae_unavailable",
            "strongest_calibration_baseline": strongest_baseline,
            "candidate_issued_mean_trajectory_iae_pp": float(
                issued_test["trajectory_iae_pp"].mean()
            ),
            "baseline_on_same_issued_clusters_mean_trajectory_iae_pp": None,
            "candidate_minus_baseline_iae_pp": None,
        }
    candidate_mean = float(issued_test["trajectory_iae_pp"].mean())
    baseline_mean = float(selected["trajectory_iae_pp"].mean())
    return {
        "evaluable": True,
        "unavailable_reason": None,
        "strongest_calibration_baseline": strongest_baseline,
        "candidate_issued_mean_trajectory_iae_pp": candidate_mean,
        "baseline_on_same_issued_clusters_mean_trajectory_iae_pp": baseline_mean,
        "candidate_minus_baseline_iae_pp": candidate_mean - baseline_mean,
    }


def _validate_protocol_rules(protocol: ValidatedSyntheticProtocol) -> None:
    decision = protocol.decision_config()
    candidate = protocol.candidate_config()
    endpoint = protocol.endpoint_config()
    seeds = dict(protocol.partition_seed_roots)
    if (
        int(
            candidate["primary_issuance_policy"]["required_eligible_test_cluster_count"]
        )
        != TEST_ISSUE_COUNT
        or int(decision["minimum_eligible_audit_cluster_count"]) != AUDIT_ISSUE_COUNT
        or int(endpoint["bootstrap"]["resamples"]) != BOOTSTRAP_RESAMPLES
        or int(endpoint["bootstrap"]["seed_root"]) != BOOTSTRAP_SEED
        or int(seeds["random_rejection_comparator"]) != RANDOM_REJECTION_SEED
    ):
        raise SyntheticProtocolError("Analysis constants drifted from frozen v1")


def analyze_synthetic_identifiability(
    score: FrozenScoreResult,
    matched_audit: MatchedPairAuditResult,
    protocol: ValidatedSyntheticProtocol,
) -> SyntheticAnalysisResult:
    """Evaluate frozen v1 gates and preserve every unavailable result explicitly."""
    _validate_protocol_rules(protocol)
    expected_days = tuple(protocol.forecast_days)
    observed_days = tuple(
        sorted(pd.to_numeric(score.point_scores["forecast_day"]).unique())
    )
    if observed_days != expected_days:
        raise SyntheticProtocolError("Analysis forecast grid differs from frozen v1")

    metrics = build_model_metrics(score.point_scores)
    strongest_baseline, baseline_summary = select_strongest_calibration_baseline(
        metrics
    )
    test = _candidate_trajectories(metrics, "test")
    audit = _candidate_trajectories(metrics, "audit")
    calibration = _candidate_trajectories(metrics, "calibration")
    test_summary = _partition_policy_summary(test, issue_count=TEST_ISSUE_COUNT)
    audit_summary = _partition_policy_summary(audit, issue_count=AUDIT_ISSUE_COUNT)

    eligible_test = test.loc[test["hard_eligible"]].copy()
    random_distribution = _random_rejection_distribution(
        eligible_test,
        issue_count=TEST_ISSUE_COUNT,
        ranking_count=RANDOM_RANKING_COUNT,
    )
    defined_random = random_distribution.loc[
        random_distribution["status"].eq("defined")
    ]
    random_fully_defined = len(defined_random) == RANDOM_RANKING_COUNT
    random_mean = (
        float(defined_random["catastrophic_rate"].mean())
        if random_fully_defined
        else None
    )
    issued_test = test.loc[test["primary_issued"]].copy()
    issued_risk = (
        float(issued_test["catastrophic_error"].mean())
        if len(issued_test) == TEST_ISSUE_COUNT
        else None
    )
    main_reduction = (
        1.0 - issued_risk / random_mean
        if issued_risk is not None and random_mean is not None and random_mean > 0.0
        else None
    )

    bootstrap = _bootstrap_risk_reduction(
        test, issue_count=TEST_ISSUE_COUNT, resamples=BOOTSTRAP_RESAMPLES
    )
    defined_bootstrap = bootstrap.loc[bootstrap["status"].eq("defined")]
    bootstrap_fully_defined = len(defined_bootstrap) == BOOTSTRAP_RESAMPLES
    lower_bound = (
        float(
            np.quantile(
                defined_bootstrap["risk_reduction_fraction"].to_numpy(dtype=float),
                0.05,
                method="linear",
            )
        )
        if bootstrap_fully_defined
        else None
    )

    iae_endpoint = _baseline_iae_endpoint(metrics, issued_test, strongest_baseline)
    finite_fraction = float(test["finite_forecast"].mean())
    minimum_count_reasons: list[str] = []
    if test_summary["hard_eligible_catastrophic_count"] < 30:
        minimum_count_reasons.append("fewer_than_30_hard_eligible_test_catastrophes")
    if test_summary["hard_eligible_count"] < TEST_ISSUE_COUNT:
        minimum_count_reasons.append("fewer_than_500_hard_eligible_test_clusters")
    if int(calibration["hard_eligible"].sum()) < 250:
        minimum_count_reasons.append(
            "fewer_than_250_hard_eligible_calibration_clusters"
        )
    if audit_summary["hard_eligible_count"] < AUDIT_ISSUE_COUNT:
        minimum_count_reasons.append("fewer_than_250_hard_eligible_audit_clusters")
    if matched_audit.qualified_pair_count < 200:
        minimum_count_reasons.append("fewer_than_200_qualified_matched_pairs")
    if finite_fraction < 0.95:
        minimum_count_reasons.append("fewer_than_95_percent_finite_test_forecasts")
    inconclusive_reasons = list(minimum_count_reasons)
    if not matched_audit.endpoint_available and (
        matched_audit.unavailable_reason not in inconclusive_reasons
    ):
        inconclusive_reasons.append(
            matched_audit.unavailable_reason or "matched_endpoint_unavailable"
        )
    if not bootstrap_fully_defined:
        inconclusive_reasons.append("bootstrap_lower_bound_undefined")

    audit_reduction = audit_summary["analytic_risk_reduction_fraction"]
    primary_gates = {
        "catastrophic_risk_reduction_at_50_percent_issuance": bool(
            main_reduction is not None
            and main_reduction >= 0.30
            and lower_bound is not None
            and lower_bound > 0.0
        ),
        "matched_prefix_both_members_rejected": bool(
            matched_audit.endpoint_available
            and matched_audit.qualified_pair_count >= 200
            and math.isfinite(float(matched_audit.both_rejected_fraction))
            and matched_audit.both_rejected_fraction >= 0.80
        ),
        "issued_trajectory_iae_noninferiority": bool(
            iae_endpoint["evaluable"]
            and float(iae_endpoint["candidate_minus_baseline_iae_pp"]) <= 0.10
        ),
    }
    safety_gates = {
        "minimum_counts_and_finite_forecasts": not minimum_count_reasons,
        "audit_directional_consistency": bool(
            audit_reduction is not None and float(audit_reduction) > 0.0
        ),
        "random_rankings_fully_defined": random_fully_defined,
        "bootstrap_fully_defined": bootstrap_fully_defined,
    }
    if inconclusive_reasons:
        status = "inconclusive_not_success"
    elif all(primary_gates.values()) and all(safety_gates.values()):
        status = "success"
    else:
        status = "failure"

    family_metrics = _family_metrics(test)
    audit_family_metrics = _family_metrics(audit)
    family_specific_reversals = [
        *_family_specific_reversal_records(family_metrics, partition="test"),
        *_family_specific_reversal_records(audit_family_metrics, partition="audit"),
    ]
    forecast_day_metrics = _forecast_day_metrics(score.point_scores)
    risk_coverage = _risk_coverage_curve(test, protocol_id=protocol.protocol_id)
    rejection_metrics = _rejection_policy_metrics(
        test, protocol_id=protocol.protocol_id
    )
    declared_variant_count = _declared_candidate_variant_count(protocol)
    member_fit_metrics = _member_fit_metrics(
        metrics, declared_variant_count=declared_variant_count
    )
    model_fit_failure_summary = _model_fit_failure_summary(
        metrics, declared_variant_count=declared_variant_count
    )
    matched_model_failures = _matched_model_failure_summary(matched_audit.pair_scores)
    noise_sensitivity_metrics = _noise_sensitivity_metrics(test)
    auc, auc_count = _disagreement_auroc(test)
    risk_coverage_area = (
        float(risk_coverage["catastrophic_rate"].mean())
        if not risk_coverage.empty
        else None
    )

    report: dict[str, Any] = {
        "status": status,
        "protocol_id": protocol.protocol_id,
        "config_canonical_sha256": protocol.config_sha256,
        "evidence_role": "synthetic_mechanism_identifiability_stress_test_only",
        "inconclusive_reasons": inconclusive_reasons,
        "primary_gates": primary_gates,
        "required_safety_gates": safety_gates,
        "test_policy": {
            **test_summary,
            "published_random_ranking_count": RANDOM_RANKING_COUNT,
            "random_rankings_fully_defined": random_fully_defined,
            "published_random_mean_catastrophic_rate": random_mean,
            "risk_reduction_fraction": main_reduction,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_defined_resample_count": int(len(defined_bootstrap)),
            "bootstrap_one_sided_95pct_lower_bound": lower_bound,
        },
        "matched_prefix_audit": {
            "endpoint_available": bool(matched_audit.endpoint_available),
            "unavailable_reason": matched_audit.unavailable_reason,
            "qualified_pair_count": int(matched_audit.qualified_pair_count),
            **matched_model_failures,
            "calibration_disagreement_threshold_pp": _finite_or_none(
                matched_audit.calibration_disagreement_threshold_pp
            ),
            "both_rejected_pair_count": int(matched_audit.both_rejected_pair_count),
            "both_rejected_fraction": _finite_or_none(
                matched_audit.both_rejected_fraction
            ),
        },
        "mean_forecast_comparison": iae_endpoint,
        "calibration_baseline_selection": _json_safe_records(baseline_summary),
        "audit_policy": audit_summary,
        "secondary": {
            "disagreement_auroc_for_catastrophic_error": auc,
            "disagreement_auroc_hard_eligible_finite_cluster_count": auc_count,
            "finite_test_forecast_fraction": finite_fraction,
            "risk_coverage_area_mean_cumulative_catastrophic_rate": (
                risk_coverage_area
            ),
            "maximum_evaluable_test_coverage_fraction": (
                float(len(eligible_test) / len(test))
            ),
            "issuance_25_50_75": _json_safe_records(rejection_metrics),
            "test_family_metrics": _json_safe_records(family_metrics),
            "audit_family_metrics": _json_safe_records(audit_family_metrics),
            "family_specific_reversal_count": len(family_specific_reversals),
            "family_specific_reversals": family_specific_reversals,
            "credible_member_and_fit_failure_slices": _json_safe_records(
                member_fit_metrics.loc[member_fit_metrics["partition"].eq("test")]
            ),
            "model_fit_failure_summary": _json_safe_records(
                pd.DataFrame(model_fit_failure_summary)
            ),
            "noise_free_versus_noisy_future_sensitivity": _json_safe_records(
                noise_sensitivity_metrics
            ),
        },
        "claim_boundary": (
            "This result is a frozen synthetic mechanism stress test. It does not "
            "validate real LFP, Hithium product, individual-cell, storage-station, "
            "operational interval, or 15-25 year accuracy claims."
        ),
    }
    _assert_json_finite(report)
    return SyntheticAnalysisResult(
        report=report,
        model_metrics=metrics,
        family_metrics=family_metrics,
        forecast_day_metrics=forecast_day_metrics,
        risk_coverage=risk_coverage,
        rejection_policy_metrics=rejection_metrics,
        member_fit_metrics=member_fit_metrics,
        noise_sensitivity_metrics=noise_sensitivity_metrics,
        random_rejection=random_distribution,
        bootstrap=bootstrap,
    )
