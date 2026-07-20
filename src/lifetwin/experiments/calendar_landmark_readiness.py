from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from collections.abc import Mapping

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
    GLOBAL_LANDMARK_POLICY,
)
from lifetwin.experiments.calendar_v3_activation_development import (
    EXPECTED_PREFIXES,
    GATE_SCENARIOS,
    PRIMARY_CANDIDATE,
    PRIMARY_COMPARATOR,
    calendar_v3_prediction_sha256,
    run_calendar_v3_activation_development,
    score_calendar_v3_predictions,
)


EXPERIMENT_ID = "naumann_calendar_landmark_readiness_locked_diagnostic_v1"
DESIGN_STATUS = "retrospective_locked_after_outcome_access"
EVIDENCE_ROLE = "reused_naumann_landmark_readiness_diagnostic_only"
DATASET_RELATIONSHIP = "reused_and_outcomes_already_inspected"
CONFIRMATION_STATUS = "blocked_reused_outcomes"
COMMON_SUPPORT_POLICY = "fixed_checkup_index_closed_interval"
COMMON_SUPPORT_START_CHECKUP = 14
COMMON_SUPPORT_END_CHECKUP = 34
CONDITION_REGRESSION_TOLERANCE_PP = 1e-12
# This digest binds the parsed, validated outcome values rather than only the
# source filename. It is populated from the frozen public Naumann CSV.
EXPECTED_CANONICAL_OUTCOME_SHA256 = (
    "17d117b8e6df3a033209971db9b82c5e9ca2d5a7ca89f45e484737196371b7eb"
)
EXPECTED_PROHIBITED_CLAIMS = (
    "confirmed_earliest_landmark_on_naumann_reuse",
    "confirmatory_superiority_on_naumann_reuse",
    "independent_external_validation",
    "hithium_product_accuracy",
    "utility_scale_storage_validation",
    "15_to_25_year_extrapolation",
)

TOP_LEVEL_PROTOCOL_KEYS = {
    "experiment_id",
    "design_status",
    "dataset_id",
    "dataset_snapshot_id",
    "label_version",
    "statistical_unit",
    "evidence_role",
    "dataset_relationship",
    "training_history_policy",
    "landmark_prefix_checkups",
    "scenarios",
    "candidate_method",
    "comparator_method",
    "common_support",
    "retrospective_signal_rule",
    "confirmation_policy",
    "prohibited_claims",
}

COMMON_SUPPORT_METRIC_COLUMNS = [
    "scenario",
    "fold_id",
    "target_condition_id",
    "prefix_checkups",
    "prefix_end_checkup_index",
    "prefix_end_days",
    "method",
    "training_state_sha256",
    "prediction_state_sha256",
    "common_support_start_checkup_index",
    "common_support_end_checkup_index",
    "common_support_point_count",
    "common_support_start_days",
    "common_support_end_days",
    "common_support_trajectory_iae_pp",
    "common_support_point_mae_pp",
    "common_support_final_true_retention_pct",
    "common_support_final_predicted_retention_pct",
    "common_support_final_error_pp",
    "common_support_final_absolute_error_pp",
]


def default_landmark_readiness_protocol() -> dict[str, object]:
    """Return a fresh copy of the locked retrospective protocol."""
    return {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "evidence_role": EVIDENCE_ROLE,
        "dataset_relationship": DATASET_RELATIONSHIP,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "landmark_prefix_checkups": list(EXPECTED_PREFIXES),
        "scenarios": list(GATE_SCENARIOS),
        "candidate_method": PRIMARY_CANDIDATE,
        "comparator_method": PRIMARY_COMPARATOR,
        "common_support": {
            "policy": COMMON_SUPPORT_POLICY,
            "start_checkup_index": COMMON_SUPPORT_START_CHECKUP,
            "end_checkup_index": COMMON_SUPPORT_END_CHECKUP,
        },
        "retrospective_signal_rule": {
            "mean_delta_threshold_pp": 0.0,
            "condition_regression_tolerance_pp": (
                CONDITION_REGRESSION_TOLERANCE_PP
            ),
            "require_all_scenarios": True,
            "require_no_condition_regressions": True,
            "minimum_unique_improved_conditions": 1,
        },
        "confirmation_policy": {
            "required_dataset_role": "independent_external_replication",
            "current_dataset_status": CONFIRMATION_STATUS,
            "confirmed_earliest_landmark_must_be_null": True,
        },
        "prohibited_claims": list(EXPECTED_PROHIBITED_CLAIMS),
    }


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    *,
    context: str,
) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} keys must be exact: missing={missing}, unknown={unknown}"
        )


def _mapping(value: object, *, context: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return dict(value)


def _strict_int(value: object, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{context} must be an integer")
    return int(value)


def _finite_float(value: object, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{context} must be numeric")
    converted = float(value)
    if not np.isfinite(converted):
        raise ValueError(f"{context} must be finite")
    return converted


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_naumann_outcome_sha256(observations: pd.DataFrame) -> str:
    """Fingerprint every authoritative coordinate and measured outcome."""
    validate_naumann_calendar_observations(observations)
    ordered = observations.sort_values(
        ["condition_id", "checkup_index"], kind="stable"
    )
    columns = (
        "condition_id",
        "checkup_index",
        "elapsed_time_s",
        "capacity_ah",
        "capacity_retention_pct",
        "capacity_loss_pct",
        "resistance_dc_ohm",
        "resistance_growth_pct",
    )
    rows: list[list[str]] = []
    for values in ordered[list(columns)].itertuples(index=False, name=None):
        rows.append(
            [
                str(values[0]),
                str(int(values[1])),
                *(float(value).hex() for value in values[2:]),
            ]
        )
    return _canonical_json_sha256({"columns": list(columns), "rows": rows})


def validate_landmark_readiness_protocol(
    protocol: Mapping[str, object],
) -> dict[str, object]:
    """Reject any drift in the locked landmarks, support, methods, or claims."""
    parsed = _mapping(protocol, context="Landmark readiness protocol")
    _require_exact_keys(
        parsed,
        TOP_LEVEL_PROTOCOL_KEYS,
        context="Landmark readiness protocol",
    )
    exact_scalars = {
        "experiment_id": EXPERIMENT_ID,
        "design_status": DESIGN_STATUS,
        "dataset_id": NAUMANN_CALENDAR_DATASET_ID,
        "dataset_snapshot_id": EXPECTED_DATASET_SNAPSHOT_ID,
        "label_version": EXPECTED_LABEL_VERSION,
        "statistical_unit": NAUMANN_STATISTICAL_UNIT,
        "evidence_role": EVIDENCE_ROLE,
        "dataset_relationship": DATASET_RELATIONSHIP,
        "training_history_policy": GLOBAL_LANDMARK_POLICY,
        "candidate_method": PRIMARY_CANDIDATE,
        "comparator_method": PRIMARY_COMPARATOR,
    }
    for key, expected in exact_scalars.items():
        if parsed[key] != expected:
            raise ValueError(f"Landmark readiness {key} must remain {expected}")

    raw_landmarks = parsed["landmark_prefix_checkups"]
    if not isinstance(raw_landmarks, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_landmarks
    ):
        raise ValueError("landmark_prefix_checkups must be an integer list")
    if raw_landmarks != list(EXPECTED_PREFIXES):
        raise ValueError(
            f"Landmark prefixes must remain {list(EXPECTED_PREFIXES)}"
        )
    if parsed["scenarios"] != list(GATE_SCENARIOS):
        raise ValueError(f"Landmark scenarios must remain {list(GATE_SCENARIOS)}")

    common_support = _mapping(parsed["common_support"], context="common_support")
    _require_exact_keys(
        common_support,
        {"policy", "start_checkup_index", "end_checkup_index"},
        context="common_support",
    )
    if common_support["policy"] != COMMON_SUPPORT_POLICY:
        raise ValueError("Common-support policy cannot change")
    start = _strict_int(
        common_support["start_checkup_index"],
        context="common_support.start_checkup_index",
    )
    end = _strict_int(
        common_support["end_checkup_index"],
        context="common_support.end_checkup_index",
    )
    if (start, end) != (
        COMMON_SUPPORT_START_CHECKUP,
        COMMON_SUPPORT_END_CHECKUP,
    ):
        raise ValueError("Common support must remain checkups 14 through 34")

    signal_rule = _mapping(
        parsed["retrospective_signal_rule"],
        context="retrospective_signal_rule",
    )
    _require_exact_keys(
        signal_rule,
        {
            "mean_delta_threshold_pp",
            "condition_regression_tolerance_pp",
            "require_all_scenarios",
            "require_no_condition_regressions",
            "minimum_unique_improved_conditions",
        },
        context="retrospective_signal_rule",
    )
    mean_threshold = _finite_float(
        signal_rule["mean_delta_threshold_pp"],
        context="mean_delta_threshold_pp",
    )
    regression_tolerance = _finite_float(
        signal_rule["condition_regression_tolerance_pp"],
        context="condition_regression_tolerance_pp",
    )
    minimum_improved = _strict_int(
        signal_rule["minimum_unique_improved_conditions"],
        context="minimum_unique_improved_conditions",
    )
    if mean_threshold != 0.0:
        raise ValueError("The retrospective mean-delta threshold must remain zero")
    if regression_tolerance != CONDITION_REGRESSION_TOLERANCE_PP:
        raise ValueError("The condition-regression tolerance cannot change")
    if signal_rule["require_all_scenarios"] is not True:
        raise ValueError("All frozen scenarios must remain required")
    if signal_rule["require_no_condition_regressions"] is not True:
        raise ValueError("The no-condition-regression rule must remain enabled")
    if minimum_improved != 1:
        raise ValueError("At least one unique improved condition remains required")

    confirmation = _mapping(
        parsed["confirmation_policy"], context="confirmation_policy"
    )
    _require_exact_keys(
        confirmation,
        {
            "required_dataset_role",
            "current_dataset_status",
            "confirmed_earliest_landmark_must_be_null",
        },
        context="confirmation_policy",
    )
    if confirmation != {
        "required_dataset_role": "independent_external_replication",
        "current_dataset_status": CONFIRMATION_STATUS,
        "confirmed_earliest_landmark_must_be_null": True,
    }:
        raise ValueError("The reused-dataset confirmation block cannot change")
    if parsed["prohibited_claims"] != list(EXPECTED_PROHIBITED_CLAIMS):
        raise ValueError("Landmark readiness prohibited claims cannot change")
    return deepcopy(parsed)


def _require_metric_schema(frame: pd.DataFrame) -> None:
    missing = sorted(set(COMMON_SUPPORT_METRIC_COLUMNS) - set(frame.columns))
    unknown = sorted(set(frame.columns) - set(COMMON_SUPPORT_METRIC_COLUMNS))
    if missing or unknown:
        raise ValueError(
            "Common-support metric schema mismatch: "
            f"missing={missing}, unknown={unknown}"
        )
    if frame.empty:
        raise ValueError("Common-support metrics cannot be empty")
    key = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "method",
    ]
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError("Common-support metric keys must be unique and non-null")
    numeric = [
        column
        for column in COMMON_SUPPORT_METRIC_COLUMNS
        if column not in {*key, "training_state_sha256", "prediction_state_sha256"}
    ]
    converted = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if converted.isna().any().any() or not np.isfinite(converted.to_numpy()).all():
        raise ValueError("Common-support metrics must contain finite numeric values")
    for column in ("training_state_sha256", "prediction_state_sha256"):
        if not frame[column].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
            raise ValueError(f"{column} must contain canonical SHA-256 values")


def _score_common_support_landmarks(
    predictions: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    """Score every landmark on the identical authoritative checkup 14..34 window."""
    parsed = validate_landmark_readiness_protocol(protocol)
    outcome_sha256 = canonical_naumann_outcome_sha256(observations)
    if outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError(
            "Naumann authoritative outcome snapshot mismatch: "
            f"expected {EXPECTED_CANONICAL_OUTCOME_SHA256}, found {outcome_sha256}"
        )

    # The public V3 scorer is the authority for pack hash, coordinates, final flags,
    # prefix boundaries, full future support, and frozen scenario/target coverage.
    score_calendar_v3_predictions(
        predictions,
        observations,
        frozen_prediction_sha256=frozen_prediction_sha256,
    )

    support = dict(parsed["common_support"])
    start = int(support["start_checkup_index"])
    end = int(support["end_checkup_index"])
    methods = (str(parsed["candidate_method"]), str(parsed["comparator_method"]))
    selected = predictions.loc[
        predictions["scenario"].astype(str).isin(parsed["scenarios"])
        & pd.to_numeric(predictions["prefix_checkups"]).isin(
            parsed["landmark_prefix_checkups"]
        )
        & predictions["method"].astype(str).isin(methods)
        & pd.to_numeric(predictions["target_checkup_index"]).between(start, end)
    ].copy()
    if selected.empty:
        raise ValueError("No V3 predictions remain on the locked common support")

    truth = observations[
        [
            "condition_id",
            "checkup_index",
            "elapsed_days",
            "capacity_retention_pct",
        ]
    ].rename(
        columns={
            "condition_id": "target_condition_id",
            "checkup_index": "target_checkup_index",
            "elapsed_days": "truth_elapsed_days",
            "capacity_retention_pct": "true_capacity_retention_pct",
        }
    )
    if truth.duplicated(["target_condition_id", "target_checkup_index"]).any():
        raise ValueError("Authoritative outcomes must be unique by condition/checkup")
    scored = selected.merge(
        truth,
        on=["target_condition_id", "target_checkup_index"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (scored["_merge"] != "both").any() or scored[
        ["truth_elapsed_days", "true_capacity_retention_pct"]
    ].isna().any().any():
        raise ValueError("Every common-support prediction must match authoritative truth")
    scored = scored.drop(columns="_merge")
    scored["prediction_error_pp"] = (
        pd.to_numeric(scored["predicted_capacity_retention_pct"])
        - pd.to_numeric(scored["true_capacity_retention_pct"])
    )

    grouping = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_checkup_index",
        "prefix_end_days",
        "method",
        "training_state_sha256",
        "prediction_state_sha256",
    ]
    expected_indices = list(range(start, end + 1))
    rows: list[dict[str, object]] = []
    for keys, group in scored.groupby(grouping, sort=True):
        ordered = group.sort_values("target_checkup_index", kind="stable")
        observed_indices = (
            pd.to_numeric(ordered["target_checkup_index"]).astype(int).tolist()
        )
        if observed_indices != expected_indices:
            raise ValueError(
                "Every landmark trajectory must contain the complete common support"
            )
        elapsed = ordered["truth_elapsed_days"].to_numpy(dtype=float)
        absolute = np.abs(ordered["prediction_error_pp"].to_numpy(dtype=float))
        if len(elapsed) < 2 or np.any(np.diff(elapsed) <= 0.0):
            raise ValueError("Common-support time coordinates must increase")
        final = ordered.iloc[-1]
        rows.append(
            {
                **dict(zip(grouping, keys, strict=True)),
                "common_support_start_checkup_index": start,
                "common_support_end_checkup_index": end,
                "common_support_point_count": len(ordered),
                "common_support_start_days": float(elapsed[0]),
                "common_support_end_days": float(elapsed[-1]),
                "common_support_trajectory_iae_pp": float(
                    np.trapezoid(absolute, elapsed) / (elapsed[-1] - elapsed[0])
                ),
                "common_support_point_mae_pp": float(absolute.mean()),
                "common_support_final_true_retention_pct": float(
                    final["true_capacity_retention_pct"]
                ),
                "common_support_final_predicted_retention_pct": float(
                    final["predicted_capacity_retention_pct"]
                ),
                "common_support_final_error_pp": float(
                    final["prediction_error_pp"]
                ),
                "common_support_final_absolute_error_pp": abs(
                    float(final["prediction_error_pp"])
                ),
            }
        )
    metrics = pd.DataFrame(rows, columns=COMMON_SUPPORT_METRIC_COLUMNS)
    metrics = metrics.sort_values(
        ["scenario", "target_condition_id", "prefix_checkups", "method"],
        kind="stable",
    ).reset_index(drop=True)
    _require_metric_schema(metrics)
    return metrics


def _paired_common_support_metrics(
    metrics: pd.DataFrame,
    *,
    protocol: Mapping[str, object],
) -> pd.DataFrame:
    _require_metric_schema(metrics)
    metrics = metrics.copy()
    prefixes = pd.to_numeric(metrics["prefix_checkups"], errors="coerce")
    if (
        prefixes.isna().any()
        or not np.equal(prefixes, np.floor(prefixes)).all()
        or set(prefixes.astype(int)) != set(EXPECTED_PREFIXES)
    ):
        raise ValueError("Metrics must contain exactly the locked landmarks")
    metrics["prefix_checkups"] = prefixes.astype(int)
    if set(metrics["scenario"].astype(str)) != set(GATE_SCENARIOS):
        raise ValueError("Metrics must contain exactly the locked scenarios")
    expected_points = (
        COMMON_SUPPORT_END_CHECKUP - COMMON_SUPPORT_START_CHECKUP + 1
    )
    if not (
        pd.to_numeric(metrics["common_support_start_checkup_index"])
        == COMMON_SUPPORT_START_CHECKUP
    ).all() or not (
        pd.to_numeric(metrics["common_support_end_checkup_index"])
        == COMMON_SUPPORT_END_CHECKUP
    ).all():
        raise ValueError("Metric rows must retain the locked common support")
    if not (
        pd.to_numeric(metrics["common_support_point_count"]) == expected_points
    ).all():
        raise ValueError("Metric rows must contain every common-support checkup")
    if not (
        pd.to_numeric(metrics["prefix_end_checkup_index"])
        == metrics["prefix_checkups"] - 1
    ).all():
        raise ValueError("Metric prefix-end indices must match their landmarks")
    candidate_method = str(protocol["candidate_method"])
    comparator_method = str(protocol["comparator_method"])
    if set(metrics["method"].astype(str)) != {candidate_method, comparator_method}:
        raise ValueError("Metrics must contain exactly the locked candidate and comparator")
    keys = [
        "scenario",
        "fold_id",
        "target_condition_id",
        "prefix_checkups",
        "prefix_end_checkup_index",
        "prefix_end_days",
        "common_support_start_checkup_index",
        "common_support_end_checkup_index",
        "common_support_point_count",
        "common_support_start_days",
        "common_support_end_days",
    ]
    state_columns = ["training_state_sha256", "prediction_state_sha256"]
    candidate = metrics.loc[
        metrics["method"] == candidate_method,
        [
            *keys,
            *state_columns,
            "common_support_trajectory_iae_pp",
            "common_support_point_mae_pp",
            "common_support_final_absolute_error_pp",
        ],
    ].rename(
        columns={
            "training_state_sha256": "candidate_training_state_sha256",
            "prediction_state_sha256": "candidate_prediction_state_sha256",
            "common_support_trajectory_iae_pp": "candidate_trajectory_iae_pp",
            "common_support_point_mae_pp": "candidate_point_mae_pp",
            "common_support_final_absolute_error_pp": (
                "candidate_final_absolute_error_pp"
            ),
        }
    )
    comparator = metrics.loc[
        metrics["method"] == comparator_method,
        [
            *keys,
            *state_columns,
            "common_support_trajectory_iae_pp",
            "common_support_point_mae_pp",
            "common_support_final_absolute_error_pp",
        ],
    ].rename(
        columns={
            "training_state_sha256": "comparator_training_state_sha256",
            "prediction_state_sha256": "comparator_prediction_state_sha256",
            "common_support_trajectory_iae_pp": "comparator_trajectory_iae_pp",
            "common_support_point_mae_pp": "comparator_point_mae_pp",
            "common_support_final_absolute_error_pp": (
                "comparator_final_absolute_error_pp"
            ),
        }
    )
    paired = candidate.merge(
        comparator,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (paired["_merge"] != "both").any():
        raise ValueError("Every condition/landmark needs both locked methods")
    paired = paired.drop(columns="_merge")
    for state in state_columns:
        if not (
            paired[f"candidate_{state}"] == paired[f"comparator_{state}"]
        ).all():
            raise ValueError("Paired landmark methods must share frozen model state")
    paired["paired_delta_iae_pp"] = (
        paired["candidate_trajectory_iae_pp"]
        - paired["comparator_trajectory_iae_pp"]
    )
    return paired.sort_values(
        ["scenario", "prefix_checkups", "target_condition_id"],
        kind="stable",
    ).reset_index(drop=True)


def _summarize_authoritative_metrics(
    common_support_metrics: pd.DataFrame,
    *,
    protocol: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarize retrospective signals without issuing a confirmation claim."""
    parsed = validate_landmark_readiness_protocol(protocol)
    paired = _paired_common_support_metrics(
        common_support_metrics,
        protocol=parsed,
    )
    expected_scenarios = [str(value) for value in parsed["scenarios"]]
    expected_prefixes = [int(value) for value in parsed["landmark_prefix_checkups"]]
    observed_groups = set(
        zip(
            paired["scenario"].astype(str),
            pd.to_numeric(paired["prefix_checkups"]).astype(int),
            strict=False,
        )
    )
    expected_groups = {
        (scenario, prefix)
        for scenario in expected_scenarios
        for prefix in expected_prefixes
    }
    if observed_groups != expected_groups:
        raise ValueError("Landmark summary requires every frozen scenario/prefix")

    rule = dict(parsed["retrospective_signal_rule"])
    mean_threshold = float(rule["mean_delta_threshold_pp"])
    tolerance = float(rule["condition_regression_tolerance_pp"])
    summary_rows: list[dict[str, object]] = []
    for (scenario, prefix), group in paired.groupby(
        ["scenario", "prefix_checkups"], sort=True
    ):
        delta = group["paired_delta_iae_pp"].to_numpy(dtype=float)
        prefix_days = pd.to_numeric(group["prefix_end_days"]).unique()
        if len(prefix_days) != 1:
            raise ValueError("A landmark must have one authoritative prefix-end day")
        better = int(np.sum(delta < -tolerance))
        worse = int(np.sum(delta > tolerance))
        equal = int(len(delta) - better - worse)
        mean_delta = float(delta.mean())
        summary_rows.append(
            {
                "scenario": str(scenario),
                "prefix_checkups": int(prefix),
                "prefix_end_days": float(prefix_days[0]),
                "independent_condition_count": len(group),
                "candidate_trajectory_iae_pp_mean": float(
                    group["candidate_trajectory_iae_pp"].mean()
                ),
                "comparator_trajectory_iae_pp_mean": float(
                    group["comparator_trajectory_iae_pp"].mean()
                ),
                "mean_paired_delta_iae_pp": mean_delta,
                "maximum_condition_regression_pp": float(delta.max()),
                "candidate_better_condition_count": better,
                "candidate_worse_condition_count": worse,
                "candidate_equal_condition_count": equal,
                "scenario_mean_improvement_met": bool(
                    mean_delta < mean_threshold
                ),
                "scenario_no_condition_regression_met": bool(worse == 0),
                "scenario_has_observed_improvement": bool(better > 0),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["prefix_checkups", "scenario"], kind="stable"
    ).reset_index(drop=True)

    landmark_decisions: list[dict[str, object]] = []
    for prefix in expected_prefixes:
        group = summary.loc[summary["prefix_checkups"] == prefix]
        scenario_order = list(group["scenario"].astype(str))
        all_scenarios_present = set(scenario_order) == set(expected_scenarios)
        paired_prefix = paired.loc[paired["prefix_checkups"] == prefix]
        improved_ids = sorted(
            paired_prefix.loc[
                paired_prefix["paired_delta_iae_pp"] < -tolerance,
                "target_condition_id",
            ]
            .astype(str)
            .unique()
        )
        all_means_improved = bool(
            all_scenarios_present and group["scenario_mean_improvement_met"].all()
        )
        no_regressions = bool(
            all_scenarios_present
            and group["scenario_no_condition_regression_met"].all()
        )
        enough_unique_improvements = bool(
            len(improved_ids) >= int(rule["minimum_unique_improved_conditions"])
        )
        signal_met = bool(
            all_scenarios_present
            and all_means_improved
            and no_regressions
            and enough_unique_improvements
        )
        landmark_decisions.append(
            {
                "prefix_checkups": prefix,
                "all_scenarios_present": all_scenarios_present,
                "all_scenarios_mean_improvement_met": all_means_improved,
                "no_condition_regressions_met": no_regressions,
                "unique_improved_condition_count": len(improved_ids),
                "unique_improved_condition_ids": improved_ids,
                "minimum_unique_improved_conditions_met": (
                    enough_unique_improvements
                ),
                "retrospective_signal_criterion_met": signal_met,
            }
        )
    passing = [
        int(row["prefix_checkups"])
        for row in landmark_decisions
        if row["retrospective_signal_criterion_met"]
    ]
    retrospective_signal = min(passing) if passing else None
    decision: dict[str, object] = {
        "status": (
            "retrospective_signal_only_confirmation_blocked"
            if retrospective_signal is not None
            else "no_retrospective_signal_confirmation_blocked"
        ),
        "experiment_id": EXPERIMENT_ID,
        "protocol_sha256": _canonical_json_sha256(parsed),
        "dataset_relationship": DATASET_RELATIONSHIP,
        "common_support": dict(parsed["common_support"]),
        "landmark_decisions": landmark_decisions,
        "retrospective_signal_landmark": retrospective_signal,
        "confirmed_earliest_landmark": None,
        "confirmation_status": CONFIRMATION_STATUS,
        "model_validation_status": "not_confirmed",
        "reason": (
            "Naumann outcomes were inspected during model development. This layer "
            "can locate a locked retrospective signal, but only a separately frozen "
            "independent-dataset protocol can confirm an earliest landmark."
        ),
        "prohibited_claims": list(parsed["prohibited_claims"]),
    }
    return summary, decision


def run_landmark_readiness(
    observations: pd.DataFrame,
    *,
    v3_config: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Regenerate, score, and summarize the locked V3 experiment end to end."""
    parsed = validate_landmark_readiness_protocol(protocol)
    outcome_sha256 = canonical_naumann_outcome_sha256(observations)
    if outcome_sha256 != EXPECTED_CANONICAL_OUTCOME_SHA256:
        raise ValueError(
            "Naumann authoritative outcome snapshot mismatch: "
            f"expected {EXPECTED_CANONICAL_OUTCOME_SHA256}, found {outcome_sha256}"
        )
    v3_run = run_calendar_v3_activation_development(
        observations,
        config=v3_config,
    )
    predictions = v3_run[1]
    prediction_sha256 = calendar_v3_prediction_sha256(predictions)
    metrics = _score_common_support_landmarks(
        predictions,
        observations,
        frozen_prediction_sha256=prediction_sha256,
        protocol=parsed,
    )
    summary, decision = _summarize_authoritative_metrics(
        metrics,
        protocol=parsed,
    )
    decision["authoritative_outcome_sha256"] = outcome_sha256
    decision["regenerated_v3_prediction_sha256"] = prediction_sha256
    decision["v3_config_sha256"] = _canonical_json_sha256(v3_config)
    decision["prediction_input_accepted_from_caller"] = False
    return metrics, summary, decision
