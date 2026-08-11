"""Post-outcome diagnostics for the frozen SNL LFP RPT benchmark.

The functions in this module can explain a completed retrospective result, but
they are deliberately kept outside the frozen predictor and cannot tune it.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.snl import RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.snl_rpt_loco import (
    BASE_MODEL_IDS,
    DECISION_COLUMNS,
    MODEL_IDS,
    SCORE_COLUMNS,
    SELECTOR_MODEL_ID,
    TARGET_TRUTH_COLUMNS,
    SNLRPTLOCOError,
    score_snl_rpt_loco,
    validate_snl_rpt_loco_config,
)


AUDIT_SCHEMA_VERSION = "lifetwin.snl_rpt_post_outcome_audit.v1"
CELL_DIAGNOSTIC_COLUMNS = (
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "future_observation_count",
    "prefix_end_equivalent_full_cycles",
    "last_scored_equivalent_full_cycles",
    "scored_horizon_equivalent_full_cycles",
    "issued",
    "selected_expert_model_id",
    "selection_mode",
    "evidence_status",
    "persistence_trajectory_iae_pp",
    "selector_trajectory_iae_pp",
    "selector_improvement_vs_persistence_pp",
    "realized_best_base_expert_model_id",
    "realized_best_base_expert_trajectory_iae_pp",
    "selector_regret_vs_realized_best_base_expert_pp",
    "selected_expert_matches_realized_best_base_expert",
)
CONDITION_DIAGNOSTIC_COLUMNS = (
    "outer_condition_id",
    "landmark_visit_count",
    "cell_count",
    "issued_cell_count",
    "mean_future_observation_count",
    "minimum_last_scored_equivalent_full_cycles",
    "maximum_last_scored_equivalent_full_cycles",
    "persistence_trajectory_iae_pp",
    "selector_trajectory_iae_pp",
    "selector_improvement_vs_persistence_pp",
    "realized_best_fixed_base_expert_model_id",
    "realized_best_fixed_base_expert_trajectory_iae_pp",
    "selector_regret_vs_realized_best_fixed_base_expert_pp",
)
MODEL_METRIC_COLUMNS = (
    "landmark_visit_count",
    "model_id",
    "issued_cell_count",
    "issued_cell_fraction",
    "issued_condition_cluster_count",
    "cell_equal_trajectory_iae_pp",
    "condition_equal_trajectory_iae_pp",
    "cell_equal_trajectory_mae_pp",
    "condition_equal_trajectory_mae_pp",
    "cell_equal_trajectory_rmse_pp",
    "condition_equal_trajectory_rmse_pp",
    "cell_equal_endpoint_absolute_error_pp",
    "condition_equal_endpoint_absolute_error_pp",
)
SELECTOR_CHOICE_COLUMNS = (
    "landmark_visit_count",
    "issued",
    "selected_expert_model_id",
    "selection_mode",
    "evidence_status",
    "cell_count",
    "cell_fraction",
    "condition_cluster_count",
)
EXTRACTION_SENSITIVITY_COLUMNS = (
    "rest_gap_hours",
    "duplicate_visit_efc",
    "is_primary_setting",
    "canonical_rpt_trajectory_sha256",
    "exact_primary_trajectory_match",
    "physical_cell_count",
    "condition_cluster_count",
    "trajectory_row_count",
    "minimum_rpt_visit_count",
    "median_rpt_visit_count",
    "maximum_rpt_visit_count",
    "visit_count_changed_cell_count",
    "maximum_absolute_visit_count_delta",
    "matched_primary_visit_count",
    "primary_visit_match_fraction",
    "candidate_visit_match_fraction",
    "mean_absolute_matched_efc_delta",
    "maximum_absolute_matched_efc_delta",
    "mean_absolute_matched_retention_delta_pp",
    "maximum_absolute_matched_retention_delta_pp",
)


class SNLRPTPostOutcomeAuditError(ValueError):
    """Raised when a persisted SNL result or audit input is inconsistent."""


def _normalize_scores_like(
    scores: pd.DataFrame,
    replayed: pd.DataFrame,
) -> pd.DataFrame:
    if tuple(scores.columns) != SCORE_COLUMNS:
        raise SNLRPTPostOutcomeAuditError("Persisted SNL score columns changed")
    normalized = scores.copy()
    for column in SCORE_COLUMNS:
        expected = replayed[column]
        if pd.api.types.is_bool_dtype(expected.dtype):
            values = normalized[column]
            if pd.api.types.is_bool_dtype(values.dtype):
                normalized[column] = values.astype(bool)
            else:
                mapping = {"true": True, "false": False}
                converted = values.astype(str).str.lower().map(mapping)
                if converted.isna().any():
                    raise SNLRPTPostOutcomeAuditError(
                        f"Invalid persisted Boolean values: {column}"
                    )
                normalized[column] = converted.astype(bool)
        elif pd.api.types.is_integer_dtype(expected.dtype):
            numeric = pd.to_numeric(normalized[column], errors="raise")
            if not np.equal(numeric, np.floor(numeric)).all():
                raise SNLRPTPostOutcomeAuditError(
                    f"Non-integer persisted SNL score values: {column}"
                )
            normalized[column] = numeric.astype(np.int64)
        elif pd.api.types.is_float_dtype(expected.dtype):
            normalized[column] = pd.to_numeric(
                normalized[column], errors="raise"
            ).astype(float)
        else:
            normalized[column] = normalized[column].astype(str)
    return normalized


def _metric_summary(
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = (
        "trajectory_iae_pp",
        "trajectory_mae_pp",
        "trajectory_rmse_pp",
        "endpoint_absolute_error_pp",
    )
    rows: list[dict[str, object]] = []
    for landmark in sorted(scores["landmark_visit_count"].unique()):
        landmark_scores = scores.loc[scores["landmark_visit_count"] == landmark]
        denominator = int(
            decisions.loc[
                decisions["landmark_visit_count"] == landmark, "cell_id"
            ].nunique()
        )
        for model_id in MODEL_IDS:
            selected = landmark_scores.loc[
                (landmark_scores["model_id"] == model_id)
                & landmark_scores["issued"]
            ]
            condition_means = selected.groupby(
                "outer_condition_id", sort=True
            )[list(metric_columns)].mean()
            record: dict[str, object] = {
                "landmark_visit_count": int(landmark),
                "model_id": model_id,
                "issued_cell_count": int(selected["cell_id"].nunique()),
                "issued_cell_fraction": (
                    float(selected["cell_id"].nunique() / denominator)
                    if denominator
                    else 0.0
                ),
                "issued_condition_cluster_count": int(len(condition_means)),
            }
            for metric in metric_columns:
                stem = metric.removesuffix("_pp")
                record[f"cell_equal_{stem}_pp"] = (
                    float(selected[metric].mean()) if len(selected) else None
                )
                record[f"condition_equal_{stem}_pp"] = (
                    float(condition_means[metric].mean())
                    if len(condition_means)
                    else None
                )
            rows.append(record)
    return pd.DataFrame(rows, columns=MODEL_METRIC_COLUMNS).sort_values(
        ["landmark_visit_count", "model_id"],
        kind="stable",
        ignore_index=True,
    )


def _choice_summary(decisions: pd.DataFrame) -> pd.DataFrame:
    totals = decisions.groupby("landmark_visit_count", sort=True).size().to_dict()
    rows: list[dict[str, object]] = []
    keys = [
        "landmark_visit_count",
        "issued",
        "selected_expert_model_id",
        "selection_mode",
        "evidence_status",
    ]
    for values, group in decisions.groupby(keys, sort=True, dropna=False):
        landmark, issued, expert, mode, status = values
        rows.append(
            {
                "landmark_visit_count": int(landmark),
                "issued": bool(issued),
                "selected_expert_model_id": str(expert),
                "selection_mode": str(mode),
                "evidence_status": str(status),
                "cell_count": len(group),
                "cell_fraction": float(len(group) / totals[int(landmark)]),
                "condition_cluster_count": int(
                    group["outer_condition_id"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows, columns=SELECTOR_CHOICE_COLUMNS).sort_values(
        [
            "landmark_visit_count",
            "issued",
            "selected_expert_model_id",
            "selection_mode",
        ],
        kind="stable",
        ignore_index=True,
    )


def _cell_diagnostics(
    truth: pd.DataFrame,
    scores: pd.DataFrame,
    decisions: pd.DataFrame,
    config: Mapping[str, object],
) -> pd.DataFrame:
    score_end = float(config["dynamic_landmarks"]["score_end_equivalent_full_cycles"])
    landmarks = tuple(
        int(value) for value in config["dynamic_landmarks"]["prefix_visit_counts"]
    )
    rows: list[dict[str, object]] = []
    score_index = scores.set_index(
        ["outer_condition_id", "cell_id", "landmark_visit_count", "model_id"]
    )
    decision_index = decisions.set_index(
        ["outer_condition_id", "cell_id", "landmark_visit_count"]
    )
    if decision_index.index.duplicated().any():
        raise SNLRPTPostOutcomeAuditError("Duplicate SNL selector decisions")
    for (outer, cell_id), cell in truth.groupby(
        ["outer_condition_id", "cell_id"], sort=True
    ):
        ordered = cell.sort_values("visit_index", kind="stable")
        for landmark in landmarks:
            prefix_end = float(ordered.iloc[landmark - 1]["equivalent_full_cycles"])
            future = ordered.loc[
                (ordered["visit_index"] >= landmark)
                & (ordered["equivalent_full_cycles"] <= score_end)
            ]
            if future.empty:
                raise SNLRPTPostOutcomeAuditError("SNL audit found an empty suffix")
            decision = decision_index.loc[(outer, cell_id, landmark)]
            issued = bool(decision["issued"])
            metric = {
                model_id: float(
                    score_index.loc[
                        (outer, cell_id, landmark, model_id), "trajectory_iae_pp"
                    ]
                )
                for model_id in MODEL_IDS
                if (outer, cell_id, landmark, model_id) in score_index.index
            }
            if "target_prefix_persistence" not in metric:
                raise SNLRPTPostOutcomeAuditError("Persistence score is missing")
            best_base = min(
                (model_id for model_id in BASE_MODEL_IDS if model_id in metric),
                key=lambda model_id: (
                    metric[model_id],
                    BASE_MODEL_IDS.index(model_id),
                ),
            )
            selector_value = metric.get(SELECTOR_MODEL_ID)
            if issued and selector_value is None:
                raise SNLRPTPostOutcomeAuditError("Issued selector score is missing")
            selected_expert = str(decision["selected_expert_model_id"])
            if issued:
                if selected_expert not in metric:
                    raise SNLRPTPostOutcomeAuditError(
                        "Selected SNL expert score is missing"
                    )
                if not math.isclose(
                    float(selector_value),
                    metric[selected_expert],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise SNLRPTPostOutcomeAuditError(
                        "Selector curve does not match its selected expert"
                    )
            rows.append(
                {
                    "outer_condition_id": str(outer),
                    "cell_id": str(cell_id),
                    "landmark_visit_count": landmark,
                    "future_observation_count": len(future),
                    "prefix_end_equivalent_full_cycles": prefix_end,
                    "last_scored_equivalent_full_cycles": float(
                        future["equivalent_full_cycles"].max()
                    ),
                    "scored_horizon_equivalent_full_cycles": float(
                        future["equivalent_full_cycles"].max() - prefix_end
                    ),
                    "issued": issued,
                    "selected_expert_model_id": selected_expert,
                    "selection_mode": str(decision["selection_mode"]),
                    "evidence_status": str(decision["evidence_status"]),
                    "persistence_trajectory_iae_pp": metric[
                        "target_prefix_persistence"
                    ],
                    "selector_trajectory_iae_pp": selector_value,
                    "selector_improvement_vs_persistence_pp": (
                        metric["target_prefix_persistence"] - float(selector_value)
                        if selector_value is not None
                        else None
                    ),
                    "realized_best_base_expert_model_id": best_base,
                    "realized_best_base_expert_trajectory_iae_pp": metric[best_base],
                    "selector_regret_vs_realized_best_base_expert_pp": (
                        float(selector_value) - metric[best_base]
                        if selector_value is not None
                        else None
                    ),
                    "selected_expert_matches_realized_best_base_expert": (
                        bool(selected_expert == best_base) if issued else False
                    ),
                }
            )
    return pd.DataFrame(rows, columns=CELL_DIAGNOSTIC_COLUMNS).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count"],
        kind="stable",
        ignore_index=True,
    )


def _condition_diagnostics(
    cells: pd.DataFrame,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (outer, landmark), group in cells.groupby(
        ["outer_condition_id", "landmark_visit_count"], sort=True
    ):
        condition_scores = scores.loc[
            (scores["outer_condition_id"] == outer)
            & (scores["landmark_visit_count"] == landmark)
            & scores["issued"]
        ]
        by_model = condition_scores.groupby("model_id", sort=True)[
            "trajectory_iae_pp"
        ].mean()
        best_base = min(
            (model_id for model_id in BASE_MODEL_IDS if model_id in by_model.index),
            key=lambda model_id: (
                float(by_model.loc[model_id]),
                BASE_MODEL_IDS.index(model_id),
            ),
        )
        selector_iae = float(by_model.loc[SELECTOR_MODEL_ID])
        persistence_iae = float(by_model.loc["target_prefix_persistence"])
        rows.append(
            {
                "outer_condition_id": str(outer),
                "landmark_visit_count": int(landmark),
                "cell_count": len(group),
                "issued_cell_count": int(group["issued"].sum()),
                "mean_future_observation_count": float(
                    group["future_observation_count"].mean()
                ),
                "minimum_last_scored_equivalent_full_cycles": float(
                    group["last_scored_equivalent_full_cycles"].min()
                ),
                "maximum_last_scored_equivalent_full_cycles": float(
                    group["last_scored_equivalent_full_cycles"].max()
                ),
                "persistence_trajectory_iae_pp": persistence_iae,
                "selector_trajectory_iae_pp": selector_iae,
                "selector_improvement_vs_persistence_pp": (
                    persistence_iae - selector_iae
                ),
                "realized_best_fixed_base_expert_model_id": best_base,
                "realized_best_fixed_base_expert_trajectory_iae_pp": float(
                    by_model.loc[best_base]
                ),
                "selector_regret_vs_realized_best_fixed_base_expert_pp": (
                    selector_iae - float(by_model.loc[best_base])
                ),
            }
        )
    return pd.DataFrame(rows, columns=CONDITION_DIAGNOSTIC_COLUMNS).sort_values(
        ["outer_condition_id", "landmark_visit_count"],
        kind="stable",
        ignore_index=True,
    )


def audit_snl_rpt_loco_result(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    scores: pd.DataFrame,
    score_summary: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Replay and diagnose the completed frozen SNL result."""
    frozen = validate_snl_rpt_loco_config(config)
    if tuple(truth.columns) != TARGET_TRUTH_COLUMNS:
        raise SNLRPTPostOutcomeAuditError("SNL audit truth columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise SNLRPTPostOutcomeAuditError("SNL audit decision columns changed")
    try:
        replayed_scores, replayed_summary = score_snl_rpt_loco(
            truth,
            predictions,
            decisions,
            prediction_manifest,
            frozen,
        )
    except SNLRPTLOCOError as exc:
        raise SNLRPTPostOutcomeAuditError(
            f"Frozen SNL result cannot be replayed: {exc}"
        ) from exc
    normalized_scores = _normalize_scores_like(scores, replayed_scores)
    if canonical_frame_sha256(
        normalized_scores, SCORE_COLUMNS
    ) != canonical_frame_sha256(replayed_scores, SCORE_COLUMNS):
        raise SNLRPTPostOutcomeAuditError("Persisted SNL score table changed")
    if canonical_json_sha256(dict(score_summary)) != canonical_json_sha256(
        replayed_summary
    ):
        raise SNLRPTPostOutcomeAuditError("Persisted SNL score summary changed")

    cells = _cell_diagnostics(truth, replayed_scores, decisions, frozen)
    conditions = _condition_diagnostics(cells, replayed_scores)
    models = _metric_summary(replayed_scores, decisions)
    choices = _choice_summary(decisions)
    landmark_summary: dict[str, object] = {}
    for landmark, group in cells.groupby("landmark_visit_count", sort=True):
        selector = models.loc[
            (models["landmark_visit_count"] == landmark)
            & (models["model_id"] == SELECTOR_MODEL_ID)
        ].iloc[0]
        fixed_base = models.loc[
            (models["landmark_visit_count"] == landmark)
            & models["model_id"].isin(BASE_MODEL_IDS)
        ].sort_values(
            ["condition_equal_trajectory_iae_pp", "model_id"], kind="stable"
        )
        best_fixed = fixed_base.iloc[0]
        condition_group = conditions.loc[
            conditions["landmark_visit_count"] == landmark
        ]
        landmark_summary[str(int(landmark))] = {
            "cell_count": len(group),
            "condition_cluster_count": int(group["outer_condition_id"].nunique()),
            "future_observation_count": {
                "minimum": int(group["future_observation_count"].min()),
                "median": float(group["future_observation_count"].median()),
                "maximum": int(group["future_observation_count"].max()),
            },
            "last_scored_equivalent_full_cycles": {
                "minimum": float(
                    group["last_scored_equivalent_full_cycles"].min()
                ),
                "median": float(
                    group["last_scored_equivalent_full_cycles"].median()
                ),
                "maximum": float(
                    group["last_scored_equivalent_full_cycles"].max()
                ),
            },
            "selector_condition_equal_trajectory_iae_pp": float(
                selector["condition_equal_trajectory_iae_pp"]
            ),
            "selector_cell_equal_trajectory_iae_pp": float(
                selector["cell_equal_trajectory_iae_pp"]
            ),
            "condition_vs_cell_weighting_gap_pp": float(
                selector["condition_equal_trajectory_iae_pp"]
                - selector["cell_equal_trajectory_iae_pp"]
            ),
            "realized_best_fixed_base_expert_model_id": str(
                best_fixed["model_id"]
            ),
            "realized_best_fixed_base_expert_condition_equal_trajectory_iae_pp": float(
                best_fixed["condition_equal_trajectory_iae_pp"]
            ),
            "selector_regret_vs_realized_best_fixed_base_expert_pp": float(
                selector["condition_equal_trajectory_iae_pp"]
                - best_fixed["condition_equal_trajectory_iae_pp"]
            ),
            "mean_cell_level_selector_regret_vs_realized_oracle_base_expert_pp": float(
                group[
                    "selector_regret_vs_realized_best_base_expert_pp"
                ].mean()
            ),
            "maximum_cell_level_selector_regret_vs_realized_oracle_base_expert_pp": float(
                group[
                    "selector_regret_vs_realized_best_base_expert_pp"
                ].max()
            ),
            "selected_expert_matches_realized_oracle_base_expert_fraction": float(
                group[
                    "selected_expert_matches_realized_best_base_expert"
                ].mean()
            ),
            "improved_condition_cluster_fraction_vs_persistence": float(
                (
                    condition_group[
                        "selector_improvement_vs_persistence_pp"
                    ]
                    > 0.0
                ).mean()
            ),
        }

    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "experiment_id": str(frozen["experiment_id"]),
        "evidence_role": "post_outcome_diagnostic_not_model_selection",
        "prediction_manifest_content_sha256": prediction_manifest.get(
            "manifest_content_sha256"
        ),
        "replayed_score_rows_sha256": canonical_frame_sha256(
            replayed_scores, SCORE_COLUMNS
        ),
        "cell_diagnostics_sha256": canonical_frame_sha256(
            cells, CELL_DIAGNOSTIC_COLUMNS
        ),
        "condition_diagnostics_sha256": canonical_frame_sha256(
            conditions, CONDITION_DIAGNOSTIC_COLUMNS
        ),
        "model_metrics_sha256": canonical_frame_sha256(
            models, MODEL_METRIC_COLUMNS
        ),
        "selector_choice_summary_sha256": canonical_frame_sha256(
            choices, SELECTOR_CHOICE_COLUMNS
        ),
        "by_landmark": landmark_summary,
        "finding": (
            "The condition-informed degradation-rate prior is the strongest fixed "
            "expert at three RPT visits. The frozen selector becomes slightly "
            "stronger at four visits; this is explanatory evidence, not permission "
            "to retune the frozen three-visit primary result."
        ),
        "claim_boundary": [
            "All diagnostics were computed after SNL outcomes were exposed.",
            "Realized best-expert rows are hindsight oracles and are not deployable.",
            "Condition clusters, not cells, are the primary equal-weight units.",
            "Cells have unequal observed suffix lengths below the 2500-EFC cap.",
            "This audit cannot confirm calendar aging, field performance, Hithium products, or 15-25 year accuracy.",
            "Aggregate results remain private until Battery Archive clarifies competition and publication scope.",
        ],
    }
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return cells, conditions, models, choices, summary


def _match_visits(
    primary: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    tolerance_efc: float,
) -> tuple[list[float], list[float]]:
    efc_deltas: list[float] = []
    retention_deltas: list[float] = []
    for cell_id, primary_cell in primary.groupby("cell_id", sort=True):
        candidate_cell = candidate.loc[candidate["cell_id"] == cell_id].sort_values(
            "equivalent_full_cycles", kind="stable"
        )
        available = list(candidate_cell.index)
        for _, primary_row in primary_cell.sort_values(
            "equivalent_full_cycles", kind="stable"
        ).iterrows():
            if not available:
                break
            nearest = min(
                available,
                key=lambda index: (
                    abs(
                        float(candidate.loc[index, "equivalent_full_cycles"])
                        - float(primary_row["equivalent_full_cycles"])
                    ),
                    int(index),
                ),
            )
            efc_delta = abs(
                float(candidate.loc[nearest, "equivalent_full_cycles"])
                - float(primary_row["equivalent_full_cycles"])
            )
            if efc_delta <= tolerance_efc:
                efc_deltas.append(efc_delta)
                retention_deltas.append(
                    abs(
                        float(candidate.loc[nearest, "capacity_retention_pct"])
                        - float(primary_row["capacity_retention_pct"])
                    )
                )
                available.remove(nearest)
    return efc_deltas, retention_deltas


def summarize_snl_rpt_extraction_sensitivity(
    primary_trajectories: pd.DataFrame,
    candidate_runs: Sequence[
        tuple[float, float, pd.DataFrame, Mapping[str, object]]
    ],
    *,
    primary_rest_gap_hours: float = 1.0,
    primary_duplicate_visit_efc: float = 10.0,
    matching_tolerance_efc: float = 25.0,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare fixed RPT-extraction threshold variants without rerunning models."""
    if tuple(primary_trajectories.columns) != RPT_TRAJECTORY_COLUMNS:
        raise SNLRPTPostOutcomeAuditError("Primary SNL RPT columns changed")
    if matching_tolerance_efc < 0.0:
        raise SNLRPTPostOutcomeAuditError("Visit matching tolerance must be non-negative")
    primary = primary_trajectories.sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    primary_hash = canonical_frame_sha256(primary, RPT_TRAJECTORY_COLUMNS)
    primary_counts = primary.groupby("cell_id", sort=True).size()
    rows: list[dict[str, object]] = []
    settings_seen: set[tuple[float, float]] = set()
    for rest_gap, duplicate_efc, trajectories, audit in candidate_runs:
        setting = (float(rest_gap), float(duplicate_efc))
        if setting in settings_seen:
            raise SNLRPTPostOutcomeAuditError(
                f"Duplicate extraction sensitivity setting: {setting}"
            )
        settings_seen.add(setting)
        if tuple(trajectories.columns) != RPT_TRAJECTORY_COLUMNS:
            raise SNLRPTPostOutcomeAuditError(
                f"SNL sensitivity columns changed for setting {setting}"
            )
        candidate = trajectories.sort_values(
            ["condition_id", "cell_id", "visit_index"],
            kind="stable",
            ignore_index=True,
        )
        candidate_hash = canonical_frame_sha256(candidate, RPT_TRAJECTORY_COLUMNS)
        audit_hash = str(audit.get("canonical_rpt_trajectory_sha256", ""))
        if audit_hash != candidate_hash:
            raise SNLRPTPostOutcomeAuditError(
                f"SNL sensitivity extraction hash mismatch for setting {setting}"
            )
        candidate_counts = candidate.groupby("cell_id", sort=True).size()
        aligned_counts = pd.concat(
            [primary_counts.rename("primary"), candidate_counts.rename("candidate")],
            axis=1,
        ).fillna(0)
        count_delta = (aligned_counts["candidate"] - aligned_counts["primary"]).abs()
        efc_delta, retention_delta = _match_visits(
            primary,
            candidate,
            tolerance_efc=matching_tolerance_efc,
        )
        matched = len(efc_delta)
        rows.append(
            {
                "rest_gap_hours": setting[0],
                "duplicate_visit_efc": setting[1],
                "is_primary_setting": bool(
                    math.isclose(setting[0], primary_rest_gap_hours)
                    and math.isclose(setting[1], primary_duplicate_visit_efc)
                ),
                "canonical_rpt_trajectory_sha256": candidate_hash,
                "exact_primary_trajectory_match": candidate_hash == primary_hash,
                "physical_cell_count": int(candidate["cell_id"].nunique()),
                "condition_cluster_count": int(candidate["condition_id"].nunique()),
                "trajectory_row_count": len(candidate),
                "minimum_rpt_visit_count": int(candidate_counts.min()),
                "median_rpt_visit_count": float(candidate_counts.median()),
                "maximum_rpt_visit_count": int(candidate_counts.max()),
                "visit_count_changed_cell_count": int((count_delta > 0).sum()),
                "maximum_absolute_visit_count_delta": int(count_delta.max()),
                "matched_primary_visit_count": matched,
                "primary_visit_match_fraction": float(matched / len(primary)),
                "candidate_visit_match_fraction": float(matched / len(candidate)),
                "mean_absolute_matched_efc_delta": (
                    float(np.mean(efc_delta)) if efc_delta else None
                ),
                "maximum_absolute_matched_efc_delta": (
                    float(np.max(efc_delta)) if efc_delta else None
                ),
                "mean_absolute_matched_retention_delta_pp": (
                    float(np.mean(retention_delta)) if retention_delta else None
                ),
                "maximum_absolute_matched_retention_delta_pp": (
                    float(np.max(retention_delta)) if retention_delta else None
                ),
            }
        )
    table = pd.DataFrame(rows, columns=EXTRACTION_SENSITIVITY_COLUMNS).sort_values(
        ["rest_gap_hours", "duplicate_visit_efc"],
        kind="stable",
        ignore_index=True,
    )
    primary_rows = table.loc[table["is_primary_setting"]]
    if len(primary_rows) != 1 or not bool(
        primary_rows.iloc[0]["exact_primary_trajectory_match"]
    ):
        raise SNLRPTPostOutcomeAuditError(
            "Sensitivity grid must replay the primary extraction exactly once"
        )
    nonprimary = table.loc[~table["is_primary_setting"]]
    summary: dict[str, object] = {
        "schema_version": "lifetwin.snl_rpt_extraction_sensitivity.v1",
        "evidence_role": "post_outcome_adapter_robustness_not_model_selection",
        "primary_rpt_trajectory_sha256": primary_hash,
        "matching_tolerance_equivalent_full_cycles": matching_tolerance_efc,
        "setting_count": len(table),
        "nonprimary_exact_match_count": int(
            nonprimary["exact_primary_trajectory_match"].sum()
        ),
        "maximum_changed_cell_count": int(
            nonprimary["visit_count_changed_cell_count"].max()
        )
        if len(nonprimary)
        else 0,
        "minimum_primary_visit_match_fraction": float(
            nonprimary["primary_visit_match_fraction"].min()
        )
        if len(nonprimary)
        else 1.0,
        "maximum_matched_retention_delta_pp": float(
            nonprimary["maximum_absolute_matched_retention_delta_pp"].max()
        )
        if len(nonprimary)
        else 0.0,
        "sensitivity_rows_sha256": canonical_frame_sha256(
            table, EXTRACTION_SENSITIVITY_COLUMNS
        ),
        "claim_boundary": [
            "Threshold variants were inspected after outcomes were exposed.",
            "The frozen primary trajectory and model result were not replaced.",
            "Visit matching is descriptive and uses a fixed 25-EFC tolerance by default.",
            "Robust extraction does not establish external predictive validity.",
        ],
    }
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return table, summary


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CELL_DIAGNOSTIC_COLUMNS",
    "CONDITION_DIAGNOSTIC_COLUMNS",
    "EXTRACTION_SENSITIVITY_COLUMNS",
    "MODEL_METRIC_COLUMNS",
    "SELECTOR_CHOICE_COLUMNS",
    "SNLRPTPostOutcomeAuditError",
    "audit_snl_rpt_loco_result",
    "summarize_snl_rpt_extraction_sensitivity",
]
