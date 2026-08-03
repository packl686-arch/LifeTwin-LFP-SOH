"""Post-outcome diagnostics for the frozen NASA evidence-weighted V3 model.

These analyses are deliberately separated from the V3 predictor. They may
explain a completed development result, but they cannot tune or confirm it.
"""

from __future__ import annotations

import json
import math
from statistics import median
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    BASE_MODEL_IDS,
    build_nasa_dynamic_gate_fold_table,
)
from lifetwin.experiments.nasa_evidence_weighted_moe_v3 import (
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    V2_COMPARISON_GATE_MODEL_ID,
    V3_MODEL_ID,
    NasaEvidenceWeightedMoeError,
    predict_nasa_evidence_weighted_moe,
    score_nasa_evidence_weighted_moe,
)
from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    PREFIX_CYCLES,
    SCORE_END_CYCLE,
    canonical_frame_sha256,
    canonical_json_sha256,
)


AUDIT_SCHEMA_VERSION = "lifetwin.nasa_v3_post_outcome_audit.v1"
ABLATION_IDS = (
    "hindsight_oracle_base_expert",
    "frozen_v3_evidence_weighted_mixture",
    "equal_weight_base_mixture",
    "pure_inverse_risk_base_mixture",
    "hard_lowest_predicted_risk_expert",
    "frozen_v2_curve_gate",
)
ABLATION_SCORE_COLUMNS = (
    "held_out_cell_id",
    "prefix_cycle",
    "audit_model_id",
    "selected_model_id",
    "weights_json",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
)
EVIDENCE_COLUMNS = (
    "held_out_cell_id",
    "prefix_cycle",
    "v3_trajectory_mae_pp",
    "v2_curve_gate_trajectory_mae_pp",
    "equal_weight_trajectory_mae_pp",
    "pure_risk_trajectory_mae_pp",
    "hard_risk_trajectory_mae_pp",
    "hindsight_oracle_trajectory_mae_pp",
    "v3_delta_vs_v2_pp",
    "v3_delta_vs_equal_weight_pp",
    "v3_regret_vs_hindsight_oracle_pp",
    "dominant_expert_model_id",
    "realized_best_expert_model_id",
    "dominant_expert_matches_realized_best",
    "risk_margin_fraction",
    "mean_neighbor_distance",
    "selection_strength",
    "mean_evidence_band_width_pp",
    "empirical_evidence_band_coverage_fraction",
    "evidence_status",
    "operational_action",
)
ATTACK_COLUMNS = (
    "attack_id",
    "held_out_cell_id",
    "prefix_cycle",
    "passed",
    "mean_absolute_prediction_delta_pp",
    "maximum_absolute_prediction_delta_pp",
    "original_operational_action",
    "attacked_operational_action",
    "attacked_mean_neighbor_distance",
    "detail",
)


class NasaV3PostOutcomeAuditError(ValueError):
    """Raised when the post-outcome audit input is inconsistent."""


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _truth_table(cycles: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cell_id in CELL_CUTOFFS:
        cell = cycles.loc[
            (cycles["cell_id"].astype(str) == cell_id)
            & (pd.to_numeric(cycles["cycle_index"]) <= SCORE_END_CYCLE)
        ].copy()
        cell["cycle_index"] = pd.to_numeric(cell["cycle_index"], errors="raise").astype(
            int
        )
        cell["discharge_capacity_ah"] = pd.to_numeric(
            cell["discharge_capacity_ah"], errors="raise"
        ).astype(float)
        cell = cell.sort_values("cycle_index", kind="stable")
        initial = cell.loc[
            cell["cycle_index"].between(1, 5), "discharge_capacity_ah"
        ].tolist()
        if len(initial) != 5:
            raise NasaV3PostOutcomeAuditError("Audit truth requires cycles 1 through 5")
        normalization = float(median(initial))
        rows.append(
            pd.DataFrame(
                {
                    "held_out_cell_id": cell_id,
                    "forecast_cycle": cell["cycle_index"].astype(int),
                    "observed_capacity_retention_pct": (
                        100.0 * cell["discharge_capacity_ah"] / normalization
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _trajectory_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
) -> tuple[float, float, float]:
    error = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
    absolute = np.abs(error)
    return (
        float(np.mean(absolute)),
        float(np.sqrt(np.mean(np.square(error)))),
        float(absolute[-1]),
    )


def _one_hot(model_id: str) -> dict[str, float]:
    return {candidate: float(candidate == model_id) for candidate in BASE_MODEL_IDS}


def _ablation_predictions(
    group: pd.DataFrame,
    observed: np.ndarray,
    v3_config: Mapping[str, object],
) -> list[tuple[str, str, dict[str, float], np.ndarray]]:
    base = {
        model_id: group.loc[group["model_id"] == model_id]
        .sort_values("forecast_cycle", kind="stable")[
            "predicted_capacity_retention_pct"
        ]
        .to_numpy(dtype=float)
        for model_id in BASE_MODEL_IDS
    }
    matrix = np.vstack([base[model_id] for model_id in BASE_MODEL_IDS])
    realized_mae = {
        model_id: float(np.mean(np.abs(values - observed)))
        for model_id, values in base.items()
    }
    realized_best = min(
        BASE_MODEL_IDS,
        key=lambda model_id: (
            realized_mae[model_id],
            BASE_MODEL_IDS.index(model_id),
        ),
    )
    v3_row = group.loc[group["model_id"] == V3_MODEL_ID].iloc[0]
    risks = {
        str(key): float(value)
        for key, value in json.loads(str(v3_row["expert_risks_json"])).items()
    }
    mixture = v3_config["mixture"]
    risk_epsilon = float(mixture["risk_epsilon_pp"])
    risk_power = float(mixture["risk_inverse_power"])
    raw_risk = np.asarray(
        [
            (risks[model_id] + risk_epsilon) ** (-risk_power)
            for model_id in BASE_MODEL_IDS
        ],
        dtype=float,
    )
    pure_risk_array = raw_risk / float(np.sum(raw_risk))
    pure_risk_weights = {
        model_id: float(value)
        for model_id, value in zip(BASE_MODEL_IDS, pure_risk_array, strict=True)
    }
    hard_risk = min(
        BASE_MODEL_IDS,
        key=lambda model_id: (risks[model_id], BASE_MODEL_IDS.index(model_id)),
    )
    uniform = {model_id: 1.0 / len(BASE_MODEL_IDS) for model_id in BASE_MODEL_IDS}
    v3_weights = {
        str(key): float(value)
        for key, value in json.loads(str(v3_row["expert_weights_json"])).items()
    }

    def mixed(weights: Mapping[str, float]) -> np.ndarray:
        array = np.asarray([weights[model_id] for model_id in BASE_MODEL_IDS])
        return np.sum(array[:, None] * matrix, axis=0)

    v2_curve = group.loc[group["model_id"] == V2_COMPARISON_GATE_MODEL_ID].sort_values(
        "forecast_cycle", kind="stable"
    )
    v2_selected = str(v2_curve["dominant_expert_model_id"].iloc[0])
    return [
        (
            "hindsight_oracle_base_expert",
            realized_best,
            _one_hot(realized_best),
            base[realized_best],
        ),
        (
            "frozen_v3_evidence_weighted_mixture",
            str(v3_row["dominant_expert_model_id"]),
            v3_weights,
            mixed(v3_weights),
        ),
        (
            "equal_weight_base_mixture",
            "none_equal_weight",
            uniform,
            mixed(uniform),
        ),
        (
            "pure_inverse_risk_base_mixture",
            min(
                BASE_MODEL_IDS,
                key=lambda model_id: (
                    -pure_risk_weights[model_id],
                    BASE_MODEL_IDS.index(model_id),
                ),
            ),
            pure_risk_weights,
            mixed(pure_risk_weights),
        ),
        (
            "hard_lowest_predicted_risk_expert",
            hard_risk,
            _one_hot(hard_risk),
            base[hard_risk],
        ),
        (
            "frozen_v2_curve_gate",
            v2_selected,
            _one_hot(v2_selected),
            v2_curve["predicted_capacity_retention_pct"].to_numpy(dtype=float),
        ),
    ]


def _rank_correlation(left: pd.Series, right: pd.Series) -> float | None:
    left_rank = left.astype(float).rank(method="average")
    right_rank = right.astype(float).rank(method="average")
    if left_rank.nunique() < 2 or right_rank.nunique() < 2:
        return None
    value = float(left_rank.corr(right_rank))
    return value if math.isfinite(value) else None


def audit_nasa_v3_result(
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    scores: pd.DataFrame,
    score_summary: Mapping[str, object],
    v2_config: Mapping[str, object],
    v3_config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Verify the committed result and compute fixed post-outcome ablations."""
    replayed_scores, replayed_summary = score_nasa_evidence_weighted_moe(
        cycles,
        predictions,
        prediction_manifest,
        v2_config,
        v3_config,
    )
    if tuple(scores.columns) != SCORE_COLUMNS:
        raise NasaV3PostOutcomeAuditError("Persisted V3 score columns changed")
    normalized_scores = scores.copy()
    for column in SCORE_COLUMNS:
        if pd.api.types.is_float_dtype(replayed_scores[column].dtype):
            normalized_scores[column] = pd.to_numeric(
                normalized_scores[column], errors="raise"
            ).astype(float)
        elif pd.api.types.is_integer_dtype(replayed_scores[column].dtype):
            normalized_scores[column] = pd.to_numeric(
                normalized_scores[column], errors="raise"
            ).astype(np.int64)
        else:
            normalized_scores[column] = normalized_scores[column].astype(str)
    if canonical_frame_sha256(
        normalized_scores,
        SCORE_COLUMNS,
    ) != canonical_frame_sha256(replayed_scores, SCORE_COLUMNS):
        raise NasaV3PostOutcomeAuditError("Persisted V3 score table changed")
    if canonical_json_sha256(dict(score_summary)) != canonical_json_sha256(
        replayed_summary
    ):
        raise NasaV3PostOutcomeAuditError("Persisted V3 score summary changed")

    truth = _truth_table(cycles)
    linked = predictions.merge(
        truth,
        on=["held_out_cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise NasaV3PostOutcomeAuditError("Audit predictions cannot link to truth")

    rows: list[dict[str, object]] = []
    for (cell_id, prefix_cycle), group in linked.groupby(
        ["held_out_cell_id", "prefix_cycle"], sort=True
    ):
        group = group.sort_values(["model_id", "forecast_cycle"], kind="stable")
        observed = (
            group.loc[group["model_id"] == V3_MODEL_ID]
            .sort_values("forecast_cycle", kind="stable")[
                "observed_capacity_retention_pct"
            ]
            .to_numpy(dtype=float)
        )
        for audit_model_id, selected, weights, predicted in _ablation_predictions(
            group,
            observed,
            v3_config,
        ):
            mae, rmse, endpoint = _trajectory_metrics(predicted, observed)
            rows.append(
                {
                    "held_out_cell_id": str(cell_id),
                    "prefix_cycle": int(prefix_cycle),
                    "audit_model_id": audit_model_id,
                    "selected_model_id": selected,
                    "weights_json": _json_text(weights),
                    "trajectory_mae_pp": mae,
                    "trajectory_rmse_pp": rmse,
                    "endpoint_absolute_error_pp": endpoint,
                }
            )
    ablation_scores = pd.DataFrame(rows, columns=ABLATION_SCORE_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "audit_model_id"],
        kind="stable",
        ignore_index=True,
    )
    prefix_summary = (
        ablation_scores.groupby(["prefix_cycle", "audit_model_id"], sort=True)[
            [
                "trajectory_mae_pp",
                "trajectory_rmse_pp",
                "endpoint_absolute_error_pp",
            ]
        ]
        .mean()
        .reset_index()
    )
    overall_summary = (
        ablation_scores.groupby("audit_model_id", sort=True)[
            [
                "trajectory_mae_pp",
                "trajectory_rmse_pp",
                "endpoint_absolute_error_pp",
            ]
        ]
        .mean()
        .reset_index()
    )

    lookup = ablation_scores.pivot(
        index=["held_out_cell_id", "prefix_cycle"],
        columns="audit_model_id",
        values="trajectory_mae_pp",
    )
    decision = (
        predictions.loc[predictions["model_id"] == V3_MODEL_ID]
        .groupby(["held_out_cell_id", "prefix_cycle"], sort=True)
        .first()
    )
    v3_scores = normalized_scores.loc[
        normalized_scores["model_id"] == V3_MODEL_ID
    ].set_index(["held_out_cell_id", "prefix_cycle"])
    evidence_rows: list[dict[str, object]] = []
    for index, v3_value in decision.iterrows():
        fold_scores = lookup.loc[index]
        realized_best = ablation_scores.loc[
            (ablation_scores["held_out_cell_id"] == index[0])
            & (ablation_scores["prefix_cycle"] == index[1])
            & (ablation_scores["audit_model_id"] == "hindsight_oracle_base_expert"),
            "selected_model_id",
        ].iloc[0]
        v3_mae = float(fold_scores["frozen_v3_evidence_weighted_mixture"])
        v2_mae = float(fold_scores["frozen_v2_curve_gate"])
        equal_mae = float(fold_scores["equal_weight_base_mixture"])
        oracle_mae = float(fold_scores["hindsight_oracle_base_expert"])
        evidence_rows.append(
            {
                "held_out_cell_id": str(index[0]),
                "prefix_cycle": int(index[1]),
                "v3_trajectory_mae_pp": v3_mae,
                "v2_curve_gate_trajectory_mae_pp": v2_mae,
                "equal_weight_trajectory_mae_pp": equal_mae,
                "pure_risk_trajectory_mae_pp": float(
                    fold_scores["pure_inverse_risk_base_mixture"]
                ),
                "hard_risk_trajectory_mae_pp": float(
                    fold_scores["hard_lowest_predicted_risk_expert"]
                ),
                "hindsight_oracle_trajectory_mae_pp": oracle_mae,
                "v3_delta_vs_v2_pp": v3_mae - v2_mae,
                "v3_delta_vs_equal_weight_pp": v3_mae - equal_mae,
                "v3_regret_vs_hindsight_oracle_pp": v3_mae - oracle_mae,
                "dominant_expert_model_id": str(v3_value["dominant_expert_model_id"]),
                "realized_best_expert_model_id": str(realized_best),
                "dominant_expert_matches_realized_best": float(
                    str(v3_value["dominant_expert_model_id"]) == str(realized_best)
                ),
                "risk_margin_fraction": float(v3_value["risk_margin_fraction"]),
                "mean_neighbor_distance": float(v3_value["mean_neighbor_distance"]),
                "selection_strength": float(v3_value["selection_strength"]),
                "mean_evidence_band_width_pp": float(
                    v3_scores.loc[index, "mean_evidence_band_width_pp"]
                ),
                "empirical_evidence_band_coverage_fraction": float(
                    v3_scores.loc[
                        index,
                        "empirical_evidence_band_coverage_fraction",
                    ]
                ),
                "evidence_status": str(v3_value["evidence_status"]),
                "operational_action": str(v3_value["operational_action"]),
            }
        )
    evidence = pd.DataFrame(evidence_rows, columns=EVIDENCE_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle"], kind="stable", ignore_index=True
    )

    action_summary = (
        evidence.groupby("operational_action", sort=True)[
            ["v3_trajectory_mae_pp", "mean_evidence_band_width_pp"]
        ]
        .agg(["count", "mean"])
        .reset_index()
    )
    action_summary.columns = [
        "operational_action",
        "fold_count",
        "mean_v3_trajectory_mae_pp",
        "width_fold_count",
        "mean_evidence_band_width_pp",
    ]
    action_records = action_summary.drop(columns="width_fold_count").to_dict(
        orient="records"
    )
    refusal = evidence.loc[evidence["operational_action"] == "refuse_recommended"]
    summary: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "evidence_role": "post_outcome_diagnostic_not_model_selection",
        "v3_prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "v3_score_sha256": str(score_summary["score_sha256"]),
        "ablation_score_sha256": canonical_frame_sha256(
            ablation_scores,
            ABLATION_SCORE_COLUMNS,
        ),
        "evidence_diagnostic_sha256": canonical_frame_sha256(
            evidence,
            EVIDENCE_COLUMNS,
        ),
        "overall_ablation_metrics": overall_summary.to_dict(orient="records"),
        "prefix_ablation_metrics": prefix_summary.to_dict(orient="records"),
        "dominant_expert_match_fraction": float(
            evidence["dominant_expert_matches_realized_best"].mean()
        ),
        "spearman_rank_correlations": {
            "mean_neighbor_distance_vs_v3_mae": _rank_correlation(
                evidence["mean_neighbor_distance"],
                evidence["v3_trajectory_mae_pp"],
            ),
            "selection_strength_vs_v3_mae": _rank_correlation(
                evidence["selection_strength"],
                evidence["v3_trajectory_mae_pp"],
            ),
            "evidence_band_width_vs_v3_mae": _rank_correlation(
                evidence["mean_evidence_band_width_pp"],
                evidence["v3_trajectory_mae_pp"],
            ),
            "risk_margin_vs_selector_regret": _rank_correlation(
                evidence["risk_margin_fraction"],
                evidence["v3_regret_vs_hindsight_oracle_pp"],
            ),
        },
        "operational_action_summary": action_records,
        "refusal_diagnostic": {
            "refusal_fold_count": len(refusal),
            "refusal_mean_v3_mae_pp": (
                float(refusal["v3_trajectory_mae_pp"].mean())
                if not refusal.empty
                else None
            ),
            "all_fold_median_v3_mae_pp": float(
                evidence["v3_trajectory_mae_pp"].median()
            ),
            "interpretation": ("covariate_ood_only_not_validated_error_ranking"),
        },
        "interval_diagnostic": {
            "mean_width_pp": float(evidence["mean_evidence_band_width_pp"].mean()),
            "mean_empirical_coverage_fraction": float(
                evidence["empirical_evidence_band_coverage_fraction"].mean()
            ),
            "formal_coverage_claim": False,
        },
        "claim_boundary": [
            "Ablations were computed after NASA outcomes were exposed.",
            (
                "The hindsight row selects the best realized single base expert; "
                "it is unusable and is not a lower bound for convex mixtures."
            ),
            "Correlations over 16 reused folds are descriptive, not inferential.",
            "Evidence bands are not calibrated confidence intervals.",
            "Refusal is a covariate-distance rule, not a validated error detector.",
        ],
    }
    return ablation_scores, prefix_summary, evidence, summary


def run_nasa_v3_attacks(
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    v2_config: Mapping[str, object],
    v3_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run deterministic firewall, mild-perturbation, and OOD attacks."""
    original = predictions.sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    attack_rows: list[dict[str, object]] = []
    for cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            attacked_cycles = cycles.copy()
            suffix = (attacked_cycles["cell_id"].astype(str) == cell_id) & (
                pd.to_numeric(attacked_cycles["cycle_index"]) > prefix_cycle
            )
            attacked_cycles.loc[suffix, "discharge_capacity_ah"] *= 0.77
            attacked_cycles.loc[suffix, "common_window_3p8_to_3p4_duration_s"] *= 0.73
            attacked_cycles.loc[suffix, "voltage_at_1p0_ah_v"] -= 0.25
            attacked_cycles.loc[suffix, "temperature_rise_c"] += 5.0
            attacked_fold = build_nasa_dynamic_gate_fold_table(
                attacked_cycles,
                v2_config,
            )
            attacked_predictions, _ = predict_nasa_evidence_weighted_moe(
                attacked_fold,
                v2_config,
                v3_config,
            )
            coordinates = (original["held_out_cell_id"] == cell_id) & (
                original["prefix_cycle"] == prefix_cycle
            )
            attacked_coordinates = (
                attacked_predictions["held_out_cell_id"] == cell_id
            ) & (attacked_predictions["prefix_cycle"] == prefix_cycle)
            before = original.loc[coordinates].reset_index(drop=True)
            after = attacked_predictions.loc[attacked_coordinates].reset_index(
                drop=True
            )
            passed = canonical_frame_sha256(
                before,
                PREDICTION_COLUMNS,
            ) == canonical_frame_sha256(after, PREDICTION_COLUMNS)
            attack_rows.append(
                {
                    "attack_id": "held_out_suffix_mutation_firewall",
                    "held_out_cell_id": cell_id,
                    "prefix_cycle": prefix_cycle,
                    "passed": passed,
                    "mean_absolute_prediction_delta_pp": 0.0 if passed else None,
                    "maximum_absolute_prediction_delta_pp": 0.0 if passed else None,
                    "original_operational_action": "not_applicable",
                    "attacked_operational_action": "not_applicable",
                    "attacked_mean_neighbor_distance": 0.0,
                    "detail": "mutated every held-out suffix measurement",
                }
            )

            mild_cycles = cycles.copy()
            prefix_mask = (mild_cycles["cell_id"].astype(str) == cell_id) & (
                pd.to_numeric(mild_cycles["cycle_index"]) <= prefix_cycle
            )
            progress = (
                pd.to_numeric(mild_cycles.loc[prefix_mask, "cycle_index"]).to_numpy(
                    dtype=float
                )
                - 1.0
            ) / max(prefix_cycle - 1, 1)
            mild_cycles.loc[prefix_mask, "discharge_capacity_ah"] *= (
                1.0 - 0.002 * progress
            )
            mild_cycles.loc[prefix_mask, "common_window_3p8_to_3p4_duration_s"] *= (
                1.0 + 0.005 * progress
            )
            mild_cycles.loc[prefix_mask, "voltage_at_1p0_ah_v"] -= 0.002 * progress
            mild_cycles.loc[prefix_mask, "temperature_rise_c"] += 0.1 * progress
            mild_fold = build_nasa_dynamic_gate_fold_table(mild_cycles, v2_config)
            mild_predictions, _ = predict_nasa_evidence_weighted_moe(
                mild_fold,
                v2_config,
                v3_config,
            )
            v3_before = before.loc[before["model_id"] == V3_MODEL_ID].sort_values(
                "forecast_cycle", kind="stable"
            )
            v3_after = mild_predictions.loc[
                (mild_predictions["held_out_cell_id"] == cell_id)
                & (mild_predictions["prefix_cycle"] == prefix_cycle)
                & (mild_predictions["model_id"] == V3_MODEL_ID)
            ].sort_values("forecast_cycle", kind="stable")
            delta = np.abs(
                v3_before["predicted_capacity_retention_pct"].to_numpy(dtype=float)
                - v3_after["predicted_capacity_retention_pct"].to_numpy(dtype=float)
            )
            original_action = str(v3_before["operational_action"].iloc[0])
            attacked_action = str(v3_after["operational_action"].iloc[0])
            attack_rows.append(
                {
                    "attack_id": "mild_prefix_sensor_drift",
                    "held_out_cell_id": cell_id,
                    "prefix_cycle": prefix_cycle,
                    "passed": bool(float(np.max(delta)) <= 2.0),
                    "mean_absolute_prediction_delta_pp": float(np.mean(delta)),
                    "maximum_absolute_prediction_delta_pp": float(np.max(delta)),
                    "original_operational_action": original_action,
                    "attacked_operational_action": attacked_action,
                    "attacked_mean_neighbor_distance": float(
                        v3_after["mean_neighbor_distance"].iloc[0]
                    ),
                    "detail": (
                        "0.2% capacity, 0.5% duration, 2 mV voltage, and "
                        "0.1 C endpoint-linear drift; 2 pp max-change diagnostic gate"
                    ),
                }
            )

    for cell_id in CELL_CUTOFFS:
        attacked_cycles = cycles.copy()
        mask = attacked_cycles["cell_id"].astype(str) == cell_id
        cycle_index = pd.to_numeric(attacked_cycles.loc[mask, "cycle_index"]).to_numpy(
            dtype=float
        )
        attacked_cycles.loc[mask, "common_window_3p8_to_3p4_duration_s"] *= (
            1.0 + cycle_index
        )
        attacked_cycles.loc[mask, "voltage_at_1p0_ah_v"] -= 0.10 * cycle_index
        attacked_cycles.loc[mask, "temperature_rise_c"] += 8.0 * cycle_index
        fold = build_nasa_dynamic_gate_fold_table(attacked_cycles, v2_config)
        attacked_predictions, _ = predict_nasa_evidence_weighted_moe(
            fold,
            v2_config,
            v3_config,
        )
        decisions = (
            attacked_predictions.loc[
                (attacked_predictions["held_out_cell_id"] == cell_id)
                & (attacked_predictions["model_id"] == V3_MODEL_ID)
            ]
            .groupby("prefix_cycle", sort=True)
            .first()
        )
        for prefix_cycle, decision in decisions.iterrows():
            refused = str(decision["operational_action"]) == "refuse_recommended"
            attack_rows.append(
                {
                    "attack_id": "severe_curve_covariate_shift",
                    "held_out_cell_id": cell_id,
                    "prefix_cycle": int(prefix_cycle),
                    "passed": refused,
                    "mean_absolute_prediction_delta_pp": 0.0,
                    "maximum_absolute_prediction_delta_pp": 0.0,
                    "original_operational_action": "not_applicable",
                    "attacked_operational_action": str(decision["operational_action"]),
                    "attacked_mean_neighbor_distance": float(
                        decision["mean_neighbor_distance"]
                    ),
                    "detail": "extreme slope shift in three curve diagnostics",
                }
            )

    missing_cycles = cycles.copy()
    missing_cycles.loc[
        (missing_cycles["cell_id"].astype(str) == "B0005")
        & (pd.to_numeric(missing_cycles["cycle_index"]) == 20),
        "voltage_at_1p0_ah_v",
    ] = np.nan
    failed_closed = False
    rejection = ""
    try:
        build_nasa_dynamic_gate_fold_table(missing_cycles, v2_config)
    except (NasaEvidenceWeightedMoeError, ValueError) as exc:
        failed_closed = True
        rejection = f"{type(exc).__name__}: {exc}"
    attack_rows.append(
        {
            "attack_id": "missing_required_curve_feature",
            "held_out_cell_id": "B0005",
            "prefix_cycle": 20,
            "passed": failed_closed,
            "mean_absolute_prediction_delta_pp": 0.0,
            "maximum_absolute_prediction_delta_pp": 0.0,
            "original_operational_action": "not_applicable",
            "attacked_operational_action": "failed_closed"
            if failed_closed
            else "accepted",
            "attacked_mean_neighbor_distance": 0.0,
            "detail": rejection or "missing value was unexpectedly accepted",
        }
    )
    attacks = pd.DataFrame(attack_rows, columns=ATTACK_COLUMNS).sort_values(
        ["attack_id", "held_out_cell_id", "prefix_cycle"],
        kind="stable",
        ignore_index=True,
    )
    by_attack = []
    for attack_id, group in attacks.groupby("attack_id", sort=True):
        finite_delta = pd.to_numeric(
            group["maximum_absolute_prediction_delta_pp"], errors="coerce"
        ).dropna()
        by_attack.append(
            {
                "attack_id": str(attack_id),
                "case_count": len(group),
                "passed_count": int(group["passed"].astype(bool).sum()),
                "all_passed": bool(group["passed"].astype(bool).all()),
                "maximum_prediction_delta_pp": (
                    float(finite_delta.max()) if not finite_delta.empty else None
                ),
                "operational_action_transition_count": int(
                    (
                        group["original_operational_action"]
                        != group["attacked_operational_action"]
                    ).sum()
                ),
            }
        )
    summary: dict[str, object] = {
        "schema_version": "lifetwin.nasa_v3_attack_summary.v1",
        "evidence_role": "post_outcome_software_and_sensitivity_audit",
        "attack_result_sha256": canonical_frame_sha256(attacks, ATTACK_COLUMNS),
        "by_attack": by_attack,
        "claim_boundary": [
            "Suffix attacks validate software noninterference, not model accuracy.",
            "The mild drift threshold is a post-outcome diagnostic, not a frozen gate.",
            "Severe OOD attacks are synthetic and do not estimate field prevalence.",
        ],
    }
    return attacks, summary


__all__ = [
    "ABLATION_SCORE_COLUMNS",
    "ATTACK_COLUMNS",
    "AUDIT_SCHEMA_VERSION",
    "EVIDENCE_COLUMNS",
    "NasaV3PostOutcomeAuditError",
    "audit_nasa_v3_result",
    "run_nasa_v3_attacks",
]
