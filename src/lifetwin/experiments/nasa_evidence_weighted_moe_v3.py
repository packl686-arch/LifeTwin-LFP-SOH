"""Post-outcome V3 evidence-weighted mixture for the NASA stress benchmark.

V3 deliberately composes the frozen V2 prediction interface instead of changing
it. NASA V1/V2 outcomes were already inspected before this protocol was written,
so all metrics from this module are retrospective development evidence only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.nasa_dynamic_gate_v2 import (
    BASE_MODEL_IDS,
    CONFIG_SEMANTIC_SHA256 as V2_CONFIG_SEMANTIC_SHA256,
    EXPERIMENT_ID as V2_EXPERIMENT_ID,
    PREDICTION_COLUMNS as V2_PREDICTION_COLUMNS,
    build_nasa_dynamic_gate_fold_table,
    predict_nasa_dynamic_gate,
    validate_nasa_dynamic_gate_config,
)
from lifetwin.experiments.nasa_prefix_loco import (
    CELL_CUTOFFS,
    DATASET_ID,
    PREFIX_CYCLES,
    PRIMARY_PREFIX_CYCLE,
    SCORE_END_CYCLE,
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.nasa_evidence_weighted_moe.config.v3"
EXPERIMENT_ID = "nasa_pcoe_evidence_weighted_moe_v3"
CONFIG_SEMANTIC_SHA256 = (
    "77b2e0465c04cc9f97499d55c12cdf3cbe177b6e4b423fff11ef5da503f9872a"
)
PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "lifetwin.nasa_evidence_weighted_moe.prediction_manifest.v3"
)
V2_COMPARISON_GATE_MODEL_ID = "nested_loco_curve_aware_mean_gate"
V3_MODEL_ID = "evidence_weighted_mixture_v3"
MODEL_IDS = (*BASE_MODEL_IDS, V2_COMPARISON_GATE_MODEL_ID, V3_MODEL_ID)

PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "forecast_cycle",
    "predicted_capacity_retention_pct",
    "evidence_band_lower_pct",
    "evidence_band_upper_pct",
    "normalization_capacity_ah",
    "prefix_row_count",
    "target_prefix_sha256",
    "dominant_expert_model_id",
    "expert_weights_json",
    "expert_risks_json",
    "nearest_cell_ids",
    "neighbor_distances_json",
    "risk_margin_fraction",
    "mean_neighbor_distance",
    "margin_support",
    "distance_support",
    "selection_strength",
    "evidence_status",
    "operational_action",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "held_out_cell_id",
    "training_cell_ids",
    "prefix_cycle",
    "score_end_cycle",
    "model_id",
    "dominant_expert_model_id",
    "evidence_status",
    "operational_action",
    "future_observation_count",
    "trajectory_iae_pp_normalized_by_cycle_horizon",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
    "empirical_evidence_band_coverage_fraction",
    "mean_evidence_band_width_pp",
    "endpoint_inside_evidence_band",
)

_ALLOWED_CLAIMS = (
    "retrospective_evidence_weighted_mixture_development",
    "future_label_firewall_software_validation",
    "descriptive_model_weight_and_refusal_analysis",
)
_PROHIBITED_CLAIMS = (
    "nasa_v3_is_preregistered_outcome_blind_confirmation",
    "attia_2020_is_still_an_unseen_external_test",
    "lfp_calendar_aging_validation",
    "fifteen_to_twenty_five_year_accuracy",
    "hithium_product_accuracy",
    "stationary_storage_field_validation",
    "formal_uncertainty_coverage",
    "inferential_significance_from_four_nasa_cells",
)


class NasaEvidenceWeightedMoeError(ValueError):
    """Raised when the frozen V3 contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NasaEvidenceWeightedMoeError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_text(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NasaEvidenceWeightedMoeError(
            "Value is not canonical finite JSON"
        ) from exc


def validate_nasa_evidence_weighted_moe_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise NasaEvidenceWeightedMoeError("V3 config must be an object")
    if canonical_json_sha256(dict(config)) != CONFIG_SEMANTIC_SHA256:
        raise NasaEvidenceWeightedMoeError("Frozen V3 config changed")
    detached = json.loads(_canonical_json_text(dict(config)))
    if detached.get("schema_version") != SCHEMA_VERSION:
        raise NasaEvidenceWeightedMoeError("V3 schema changed")
    if detached.get("experiment_id") != EXPERIMENT_ID:
        raise NasaEvidenceWeightedMoeError("V3 experiment identity changed")
    if detached["base_protocol"]["config_semantic_sha256"] != V2_CONFIG_SEMANTIC_SHA256:
        raise NasaEvidenceWeightedMoeError("V3 base-protocol hash changed")
    if detached["base_protocol"]["experiment_id"] != V2_EXPERIMENT_ID:
        raise NasaEvidenceWeightedMoeError("V3 base experiment changed")
    return detached


def load_nasa_evidence_weighted_moe_config(
    path: str | Path,
) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                NasaEvidenceWeightedMoeError(f"Non-finite JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise NasaEvidenceWeightedMoeError("Cannot load V3 config") from exc
    if not isinstance(value, Mapping):
        raise NasaEvidenceWeightedMoeError("V3 config must be an object")
    return validate_nasa_evidence_weighted_moe_config(value)


def _validate_v2_config(config: Mapping[str, object]) -> dict[str, object]:
    parsed = validate_nasa_dynamic_gate_config(config)
    if canonical_json_sha256(parsed) != V2_CONFIG_SEMANTIC_SHA256:
        raise NasaEvidenceWeightedMoeError("V2 config hash changed under V3")
    return parsed


def _piecewise_support(
    value: float,
    *,
    full_support_at: float,
    zero_support_at: float,
    increasing: bool,
) -> float:
    if zero_support_at <= full_support_at:
        raise NasaEvidenceWeightedMoeError("Invalid support thresholds")
    if increasing:
        return float(
            np.clip(
                (value - full_support_at) / (zero_support_at - full_support_at),
                0.0,
                1.0,
            )
        )
    return float(
        np.clip(
            (zero_support_at - value) / (zero_support_at - full_support_at),
            0.0,
            1.0,
        )
    )


def _one_hot_weights(model_id: str) -> dict[str, float]:
    if model_id not in BASE_MODEL_IDS:
        raise NasaEvidenceWeightedMoeError("Unknown base model for one-hot weights")
    return {candidate: float(candidate == model_id) for candidate in BASE_MODEL_IDS}


def _v3_weight_diagnostics(
    v2_group: pd.DataFrame,
    config: Mapping[str, object],
) -> dict[str, object]:
    gate = v2_group.loc[v2_group["model_id"] == V2_COMPARISON_GATE_MODEL_ID].iloc[0]
    nearest_cells = str(gate["gate_nearest_cell_ids"]).split(";")
    if len(nearest_cells) != 2:
        raise NasaEvidenceWeightedMoeError("V2 gate does not expose two neighbors")
    distances_value = json.loads(str(gate["gate_neighbor_distances_json"]))
    distances = {cell_id: float(distances_value[cell_id]) for cell_id in nearest_cells}

    mixture = config["mixture"]
    distance_epsilon = float(mixture["neighbor_distance_epsilon"])
    inverse_distance = np.asarray(
        [1.0 / (distances[cell_id] + distance_epsilon) for cell_id in nearest_cells],
        dtype=float,
    )
    neighbor_weights = inverse_distance / float(np.sum(inverse_distance))
    risks: dict[str, float] = {}
    for model_id in BASE_MODEL_IDS:
        base = v2_group.loc[v2_group["model_id"] == model_id].iloc[0]
        per_cell = json.loads(str(base["gate_training_mae_json"]))
        errors = np.asarray(
            [float(per_cell[cell_id]) for cell_id in nearest_cells],
            dtype=float,
        )
        mean_error = float(np.sum(neighbor_weights * errors))
        dispersion = float(np.sum(neighbor_weights * np.abs(errors - mean_error)))
        risks[model_id] = (
            mean_error + float(mixture["risk_dispersion_penalty"]) * dispersion
        )

    ordered_risks = sorted(
        risks.items(),
        key=lambda item: (item[1], BASE_MODEL_IDS.index(item[0])),
    )
    best_risk = ordered_risks[0][1]
    second_risk = ordered_risks[1][1]
    risk_margin = float(
        (second_risk - best_risk) / max(best_risk, float(mixture["risk_epsilon_pp"]))
    )
    margin_support = _piecewise_support(
        risk_margin,
        full_support_at=float(mixture["relative_margin_full_uniform_below"]),
        zero_support_at=float(mixture["relative_margin_full_evidence_above"]),
        increasing=True,
    )
    mean_distance = float(np.mean(list(distances.values())))
    distance_support = _piecewise_support(
        mean_distance,
        full_support_at=float(mixture["distance_full_evidence_below"]),
        zero_support_at=float(mixture["distance_full_uniform_above"]),
        increasing=False,
    )
    selection_strength = margin_support * distance_support

    risk_epsilon = float(mixture["risk_epsilon_pp"])
    risk_power = float(mixture["risk_inverse_power"])
    raw_risk_weights = np.asarray(
        [
            (risks[model_id] + risk_epsilon) ** (-risk_power)
            for model_id in BASE_MODEL_IDS
        ],
        dtype=float,
    )
    risk_weights = raw_risk_weights / float(np.sum(raw_risk_weights))
    uniform = np.full(len(BASE_MODEL_IDS), 1.0 / len(BASE_MODEL_IDS))
    final_weights_array = uniform + selection_strength * (risk_weights - uniform)
    final_weights_array = final_weights_array / float(np.sum(final_weights_array))
    weights = {
        model_id: float(value)
        for model_id, value in zip(
            BASE_MODEL_IDS,
            final_weights_array,
            strict=True,
        )
    }
    dominant_expert = min(
        BASE_MODEL_IDS,
        key=lambda model_id: (-weights[model_id], BASE_MODEL_IDS.index(model_id)),
    )

    if mean_distance >= float(
        config["evidence_band"]["refusal_mean_neighbor_distance"]
    ):
        evidence_status = "out_of_domain"
        operational_action = "refuse_recommended"
    elif distance_support == 0.0:
        evidence_status = "distance_unsupported_equal_blend"
        operational_action = "predict_with_warning"
    elif margin_support == 0.0:
        evidence_status = "risk_ambiguous_equal_blend"
        operational_action = "predict_with_warning"
    elif selection_strength < 1.0:
        evidence_status = "partial_evidence_blend"
        operational_action = "predict_with_warning"
    else:
        evidence_status = "supported_risk_weighted_blend"
        operational_action = "predict"
    return {
        "nearest_cells": nearest_cells,
        "distances": distances,
        "risks": risks,
        "weights": weights,
        "dominant_expert": dominant_expert,
        "risk_margin": risk_margin,
        "mean_distance": mean_distance,
        "margin_support": margin_support,
        "distance_support": distance_support,
        "selection_strength": selection_strength,
        "evidence_status": evidence_status,
        "operational_action": operational_action,
    }


def _rows_for_fold(
    v2_group: pd.DataFrame,
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    first = v2_group.iloc[0]
    held_out_cell_id = str(first["held_out_cell_id"])
    prefix_cycle = int(first["prefix_cycle"])
    diagnostics = _v3_weight_diagnostics(v2_group, config)
    weights = diagnostics["weights"]
    forecast_cycles = sorted(v2_group["forecast_cycle"].astype(int).unique())
    base_by_model = {
        model_id: v2_group.loc[v2_group["model_id"] == model_id]
        .sort_values("forecast_cycle", kind="stable")
        .reset_index(drop=True)
        for model_id in BASE_MODEL_IDS
    }
    for model_id, frame in base_by_model.items():
        if frame["forecast_cycle"].astype(int).tolist() != forecast_cycles:
            raise NasaEvidenceWeightedMoeError(
                f"V2 base forecast support changed for {model_id}"
            )

    centers = np.vstack(
        [
            base_by_model[model_id]["predicted_capacity_retention_pct"].to_numpy(
                dtype=float
            )
            for model_id in BASE_MODEL_IDS
        ]
    )
    lower_bounds = np.vstack(
        [
            base_by_model[model_id]["evidence_band_lower_pct"].to_numpy(dtype=float)
            for model_id in BASE_MODEL_IDS
        ]
    )
    upper_bounds = np.vstack(
        [
            base_by_model[model_id]["evidence_band_upper_pct"].to_numpy(dtype=float)
            for model_id in BASE_MODEL_IDS
        ]
    )
    weight_array = np.asarray([weights[model_id] for model_id in BASE_MODEL_IDS])
    mixture_center = np.sum(weight_array[:, None] * centers, axis=0)
    combined_lower = np.sum(weight_array[:, None] * lower_bounds, axis=0)
    combined_upper = np.sum(weight_array[:, None] * upper_bounds, axis=0)
    disagreement = np.sqrt(
        np.sum(
            weight_array[:, None] * np.square(centers - mixture_center[None, :]),
            axis=0,
        )
    )
    band_config = config["evidence_band"]
    base_half_width = np.maximum(
        mixture_center - combined_lower,
        combined_upper - mixture_center,
    )
    low_evidence_multiplier = 1.0 + float(
        band_config["low_evidence_half_width_multiplier"]
    ) * (1.0 - float(diagnostics["selection_strength"]))
    mixture_half_width = (
        base_half_width * low_evidence_multiplier
        + float(band_config["model_disagreement_standard_deviation_multiplier"])
        * disagreement
    )
    mixture_lower = np.clip(mixture_center - mixture_half_width, 0.0, 110.0)
    mixture_upper = np.clip(mixture_center + mixture_half_width, 0.0, 110.0)

    shared = {
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "held_out_cell_id": held_out_cell_id,
        "training_cell_ids": str(first["training_cell_ids"]),
        "prefix_cycle": prefix_cycle,
        "score_end_cycle": SCORE_END_CYCLE,
        "normalization_capacity_ah": float(first["normalization_capacity_ah"]),
        "prefix_row_count": int(first["prefix_row_count"]),
        "target_prefix_sha256": str(first["target_prefix_sha256"]),
    }
    rows: list[dict[str, object]] = []
    for model_id in (*BASE_MODEL_IDS, V2_COMPARISON_GATE_MODEL_ID):
        source = v2_group.loc[v2_group["model_id"] == model_id].sort_values(
            "forecast_cycle", kind="stable"
        )
        selected = str(source["selected_base_model_id"].iloc[0])
        one_hot = _one_hot_weights(selected)
        for row in source.itertuples(index=False):
            rows.append(
                {
                    **shared,
                    "model_id": model_id,
                    "forecast_cycle": int(row.forecast_cycle),
                    "predicted_capacity_retention_pct": float(
                        row.predicted_capacity_retention_pct
                    ),
                    "evidence_band_lower_pct": float(row.evidence_band_lower_pct),
                    "evidence_band_upper_pct": float(row.evidence_band_upper_pct),
                    "dominant_expert_model_id": selected,
                    "expert_weights_json": _canonical_json_text(one_hot),
                    "expert_risks_json": "{}",
                    "nearest_cell_ids": "not_applicable",
                    "neighbor_distances_json": "{}",
                    "risk_margin_fraction": 0.0,
                    "mean_neighbor_distance": 0.0,
                    "margin_support": 0.0,
                    "distance_support": 0.0,
                    "selection_strength": 0.0,
                    "evidence_status": "v2_comparator",
                    "operational_action": "predict",
                }
            )

    weights_json = _canonical_json_text(weights)
    risks_json = _canonical_json_text(diagnostics["risks"])
    distances_json = _canonical_json_text(diagnostics["distances"])
    nearest_text = ";".join(diagnostics["nearest_cells"])
    for index, forecast_cycle in enumerate(forecast_cycles):
        rows.append(
            {
                **shared,
                "model_id": V3_MODEL_ID,
                "forecast_cycle": forecast_cycle,
                "predicted_capacity_retention_pct": float(mixture_center[index]),
                "evidence_band_lower_pct": float(mixture_lower[index]),
                "evidence_band_upper_pct": float(mixture_upper[index]),
                "dominant_expert_model_id": str(diagnostics["dominant_expert"]),
                "expert_weights_json": weights_json,
                "expert_risks_json": risks_json,
                "nearest_cell_ids": nearest_text,
                "neighbor_distances_json": distances_json,
                "risk_margin_fraction": float(diagnostics["risk_margin"]),
                "mean_neighbor_distance": float(diagnostics["mean_distance"]),
                "margin_support": float(diagnostics["margin_support"]),
                "distance_support": float(diagnostics["distance_support"]),
                "selection_strength": float(diagnostics["selection_strength"]),
                "evidence_status": str(diagnostics["evidence_status"]),
                "operational_action": str(diagnostics["operational_action"]),
            }
        )
    return rows


def predict_nasa_evidence_weighted_moe(
    fold_table: pd.DataFrame,
    v2_config: Mapping[str, object],
    v3_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compose frozen V2 outputs into deterministic V3 mixture predictions."""
    parsed_v2 = _validate_v2_config(v2_config)
    parsed_v3 = validate_nasa_evidence_weighted_moe_config(v3_config)
    v2_predictions, v2_manifest = predict_nasa_dynamic_gate(
        fold_table,
        parsed_v2,
    )
    if tuple(v2_predictions.columns) != V2_PREDICTION_COLUMNS:
        raise NasaEvidenceWeightedMoeError("V2 prediction schema changed")

    rows: list[dict[str, object]] = []
    for held_out_cell_id in CELL_CUTOFFS:
        for prefix_cycle in PREFIX_CYCLES:
            group = v2_predictions.loc[
                (v2_predictions["held_out_cell_id"] == held_out_cell_id)
                & (v2_predictions["prefix_cycle"] == prefix_cycle)
            ]
            rows.extend(_rows_for_fold(group, parsed_v3))
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": "retrospective_method_development_only",
        "v3_config_semantic_sha256": canonical_json_sha256(parsed_v3),
        "v2_config_semantic_sha256": canonical_json_sha256(parsed_v2),
        "fold_table_sha256": str(v2_manifest["fold_table_sha256"]),
        "v2_prediction_sha256": str(v2_manifest["prediction_sha256"]),
        "v2_manifest_sha256": canonical_json_sha256(v2_manifest),
        "prediction_sha256": canonical_frame_sha256(
            predictions,
            PREDICTION_COLUMNS,
        ),
        "prediction_row_count": len(predictions),
        "held_out_cell_ids": list(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "model_ids": list(MODEL_IDS),
        "score_end_cycle": SCORE_END_CYCLE,
        "target_future_outcomes_used": False,
        "outer_fold_training_histories_used": True,
        "outcomes_previously_exposed_for_development": True,
        "inference_scope": "retrospective_development_only",
    }
    return predictions, manifest


def _validated_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise NasaEvidenceWeightedMoeError("V3 prediction columns changed")
    result = predictions.copy()
    numeric_columns = (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "predicted_capacity_retention_pct",
        "evidence_band_lower_pct",
        "evidence_band_upper_pct",
        "normalization_capacity_ah",
        "prefix_row_count",
        "risk_margin_fraction",
        "mean_neighbor_distance",
        "margin_support",
        "distance_support",
        "selection_strength",
    )
    string_columns = tuple(
        column for column in PREDICTION_COLUMNS if column not in numeric_columns
    )
    if result.loc[:, string_columns].isna().any().any():
        raise NasaEvidenceWeightedMoeError("V3 prediction strings cannot be null")
    for column in string_columns:
        result[column] = result[column].astype(str)
    numeric = result.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if (
        numeric.isna().any().any()
        or not np.isfinite(numeric.to_numpy(dtype=float)).all()
    ):
        raise NasaEvidenceWeightedMoeError("V3 predictions must be finite")
    for column in (
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    ):
        values = numeric[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise NasaEvidenceWeightedMoeError(f"{column} must be integral")
        result[column] = values.astype(np.int64)
    for column in set(numeric_columns) - {
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    }:
        result[column] = numeric[column].astype(float)

    if set(result["experiment_id"]) != {EXPERIMENT_ID}:
        raise NasaEvidenceWeightedMoeError("V3 experiment identity changed")
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise NasaEvidenceWeightedMoeError("V3 dataset identity changed")
    if set(result["held_out_cell_id"]) != set(CELL_CUTOFFS):
        raise NasaEvidenceWeightedMoeError("V3 held-out cells changed")
    if set(result["prefix_cycle"].astype(int)) != set(PREFIX_CYCLES):
        raise NasaEvidenceWeightedMoeError("V3 prefixes changed")
    if set(result["model_id"]) != set(MODEL_IDS):
        raise NasaEvidenceWeightedMoeError("V3 model registry changed")
    if not set(result["dominant_expert_model_id"]).issubset(set(BASE_MODEL_IDS)):
        raise NasaEvidenceWeightedMoeError("V3 dominant expert is unknown")
    expected_rows = (
        len(CELL_CUTOFFS)
        * len(MODEL_IDS)
        * sum(SCORE_END_CYCLE - prefix_cycle for prefix_cycle in PREFIX_CYCLES)
    )
    if len(result) != expected_rows:
        raise NasaEvidenceWeightedMoeError("V3 prediction cardinality changed")
    for column in (
        "predicted_capacity_retention_pct",
        "evidence_band_lower_pct",
        "evidence_band_upper_pct",
    ):
        if not result[column].between(0.0, 110.0).all():
            raise NasaEvidenceWeightedMoeError(f"{column} exceeds bounds")
    if not (
        (
            result["evidence_band_lower_pct"]
            <= result["predicted_capacity_retention_pct"]
        )
        & (
            result["predicted_capacity_retention_pct"]
            <= result["evidence_band_upper_pct"]
        )
    ).all():
        raise NasaEvidenceWeightedMoeError("V3 evidence band excludes its center")
    for column in ("margin_support", "distance_support", "selection_strength"):
        if not result[column].between(0.0, 1.0).all():
            raise NasaEvidenceWeightedMoeError(f"{column} must lie in [0, 1]")
    if (result["risk_margin_fraction"] < 0.0).any() or (
        result["mean_neighbor_distance"] < 0.0
    ).any():
        raise NasaEvidenceWeightedMoeError("V3 evidence diagnostics cannot be negative")
    if result.duplicated(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"]
    ).any():
        raise NasaEvidenceWeightedMoeError("V3 prediction coordinates are duplicated")

    json_columns = (
        "expert_weights_json",
        "expert_risks_json",
        "neighbor_distances_json",
    )
    for column in json_columns:
        for value in result[column].unique():
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise NasaEvidenceWeightedMoeError(
                    f"{column} is not valid JSON"
                ) from exc
            if _canonical_json_text(decoded) != value:
                raise NasaEvidenceWeightedMoeError(f"{column} is not canonical JSON")

    for held_out_cell_id in CELL_CUTOFFS:
        expected_training = ";".join(
            cell_id for cell_id in CELL_CUTOFFS if cell_id != held_out_cell_id
        )
        for prefix_cycle in PREFIX_CYCLES:
            for model_id in MODEL_IDS:
                group = result.loc[
                    (result["held_out_cell_id"] == held_out_cell_id)
                    & (result["prefix_cycle"] == prefix_cycle)
                    & (result["model_id"] == model_id)
                ].sort_values("forecast_cycle", kind="stable")
                if group["forecast_cycle"].astype(int).tolist() != list(
                    range(prefix_cycle + 1, SCORE_END_CYCLE + 1)
                ):
                    raise NasaEvidenceWeightedMoeError("V3 forecast support changed")
                invariant_columns = tuple(
                    column
                    for column in PREDICTION_COLUMNS
                    if column
                    not in {
                        "forecast_cycle",
                        "predicted_capacity_retention_pct",
                        "evidence_band_lower_pct",
                        "evidence_band_upper_pct",
                    }
                )
                if any(
                    group[column].nunique(dropna=False) != 1
                    for column in invariant_columns
                ):
                    raise NasaEvidenceWeightedMoeError(
                        "V3 fold metadata changes within a trajectory"
                    )
                if set(group["training_cell_ids"]) != {expected_training}:
                    raise NasaEvidenceWeightedMoeError(
                        "V3 training-cell identities changed"
                    )
                if set(group["prefix_row_count"].astype(int)) != {prefix_cycle}:
                    raise NasaEvidenceWeightedMoeError("V3 prefix count changed")
                weights = json.loads(str(group["expert_weights_json"].iloc[0]))
                if set(weights) != set(BASE_MODEL_IDS):
                    raise NasaEvidenceWeightedMoeError(
                        "V3 expert-weight registry changed"
                    )
                weight_values = np.asarray(list(weights.values()), dtype=float)
                if (weight_values < 0.0).any() or not math.isclose(
                    float(np.sum(weight_values)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise NasaEvidenceWeightedMoeError(
                        "V3 expert weights must be a probability vector"
                    )
                if model_id == V3_MODEL_ID:
                    risks = json.loads(str(group["expert_risks_json"].iloc[0]))
                    if set(risks) != set(BASE_MODEL_IDS):
                        raise NasaEvidenceWeightedMoeError(
                            "V3 expert-risk registry changed"
                        )
                    nearest = str(group["nearest_cell_ids"].iloc[0]).split(";")
                    if len(nearest) != 2 or not set(nearest).issubset(
                        set(expected_training.split(";"))
                    ):
                        raise NasaEvidenceWeightedMoeError(
                            "V3 nearest-cell metadata changed"
                        )
                elif json.loads(str(group["expert_risks_json"].iloc[0])) != {}:
                    raise NasaEvidenceWeightedMoeError(
                        "V2 comparator unexpectedly contains V3 risks"
                    )
    return result.sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def _validate_prediction_manifest(
    manifest: Mapping[str, object],
    *,
    fold_table: pd.DataFrame,
    predictions: pd.DataFrame,
    v2_config: Mapping[str, object],
    v3_config: Mapping[str, object],
) -> None:
    expected_predictions, expected = predict_nasa_evidence_weighted_moe(
        fold_table,
        v2_config,
        v3_config,
    )
    actual_prediction_sha256 = canonical_frame_sha256(
        predictions,
        PREDICTION_COLUMNS,
    )
    expected_prediction_sha256 = canonical_frame_sha256(
        expected_predictions,
        PREDICTION_COLUMNS,
    )
    if actual_prediction_sha256 != expected_prediction_sha256:
        raise NasaEvidenceWeightedMoeError(
            "V3 predictions differ from the deterministic frozen replay"
        )
    if set(manifest) != set(expected):
        raise NasaEvidenceWeightedMoeError("V3 manifest keys changed")
    for key, expected_value in expected.items():
        if manifest[key] != expected_value:
            raise NasaEvidenceWeightedMoeError(f"V3 manifest mismatch for {key}")


def _normalization_capacity(cell: pd.DataFrame) -> float:
    first_five = cell.loc[
        cell["cycle_index"].between(1, 5), "discharge_capacity_ah"
    ].astype(float)
    if len(first_five) != 5:
        raise NasaEvidenceWeightedMoeError(
            "V3 scoring requires exact normalization cycles 1 to 5"
        )
    value = float(median(first_five.tolist()))
    if not math.isfinite(value) or value <= 0.0:
        raise NasaEvidenceWeightedMoeError("V3 normalization must be positive")
    return value


def score_nasa_evidence_weighted_moe(
    cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    v2_config: Mapping[str, object],
    v3_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score committed V3 predictions after rebuilding the frozen fold input."""
    parsed_v2 = _validate_v2_config(v2_config)
    parsed_v3 = validate_nasa_evidence_weighted_moe_config(v3_config)
    fold_table = build_nasa_dynamic_gate_fold_table(cycles, parsed_v2)
    ordered_predictions = _validated_predictions(predictions)
    _validate_prediction_manifest(
        prediction_manifest,
        fold_table=fold_table,
        predictions=ordered_predictions,
        v2_config=parsed_v2,
        v3_config=parsed_v3,
    )

    truth_frames: list[pd.DataFrame] = []
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
        normalization = _normalization_capacity(cell)
        truth_frames.append(
            pd.DataFrame(
                {
                    "held_out_cell_id": cell_id,
                    "forecast_cycle": cell["cycle_index"],
                    "observed_capacity_retention_pct": (
                        100.0 * cell["discharge_capacity_ah"] / normalization
                    ),
                }
            )
        )
    truth = pd.concat(truth_frames, ignore_index=True)
    linked = ordered_predictions.merge(
        truth,
        on=["held_out_cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise NasaEvidenceWeightedMoeError("V3 predictions cannot link to truth")

    score_rows: list[dict[str, object]] = []
    for (cell_id, prefix_cycle, model_id), group in linked.groupby(
        ["held_out_cell_id", "prefix_cycle", "model_id"],
        sort=True,
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        forecast_cycle = group["forecast_cycle"].to_numpy(dtype=float)
        observed = group["observed_capacity_retention_pct"].to_numpy(dtype=float)
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        lower = group["evidence_band_lower_pct"].to_numpy(dtype=float)
        upper = group["evidence_band_upper_pct"].to_numpy(dtype=float)
        error = predicted - observed
        absolute_error = np.abs(error)
        horizon = float(forecast_cycle[-1] - forecast_cycle[0])
        iae = (
            float(np.trapezoid(absolute_error, forecast_cycle) / horizon)
            if horizon > 0.0
            else float(absolute_error[0])
        )
        inside = (observed >= lower) & (observed <= upper)
        score_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "held_out_cell_id": str(cell_id),
                "training_cell_ids": str(group["training_cell_ids"].iloc[0]),
                "prefix_cycle": int(prefix_cycle),
                "score_end_cycle": SCORE_END_CYCLE,
                "model_id": str(model_id),
                "dominant_expert_model_id": str(
                    group["dominant_expert_model_id"].iloc[0]
                ),
                "evidence_status": str(group["evidence_status"].iloc[0]),
                "operational_action": str(group["operational_action"].iloc[0]),
                "future_observation_count": len(group),
                "trajectory_iae_pp_normalized_by_cycle_horizon": iae,
                "trajectory_mae_pp": float(np.mean(absolute_error)),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(absolute_error[-1]),
                "empirical_evidence_band_coverage_fraction": float(np.mean(inside)),
                "mean_evidence_band_width_pp": float(np.mean(upper - lower)),
                "endpoint_inside_evidence_band": float(inside[-1]),
            }
        )
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["held_out_cell_id", "prefix_cycle", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    metric_columns = list(SCORE_COLUMNS[-7:])
    aggregate = (
        scores.groupby(["prefix_cycle", "model_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    overall = scores.groupby("model_id", sort=True)[metric_columns].mean().reset_index()
    v3_overall = float(
        overall.loc[
            overall["model_id"] == V3_MODEL_ID,
            "trajectory_mae_pp",
        ].iloc[0]
    )
    v2_overall = float(
        overall.loc[
            overall["model_id"] == V2_COMPARISON_GATE_MODEL_ID,
            "trajectory_mae_pp",
        ].iloc[0]
    )
    gate = parsed_v3["evaluation"]["nasa_development_promotion_gate"]
    overall_passed = v3_overall <= float(gate["maximum_overall_mean_trajectory_mae_pp"])
    prefix_deltas: list[dict[str, object]] = []
    prefix_passed = True
    for prefix_cycle in PREFIX_CYCLES:
        v3_value = float(
            aggregate.loc[
                (aggregate["prefix_cycle"] == prefix_cycle)
                & (aggregate["model_id"] == V3_MODEL_ID),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        v2_value = float(
            aggregate.loc[
                (aggregate["prefix_cycle"] == prefix_cycle)
                & (aggregate["model_id"] == V2_COMPARISON_GATE_MODEL_ID),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        delta = v3_value - v2_value
        passed = delta <= float(
            gate["maximum_prefix_mae_degradation_vs_v2_curve_gate_pp"]
        )
        prefix_passed = prefix_passed and passed
        prefix_deltas.append(
            {
                "prefix_cycle": prefix_cycle,
                "v3_mae_pp": v3_value,
                "v2_curve_gate_mae_pp": v2_value,
                "delta_pp": delta,
                "passed": passed,
            }
        )

    decisions = (
        ordered_predictions.loc[ordered_predictions["model_id"] == V3_MODEL_ID]
        .groupby(["held_out_cell_id", "prefix_cycle"], sort=True)
        .first()
    )
    action_counts = {
        str(key): int(value)
        for key, value in decisions["operational_action"].value_counts().items()
    }
    evidence_counts = {
        str(key): int(value)
        for key, value in decisions["evidence_status"].value_counts().items()
    }
    weight_records: list[dict[str, object]] = []
    for prefix_cycle, group in decisions.reset_index().groupby(
        "prefix_cycle", sort=True
    ):
        decoded = [json.loads(value) for value in group["expert_weights_json"].tolist()]
        weight_records.append(
            {
                "prefix_cycle": int(prefix_cycle),
                **{
                    model_id: float(np.mean([weights[model_id] for weights in decoded]))
                    for model_id in BASE_MODEL_IDS
                },
            }
        )

    summary: dict[str, object] = {
        "schema_version": "lifetwin.nasa_evidence_weighted_moe.score_summary.v3",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "development_status": (
            "post_nasa_v2_outcome_development_not_independent_confirmation"
        ),
        "prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "score_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "fold_count": len(CELL_CUTOFFS),
        "prefix_cycles": list(PREFIX_CYCLES),
        "primary_prefix_cycle": PRIMARY_PREFIX_CYCLE,
        "score_end_cycle": SCORE_END_CYCLE,
        "model_ids": list(MODEL_IDS),
        "aggregate_metrics": json.loads(
            _canonical_json_text(aggregate.to_dict(orient="records"))
        ),
        "overall_metrics": json.loads(
            _canonical_json_text(overall.to_dict(orient="records"))
        ),
        "v3_vs_v2_curve_gate": {
            "v3_overall_mae_pp": v3_overall,
            "v2_overall_mae_pp": v2_overall,
            "relative_mae_change_fraction": (v3_overall - v2_overall) / v2_overall,
            "prefix_deltas": prefix_deltas,
        },
        "nasa_development_promotion_gate": {
            "status": "passed" if overall_passed and prefix_passed else "failed",
            "overall_gate_passed": overall_passed,
            "all_prefix_gates_passed": prefix_passed,
            "thresholds": gate,
            "interpretation": "development_only_not_external_confirmation",
        },
        "operational_action_counts": action_counts,
        "evidence_status_counts": evidence_counts,
        "mean_expert_weights_by_prefix": weight_records,
        "evidence_band_scope": "descriptive_not_formal_coverage",
        "allowed_claims": list(_ALLOWED_CLAIMS),
        "prohibited_claims": list(_PROHIBITED_CLAIMS),
    }
    return scores, summary


__all__ = [
    "CONFIG_SEMANTIC_SHA256",
    "EXPERIMENT_ID",
    "MODEL_IDS",
    "NasaEvidenceWeightedMoeError",
    "PREDICTION_COLUMNS",
    "SCORE_COLUMNS",
    "V2_COMPARISON_GATE_MODEL_ID",
    "V3_MODEL_ID",
    "load_nasa_evidence_weighted_moe_config",
    "predict_nasa_evidence_weighted_moe",
    "score_nasa_evidence_weighted_moe",
    "validate_nasa_evidence_weighted_moe_config",
]
