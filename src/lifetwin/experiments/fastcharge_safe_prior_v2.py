"""Post-V1 safe-prior development model for FastCharge cycle-300 trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.data.fastcharge_portability import (
    CANONICAL_CYCLE_COLUMNS,
    DATASET_ID,
    TARGET_PREFIX_COLUMNS,
    build_fastcharge_prediction_inputs,
)
from lifetwin.experiments.fastcharge_trajectory_portability import (
    BASE_MODEL_IDS,
    CALIBRATION_COLUMNS,
    FEATURE_IDS,
    PREDICTION_COLUMNS,
    SCORE_COLUMNS,
    _expert_predictions_for_target,
    _json_text,
    _normalization_capacity,
    _reference_error_table,
    _retention,
    _streaming_frame_sha256,
    _training_resources,
    _trajectory_signature,
    _validated_full_cycles,
    _validated_target_prefixes,
    _validated_training,
    _weight_diagnostics,
)
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


SCHEMA_VERSION = "lifetwin.fastcharge_lfp_safe_prior.config.v2"
EXPERIMENT_ID = "fastcharge_lfp_safe_prior_trajectory_v2"
CONFIG_SEMANTIC_SHA256 = (
    "fddf507e6aa812d9360b4765f5b45f3771a20838ea49b5b94246be9c6312263f"
)
PRIOR_MODEL_ID = "safe_global_prior_mixture"
HARD_MODEL_ID = "safe_hard_local_risk_selector"
MOE_MODEL_ID = "safe_prior_local_evidence_moe"
MODEL_IDS = (*BASE_MODEL_IDS, PRIOR_MODEL_ID, HARD_MODEL_ID, MOE_MODEL_ID)
PREDICTION_MANIFEST_SCHEMA_VERSION = (
    "lifetwin.fastcharge_safe_prior_prediction_manifest.v2"
)
QUALIFICATION_COLUMNS = (
    "prefix_cycle",
    "model_id",
    "training_loo_mean_mae_pp",
    "persistence_reference_mae_pp",
    "eligible_for_safe_pool",
    "safe_prior_weight",
)


class FastChargeSafePriorV2Error(ValueError):
    """Raised when the frozen safe-prior V2 contract is violated."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FastChargeSafePriorV2Error(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def validate_fastcharge_safe_prior_v2_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, Mapping):
        raise FastChargeSafePriorV2Error("V2 config must be an object")
    if canonical_json_sha256(dict(config)) != CONFIG_SEMANTIC_SHA256:
        raise FastChargeSafePriorV2Error("Frozen safe-prior V2 config changed")
    detached = json.loads(_json_text(dict(config)))
    if detached.get("schema_version") != SCHEMA_VERSION:
        raise FastChargeSafePriorV2Error("V2 config schema changed")
    if detached.get("experiment_id") != EXPERIMENT_ID:
        raise FastChargeSafePriorV2Error("V2 experiment identity changed")
    if tuple(detached["base_experts"]["model_ids"]) != BASE_MODEL_IDS:
        raise FastChargeSafePriorV2Error("V2 base model registry changed")
    if tuple(detached["similarity"]["feature_ids"]) != FEATURE_IDS:
        raise FastChargeSafePriorV2Error("V2 feature registry changed")
    if detached["safe_prior"]["model_id"] != MOE_MODEL_ID:
        raise FastChargeSafePriorV2Error("V2 primary model changed")
    if detached["dataset"]["dataset_id"] != DATASET_ID:
        raise FastChargeSafePriorV2Error("V2 dataset identity changed")
    return detached


def load_fastcharge_safe_prior_v2_config(
    path: str | Path,
) -> dict[str, object]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                FastChargeSafePriorV2Error(f"Non-finite JSON constant: {value}")
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise FastChargeSafePriorV2Error("Cannot load safe-prior V2 config") from exc
    if not isinstance(value, Mapping):
        raise FastChargeSafePriorV2Error("V2 config must be an object")
    return validate_fastcharge_safe_prior_v2_config(value)


def _core_config(config: Mapping[str, object]) -> dict[str, object]:
    """Adapt V2 names to the frozen V1 core without changing either protocol."""
    result = json.loads(_json_text(dict(config)))
    local = result["local_evidence"]
    distance = local["distance_support_thresholds"]
    result["mixture"] = {
        "risk_reference_neighbor_count": local["risk_reference_neighbor_count"],
        "risk_dispersion_penalty": local["risk_dispersion_penalty"],
        "risk_epsilon_pp": local["risk_epsilon_pp"],
        "risk_inverse_power": local["risk_inverse_power"],
        "relative_margin_full_uniform_below": local["relative_margin_full_prior_below"],
        "relative_margin_full_evidence_above": local[
            "relative_margin_full_local_above"
        ],
        "distance_support_thresholds": {
            "full_evidence_training_loo_quantile": distance[
                "full_evidence_training_loo_quantile"
            ],
            "full_uniform_training_loo_quantile": distance[
                "full_prior_training_loo_quantile"
            ],
            "refusal_training_loo_quantile": distance["refusal_training_loo_quantile"],
            "refusal_multiplier": distance["refusal_multiplier"],
        },
        "feature_jackknife_stability": local["feature_jackknife_stability"],
    }
    return result


def _prefix_cycles(config: Mapping[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in config["split_and_firewall"]["prefix_cycles"])


def _score_end(config: Mapping[str, object]) -> int:
    return int(config["split_and_firewall"]["score_end_cycle"])


def _safe_prior(
    reference_errors: Mapping[str, Mapping[str, float]],
    config: Mapping[str, object],
) -> dict[str, object]:
    global_risks = {
        model_id: float(
            np.mean([errors[model_id] for errors in reference_errors.values()])
        )
        for model_id in BASE_MODEL_IDS
    }
    persistence = global_risks["target_prefix_persistence"]
    rules = config["safe_prior"]["eligibility_requires_both"]
    eligible = [
        model_id
        for model_id in BASE_MODEL_IDS
        if (
            global_risks[model_id]
            <= persistence * float(rules["maximum_relative_mae_vs_persistence"])
            and global_risks[model_id]
            <= persistence + float(rules["maximum_absolute_mae_above_persistence_pp"])
        )
    ]
    persistence_id = "target_prefix_persistence"
    if bool(config["safe_prior"]["persistence_anchor_always_included"]):
        eligible = sorted(
            set(eligible) | {persistence_id},
            key=BASE_MODEL_IDS.index,
        )
    if not eligible:
        raise FastChargeSafePriorV2Error("Safe expert pool is empty")
    epsilon = float(config["safe_prior"]["risk_epsilon_pp"])
    power = float(config["safe_prior"]["risk_inverse_power"])
    raw = {
        model_id: (global_risks[model_id] + epsilon) ** (-power)
        for model_id in eligible
    }
    denominator = float(sum(raw.values()))
    weights = {
        model_id: (float(raw[model_id] / denominator) if model_id in raw else 0.0)
        for model_id in BASE_MODEL_IDS
    }
    return {
        "global_risks": global_risks,
        "eligible": eligible,
        "weights": weights,
    }


def _restricted_local_weights(
    local_risks: Mapping[str, float],
    eligible: Sequence[str],
    config: Mapping[str, object],
) -> dict[str, float]:
    epsilon = float(config["safe_prior"]["risk_epsilon_pp"])
    power = float(config["safe_prior"]["risk_inverse_power"])
    raw = {
        model_id: (local_risks[model_id] + epsilon) ** (-power) for model_id in eligible
    }
    denominator = float(sum(raw.values()))
    return {
        model_id: (float(raw[model_id] / denominator) if model_id in raw else 0.0)
        for model_id in BASE_MODEL_IDS
    }


def _safe_model_predictions(
    experts: Mapping[str, np.ndarray],
    diagnostics: Mapping[str, object],
    prior: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    eligible = [str(value) for value in prior["eligible"]]
    prior_weights = {
        model_id: float(prior["weights"][model_id]) for model_id in BASE_MODEL_IDS
    }
    local_weights = _restricted_local_weights(
        diagnostics["risks"],
        eligible,
        config,
    )
    strength = float(diagnostics["selection_strength"])
    final_weights = {
        model_id: float(
            prior_weights[model_id]
            + strength * (local_weights[model_id] - prior_weights[model_id])
        )
        for model_id in BASE_MODEL_IDS
    }
    hard_model = min(
        eligible,
        key=lambda model_id: (
            float(diagnostics["risks"][model_id]),
            BASE_MODEL_IDS.index(model_id),
        ),
    )
    matrix = np.vstack([experts[model_id] for model_id in BASE_MODEL_IDS])

    def mixed(weights: Mapping[str, float]) -> np.ndarray:
        array = np.asarray([weights[model_id] for model_id in BASE_MODEL_IDS])
        return np.sum(array[:, None] * matrix, axis=0)

    predictions = {
        **experts,
        PRIOR_MODEL_ID: mixed(prior_weights),
        HARD_MODEL_ID: experts[hard_model],
        MOE_MODEL_ID: mixed(final_weights),
    }
    if tuple(predictions) != MODEL_IDS:
        raise FastChargeSafePriorV2Error("V2 prediction registry changed")
    metadata = {
        "eligible": eligible,
        "prior_weights": prior_weights,
        "local_weights": local_weights,
        "final_weights": final_weights,
        "hard_model": hard_model,
        "dominant": min(
            BASE_MODEL_IDS,
            key=lambda model_id: (
                -final_weights[model_id],
                BASE_MODEL_IDS.index(model_id),
            ),
        ),
    }
    return (
        {
            model_id: np.clip(values.astype(float), 0.0, 110.0)
            for model_id, values in predictions.items()
        },
        metadata,
    )


def _one_hot(model_id: str) -> dict[str, float]:
    return {candidate: float(candidate == model_id) for candidate in BASE_MODEL_IDS}


def _model_metadata(
    model_id: str,
    diagnostics: Mapping[str, object],
    safe_metadata: Mapping[str, object],
) -> dict[str, object]:
    if model_id in BASE_MODEL_IDS:
        return {
            "dominant": model_id,
            "weights": _one_hot(model_id),
            "status": "base_expert_comparator",
            "action": "predict",
        }
    if model_id == PRIOR_MODEL_ID:
        return {
            "dominant": min(
                BASE_MODEL_IDS,
                key=lambda candidate: (
                    -safe_metadata["prior_weights"][candidate],
                    BASE_MODEL_IDS.index(candidate),
                ),
            ),
            "weights": safe_metadata["prior_weights"],
            "status": "safe_global_prior_comparator",
            "action": "predict",
        }
    if model_id == HARD_MODEL_ID:
        hard = str(safe_metadata["hard_model"])
        return {
            "dominant": hard,
            "weights": _one_hot(hard),
            "status": "safe_hard_risk_comparator",
            "action": "predict",
        }
    if model_id != MOE_MODEL_ID:
        raise FastChargeSafePriorV2Error("Unknown V2 model")
    return {
        "dominant": safe_metadata["dominant"],
        "weights": safe_metadata["final_weights"],
        "status": diagnostics["evidence_status"],
        "action": diagnostics["operational_action"],
    }


def _calibration(
    resources: Mapping[str, object],
    config: Mapping[str, object],
    core_config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[tuple[int, str], np.ndarray]]:
    cell_ids = sorted(resources["cells"])
    coverage = float(config["uncertainty"]["nominal_pointwise_coverage"])
    quantile_level = min(
        1.0,
        math.ceil((len(cell_ids) + 1) * coverage) / len(cell_ids),
    )
    quantiles: dict[tuple[int, str], np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for prefix_cycle in _prefix_cycles(config):
        forecast = np.arange(prefix_cycle + 1, _score_end(config) + 1, dtype=int)
        residuals = {model_id: [] for model_id in MODEL_IDS}
        for target_cell_id in cell_ids:
            references = [cell_id for cell_id in cell_ids if cell_id != target_cell_id]
            reference_errors = _reference_error_table(
                prefix_cycle,
                references,
                resources,
                core_config,
            )
            prior = _safe_prior(reference_errors, config)
            signatures = resources["signatures"][prefix_cycle]
            diagnostics = _weight_diagnostics(
                signatures[target_cell_id],
                {cell_id: signatures[cell_id] for cell_id in references},
                reference_errors,
                resources["distance_thresholds"][prefix_cycle],
                core_config,
            )
            prefix = resources["cells"][target_cell_id].loc[
                resources["cells"][target_cell_id]["cycle_index"] <= prefix_cycle
            ]
            experts, _ = _expert_predictions_for_target(
                prefix,
                signatures[target_cell_id],
                references,
                prefix_cycle,
                resources,
                core_config,
            )
            predictions, _ = _safe_model_predictions(
                experts,
                diagnostics,
                prior,
                config,
            )
            observed = resources["retentions"][target_cell_id][forecast - 1]
            for model_id in MODEL_IDS:
                residuals[model_id].append(np.abs(predictions[model_id] - observed))
        for model_id in MODEL_IDS:
            matrix = np.vstack(residuals[model_id])
            values = np.quantile(
                matrix,
                quantile_level,
                axis=0,
                method="higher",
            ).astype(float)
            quantiles[(prefix_cycle, model_id)] = values
            for index, forecast_cycle in enumerate(forecast):
                rows.append(
                    {
                        "prefix_cycle": prefix_cycle,
                        "model_id": model_id,
                        "forecast_cycle": int(forecast_cycle),
                        "calibration_cell_count": len(cell_ids),
                        "absolute_residual_quantile_level": quantile_level,
                        "calibration_half_width_pp": float(values[index]),
                    }
                )
    table = pd.DataFrame(rows, columns=CALIBRATION_COLUMNS).sort_values(
        ["prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    return table, quantiles


def predict_fastcharge_safe_prior_v2(
    training_cycles: pd.DataFrame,
    target_prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame]:
    parsed = validate_fastcharge_safe_prior_v2_config(config)
    core = _core_config(parsed)
    training = _validated_training(training_cycles, core)
    prefixes = _validated_target_prefixes(
        target_prefixes,
        set(training["cell_id"]),
        core,
    )
    resources = _training_resources(training, core)
    calibration, quantiles = _calibration(resources, parsed, core)
    training_ids = sorted(resources["cells"])
    reference_errors = {
        prefix_cycle: _reference_error_table(
            prefix_cycle,
            training_ids,
            resources,
            core,
        )
        for prefix_cycle in _prefix_cycles(parsed)
    }
    priors = {
        prefix_cycle: _safe_prior(reference_errors[prefix_cycle], parsed)
        for prefix_cycle in _prefix_cycles(parsed)
    }
    qualification_rows: list[dict[str, object]] = []
    for prefix_cycle in _prefix_cycles(parsed):
        prior = priors[prefix_cycle]
        persistence = prior["global_risks"]["target_prefix_persistence"]
        for model_id in BASE_MODEL_IDS:
            qualification_rows.append(
                {
                    "prefix_cycle": prefix_cycle,
                    "model_id": model_id,
                    "training_loo_mean_mae_pp": float(prior["global_risks"][model_id]),
                    "persistence_reference_mae_pp": float(persistence),
                    "eligible_for_safe_pool": bool(model_id in prior["eligible"]),
                    "safe_prior_weight": float(prior["weights"][model_id]),
                }
            )
    qualification = pd.DataFrame(
        qualification_rows, columns=QUALIFICATION_COLUMNS
    ).sort_values(["prefix_cycle", "model_id"], kind="stable", ignore_index=True)
    rows: list[dict[str, object]] = []
    for (paper_split, cell_id, prefix_cycle), prefix in prefixes.groupby(
        ["paper_split", "cell_id", "prefix_cycle"], sort=True
    ):
        prefix_cycle = int(prefix_cycle)
        prefix = prefix.sort_values("cycle_index", kind="stable")
        signature = _trajectory_signature(prefix, core)
        diagnostics = _weight_diagnostics(
            signature,
            resources["signatures"][prefix_cycle],
            reference_errors[prefix_cycle],
            resources["distance_thresholds"][prefix_cycle],
            core,
        )
        experts, normalization = _expert_predictions_for_target(
            prefix,
            signature,
            training_ids,
            prefix_cycle,
            resources,
            core,
        )
        predictions, safe_metadata = _safe_model_predictions(
            experts,
            diagnostics,
            priors[prefix_cycle],
            parsed,
        )
        forecast = np.arange(prefix_cycle + 1, _score_end(parsed) + 1, dtype=int)
        prefix_hash = canonical_frame_sha256(
            prefix.loc[:, TARGET_PREFIX_COLUMNS].reset_index(drop=True),
            TARGET_PREFIX_COLUMNS,
        )
        for model_id in MODEL_IDS:
            metadata = _model_metadata(model_id, diagnostics, safe_metadata)
            center = predictions[model_id]
            half_width = quantiles[(prefix_cycle, model_id)]
            lower = np.clip(center - half_width, 0.0, 110.0)
            upper = np.clip(center + half_width, 0.0, 110.0)
            is_base = model_id in BASE_MODEL_IDS
            weights_json = _json_text(metadata["weights"])
            risks_json = _json_text({} if is_base else diagnostics["risks"])
            nearest = "not_applicable" if is_base else ";".join(diagnostics["nearest"])
            distances_json = _json_text({} if is_base else diagnostics["distances"])
            shared = {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "paper_split": str(paper_split),
                "cell_id": str(cell_id),
                "training_split": str(parsed["split_and_firewall"]["training_split"]),
                "prefix_cycle": prefix_cycle,
                "score_end_cycle": _score_end(parsed),
                "model_id": model_id,
                "normalization_capacity_ah": normalization,
                "prefix_row_count": len(prefix),
                "target_prefix_sha256": prefix_hash,
                "dominant_expert_model_id": str(metadata["dominant"]),
                "expert_weights_json": weights_json,
                "expert_risks_json": risks_json,
                "nearest_training_cell_ids": nearest,
                "neighbor_distances_json": distances_json,
                "risk_margin_fraction": 0.0
                if is_base
                else float(diagnostics["risk_margin"]),
                "mean_neighbor_distance": 0.0
                if is_base
                else float(diagnostics["mean_distance"]),
                "feature_jackknife_instability_l1": 0.0
                if is_base
                else float(diagnostics["instability"]),
                "margin_support": 0.0
                if is_base
                else float(diagnostics["margin_support"]),
                "distance_support": 0.0
                if is_base
                else float(diagnostics["distance_support"]),
                "stability_support": 0.0
                if is_base
                else float(diagnostics["stability_support"]),
                "selection_strength": 0.0
                if is_base
                else float(diagnostics["selection_strength"]),
                "evidence_status": str(metadata["status"]),
                "operational_action": str(metadata["action"]),
            }
            for index, forecast_cycle in enumerate(forecast):
                rows.append(
                    {
                        **shared,
                        "forecast_cycle": int(forecast_cycle),
                        "predicted_capacity_retention_pct": float(center[index]),
                        "interval_lower_pct": float(lower[index]),
                        "interval_upper_pct": float(upper[index]),
                        "calibration_half_width_pp": float(half_width[index]),
                    }
                )
    prediction_frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": PREDICTION_MANIFEST_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": parsed["evidence_role"],
        "config_semantic_sha256": canonical_json_sha256(parsed),
        "training_cycle_sha256": canonical_frame_sha256(
            training, CANONICAL_CYCLE_COLUMNS
        ),
        "target_prefix_sha256": canonical_frame_sha256(prefixes, TARGET_PREFIX_COLUMNS),
        "calibration_table_sha256": canonical_frame_sha256(
            calibration, CALIBRATION_COLUMNS
        ),
        "qualification_table_sha256": canonical_frame_sha256(
            qualification, QUALIFICATION_COLUMNS
        ),
        "prediction_sha256": _streaming_frame_sha256(
            prediction_frame, PREDICTION_COLUMNS
        ),
        "prediction_row_count": len(prediction_frame),
        "training_cell_count": int(training["cell_id"].nunique()),
        "target_cell_count": int(prefixes["cell_id"].nunique()),
        "target_cells_by_split": {
            str(key): int(value)
            for key, value in prefixes.groupby("paper_split")["cell_id"]
            .nunique()
            .items()
        },
        "prefix_cycles": list(_prefix_cycles(parsed)),
        "score_end_cycle": _score_end(parsed),
        "model_ids": list(MODEL_IDS),
        "training_derived_distance_thresholds": {
            str(key): value for key, value in resources["distance_thresholds"].items()
        },
        "safe_pool_by_prefix": {
            str(prefix_cycle): priors[prefix_cycle]["eligible"]
            for prefix_cycle in _prefix_cycles(parsed)
        },
        "evaluation_target_future_outcomes_used": False,
        "complete_training_histories_used": True,
        "interval_calibration_target_suffix_used": False,
        "cycle_201_to_300_target_scores_exposed_before_prediction": False,
        "outcome_exposed_through_cycle_200": True,
        "inference_scope": "post_v1_development_not_independent_confirmation",
    }
    return prediction_frame, manifest, calibration, qualification


def _normalized_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise FastChargeSafePriorV2Error("V2 prediction columns changed")
    result = predictions.copy()
    integer_columns = {
        "prefix_cycle",
        "score_end_cycle",
        "forecast_cycle",
        "prefix_row_count",
    }
    float_columns = {
        "predicted_capacity_retention_pct",
        "interval_lower_pct",
        "interval_upper_pct",
        "calibration_half_width_pp",
        "normalization_capacity_ah",
        "risk_margin_fraction",
        "mean_neighbor_distance",
        "feature_jackknife_instability_l1",
        "margin_support",
        "distance_support",
        "stability_support",
        "selection_strength",
    }
    numeric_columns = integer_columns | float_columns
    for column in set(PREDICTION_COLUMNS) - numeric_columns:
        if result[column].isna().any():
            raise FastChargeSafePriorV2Error(
                f"V2 prediction string {column} cannot be null"
            )
        result[column] = result[column].astype(str)
    for column in numeric_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            raise FastChargeSafePriorV2Error(
                f"V2 prediction numeric {column} must be finite"
            )
        if column in integer_columns:
            raw = values.to_numpy(dtype=float)
            if not np.equal(raw, np.floor(raw)).all():
                raise FastChargeSafePriorV2Error(
                    f"V2 prediction {column} must be integral"
                )
            result[column] = raw.astype(np.int64)
        else:
            result[column] = values.astype(float)
    if set(result["experiment_id"]) != {EXPERIMENT_ID}:
        raise FastChargeSafePriorV2Error("V2 prediction experiment changed")
    if set(result["dataset_id"]) != {DATASET_ID}:
        raise FastChargeSafePriorV2Error("V2 prediction dataset changed")
    if set(result["model_id"]) != set(MODEL_IDS):
        raise FastChargeSafePriorV2Error("V2 prediction model registry changed")
    if not (
        (result["interval_lower_pct"] <= result["predicted_capacity_retention_pct"])
        & (result["predicted_capacity_retention_pct"] <= result["interval_upper_pct"])
    ).all():
        raise FastChargeSafePriorV2Error("V2 interval excludes its center")
    for column in (
        "predicted_capacity_retention_pct",
        "interval_lower_pct",
        "interval_upper_pct",
    ):
        if not result[column].between(0.0, 110.0).all():
            raise FastChargeSafePriorV2Error(f"V2 prediction {column} exceeds bounds")
    for column in (
        "margin_support",
        "distance_support",
        "stability_support",
        "selection_strength",
    ):
        if not result[column].between(0.0, 1.0).all():
            raise FastChargeSafePriorV2Error(f"V2 {column} must lie in [0, 1]")
    if result.duplicated(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"]
    ).any():
        raise FastChargeSafePriorV2Error("V2 prediction coordinates duplicate")
    for value in result["expert_weights_json"].unique():
        weights = json.loads(value)
        if _json_text(weights) != value or set(weights) != set(BASE_MODEL_IDS):
            raise FastChargeSafePriorV2Error("V2 expert-weight registry changed")
        numeric = np.asarray(list(weights.values()), dtype=float)
        if (numeric < 0.0).any() or not math.isclose(
            float(np.sum(numeric)), 1.0, rel_tol=0.0, abs_tol=1e-12
        ):
            raise FastChargeSafePriorV2Error("V2 expert weights must sum to one")
    return result.sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )


def _validate_replay(
    predictions: pd.DataFrame,
    manifest: Mapping[str, object],
    training: pd.DataFrame,
    prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> None:
    expected, expected_manifest, _, _ = predict_fastcharge_safe_prior_v2(
        training,
        prefixes,
        config,
    )
    if _streaming_frame_sha256(
        predictions, PREDICTION_COLUMNS
    ) != _streaming_frame_sha256(expected, PREDICTION_COLUMNS):
        raise FastChargeSafePriorV2Error(
            "V2 predictions differ from deterministic frozen replay"
        )
    if set(manifest) != set(expected_manifest):
        raise FastChargeSafePriorV2Error("V2 manifest keys changed")
    for key, expected_value in expected_manifest.items():
        if manifest[key] != expected_value:
            raise FastChargeSafePriorV2Error(f"V2 manifest mismatch for {key}")


def score_fastcharge_safe_prior_v2(
    full_cycles: pd.DataFrame,
    predictions: pd.DataFrame,
    prediction_manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    parsed = validate_fastcharge_safe_prior_v2_config(config)
    core = _core_config(parsed)
    cycles = _validated_full_cycles(full_cycles, core)
    training, prefixes, _ = build_fastcharge_prediction_inputs(cycles, parsed)
    training = _validated_training(training, core)
    prefixes = _validated_target_prefixes(
        prefixes,
        set(training["cell_id"]),
        core,
    )
    ordered = _normalized_predictions(predictions)
    expected_rows = (
        int(parsed["split_and_firewall"]["expected_total_evaluation_cells"])
        * len(MODEL_IDS)
        * sum(_score_end(parsed) - prefix for prefix in _prefix_cycles(parsed))
    )
    if len(ordered) != expected_rows:
        raise FastChargeSafePriorV2Error("V2 prediction row count changed")
    _validate_replay(ordered, prediction_manifest, training, prefixes, parsed)

    evaluation_splits = set(parsed["split_and_firewall"]["evaluation_splits"])
    truth_rows: list[pd.DataFrame] = []
    for cell_id, cell in cycles.loc[
        cycles["paper_split"].isin(evaluation_splits)
    ].groupby("cell_id", sort=True):
        cell = cell.sort_values("cycle_index", kind="stable")
        normalization = _normalization_capacity(cell)
        truth_rows.append(
            pd.DataFrame(
                {
                    "paper_split": str(cell["paper_split"].iloc[0]),
                    "cell_id": str(cell_id),
                    "forecast_cycle": cell["cycle_index"].astype(int),
                    "observed_capacity_retention_pct": _retention(cell, normalization),
                }
            )
        )
    truth = pd.concat(truth_rows, ignore_index=True)
    linked = ordered.merge(
        truth,
        on=["paper_split", "cell_id", "forecast_cycle"],
        how="left",
        validate="many_to_one",
    )
    if linked["observed_capacity_retention_pct"].isna().any():
        raise FastChargeSafePriorV2Error("V2 predictions cannot link to truth")
    score_rows: list[dict[str, object]] = []
    for (paper_split, cell_id, prefix_cycle, model_id), group in linked.groupby(
        ["paper_split", "cell_id", "prefix_cycle", "model_id"], sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        observed = group["observed_capacity_retention_pct"].to_numpy(dtype=float)
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        lower = group["interval_lower_pct"].to_numpy(dtype=float)
        upper = group["interval_upper_pct"].to_numpy(dtype=float)
        error = predicted - observed
        absolute = np.abs(error)
        inside = (observed >= lower) & (observed <= upper)
        score_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_id": DATASET_ID,
                "paper_split": str(paper_split),
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix_cycle),
                "score_end_cycle": _score_end(parsed),
                "model_id": str(model_id),
                "dominant_expert_model_id": str(
                    group["dominant_expert_model_id"].iloc[0]
                ),
                "evidence_status": str(group["evidence_status"].iloc[0]),
                "operational_action": str(group["operational_action"].iloc[0]),
                "future_observation_count": len(group),
                "trajectory_mae_pp": float(np.mean(absolute)),
                "trajectory_rmse_pp": float(np.sqrt(np.mean(np.square(error)))),
                "endpoint_absolute_error_pp": float(absolute[-1]),
                "empirical_interval_coverage_fraction": float(np.mean(inside)),
                "mean_interval_width_pp": float(np.mean(upper - lower)),
                "endpoint_inside_interval": float(inside[-1]),
            }
        )
    scores = pd.DataFrame(score_rows, columns=SCORE_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    metric_columns = list(SCORE_COLUMNS[-6:])
    by_split_prefix = (
        scores.groupby(["paper_split", "prefix_cycle", "model_id"], sort=True)[
            metric_columns
        ]
        .mean()
        .reset_index()
    )
    by_split = (
        scores.groupby(["paper_split", "model_id"], sort=True)[metric_columns]
        .mean()
        .reset_index()
    )
    overall = scores.groupby("model_id", sort=True)[metric_columns].mean().reset_index()

    def metric(model_id: str, name: str) -> float:
        return float(overall.loc[overall["model_id"] == model_id, name].iloc[0])

    moe_mae = metric(MOE_MODEL_ID, "trajectory_mae_pp")
    analog_mae = metric("nearest_neighbor_delta_transfer", "trajectory_mae_pp")
    persistence_mae = metric("target_prefix_persistence", "trajectory_mae_pp")
    gate = parsed["evaluation"]["frozen_development_gate"]
    analog_overall_pass = (moe_mae - analog_mae) <= float(
        gate["maximum_overall_mae_degradation_vs_nearest_neighbor_transfer_pp"]
    )
    split_deltas: list[dict[str, object]] = []
    split_pass = True
    for paper_split in parsed["split_and_firewall"]["evaluation_splits"]:
        moe_value = float(
            by_split.loc[
                (by_split["paper_split"] == paper_split)
                & (by_split["model_id"] == MOE_MODEL_ID),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        analog_value = float(
            by_split.loc[
                (by_split["paper_split"] == paper_split)
                & (by_split["model_id"] == "nearest_neighbor_delta_transfer"),
                "trajectory_mae_pp",
            ].iloc[0]
        )
        delta = moe_value - analog_value
        passed = delta <= float(
            gate["maximum_each_split_mae_degradation_vs_nearest_neighbor_transfer_pp"]
        )
        split_pass = split_pass and passed
        split_deltas.append(
            {
                "paper_split": paper_split,
                "moe_mae_pp": moe_value,
                "nearest_neighbor_transfer_mae_pp": analog_value,
                "delta_pp": delta,
                "passed": passed,
            }
        )
    improvement = (persistence_mae - moe_mae) / persistence_mae
    persistence_pass = improvement >= float(
        gate["minimum_relative_mae_improvement_vs_persistence"]
    )
    coverage = metric(MOE_MODEL_ID, "empirical_interval_coverage_fraction")
    width = metric(MOE_MODEL_ID, "mean_interval_width_pp")
    coverage_pass = coverage >= float(
        gate["minimum_empirical_interval_coverage_fraction"]
    )
    width_pass = width <= float(gate["maximum_mean_interval_width_pp"])
    passed_all = all(
        (
            analog_overall_pass,
            split_pass,
            persistence_pass,
            coverage_pass,
            width_pass,
        )
    )
    decisions = (
        ordered.loc[ordered["model_id"] == MOE_MODEL_ID]
        .groupby(["paper_split", "cell_id", "prefix_cycle"], sort=True)
        .first()
    )
    summary: dict[str, object] = {
        "schema_version": "lifetwin.fastcharge_safe_prior_score_summary.v2",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": DATASET_ID,
        "evidence_role": parsed["evidence_role"],
        "prediction_sha256": str(prediction_manifest["prediction_sha256"]),
        "score_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "training_cell_count": int(training["cell_id"].nunique()),
        "evaluation_cell_count": int(prefixes["cell_id"].nunique()),
        "evaluation_cells_by_split": {
            str(key): int(value)
            for key, value in prefixes.groupby("paper_split")["cell_id"]
            .nunique()
            .items()
        },
        "prefix_cycles": list(_prefix_cycles(parsed)),
        "score_end_cycle": _score_end(parsed),
        "model_ids": list(MODEL_IDS),
        "safe_pool_by_prefix": prediction_manifest["safe_pool_by_prefix"],
        "by_split_prefix_metrics": json.loads(
            _json_text(by_split_prefix.to_dict(orient="records"))
        ),
        "by_split_metrics": json.loads(_json_text(by_split.to_dict(orient="records"))),
        "overall_metrics": json.loads(_json_text(overall.to_dict(orient="records"))),
        "primary_comparison": {
            "safe_prior_moe_overall_mae_pp": moe_mae,
            "nearest_neighbor_transfer_overall_mae_pp": analog_mae,
            "delta_vs_nearest_neighbor_transfer_pp": moe_mae - analog_mae,
            "persistence_overall_mae_pp": persistence_mae,
            "relative_improvement_vs_persistence": improvement,
            "split_deltas_vs_nearest_neighbor_transfer": split_deltas,
        },
        "interval_diagnostic": {
            "safe_prior_moe_empirical_coverage_fraction": coverage,
            "safe_prior_moe_mean_width_pp": width,
            "nominal_pointwise_coverage": parsed["uncertainty"][
                "nominal_pointwise_coverage"
            ],
            "formal_exchangeable_coverage_claim": False,
        },
        "frozen_development_gate": {
            "status": "passed" if passed_all else "failed",
            "overall_analog_noninferiority_passed": analog_overall_pass,
            "all_split_analog_noninferiority_passed": split_pass,
            "persistence_improvement_passed": persistence_pass,
            "interval_coverage_passed": coverage_pass,
            "interval_width_passed": width_pass,
            "thresholds": gate,
            "interpretation": "post_v1_development_not_independent_confirmation",
        },
        "operational_action_counts": {
            str(key): int(value)
            for key, value in decisions["operational_action"].value_counts().items()
        },
        "evidence_status_counts": {
            str(key): int(value)
            for key, value in decisions["evidence_status"].value_counts().items()
        },
        "allowed_claims": list(parsed["claim_boundaries"]["allowed_claims"]),
        "prohibited_claims": list(parsed["claim_boundaries"]["prohibited_claims"]),
    }
    return scores, summary


__all__ = [
    "CONFIG_SEMANTIC_SHA256",
    "EXPERIMENT_ID",
    "FastChargeSafePriorV2Error",
    "HARD_MODEL_ID",
    "MODEL_IDS",
    "MOE_MODEL_ID",
    "PRIOR_MODEL_ID",
    "QUALIFICATION_COLUMNS",
    "load_fastcharge_safe_prior_v2_config",
    "predict_fastcharge_safe_prior_v2",
    "score_fastcharge_safe_prior_v2",
    "validate_fastcharge_safe_prior_v2_config",
]
