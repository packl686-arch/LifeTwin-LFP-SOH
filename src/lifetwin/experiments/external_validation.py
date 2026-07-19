from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.models.probabilistic import LogNormalAFT


SOURCE_MODEL_NAME = "variance_aft_source_only"
NULL_MODEL_NAME = "intercept_aft_source_only"
EXTERNAL_MODEL_NAMES = (SOURCE_MODEL_NAME, NULL_MODEL_NAME)
TARGET_FORBIDDEN_COLUMNS = {
    "cycle_life",
    "is_censored",
    "event_observed",
    "num_cycles",
    "observed_time",
    "replicate_id",
}
EXTERNAL_PREDICTION_KEY = ["dataset_id", "cell_id", "protocol_id"]
SOURCE_CROSSWALK_SHA256_COLUMN = "authoritative_crosswalk_sha256"
EXTERNAL_PREDICTION_SCHEMA_VERSION = "attia_external_predictions_v2"


def external_prediction_artifact_sha256(predictions: pd.DataFrame) -> str:
    required = {*EXTERNAL_PREDICTION_KEY, "prediction_schema_version"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing external prediction hash keys: {missing}")
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS & set(predictions.columns))
    if leaked:
        raise ValueError(f"Frozen target predictions contain outcomes: {leaked}")
    if predictions[EXTERNAL_PREDICTION_KEY].isna().any().any():
        raise ValueError("External prediction identities cannot be null")
    if predictions.duplicated(EXTERNAL_PREDICTION_KEY).any():
        raise ValueError("External prediction identities must be unique")
    if set(predictions["prediction_schema_version"].astype(str)) != {
        EXTERNAL_PREDICTION_SCHEMA_VERSION
    }:
        raise ValueError("External prediction schema version is not frozen")
    normalized = predictions.sort_values(EXTERNAL_PREDICTION_KEY, kind="stable")
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        # Twelve significant digits remain stable after CSV parse/serialize while
        # preserving substantially more precision than the reported metrics need.
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def external_target_feature_identity_sha256(
    target: pd.DataFrame,
    *,
    feature_column: str,
) -> str:
    """Hash the target identities and predictor carried into frozen predictions."""
    columns = ["dataset_id", "cell_id", "test_id", "protocol_id", feature_column]
    missing = sorted(set(columns) - set(target.columns))
    if missing:
        raise ValueError(f"Missing target feature identity columns: {missing}")
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS & set(target.columns))
    if leaked:
        raise ValueError(f"Target feature identity contains outcomes: {leaked}")
    normalized = target[columns].copy()
    if normalized.isna().any().any():
        raise ValueError("Target feature identities cannot be null")
    if normalized["cell_id"].duplicated().any() or normalized["test_id"].duplicated().any():
        raise ValueError("Target feature identities must be unique")
    normalized[feature_column] = pd.to_numeric(
        normalized[feature_column],
        errors="coerce",
    )
    if normalized[feature_column].isna().any() or not np.isfinite(
        normalized[feature_column].to_numpy(dtype=float)
    ).all():
        raise ValueError("Target feature identity values must be finite")
    payload = normalized.sort_values("cell_id", kind="stable").to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def external_outcome_artifact_sha256(outcomes: pd.DataFrame) -> str:
    """Hash the full external outcome pack before any scoring join."""
    required = {"cell_id", "protocol_id", "cycle_life", "event_observed"}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"Missing external outcome hash columns: {missing}")
    if outcomes["cell_id"].isna().any() or outcomes["cell_id"].duplicated().any():
        raise ValueError("External outcome hash identities must be unique and non-null")
    normalized = outcomes[sorted(outcomes.columns)].sort_values(
        "cell_id",
        kind="stable",
    )
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.12g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_source_frame(
    source: pd.DataFrame,
    *,
    feature_column: str,
    label_column: str,
    expected_crosswalk_sha256: str,
) -> pd.DataFrame:
    required = {
        "cell_id",
        feature_column,
        label_column,
        SOURCE_CROSSWALK_SHA256_COLUMN,
    }
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Missing source model columns: {missing}")
    if source["cell_id"].isna().any() or source["cell_id"].duplicated().any():
        raise ValueError("Source cell identities must be unique and non-null")
    crosswalk_hashes = set(
        source[SOURCE_CROSSWALK_SHA256_COLUMN].dropna().astype(str)
    )
    if len(crosswalk_hashes) != 1 or source[
        SOURCE_CROSSWALK_SHA256_COLUMN
    ].isna().any():
        raise ValueError(
            "Source features must carry one authoritative crosswalk SHA-256"
        )
    if crosswalk_hashes != {str(expected_crosswalk_sha256)}:
        raise ValueError("Source authoritative crosswalk SHA-256 does not match")
    ordered = source[["cell_id", feature_column, label_column]].copy()
    ordered[[feature_column, label_column]] = ordered[
        [feature_column, label_column]
    ].apply(pd.to_numeric, errors="coerce")
    if ordered[[feature_column, label_column]].isna().any().any():
        raise ValueError("Source features and lifetime labels must be numeric")
    numeric = ordered[[feature_column, label_column]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (ordered[label_column] <= 0).any():
        raise ValueError("Source features must be finite and lifetimes positive")
    return ordered.sort_values("cell_id", kind="stable").reset_index(drop=True)


def _validate_target_feature_pack(
    target: pd.DataFrame,
    *,
    feature_column: str,
) -> pd.DataFrame:
    leaked = sorted(TARGET_FORBIDDEN_COLUMNS & set(target.columns))
    if leaked:
        raise ValueError(f"Target feature pack contains prohibited outcome fields: {leaked}")
    required = {
        "dataset_id",
        "cell_id",
        "test_id",
        "protocol_id",
        feature_column,
    }
    missing = sorted(required - set(target.columns))
    if missing:
        raise ValueError(f"Missing target feature-pack columns: {missing}")
    if target[["dataset_id", "cell_id", "test_id", "protocol_id"]].isna().any().any():
        raise ValueError("Target feature-pack identities cannot be null")
    if target["cell_id"].duplicated().any() or target["test_id"].duplicated().any():
        raise ValueError("Target feature pack must contain one row per cell and test")
    ordered = target.copy()
    ordered[feature_column] = pd.to_numeric(ordered[feature_column], errors="coerce")
    if ordered[feature_column].isna().any() or not np.isfinite(
        ordered[feature_column].to_numpy(dtype=float)
    ).all():
        raise ValueError("Target features must be finite numeric values")
    return ordered.sort_values("cell_id", kind="stable").reset_index(drop=True)


def _model_manifest(model: LogNormalAFT, *, feature_column: str) -> dict[str, object]:
    return {
        "feature_columns": [feature_column],
        "feature_mean": [float(value) for value in model.feature_mean_],
        "feature_scale": [float(value) for value in model.feature_scale_],
        "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_],
        "scaled_intercept": float(model.scaled_intercept_),
        "scaled_coefficients": [float(value) for value in model.scaled_coef_],
        "sigma_log_life": float(model.sigma_),
        "fit": model.fit_summary_.to_dict(),
        "l2_penalty": float(model.l2_penalty),
    }


def build_source_only_external_predictions(
    source: pd.DataFrame,
    target_features: pd.DataFrame,
    *,
    feature_column: str = "log10_delta_q_variance",
    label_column: str = "cycle_life",
    l2_penalty: float = 1e-4,
    expected_source_crosswalk_sha256: str,
    expected_source_count: int | None = 124,
    expected_target_count: int | None = 45,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit only on source outcomes and freeze label-free target predictions."""
    source_frame = _validate_source_frame(
        source,
        feature_column=feature_column,
        label_column=label_column,
        expected_crosswalk_sha256=expected_source_crosswalk_sha256,
    )
    target = _validate_target_feature_pack(
        target_features,
        feature_column=feature_column,
    )
    if expected_source_count is not None and len(source_frame) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} source cells, found {len(source_frame)}"
        )
    if expected_target_count is not None and len(target) != expected_target_count:
        raise ValueError(
            f"Expected {expected_target_count} target cells, found {len(target)}"
        )
    overlap = set(source_frame["cell_id"].astype(str)) & set(target["cell_id"].astype(str))
    if overlap:
        raise ValueError(f"Source and target cell identities overlap: {sorted(overlap)}")

    source_feature = source_frame[[feature_column]].to_numpy(dtype=float)
    source_lifetime = source_frame[label_column].to_numpy(dtype=float)
    observed = np.zeros(len(source_frame), dtype=bool)
    variance_model = LogNormalAFT(l2_penalty=l2_penalty).fit(
        source_feature,
        source_lifetime,
        is_censored=observed,
    )
    null_model = LogNormalAFT(l2_penalty=l2_penalty).fit(
        np.zeros((len(source_frame), 1), dtype=float),
        source_lifetime,
        is_censored=observed,
    )
    target_feature = target[[feature_column]].to_numpy(dtype=float)
    target_null = np.zeros((len(target), 1), dtype=float)
    quantile_levels = np.array([0.1, 0.5, 0.9])
    variance_quantiles = variance_model.predict_quantile(
        target_feature,
        quantile_levels,
    )
    null_quantiles = null_model.predict_quantile(target_null, quantile_levels)

    identity_columns = [
        column
        for column in (
            "dataset_id",
            "cell_id",
            "test_id",
            "protocol_id",
            "campaign_id",
            "source_cell_id",
            feature_column,
        )
        if column in target
    ]
    predictions = target[identity_columns].copy()
    predictions["prediction_schema_version"] = EXTERNAL_PREDICTION_SCHEMA_VERSION
    for model_name, model, matrix, quantiles in (
        (SOURCE_MODEL_NAME, variance_model, target_feature, variance_quantiles),
        (NULL_MODEL_NAME, null_model, target_null, null_quantiles),
    ):
        predictions[f"{model_name}_log_location"] = model.predict_log_location(matrix)
        predictions[f"{model_name}_predictive_sigma"] = float(model.sigma_)
        predictions[f"{model_name}_p10"] = quantiles[:, 0]
        predictions[f"{model_name}_p50"] = quantiles[:, 1]
        predictions[f"{model_name}_p90"] = quantiles[:, 2]
    predictions = predictions.sort_values("cell_id", kind="stable").reset_index(drop=True)
    frozen_hash = external_prediction_artifact_sha256(predictions)
    source_mean = float(source_frame[feature_column].mean())
    source_std = float(source_frame[feature_column].std(ddof=0))
    target_mean = float(target[feature_column].mean())
    manifest: dict[str, object] = {
        "status": "predictions_frozen_before_target_outcome_access",
        "prediction_schema_version": EXTERNAL_PREDICTION_SCHEMA_VERSION,
        "source_cell_count": len(source_frame),
        "target_cell_count": len(target),
        "feature_column": feature_column,
        "label_column_used_on_source_only": label_column,
        "target_outcome_columns_seen": [],
        "l2_penalty": float(l2_penalty),
        "source_label_authority": {
            "column": SOURCE_CROSSWALK_SHA256_COLUMN,
            "sha256": str(expected_source_crosswalk_sha256),
            "verified_before_fit": True,
        },
        "target_feature_identity_sha256": external_target_feature_identity_sha256(
            target,
            feature_column=feature_column,
        ),
        "models": {
            SOURCE_MODEL_NAME: _model_manifest(
                variance_model,
                feature_column=feature_column,
            ),
            NULL_MODEL_NAME: _model_manifest(
                null_model,
                feature_column="constant_zero",
            ),
        },
        "label_free_shift_diagnostic": {
            "source_feature_mean": source_mean,
            "source_feature_std": source_std,
            "target_feature_mean": target_mean,
            "target_mean_shift_in_source_sd": (
                (target_mean - source_mean) / source_std if source_std > 0 else None
            ),
            "target_below_source_min_count": int(
                (target[feature_column] < source_frame[feature_column].min()).sum()
            ),
            "target_above_source_max_count": int(
                (target[feature_column] > source_frame[feature_column].max()).sum()
            ),
        },
        "frozen_prediction_sha256": frozen_hash,
        "guardrails": {
            "target_used_for_scaling": False,
            "target_used_for_feature_selection": False,
            "target_used_for_l2_selection": False,
            "target_used_for_sigma_estimation": False,
            "target_used_for_calibration": False,
        },
    }
    return predictions, manifest


def _lognormal_row_nll(
    log_location: np.ndarray,
    sigma: np.ndarray,
    lifetime: np.ndarray,
) -> np.ndarray:
    if (
        not np.isfinite(log_location).all()
        or not np.isfinite(sigma).all()
        or not np.isfinite(lifetime).all()
        or (sigma <= 0).any()
        or (lifetime <= 0).any()
    ):
        raise ValueError("NLL inputs must be finite with positive scale and lifetime")
    log_lifetime = np.log(lifetime)
    z_score = (log_lifetime - log_location) / sigma
    return (
        log_lifetime
        + np.log(sigma)
        + 0.5 * z_score**2
        + 0.5 * math.log(2.0 * math.pi)
    )


def _metric_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    lifetime = scored["cycle_life"].to_numpy(dtype=float)
    for model_name in EXTERNAL_MODEL_NAMES:
        median = scored[f"{model_name}_p50"].to_numpy(dtype=float)
        error = median - lifetime
        nll = _lognormal_row_nll(
            scored[f"{model_name}_log_location"].to_numpy(dtype=float),
            scored[f"{model_name}_predictive_sigma"].to_numpy(dtype=float),
            lifetime,
        )
        coverage = (
            (lifetime >= scored[f"{model_name}_p10"].to_numpy(dtype=float))
            & (lifetime <= scored[f"{model_name}_p90"].to_numpy(dtype=float))
        )
        for index, source_row in enumerate(scored.itertuples(index=False)):
            rows.append(
                {
                    "cell_id": str(source_row.cell_id),
                    "protocol_id": str(source_row.protocol_id),
                    "model": model_name,
                    "cycle_life": float(lifetime[index]),
                    "predicted_median": float(median[index]),
                    "absolute_error": abs(float(error[index])),
                    "absolute_percentage_error": abs(float(error[index]))
                    / float(lifetime[index]),
                    "squared_error": float(error[index] ** 2),
                    "negative_log_likelihood": float(nll[index]),
                    "raw_central_80_covered": bool(coverage[index]),
                }
            )
    return pd.DataFrame(rows)


def _aggregate_metrics(rows: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for keys, group in rows.groupby([*group_columns, "model"], sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip([*group_columns, "model"], keys, strict=True))
        output.append(
            {
                **identity,
                "cell_count": len(group),
                "mean_negative_log_likelihood": float(
                    group["negative_log_likelihood"].mean()
                ),
                "mae_cycles": float(group["absolute_error"].mean()),
                "rmse_cycles": float(np.sqrt(group["squared_error"].mean())),
                "mape_fraction": float(group["absolute_percentage_error"].mean()),
                "raw_central_80_coverage": float(
                    group["raw_central_80_covered"].mean()
                ),
            }
        )
    return pd.DataFrame(output)


def _protocol_bootstrap(
    protocol_metrics: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    pivot = protocol_metrics.pivot(
        index="protocol_id",
        columns="model",
        values=["mean_negative_log_likelihood", "mape_fraction"],
    )
    if pivot.isna().any().any():
        raise ValueError("Every protocol must contain both external models")
    delta_nll = (
        pivot[("mean_negative_log_likelihood", SOURCE_MODEL_NAME)]
        - pivot[("mean_negative_log_likelihood", NULL_MODEL_NAME)]
    ).to_numpy(dtype=float)
    delta_mape = (
        pivot[("mape_fraction", SOURCE_MODEL_NAME)]
        - pivot[("mape_fraction", NULL_MODEL_NAME)]
    ).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta_nll), size=(resamples, len(delta_nll)))
    bootstrap_nll = delta_nll[indices].mean(axis=1)
    bootstrap_mape = delta_mape[indices].mean(axis=1)
    return {
        "unit": "protocol",
        "protocol_count": len(delta_nll),
        "resamples": int(resamples),
        "seed": int(seed),
        "delta_definition": "variance_aft_minus_intercept_aft; negative_is_better",
        "delta_nll_mean": float(delta_nll.mean()),
        "delta_nll_95_interval": [
            float(np.quantile(bootstrap_nll, 0.025)),
            float(np.quantile(bootstrap_nll, 0.975)),
        ],
        "delta_mape_mean": float(delta_mape.mean()),
        "delta_mape_95_interval": [
            float(np.quantile(bootstrap_mape, 0.025)),
            float(np.quantile(bootstrap_mape, 0.975)),
        ],
    }


def score_source_only_external_predictions(
    predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
    expected_outcome_sha256: str,
    expected_outcome_identity: Mapping[str, object],
    validation_thresholds: dict[str, object],
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 20260719,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Verify the frozen pack before joining authoritative external outcomes."""
    observed_hash = external_prediction_artifact_sha256(predictions)
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Frozen external prediction hash does not match prediction content")
    observed_outcome_hash = external_outcome_artifact_sha256(outcomes)
    if observed_outcome_hash != expected_outcome_sha256:
        raise ValueError("External outcome pack hash does not match frozen authority")
    if not expected_outcome_identity:
        raise ValueError("External outcome authority identity cannot be empty")
    for column, expected in expected_outcome_identity.items():
        if column not in outcomes:
            raise ValueError(f"Missing external outcome authority column: {column}")
        observed = set(outcomes[column].tolist())
        if observed != {expected}:
            raise ValueError(
                f"External outcome authority mismatch for {column}: {observed}"
            )
    required_outcomes = {"cell_id", "protocol_id", "cycle_life", "event_observed"}
    missing = sorted(required_outcomes - set(outcomes.columns))
    if missing:
        raise ValueError(f"Missing external outcome columns: {missing}")
    outcome_frame = outcomes[list(required_outcomes)].copy()
    if outcome_frame["cell_id"].isna().any() or outcome_frame["cell_id"].duplicated().any():
        raise ValueError("External outcome cell identities must be unique and non-null")
    outcome_frame["cycle_life"] = pd.to_numeric(
        outcome_frame["cycle_life"],
        errors="coerce",
    )
    if (
        outcome_frame["cycle_life"].isna().any()
        or not np.isfinite(outcome_frame["cycle_life"].to_numpy(dtype=float)).all()
        or (outcome_frame["cycle_life"] <= 0).any()
    ):
        raise ValueError("External lifetimes must be finite and positive")
    if outcome_frame["event_observed"].dtype != bool or not outcome_frame[
        "event_observed"
    ].all():
        raise ValueError("This Attia validation cohort must contain 45 observed events")
    outcome_frame = outcome_frame.rename(columns={"protocol_id": "outcome_protocol_id"})
    scored = predictions.merge(
        outcome_frame,
        on="cell_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (scored["_merge"] != "both").any() or scored["cycle_life"].isna().any():
        raise ValueError("Every frozen target prediction must match one outcome")
    if not (
        scored["protocol_id"].astype(str)
        == scored["outcome_protocol_id"].astype(str)
    ).all():
        raise ValueError("Prediction and outcome protocol identities disagree")
    scored = scored.drop(columns=["_merge", "outcome_protocol_id"])
    if len(scored) != 45 or scored["protocol_id"].nunique() != 9:
        raise ValueError("Expected the 45-cell, 9-protocol Attia validation cohort")
    protocol_counts = scored["protocol_id"].value_counts()
    if not (protocol_counts == 5).all():
        raise ValueError("Each Attia validation protocol must contain five cells")

    cell_metric_rows = _metric_rows(scored)
    protocol_metrics = _aggregate_metrics(cell_metric_rows, ["protocol_id"])
    cell_metrics = _aggregate_metrics(cell_metric_rows, [])
    bootstrap = _protocol_bootstrap(
        protocol_metrics,
        resamples=bootstrap_resamples,
        seed=bootstrap_seed,
    )
    pivot = protocol_metrics.pivot(
        index="protocol_id",
        columns="model",
        values=["mean_negative_log_likelihood", "mape_fraction"],
    )
    variance_nll = float(
        pivot[("mean_negative_log_likelihood", SOURCE_MODEL_NAME)].mean()
    )
    null_nll = float(pivot[("mean_negative_log_likelihood", NULL_MODEL_NAME)].mean())
    variance_mape = float(pivot[("mape_fraction", SOURCE_MODEL_NAME)].mean())
    null_mape = float(pivot[("mape_fraction", NULL_MODEL_NAME)].mean())
    nll_delta = variance_nll - null_nll
    mape_improvement = 1.0 - variance_mape / null_mape
    improved_protocols = int(
        (
            pivot[("mape_fraction", SOURCE_MODEL_NAME)]
            < pivot[("mape_fraction", NULL_MODEL_NAME)]
        ).sum()
    )
    passed = (
        nll_delta < float(validation_thresholds["maximum_delta_nll"])
        and mape_improvement
        >= float(validation_thresholds["minimum_mape_improvement_fraction"])
        and improved_protocols
        >= int(validation_thresholds["minimum_protocols_with_mape_improvement"])
    )
    result: dict[str, object] = {
        "status": "external_signal_gate_passed" if passed else "external_signal_gate_failed",
        "cohort": {
            "cell_count": len(scored),
            "protocol_count": int(scored["protocol_id"].nunique()),
            "cells_per_protocol": sorted(protocol_counts.unique().astype(int).tolist()),
            "observed_event_count": int(scored["event_observed"].sum()),
            "right_censored_count": 0,
        },
        "protocol_balanced_primary_metrics": {
            SOURCE_MODEL_NAME: {
                "mean_negative_log_likelihood": variance_nll,
                "mape_fraction": variance_mape,
            },
            NULL_MODEL_NAME: {
                "mean_negative_log_likelihood": null_nll,
                "mape_fraction": null_mape,
            },
            "delta_nll": nll_delta,
            "mape_improvement_fraction": mape_improvement,
            "protocols_with_mape_improvement": improved_protocols,
        },
        "cell_pooled_descriptive_metrics": cell_metrics.to_dict(orient="records"),
        "protocol_cluster_bootstrap": bootstrap,
        "signal_gate": {
            "status": "passed" if passed else "failed",
            "thresholds": dict(validation_thresholds),
        },
        "calibration_gate": {
            "status": "blocked_insufficient_external_groups",
            "external_group_count": 9,
            "minimum_required_groups": 30,
            "raw_interval_coverage_is_diagnostic_only": True,
        },
        "censoring_gate": {
            "status": "not_applicable_all_events_observed",
            "right_censored_count": 0,
            "claim": "This cohort does not validate right-censoring behavior.",
        },
        "prediction_firewall": {
            "verified_prediction_sha256": observed_hash,
            "outcome_join_after_hash_verification": True,
            "verified_outcome_sha256": observed_outcome_hash,
            "outcome_authority_verified_before_join": True,
        },
        "claim_boundary": (
            "Retrospective same-cell-model, same-laboratory, later-campaign protocol "
            "transfer only; not an independent product, storage, Hithium, or 15-25 year "
            "validation."
        ),
    }
    return result, cell_metric_rows, protocol_metrics
