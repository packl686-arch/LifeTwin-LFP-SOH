from __future__ import annotations

import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from lifetwin.evaluation.probabilistic import (
    evaluate_prediction_interval,
    evaluate_quantile,
)
from lifetwin.models.baselines import regression_metrics
from lifetwin.models.probabilistic import LogNormalAFT
from lifetwin.models.reference import (
    GaussianOffsetPosterior,
    prior_predictive_quantiles,
)


QUANTILE_LEVELS = np.array([0.1, 0.5, 0.9])
METRIC_COLUMNS = (
    "mape",
    "mae",
    "rmse",
    "negative_log_likelihood",
    "central_80_coverage",
    "central_80_mean_width",
    "mean_pinball_loss",
    "mean_log_error",
)
MODEL_NAMES = (
    "zero_shot",
    "target_reference_shrinkage",
    "target_reference_unshrunk",
    "source_reference_negative_control",
)
LOWER_IS_BETTER_METRICS = {
    "mape",
    "mae",
    "rmse",
    "negative_log_likelihood",
    "mean_pinball_loss",
}
PARTITION_COLUMNS = ("target_batch", "repeat", "k")


def _stable_rank(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def nested_reference_partition(
    frame: pd.DataFrame,
    *,
    maximum_reference_count: int,
    repeat_index: int,
    seed: int,
    domain_value: object,
    identity_column: str = "cell_id",
) -> tuple[list[str], list[str]]:
    """Return a stable max-k reference order and a common query set."""
    if identity_column not in frame:
        raise ValueError(f"Missing identity column: {identity_column}")
    if frame.empty or frame[identity_column].isna().any():
        raise ValueError("Reference partition identities must be non-null")
    if frame[identity_column].duplicated().any():
        raise ValueError("Reference partition identities must be unique")
    if maximum_reference_count < 1:
        raise ValueError("maximum_reference_count must be positive")
    if repeat_index < 0:
        raise ValueError("repeat_index cannot be negative")
    identities = [str(value) for value in frame[identity_column]]
    if len(identities) <= maximum_reference_count:
        raise ValueError("A common query requires more target rows than max k")
    ranked = sorted(
        identities,
        key=lambda identity: _stable_rank(
            seed, domain_value, repeat_index, identity
        ),
    )
    return ranked[:maximum_reference_count], ranked[maximum_reference_count:]


def _prediction_metrics(
    truth: np.ndarray,
    quantiles: np.ndarray,
    *,
    negative_log_likelihood: float,
) -> dict[str, float]:
    point = regression_metrics(truth, quantiles[:, 1])
    interval = evaluate_prediction_interval(
        truth,
        quantiles[:, 0],
        quantiles[:, 2],
        nominal_coverage=0.8,
    )
    pinball = np.mean(
        [
            evaluate_quantile(
                truth,
                quantiles[:, index],
                float(level),
            ).pinball_loss
            for index, level in enumerate(QUANTILE_LEVELS)
        ]
    )
    return {
        "mape": point.mape,
        "mae": point.mae,
        "rmse": point.rmse,
        "negative_log_likelihood": float(negative_log_likelihood),
        "central_80_coverage": interval.empirical_coverage,
        "central_80_mean_width": interval.mean_width,
        "mean_pinball_loss": float(pinball),
        "mean_log_error": float(np.mean(np.log(quantiles[:, 1]) - np.log(truth))),
    }


def _lognormal_negative_log_likelihood(
    log_location: np.ndarray,
    predictive_sigma: np.ndarray,
    truth: np.ndarray,
) -> float:
    location = np.asarray(log_location, dtype=float)
    sigma = np.asarray(predictive_sigma, dtype=float)
    duration = np.asarray(truth, dtype=float)
    if location.shape != duration.shape or sigma.shape != duration.shape:
        raise ValueError("NLL locations, sigmas, and truth must have matching shapes")
    if (
        not np.isfinite(location).all()
        or not np.isfinite(sigma).all()
        or not np.isfinite(duration).all()
        or (sigma <= 0).any()
        or (duration <= 0).any()
    ):
        raise ValueError("NLL inputs must be finite with positive sigma and truth")
    log_time = np.log(duration)
    z_score = (log_time - location) / sigma
    return float(
        np.mean(
            log_time
            + np.log(sigma)
            + 0.5 * z_score**2
            + 0.5 * math.log(2 * math.pi)
        )
    )


def prediction_artifact_sha256(
    predictions: pd.DataFrame,
    *,
    identity_column: str = "cell_id",
) -> str:
    """Hash an ID-bound, label-free prediction table in canonical row order."""
    key_columns = ["target_batch", "repeat", "k", identity_column]
    missing = [column for column in key_columns if column not in predictions]
    if missing:
        raise ValueError(f"Missing prediction hash key columns: {missing}")
    if predictions[key_columns].isna().any().any():
        raise ValueError("Prediction hash keys cannot be null")
    if predictions.duplicated(key_columns).any():
        raise ValueError("Prediction hash keys must be unique")
    normalized = predictions.sort_values(key_columns, kind="stable")
    payload = normalized.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_batch_reference_proxy_predictions(
    predictions: pd.DataFrame,
    query_labels: pd.DataFrame,
    *,
    frozen_prediction_sha256: str,
    identity_column: str = "cell_id",
    label_column: str = "cycle_life",
) -> pd.DataFrame:
    """Join query outcomes only after verifying the frozen prediction artifact."""
    if label_column in predictions:
        raise ValueError("Frozen predictions must not contain query outcomes")
    observed_hash = prediction_artifact_sha256(
        predictions,
        identity_column=identity_column,
    )
    if observed_hash != frozen_prediction_sha256:
        raise ValueError("Frozen prediction hash does not match prediction content")
    required_labels = {identity_column, label_column}
    missing = sorted(required_labels - set(query_labels.columns))
    if missing:
        raise ValueError(f"Missing query scoring columns: {missing}")
    labels = query_labels[[identity_column, label_column]].copy()
    if labels[identity_column].isna().any() or labels[identity_column].duplicated().any():
        raise ValueError("Query scoring identities must be unique and non-null")
    labels[identity_column] = labels[identity_column].astype(str)
    labels[label_column] = pd.to_numeric(labels[label_column], errors="coerce")
    if (
        labels[label_column].isna().any()
        or not np.isfinite(labels[label_column].to_numpy(dtype=float)).all()
        or (labels[label_column] <= 0).any()
    ):
        raise ValueError("Query scoring lifetimes must be finite and positive")
    scored = predictions.merge(
        labels,
        on=identity_column,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (scored["_merge"] != "both").any() or scored[label_column].isna().any():
        raise ValueError("Every frozen prediction must match exactly one query label")
    scored = scored.drop(columns="_merge")

    metric_rows: list[dict[str, object]] = []
    for (target_batch, repeat, k_shot), rows in scored.groupby(
        ["target_batch", "repeat", "k"], sort=True
    ):
        truth = rows[label_column].to_numpy(dtype=float)
        for model_name in MODEL_NAMES:
            quantiles = rows[
                [
                    f"{model_name}_p10",
                    f"{model_name}_p50",
                    f"{model_name}_p90",
                ]
            ].to_numpy(dtype=float)
            negative_log_likelihood = _lognormal_negative_log_likelihood(
                rows[f"{model_name}_log_location"].to_numpy(dtype=float),
                rows[f"{model_name}_predictive_sigma"].to_numpy(dtype=float),
                truth,
            )
            metric_rows.append(
                {
                    "target_batch": target_batch,
                    "repeat": int(repeat),
                    "k": int(k_shot),
                    "model": model_name,
                    "query_count": len(rows),
                    **_prediction_metrics(
                        truth,
                        quantiles,
                        negative_log_likelihood=negative_log_likelihood,
                    ),
                }
            )
    return pd.DataFrame(metric_rows)


def score_partitioned_batch_reference_proxy_predictions(
    predictions: pd.DataFrame,
    query_labels: pd.DataFrame,
    partition_freezes: pd.DataFrame,
    *,
    identity_column: str = "cell_id",
    label_column: str = "cycle_life",
) -> pd.DataFrame:
    """Verify and score each cross-fit partition independently.

    Repeated diagnostic partitions may reuse one cell as a reference in one
    repeat and a query in another. The leakage boundary is therefore the
    target-batch/repeat/k partition, not the aggregate prediction file.
    """
    freeze_columns = [*PARTITION_COLUMNS, "partition_prediction_sha256"]
    missing = [column for column in freeze_columns if column not in partition_freezes]
    if missing:
        raise ValueError(f"Missing partition freeze columns: {missing}")
    freezes = partition_freezes[freeze_columns].copy()
    if freezes[freeze_columns].isna().any().any():
        raise ValueError("Partition freeze keys and hashes cannot be null")
    if freezes.duplicated(list(PARTITION_COLUMNS)).any():
        raise ValueError("Partition freezes must be unique")

    grouped_predictions = {
        tuple(key): rows.copy()
        for key, rows in predictions.groupby(list(PARTITION_COLUMNS), sort=True)
    }
    freeze_map = {
        tuple(row[column] for column in PARTITION_COLUMNS): str(
            row["partition_prediction_sha256"]
        )
        for row in freezes.to_dict(orient="records")
    }
    if set(grouped_predictions) != set(freeze_map):
        raise ValueError("Partition freeze keys do not match prediction partitions")

    metric_frames: list[pd.DataFrame] = []
    for key in sorted(grouped_predictions, key=lambda value: (str(value[0]), *value[1:])):
        metric_frames.append(
            score_batch_reference_proxy_predictions(
                grouped_predictions[key],
                query_labels,
                frozen_prediction_sha256=freeze_map[key],
                identity_column=identity_column,
                label_column=label_column,
            )
        )
    return pd.concat(metric_frames, ignore_index=True).reset_index(drop=True)


def _proxy_evidence_gate(
    observed_target_batch_count: int,
    *,
    minimum_target_batch_count: int = 30,
) -> dict[str, object]:
    """Keep industrial evidence blocked while reporting the count check honestly."""
    count_status = (
        "passed"
        if observed_target_batch_count >= minimum_target_batch_count
        else "blocked"
    )
    blocking_reasons = ["proxy_dataset_not_target_industrial_evidence"]
    if count_status == "blocked":
        blocking_reasons.append("insufficient_independent_target_batches")
    return {
        "status": "blocked",
        "scope": "industrial_external_validity",
        "independent_target_batch_count_check": {
            "status": count_status,
            "observed": int(observed_target_batch_count),
            "minimum": int(minimum_target_batch_count),
        },
        "blocking_reasons": blocking_reasons,
        "reason": (
            "A repeated public-data batch proxy is not target-industrial evidence; "
            "the independent-domain count is reported as a separate check."
        ),
    }


def _comparison_rule(metric: str) -> str:
    if metric in LOWER_IS_BETTER_METRICS:
        return "lower"
    if metric == "central_80_coverage":
        return "closer_to_nominal_0.8"
    if metric == "mean_log_error":
        return "closer_to_zero"
    if metric == "central_80_mean_width":
        return "descriptive_only_without_coverage_constraint"
    raise ValueError(f"Unknown comparison metric: {metric}")


def _favorable_mask(
    baseline: pd.Series | np.ndarray,
    comparator: pd.Series | np.ndarray,
    *,
    metric: str,
) -> np.ndarray | None:
    baseline_values = np.asarray(baseline, dtype=float)
    comparator_values = np.asarray(comparator, dtype=float)
    if metric in LOWER_IS_BETTER_METRICS:
        return comparator_values < baseline_values
    if metric == "central_80_coverage":
        return np.abs(comparator_values - 0.8) < np.abs(baseline_values - 0.8)
    if metric == "mean_log_error":
        return np.abs(comparator_values) < np.abs(baseline_values)
    if metric == "central_80_mean_width":
        return None
    raise ValueError(f"Unknown comparison metric: {metric}")


def _posterior_from_rows(
    model: LogNormalAFT,
    rows: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    prior_std: float,
    shrink: bool,
) -> GaussianOffsetPosterior:
    location = model.predict_log_location(rows[feature_columns].to_numpy(dtype=float))
    residuals = np.log(rows[label_column].to_numpy(dtype=float)) - location
    if shrink:
        return GaussianOffsetPosterior.from_normal_prior(
            residuals,
            residual_sigma=model.sigma_,
            prior_std=prior_std,
        )
    return GaussianOffsetPosterior.from_flat_prior(
        residuals,
        residual_sigma=model.sigma_,
    )


def build_reference_proxy_prediction_state(
    model: LogNormalAFT,
    *,
    target_references: pd.DataFrame,
    source_controls: pd.DataFrame,
    query_features: np.ndarray,
    feature_columns: Sequence[str],
    label_column: str,
    prior_std: float,
    quantile_levels: np.ndarray = QUANTILE_LEVELS,
) -> dict[str, object]:
    """Create all proxy predictions without accepting query outcomes."""
    features = list(feature_columns)
    base_location = model.predict_log_location(query_features)
    target_posterior = _posterior_from_rows(
        model,
        target_references,
        feature_columns=features,
        label_column=label_column,
        prior_std=prior_std,
        shrink=True,
    )
    unshrunk_posterior = _posterior_from_rows(
        model,
        target_references,
        feature_columns=features,
        label_column=label_column,
        prior_std=prior_std,
        shrink=False,
    )
    control_posterior = _posterior_from_rows(
        model,
        source_controls,
        feature_columns=features,
        label_column=label_column,
        prior_std=prior_std,
        shrink=True,
    )
    quantiles = {
        "zero_shot": prior_predictive_quantiles(
            base_location,
            residual_sigma=model.sigma_,
            prior_std=prior_std,
            quantiles=quantile_levels,
        ),
        "target_reference_shrinkage": target_posterior.predict_quantile(
            base_location, quantile_levels
        ),
        "target_reference_unshrunk": unshrunk_posterior.predict_quantile(
            base_location, quantile_levels
        ),
        "source_reference_negative_control": control_posterior.predict_quantile(
            base_location, quantile_levels
        ),
    }
    locations = {
        "zero_shot": base_location,
        "target_reference_shrinkage": target_posterior.predict_log_location(
            base_location
        ),
        "target_reference_unshrunk": unshrunk_posterior.predict_log_location(
            base_location
        ),
        "source_reference_negative_control": control_posterior.predict_log_location(
            base_location
        ),
    }
    sigmas = {
        "zero_shot": math.hypot(model.sigma_, prior_std),
        "target_reference_shrinkage": target_posterior.predictive_sigma,
        "target_reference_unshrunk": unshrunk_posterior.predictive_sigma,
        "source_reference_negative_control": control_posterior.predictive_sigma,
    }
    return {
        "base_location": base_location,
        "quantiles": quantiles,
        "locations": locations,
        "sigmas": sigmas,
        "target_posterior": target_posterior,
        "unshrunk_posterior": unshrunk_posterior,
        "control_posterior": control_posterior,
    }


def _aggregate_metric_rows(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"repeat_count": int(frame["repeat"].nunique())}
    for metric in METRIC_COLUMNS:
        values = frame[metric].to_numpy(dtype=float)
        result[metric] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return result


def _equal_batch_summary(metric_frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    for k_shot, k_rows in metric_frame.groupby("k", sort=True):
        model_summary: dict[str, object] = {}
        for model_name, model_rows in k_rows.groupby("model", sort=True):
            batch_means = model_rows.groupby("target_batch")[list(METRIC_COLUMNS)].mean()
            model_summary[model_name] = {
                "target_batch_count": len(batch_means),
                **{
                    metric: float(batch_means[metric].mean())
                    for metric in METRIC_COLUMNS
                },
            }
        paired = k_rows.pivot_table(
            index=["target_batch", "repeat"],
            columns="model",
            values=list(METRIC_COLUMNS),
            aggfunc="first",
        )
        deltas: dict[str, object] = {}
        for comparator in (
            "target_reference_shrinkage",
            "target_reference_unshrunk",
            "source_reference_negative_control",
        ):
            comparator_delta: dict[str, object] = {}
            for metric in METRIC_COLUMNS:
                baseline_values = paired[(metric, "zero_shot")]
                comparator_values = paired[(metric, comparator)]
                difference = comparator_values - baseline_values
                batch_delta = difference.groupby(level="target_batch").mean()
                baseline_by_batch = baseline_values.groupby(
                    level="target_batch"
                ).mean()
                comparator_by_batch = comparator_values.groupby(
                    level="target_batch"
                ).mean()
                favorable = _favorable_mask(
                    baseline_by_batch,
                    comparator_by_batch,
                    metric=metric,
                )
                comparator_delta[f"delta_{metric}"] = float(batch_delta.mean())
                comparator_delta[f"delta_{metric}_by_batch"] = {
                    str(batch): float(value) for batch, value in batch_delta.items()
                }
                comparator_delta[f"comparison_rule_{metric}"] = _comparison_rule(
                    metric
                )
                comparator_delta[f"favorable_batch_count_{metric}"] = (
                    None if favorable is None else int(favorable.sum())
                )
                comparator_delta[f"decreased_batch_count_{metric}"] = int(
                    (batch_delta < 0).sum()
                )
            deltas[f"{comparator}_minus_zero_shot"] = comparator_delta
        summary[str(int(k_shot))] = {
            "models": model_summary,
            "equal_batch_paired_deltas": deltas,
        }
    return summary


def _training_domain_stability(fits: dict[str, object]) -> dict[str, object]:
    coefficient_names = sorted(
        {
            name
            for fit in fits.values()
            for name in fit["coefficients"]
        }
    )
    coefficient_summary: dict[str, object] = {}
    for name in coefficient_names:
        by_fold = {
            target: float(fit["coefficients"][name])
            for target, fit in fits.items()
        }
        values = np.asarray(list(by_fold.values()), dtype=float)
        nonzero = values[np.abs(values) > 1e-12]
        coefficient_summary[name] = {
            "by_held_out_target_batch": by_fold,
            "mean": float(values.mean()),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "sign_consistent": bool(
                len(nonzero) == 0
                or np.all(nonzero > 0)
                or np.all(nonzero < 0)
            ),
        }
    sigma_by_fold = {
        target: float(fit["residual_sigma"]) for target, fit in fits.items()
    }
    sigma_values = np.asarray(list(sigma_by_fold.values()), dtype=float)
    return {
        "status": "diagnostic_only",
        "outer_fold_count": len(fits),
        "perturbation": "leave_one_batch_out_training_domain_composition",
        "coefficients": coefficient_summary,
        "residual_sigma": {
            "by_held_out_target_batch": sigma_by_fold,
            "mean": float(sigma_values.mean()),
            "minimum": float(sigma_values.min()),
            "maximum": float(sigma_values.max()),
            "max_to_min_ratio": float(sigma_values.max() / sigma_values.min()),
        },
        "inference_warning": (
            "Only three batches produce three overlapping training-domain folds; "
            "this diagnoses sensitivity but cannot estimate deployment stability."
        ),
    }


def _joint_partition_stability(metric_frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    comparators = (
        "target_reference_shrinkage",
        "target_reference_unshrunk",
        "source_reference_negative_control",
    )
    for (target_batch, k_shot), rows in metric_frame.groupby(
        ["target_batch", "k"], sort=True
    ):
        pivot = rows.pivot_table(
            index="repeat",
            columns="model",
            values=list(METRIC_COLUMNS),
            aggfunc="first",
        )
        target_result = result.setdefault(str(target_batch), {})
        k_result: dict[str, object] = {}
        for comparator in comparators:
            comparator_result: dict[str, object] = {}
            for metric in METRIC_COLUMNS:
                baseline = pivot[(metric, "zero_shot")].to_numpy(dtype=float)
                comparison = pivot[(metric, comparator)].to_numpy(dtype=float)
                delta = comparison - baseline
                favorable = _favorable_mask(
                    baseline,
                    comparison,
                    metric=metric,
                )
                comparator_result[f"delta_{metric}"] = {
                    "mean": float(delta.mean()),
                    "median": float(np.median(delta)),
                    "p05": float(np.quantile(delta, 0.05)),
                    "p95": float(np.quantile(delta, 0.95)),
                    "comparison_rule": _comparison_rule(metric),
                    "favorable_repeat_fraction": (
                        None if favorable is None else float(np.mean(favorable))
                    ),
                    "decreased_repeat_fraction": float(np.mean(delta < 0)),
                }
            k_result[comparator] = comparator_result
        target_result[str(int(k_shot))] = k_result
    return {
        "status": "diagnostic_only",
        "unit": "label_independent_joint_reference_query_partition_repeat",
        "inference_warning": (
            "Each repeat changes both the max-k reference set and its complementary "
            "query composition. Repeats reuse cells and are not independent evidence; "
            "percentiles measure joint partition sensitivity, not reference-only noise."
        ),
        "by_target_batch": result,
    }


def run_batch_reference_proxy_experiment(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
    repeats: int = 200,
    seed: int = 42,
    l2_penalty: float = 1e-4,
    prior_scale_multiplier: float = 1.0,
    survival_times: Sequence[float] = (500.0, 1000.0, 1500.0),
    identity_column: str = "cell_id",
    batch_column: str = "batch_id",
    label_column: str = "cycle_life",
    censor_column: str = "is_censored",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a leave-one-batch-out reference-cell software proxy.

    Target reference labels update only a batch-level log-life intercept. All k
    values in one repeat use nested references and the same query rows outside
    the maximum-k set, so paired comparisons are meaningful.
    """
    features = list(feature_columns)
    k_shots = tuple(sorted(int(value) for value in k_values))
    if not features or not k_shots or any(value < 1 for value in k_shots):
        raise ValueError("Features and positive k_values are required")
    if len(set(features)) != len(features):
        raise ValueError("feature_columns must be unique")
    protected_features = {
        identity_column,
        batch_column,
        label_column,
        censor_column,
    }
    protected_overlap = sorted(set(features) & protected_features)
    if protected_overlap:
        raise ValueError(
            f"Batch proxy features cannot contain protected columns: {protected_overlap}"
        )
    if len(set(k_shots)) != len(k_shots):
        raise ValueError("k_values must be unique")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not math.isfinite(prior_scale_multiplier) or prior_scale_multiplier <= 0:
        raise ValueError("prior_scale_multiplier must be finite and positive")
    survival_grid = tuple(float(value) for value in survival_times)
    if (
        not survival_grid
        or any(not math.isfinite(value) or value <= 0 for value in survival_grid)
        or len(set(survival_grid)) != len(survival_grid)
    ):
        raise ValueError("survival_times must be unique finite positive values")
    required = {
        identity_column,
        batch_column,
        label_column,
        censor_column,
        *features,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing batch-proxy columns: {missing}")
    working = frame.copy()
    if working[identity_column].isna().any() or working[identity_column].duplicated().any():
        raise ValueError("Batch proxy requires unique, non-null cell identities")
    working[identity_column] = working[identity_column].astype(str)
    if working[batch_column].isna().any() or working[batch_column].nunique() < 2:
        raise ValueError("Batch proxy requires at least two non-null target domains")
    if working[censor_column].isna().any() or working[censor_column].dtype != bool:
        raise ValueError(f"{censor_column} must contain non-null boolean values")
    if working[censor_column].any():
        raise ValueError("Batch proxy reference and query cells must have observed EOL")
    numeric = working[[*features, label_column]].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Batch proxy features and labels must be finite numeric values")
    if (numeric[label_column] <= 0).any():
        raise ValueError("Batch proxy lifetimes must be positive")
    working[[*features, label_column]] = numeric
    maximum_k = max(k_shots)
    batch_sizes = working.groupby(batch_column)[identity_column].nunique()
    if (batch_sizes <= maximum_k).any():
        raise ValueError("Every target batch needs at least max(k)+1 cells")

    feature_frame = working.drop(columns=[label_column, censor_column])
    label_vault = working[
        [identity_column, label_column, censor_column]
    ].set_index(identity_column, drop=False)
    prediction_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    fit_summaries: dict[str, object] = {}
    partition_checks = {
        "reference_query_disjoint": True,
        "nested_reference_sets": True,
        "common_query_across_k": True,
    }
    for target_batch in sorted(feature_frame[batch_column].unique(), key=str):
        source_features = feature_frame.loc[
            feature_frame[batch_column] != target_batch
        ]
        target_features = feature_frame.loc[
            feature_frame[batch_column] == target_batch
        ]
        source_ids = source_features[identity_column].tolist()
        source = source_features.merge(
            label_vault.loc[source_ids, [label_column, censor_column]],
            left_on=identity_column,
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        base_model = LogNormalAFT(l2_penalty=l2_penalty).fit(
            source[features].to_numpy(dtype=float),
            source[label_column].to_numpy(dtype=float),
            is_censored=source[censor_column].to_numpy(dtype=bool),
        )
        prior_std = base_model.sigma_ * prior_scale_multiplier
        fit_summaries[str(target_batch)] = {
            **base_model.fit_summary_.to_dict(),
            "source_batch_count": int(source[batch_column].nunique()),
            "source_cell_count": len(source),
            "target_cell_count": len(target_features),
            "residual_sigma": base_model.sigma_,
            "residual_sigma_interpretation": (
                "pooled source-AFT residual scale without explicit source-batch effects"
            ),
            "variance_decomposition_identified": False,
            "prior_std": prior_std,
            "prior_scale_multiplier": prior_scale_multiplier,
            "coefficients": dict(zip(features, base_model.coef_, strict=True)),
            "intercept": base_model.intercept_,
        }
        target_by_id = target_features.set_index(identity_column, drop=False)
        source_by_id = source.set_index(identity_column, drop=False)
        for repeat_index in range(repeats):
            target_reference_order, common_query_ids = nested_reference_partition(
                target_features,
                maximum_reference_count=maximum_k,
                repeat_index=repeat_index,
                seed=seed,
                domain_value=target_batch,
                identity_column=identity_column,
            )
            source_reference_order, _ = nested_reference_partition(
                source_features,
                maximum_reference_count=maximum_k,
                repeat_index=repeat_index,
                seed=seed + 1_000_003,
                domain_value=f"source_for_{target_batch}",
                identity_column=identity_column,
            )
            maximum_reference_set = set(target_reference_order)
            query_set = set(common_query_ids)
            partition_checks["reference_query_disjoint"] &= not bool(
                maximum_reference_set & query_set
            )
            nested_sets = [
                set(target_reference_order[:k_shot]) for k_shot in k_shots
            ]
            partition_checks["nested_reference_sets"] &= all(
                nested_sets[index] < nested_sets[index + 1]
                for index in range(len(nested_sets) - 1)
            )
            query = target_by_id.loc[common_query_ids, [
                identity_column,
                batch_column,
                *features,
            ]]
            query_features = query[features].to_numpy(dtype=float)
            for k_shot in k_shots:
                reference_ids = target_reference_order[:k_shot]
                control_ids = source_reference_order[:k_shot]
                references = target_by_id.loc[reference_ids].reset_index(
                    drop=True
                ).merge(
                    label_vault.loc[reference_ids, [label_column, censor_column]],
                    left_on=identity_column,
                    right_index=True,
                    how="left",
                    validate="one_to_one",
                )
                controls = source_by_id.loc[control_ids]
                prediction_state = build_reference_proxy_prediction_state(
                    base_model,
                    target_references=references,
                    source_controls=controls,
                    query_features=query_features,
                    feature_columns=features,
                    label_column=label_column,
                    prior_std=prior_std,
                )
                model_predictions = prediction_state["quantiles"]
                model_locations = prediction_state["locations"]
                model_sigmas = prediction_state["sigmas"]
                target_posterior = prediction_state["target_posterior"]
                unshrunk_posterior = prediction_state["unshrunk_posterior"]
                control_posterior = prediction_state["control_posterior"]
                prediction = query[[identity_column, batch_column]].reset_index(
                    drop=True
                )
                prediction.insert(0, "k", k_shot)
                prediction.insert(0, "repeat", repeat_index)
                prediction.insert(0, "target_batch", target_batch)
                for model_name, quantiles in model_predictions.items():
                    prediction[f"{model_name}_p10"] = quantiles[:, 0]
                    prediction[f"{model_name}_p50"] = quantiles[:, 1]
                    prediction[f"{model_name}_p90"] = quantiles[:, 2]
                    prediction[f"{model_name}_log_location"] = model_locations[
                        model_name
                    ]
                    prediction[f"{model_name}_predictive_sigma"] = model_sigmas[
                        model_name
                    ]
                    for survival_time in survival_grid:
                        z_score = (
                            math.log(survival_time) - model_locations[model_name]
                        ) / model_sigmas[model_name]
                        time_label = str(survival_time).replace(".", "p")
                        prediction[f"{model_name}_survival_{time_label}"] = ndtr(
                            -z_score
                        )
                prediction["target_posterior_mean"] = target_posterior.posterior_mean
                prediction["target_posterior_std"] = target_posterior.posterior_std
                prediction["target_predictive_sigma"] = target_posterior.predictive_sigma
                prediction["unshrunk_posterior_mean"] = (
                    unshrunk_posterior.posterior_mean
                )
                prediction["control_posterior_mean"] = control_posterior.posterior_mean
                prediction_frames.append(prediction)
                partition_prediction_sha256 = prediction_artifact_sha256(
                    prediction,
                    identity_column=identity_column,
                )
                selection_rows.append(
                    {
                        "target_batch": target_batch,
                        "repeat": repeat_index,
                        "k": k_shot,
                        "reference_ids": "|".join(reference_ids),
                        "source_control_ids": "|".join(control_ids),
                        "common_query_count": len(common_query_ids),
                        "reference_set_sha256": hashlib.sha256(
                            "|".join(reference_ids).encode("utf-8")
                        ).hexdigest(),
                        "common_query_sha256": hashlib.sha256(
                            "|".join(common_query_ids).encode("utf-8")
                        ).hexdigest(),
                        "partition_prediction_sha256": partition_prediction_sha256,
                    }
                )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["target_batch", "repeat", "k", identity_column],
        kind="stable",
    ).reset_index(drop=True)
    selections = pd.DataFrame(selection_rows)
    for _, rows in selections.groupby(["target_batch", "repeat"], sort=False):
        partition_checks["common_query_across_k"] &= (
            rows["common_query_sha256"].nunique() == 1
        )
    frozen_prediction_sha256 = prediction_artifact_sha256(
        predictions,
        identity_column=identity_column,
    )
    reference_role_ids = {
        identity
        for value in selections["reference_ids"]
        for identity in str(value).split("|")
    }
    query_role_ids = set(predictions[identity_column].astype(str))
    cross_repeat_role_reused_cell_count = len(reference_role_ids & query_role_ids)
    # Each cross-fit partition is frozen before its query outcomes enter scoring.
    # The aggregate hash below is an artifact-integrity hash because cells may
    # change roles across diagnostic repeats.
    query_scoring_labels = label_vault.reset_index(drop=True)[
        [identity_column, label_column]
    ]
    metrics = score_partitioned_batch_reference_proxy_predictions(
        predictions,
        query_scoring_labels,
        selections,
        identity_column=identity_column,
        label_column=label_column,
    )
    partition_hashes_verified_before_scoring = True
    by_target: dict[str, object] = {}
    for (target_batch, k_shot, model_name), rows in metrics.groupby(
        ["target_batch", "k", "model"], sort=True
    ):
        target_result = by_target.setdefault(str(target_batch), {})
        k_result = target_result.setdefault(str(int(k_shot)), {})
        k_result[str(model_name)] = _aggregate_metric_rows(rows)
    result = {
        "status": "proxy_only",
        "estimand": (
            "query-cell lifetime in a held-out MATR batch after observing complete "
            "EOL labels for k target-batch reference cells"
        ),
        "reference_design": {
            "nested_reference_sets": True,
            "common_query_outside_maximum_k": True,
            "reference_counts": list(k_shots),
            "repeats": repeats,
            "seed": seed,
            "selection_uses_labels": False,
            "role_assignment_scope": "repeat_specific_diagnostic_cross_fit",
            "same_cell_may_change_role_across_repeats": True,
        },
        "model": {
            "base_distribution": "log_normal_aft",
            "features": features,
            "l2_penalty": l2_penalty,
            "target_update": "conjugate_normal_batch_intercept",
            "prior_std_rule": "residual_sigma * prior_scale_multiplier",
            "prior_scale_multiplier": prior_scale_multiplier,
            "predictive_variance": "residual_sigma^2 + posterior_batch_variance",
            "residual_sigma_interpretation": (
                "pooled source-AFT residual heuristic; within-batch cell noise and "
                "remaining source-batch shift are not identified separately"
            ),
            "variance_decomposition_identified": False,
            "coefficient_uncertainty_included": False,
            "survival_times": list(survival_grid),
        },
        "fits": fit_summaries,
        "training_domain_stability": _training_domain_stability(fit_summaries),
        "joint_partition_stability": _joint_partition_stability(metrics),
        "prediction_freeze": {
            "status": "per_partition_frozen_before_partition_scoring",
            "scope": "target_batch_repeat_k_cross_fit_partition",
            "sha256": frozen_prediction_sha256,
            "global_artifact_sha256": frozen_prediction_sha256,
            "row_count": len(predictions),
            "key_columns": ["target_batch", "repeat", "k", identity_column],
            "partition_count": len(selections),
            "partition_hash_column": "partition_prediction_sha256",
            "query_outcome_columns_present": False,
            "global_outcome_blind": False,
            "cross_repeat_role_reuse": cross_repeat_role_reused_cell_count > 0,
            "cross_repeat_role_reused_cell_count": (
                cross_repeat_role_reused_cell_count
            ),
            "phase_isolation": (
                "single_process_explicit_per_partition_prediction_then_scoring"
            ),
            "interpretation": (
                "Per-partition cross-fitting prevents a query cell's own label from "
                "entering that partition's prediction. The aggregate hash is for file "
                "integrity, not a claim of global outcome blindness across repeats."
            ),
        },
        "by_target_batch": by_target,
        "equal_batch_summary": _equal_batch_summary(metrics),
        "software_gate": {
            "status": (
                "passed"
                if all(partition_checks.values())
                and label_column not in predictions
                and prediction_artifact_sha256(
                    predictions,
                    identity_column=identity_column,
                )
                == frozen_prediction_sha256
                else "failed"
            ),
            "checks": {
                **partition_checks,
                "feature_columns_exclude_protected_columns": True,
                "reference_selection_uses_label_free_identity_frame": True,
                "prediction_artifact_excludes_query_outcomes": label_column
                not in predictions,
                "per_partition_id_bound_hash_verified_before_partition_scoring": (
                    partition_hashes_verified_before_scoring
                ),
                "global_artifact_integrity_hash_verified": prediction_artifact_sha256(
                    predictions,
                    identity_column=identity_column,
                )
                == frozen_prediction_sha256,
                "predictive_variance_includes_posterior_batch_variance": True,
            },
            "scope": "per_partition_cross_fit_software_integrity",
            "disclosure": (
                "Cells can change reference/query roles across repeats; this Gate does "
                "not claim a globally outcome-blind prediction artifact."
            ),
        },
        "evidence_gate": _proxy_evidence_gate(
            int(working[batch_column].nunique()),
            minimum_target_batch_count=30,
        ),
        "limitations": [
            "This is a batch proxy, not a same-protocol reference-cell benchmark.",
            "The prior scale is pre-registered because two source batches per fold cannot identify tau reliably.",
            "The pooled AFT residual scale does not identify within-batch sigma separately from remaining source-batch shift.",
            "Repeated cells and repeats are not independent statistical units.",
            "Joint-partition repeats change both reference and query composition.",
            "Reference and query cells can share protocols; protocol-cluster dependence is not modeled.",
            "Cross-fit partitions are phase-separated and hashed in one process, not isolated services.",
            "Cells can change reference/query roles across repeats, so the global prediction hash is not outcome-blind.",
            "Reference cells require complete EOL labels; online censored updating is not implemented.",
            "Results cannot validate Hithium storage cells or 15-25 year lifetime.",
        ],
    }
    return result, predictions, selections, metrics


def run_batch_reference_prior_scale_sensitivity(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    prior_scale_multipliers: Sequence[float],
    primary_prior_scale_multiplier: float,
    k_values: Sequence[int] = (1, 3, 5, 10),
    repeats: int = 200,
    seed: int = 42,
    l2_penalty: float = 1e-4,
    survival_times: Sequence[float] = (500.0, 1000.0, 1500.0),
    identity_column: str = "cell_id",
    batch_column: str = "batch_id",
    label_column: str = "cycle_life",
    censor_column: str = "is_censored",
) -> tuple[dict[str, object], pd.DataFrame]:
    """Repeat the batch proxy over pre-specified prior-scale multipliers."""
    multipliers = tuple(sorted(float(value) for value in prior_scale_multipliers))
    if (
        not multipliers
        or len(set(multipliers)) != len(multipliers)
        or any(not math.isfinite(value) or value <= 0 for value in multipliers)
    ):
        raise ValueError("prior_scale_multipliers must be unique finite positive values")
    if primary_prior_scale_multiplier not in multipliers:
        raise ValueError("Primary prior scale must be included in sensitivity values")

    scale_results: dict[str, object] = {}
    detail_rows: list[dict[str, object]] = []
    for multiplier in multipliers:
        result, _, _, _ = run_batch_reference_proxy_experiment(
            frame,
            feature_columns=feature_columns,
            k_values=k_values,
            repeats=repeats,
            seed=seed,
            l2_penalty=l2_penalty,
            prior_scale_multiplier=multiplier,
            survival_times=survival_times,
            identity_column=identity_column,
            batch_column=batch_column,
            label_column=label_column,
            censor_column=censor_column,
        )
        scale_key = format(multiplier, ".12g")
        scale_results[scale_key] = {
            "equal_batch_summary": result["equal_batch_summary"],
            "training_domain_stability": result["training_domain_stability"],
            "evidence_gate": result["evidence_gate"],
        }
        for k_key, k_summary in result["equal_batch_summary"].items():
            deltas = k_summary["equal_batch_paired_deltas"]
            for comparator_key, comparator_values in deltas.items():
                comparator = comparator_key.removesuffix("_minus_zero_shot")
                for metric in METRIC_COLUMNS:
                    by_batch = comparator_values[f"delta_{metric}_by_batch"]
                    detail_rows.append(
                        {
                            "prior_scale_multiplier": multiplier,
                            "is_primary_scale": multiplier
                            == primary_prior_scale_multiplier,
                            "k": int(k_key),
                            "comparator": comparator,
                            "metric": metric,
                            "equal_batch_delta": comparator_values[
                                f"delta_{metric}"
                            ],
                            "comparison_rule": comparator_values[
                                f"comparison_rule_{metric}"
                            ],
                            "favorable_batch_count": comparator_values[
                                f"favorable_batch_count_{metric}"
                            ],
                            "decreased_batch_count": comparator_values[
                                f"decreased_batch_count_{metric}"
                            ],
                            "target_batch_count": len(by_batch),
                            "minimum_batch_delta": min(by_batch.values()),
                            "maximum_batch_delta": max(by_batch.values()),
                        }
                    )
    return (
        {
            "status": "diagnostic_only",
            "estimand": (
                "sensitivity of batch-balanced proxy metrics to the pre-specified "
                "batch-offset prior standard deviation"
            ),
            "prior_scale_multipliers": list(multipliers),
            "primary_prior_scale_multiplier": primary_prior_scale_multiplier,
            "selection_rule": "pre_specified_grid_not_selected_on_query_metrics",
            "scales": scale_results,
            "evidence_gate": _proxy_evidence_gate(
                int(frame[batch_column].nunique()),
                minimum_target_batch_count=30,
            ),
            "limitations": [
                "This public-data sensitivity diagnostic is not target-industrial evidence.",
                "The independent target-domain count is a separate evidence-Gate check.",
                "Sensitivity agreement cannot replace independent target-domain validation.",
                "The analysis does not propagate feature-coefficient uncertainty.",
                "No storage-product or 15-25 year claim is supported.",
            ],
        },
        pd.DataFrame(detail_rows),
    )
