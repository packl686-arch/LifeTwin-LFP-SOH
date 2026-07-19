from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.evaluation.ipcw import (
    FrozenIPCWPolicy,
    IPCWBrierEvaluation,
    ReverseKaplanMeierCensoring,
    evaluate_ipcw_brier,
    freeze_ipcw_policy,
)
from lifetwin.models.probabilistic import LogNormalAFT


@dataclass(frozen=True)
class SyntheticIPCWPredictionState:
    model: LogNormalAFT
    policy: FrozenIPCWPolicy
    test_ids: tuple[str, ...]
    survival_probabilities: np.ndarray
    prediction_sha256: str


def generate_synthetic_survival_cohort(
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
    lifetime_intercept: float,
    lifetime_coefficients: Sequence[float],
    lifetime_sigma: float,
    censor_intercept: float,
    censor_sigma: float,
    censor_feature_coefficients: Sequence[float],
    censor_lifetime_residual_coupling: float = 0.0,
) -> pd.DataFrame:
    """Generate known latent lifetimes for a synthetic IPCW software audit."""
    counts = (train_count, validation_count, test_count)
    if any(count < 2 for count in counts):
        raise ValueError("Every synthetic split needs at least two rows")
    lifetime_beta = np.asarray(lifetime_coefficients, dtype=float)
    censor_beta = np.asarray(censor_feature_coefficients, dtype=float)
    if lifetime_beta.ndim != 1 or not len(lifetime_beta):
        raise ValueError("lifetime_coefficients must be a non-empty vector")
    if censor_beta.shape != lifetime_beta.shape:
        raise ValueError("Censor and lifetime coefficient vectors must match")
    numeric = np.concatenate(
        (
            lifetime_beta,
            censor_beta,
            np.array(
                [
                    lifetime_intercept,
                    lifetime_sigma,
                    censor_intercept,
                    censor_sigma,
                    censor_lifetime_residual_coupling,
                ]
            ),
        )
    )
    if not np.isfinite(numeric).all() or lifetime_sigma <= 0 or censor_sigma <= 0:
        raise ValueError("Synthetic distribution parameters must be finite with positive scales")

    total = sum(counts)
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(total, len(lifetime_beta)))
    lifetime_residual = rng.normal(scale=lifetime_sigma, size=total)
    log_lifetime = lifetime_intercept + features @ lifetime_beta + lifetime_residual
    log_censor_time = (
        censor_intercept
        + features @ censor_beta
        + censor_lifetime_residual_coupling * lifetime_residual
        + rng.normal(scale=censor_sigma, size=total)
    )
    true_lifetime = np.exp(log_lifetime)
    censor_time = np.exp(log_censor_time)
    is_censored = censor_time < true_lifetime
    observed_time = np.minimum(true_lifetime, censor_time)
    split = np.repeat(("train", "validation", "test"), counts)

    result = pd.DataFrame(
        {
            "row_id": [f"synthetic_{seed}_{index:06d}" for index in range(total)],
            "split": split,
            "true_lifetime": true_lifetime,
            "censor_time": censor_time,
            "observed_time": observed_time,
            "is_censored": is_censored,
        }
    )
    for index in range(features.shape[1]):
        result[f"feature_{index + 1}"] = features[:, index]
    return result


def build_synthetic_ipcw_prediction_state(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test_features: np.ndarray,
    *,
    test_ids: Sequence[object],
    feature_columns: Sequence[str],
    evaluation_times: Sequence[float],
    l2_penalty: float,
    policy_parameters: Mapping[str, float | int],
) -> SyntheticIPCWPredictionState:
    """Freeze the model, censoring policy, and test predictions before scoring."""
    features = list(feature_columns)
    required = {"row_id", "split", "observed_time", "is_censored", *features}
    for name, frame in (("train", train), ("validation", validation)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing {name} columns: {missing}")
        if frame["is_censored"].isna().any() or frame["is_censored"].dtype != bool:
            raise ValueError(f"{name} is_censored must contain non-null booleans")
    if set(train["split"]) != {"train"}:
        raise ValueError("train may contain only split=train")
    if set(validation["split"]) != {"validation"}:
        raise ValueError("validation may contain only split=validation")

    model = LogNormalAFT(l2_penalty=l2_penalty).fit(
        train[features].to_numpy(dtype=float),
        train["observed_time"].to_numpy(dtype=float),
        is_censored=train["is_censored"].to_numpy(dtype=bool),
    )
    reference = pd.concat((train, validation), ignore_index=True)
    censoring_model = ReverseKaplanMeierCensoring.fit(
        reference["observed_time"].to_numpy(dtype=float),
        reference["is_censored"].to_numpy(dtype=bool),
        reference_ids=reference["row_id"].to_numpy(),
        source_splits=reference["split"].to_numpy(),
    )
    policy = freeze_ipcw_policy(
        censoring_model,
        evaluation_times,
        min_censor_survival=float(policy_parameters["min_censor_survival"]),
        max_weight=float(policy_parameters["max_weight"]),
        min_effective_sample_size=float(
            policy_parameters["min_effective_sample_size"]
        ),
        min_effective_sample_fraction=float(
            policy_parameters["min_effective_sample_fraction"]
        ),
        min_event_count=int(policy_parameters["min_event_count"]),
        min_alive_count=int(policy_parameters["min_alive_count"]),
        max_clipped_fraction=float(policy_parameters["max_clipped_fraction"]),
    )
    matrix = np.asarray(test_features, dtype=float)
    identities = np.asarray(test_ids, dtype=object)
    if identities.ndim != 1 or len(identities) != len(matrix):
        raise ValueError("test_ids must be one-dimensional and match test features")
    if any(value is None or str(value) == "" for value in identities):
        raise ValueError("test_ids must be non-null and non-empty")
    normalized_ids = tuple(str(value) for value in identities)
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("test_ids must be unique")
    canonical_order = np.argsort(np.asarray(normalized_ids), kind="stable")
    canonical_predictions = np.column_stack(
        [
            model.survival_probability(matrix[canonical_order], time)
            for time in policy.evaluation_times
        ]
    )
    predictions = np.empty_like(canonical_predictions)
    predictions[canonical_order] = canonical_predictions
    canonical_rows = list(
        zip(
            (normalized_ids[index] for index in canonical_order),
            canonical_predictions.tolist(),
            strict=True,
        )
    )
    payload = {
        "policy_sha256": policy.policy_sha256,
        "shape": list(predictions.shape),
        "id_bound_survival_probabilities": canonical_rows,
    }
    prediction_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return SyntheticIPCWPredictionState(
        model=model,
        policy=policy,
        test_ids=normalized_ids,
        survival_probabilities=predictions,
        prediction_sha256=prediction_sha256,
    )


def _oracle_brier(
    true_lifetime: np.ndarray,
    predictions: np.ndarray,
    evaluation_times: Sequence[float],
) -> np.ndarray:
    return np.asarray(
        [
            np.mean(
                (
                    (true_lifetime > evaluation_time).astype(float)
                    - predictions[:, column]
                )
                ** 2
            )
            for column, evaluation_time in enumerate(evaluation_times)
        ],
        dtype=float,
    )


def _naive_known_status_brier(
    observed_time: np.ndarray,
    is_censored: np.ndarray,
    predictions: np.ndarray,
    evaluation_times: Sequence[float],
) -> np.ndarray:
    scores: list[float] = []
    for column, evaluation_time in enumerate(evaluation_times):
        observed_event = (~is_censored) & (observed_time <= evaluation_time)
        alive = observed_time > evaluation_time
        known = observed_event | alive
        if not known.any():
            scores.append(math.nan)
            continue
        known_truth = alive[known].astype(float)
        scores.append(float(np.mean((known_truth - predictions[known, column]) ** 2)))
    return np.asarray(scores, dtype=float)


def _integrated_score(scores: np.ndarray, grid: np.ndarray) -> float:
    return float(np.trapezoid(scores, grid) / (grid[-1] - grid[0]))


def _summarize_values(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _run_synthetic_scenario(
    *,
    scenario: Mapping[str, object],
    repetitions: int,
    seed: int,
    counts: Mapping[str, int],
    distribution: Mapping[str, object],
    evaluation_times: Sequence[float],
    l2_penalty: float,
    policy_parameters: Mapping[str, float | int],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    grid = np.asarray(evaluation_times, dtype=float)
    detail_rows: list[dict[str, object]] = []
    repetition_summaries: list[dict[str, object]] = []
    for repeat in range(repetitions):
        repeat_seed = seed + repeat * 104_729
        cohort = generate_synthetic_survival_cohort(
            train_count=int(counts["train"]),
            validation_count=int(counts["validation"]),
            test_count=int(counts["test"]),
            seed=repeat_seed,
            lifetime_intercept=float(distribution["lifetime_intercept"]),
            lifetime_coefficients=distribution["lifetime_coefficients"],
            lifetime_sigma=float(distribution["lifetime_sigma"]),
            censor_intercept=float(
                scenario.get("censor_intercept", distribution["censor_intercept"])
            ),
            censor_sigma=float(
                scenario.get("censor_sigma", distribution["censor_sigma"])
            ),
            censor_feature_coefficients=scenario["censor_feature_coefficients"],
            censor_lifetime_residual_coupling=float(
                scenario["censor_lifetime_residual_coupling"]
            ),
        )
        feature_columns = [
            column for column in cohort.columns if column.startswith("feature_")
        ]
        train = cohort.loc[cohort["split"] == "train"]
        validation = cohort.loc[cohort["split"] == "validation"]
        test = cohort.loc[cohort["split"] == "test"]

        # Only test features cross this boundary. Test outcomes are attached after
        # the state and its hashes are frozen.
        state = build_synthetic_ipcw_prediction_state(
            train,
            validation,
            test[feature_columns].to_numpy(dtype=float),
            test_ids=test["row_id"].to_numpy(),
            feature_columns=feature_columns,
            evaluation_times=grid,
            l2_penalty=l2_penalty,
            policy_parameters=policy_parameters,
        )
        if state.test_ids != tuple(str(value) for value in test["row_id"]):
            raise RuntimeError("Frozen synthetic prediction IDs lost row alignment")
        evaluation: IPCWBrierEvaluation = evaluate_ipcw_brier(
            test["observed_time"].to_numpy(dtype=float),
            test["is_censored"].to_numpy(dtype=bool),
            state.survival_probabilities,
            state.policy,
        )
        oracle = _oracle_brier(
            test["true_lifetime"].to_numpy(dtype=float),
            state.survival_probabilities,
            grid,
        )
        naive = _naive_known_status_brier(
            test["observed_time"].to_numpy(dtype=float),
            test["is_censored"].to_numpy(dtype=bool),
            state.survival_probabilities,
            grid,
        )
        oracle_ibs = _integrated_score(oracle, grid)
        naive_ibs = _integrated_score(naive, grid)
        time_errors: list[float] = []
        for column, point in enumerate(evaluation.time_points):
            error = point.raw_brier_score - float(oracle[column])
            time_errors.append(abs(error))
            detail_rows.append(
                {
                    "scenario": str(scenario["name"]),
                    "gate_role": str(scenario["gate_role"]),
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "evaluation_time": point.evaluation_time,
                    "ipcw_brier": point.raw_brier_score,
                    "clipped_ipcw_brier": point.clipped_brier_score,
                    "oracle_brier": float(oracle[column]),
                    "naive_known_status_brier": float(naive[column]),
                    "ipcw_minus_oracle": error,
                    "absolute_ipcw_error": abs(error),
                    "censor_survival": point.censor_survival,
                    "effective_sample_size": point.effective_sample_size,
                    "effective_sample_fraction": point.effective_sample_fraction,
                    "clipped_fraction": point.clipped_fraction,
                    "time_point_gate_status": point.gate_status,
                    "policy_sha256": state.policy.policy_sha256,
                    "prediction_sha256": state.prediction_sha256,
                }
            )
        repetition_summaries.append(
            {
                "repeat": repeat,
                "seed": repeat_seed,
                "test_censoring_fraction": float(test["is_censored"].mean()),
                "ipcw_status": evaluation.status,
                "ipcw_ibs": evaluation.integrated_brier_score,
                "clipped_ipcw_ibs": evaluation.clipped_integrated_brier_score,
                "oracle_ibs": oracle_ibs,
                "naive_known_status_ibs": naive_ibs,
                "ipcw_minus_oracle_ibs": evaluation.integrated_brier_score
                - oracle_ibs,
                "naive_minus_oracle_ibs": naive_ibs - oracle_ibs,
                "mean_absolute_time_point_error": float(np.mean(time_errors)),
                "maximum_absolute_time_point_error": float(np.max(time_errors)),
                "policy_sha256": state.policy.policy_sha256,
                "prediction_sha256": state.prediction_sha256,
                "fit": {
                    **state.model.fit_summary_.to_dict(),
                    "sigma_log_life": state.model.sigma_,
                    "coefficients": dict(
                        zip(feature_columns, state.model.coef_, strict=True)
                    ),
                },
            }
        )

    integrated_errors = [
        abs(float(item["ipcw_minus_oracle_ibs"])) for item in repetition_summaries
    ]
    time_errors = [
        float(row["absolute_ipcw_error"])
        for row in detail_rows
    ]
    marginal_independence_by_construction = bool(
        np.allclose(
            np.asarray(scenario["censor_feature_coefficients"], dtype=float),
            0.0,
        )
        and float(scenario["censor_lifetime_residual_coupling"]) == 0.0
    )
    scenario_result = {
        "name": str(scenario["name"]),
        "gate_role": str(scenario["gate_role"]),
        "censoring_mechanism": {
            "intercept": float(
                scenario.get("censor_intercept", distribution["censor_intercept"])
            ),
            "sigma": float(
                scenario.get("censor_sigma", distribution["censor_sigma"])
            ),
            "feature_coefficients": list(scenario["censor_feature_coefficients"]),
            "lifetime_residual_coupling": float(
                scenario["censor_lifetime_residual_coupling"]
            ),
        },
        "marginal_independence_assumption": {
            "status": (
                "satisfied_by_construction"
                if marginal_independence_by_construction
                else "violated_by_construction"
            ),
            "numeric_weight_gates_can_verify_this_assumption": False,
        },
        "repetitions": repetition_summaries,
        "summary": {
            "test_censoring_fraction": _summarize_values(
                [float(item["test_censoring_fraction"]) for item in repetition_summaries]
            ),
            "absolute_integrated_error": _summarize_values(integrated_errors),
            "absolute_time_point_error": _summarize_values(time_errors),
            "ipcw_minus_oracle_ibs": _summarize_values(
                [float(item["ipcw_minus_oracle_ibs"]) for item in repetition_summaries]
            ),
            "naive_minus_oracle_ibs": _summarize_values(
                [float(item["naive_minus_oracle_ibs"]) for item in repetition_summaries]
            ),
            "all_policy_gates_passed": all(
                item["ipcw_status"] == "passed" for item in repetition_summaries
            ),
        },
    }
    return scenario_result, detail_rows


def run_synthetic_ipcw_validation(
    *,
    scenarios: Sequence[Mapping[str, object]],
    repetitions: int,
    seed: int,
    counts: Mapping[str, int],
    distribution: Mapping[str, object],
    evaluation_times: Sequence[float],
    l2_penalty: float,
    policy_parameters: Mapping[str, float | int],
    validation_thresholds: Mapping[str, float],
) -> tuple[dict[str, object], pd.DataFrame]:
    """Validate classic IPCW against latent-lifetime oracle scores."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if not scenarios:
        raise ValueError("At least one synthetic scenario is required")
    scenario_names = [str(item["name"]) for item in scenarios]
    if len(set(scenario_names)) != len(scenario_names):
        raise ValueError("Synthetic scenario names must be unique")
    allowed_roles = {"validation", "limitation_stress"}
    for scenario in scenarios:
        role = str(scenario["gate_role"])
        if role not in allowed_roles:
            raise ValueError(f"Unexpected synthetic gate_role: {role}")
        coefficients = np.asarray(
            scenario["censor_feature_coefficients"], dtype=float
        )
        coupling = float(scenario["censor_lifetime_residual_coupling"])
        if role == "validation" and (
            not np.allclose(coefficients, 0.0) or coupling != 0.0
        ):
            raise ValueError(
                "Marginal IPCW validation scenarios must use independent censoring; "
                "dependent mechanisms belong in limitation_stress"
            )

    scenario_results: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        result, rows = _run_synthetic_scenario(
            scenario=scenario,
            repetitions=repetitions,
            seed=seed,
            counts=counts,
            distribution=distribution,
            evaluation_times=evaluation_times,
            l2_penalty=l2_penalty,
            policy_parameters=policy_parameters,
        )
        scenario_results.append(result)
        detail_rows.extend(rows)

    validation_scenarios = [
        item for item in scenario_results if item["gate_role"] == "validation"
    ]
    if not validation_scenarios:
        raise ValueError("At least one scenario must have gate_role=validation")
    mean_absolute_ibs_error = max(
        float(item["summary"]["absolute_integrated_error"]["mean"])
        for item in validation_scenarios
    )
    maximum_absolute_ibs_error = max(
        float(item["summary"]["absolute_integrated_error"]["maximum"])
        for item in validation_scenarios
    )
    mean_absolute_time_error = max(
        float(item["summary"]["absolute_time_point_error"]["mean"])
        for item in validation_scenarios
    )
    checks = {
        "all_policy_gates_passed": all(
            bool(item["summary"]["all_policy_gates_passed"])
            for item in validation_scenarios
        ),
        "mean_absolute_ibs_error": mean_absolute_ibs_error
        <= float(validation_thresholds["maximum_mean_absolute_ibs_error"]),
        "maximum_absolute_ibs_error": maximum_absolute_ibs_error
        <= float(validation_thresholds["maximum_single_repeat_absolute_ibs_error"]),
        "mean_absolute_time_point_error": mean_absolute_time_error
        <= float(validation_thresholds["maximum_mean_absolute_time_point_error"]),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "evidence_role": "synthetic_software_validation_only",
        "estimand": (
            "IPCW time-dependent Brier and IBS agreement with complete latent-life "
            "oracle scores under a pre-specified synthetic censoring mechanism"
        ),
        "method": {
            "score": "Graf-style marginal IPCW Brier score",
            "event_weight": "1 / G(Y-)",
            "alive_weight": "1 / G(t)",
            "denominator": "original_test_row_count",
            "integration": "trapezoid_on_fixed_grid",
            "censoring_estimator": "reverse_Kaplan_Meier_on_train_plus_validation",
            "test_outcome_access": "after_model_policy_and_predictions_are_frozen",
        },
        "design": {
            "repetitions": repetitions,
            "seed": seed,
            "counts_per_repetition": dict(counts),
            "evaluation_times": [float(value) for value in evaluation_times],
            "distribution": dict(distribution),
            "policy": dict(policy_parameters),
        },
        "validation_gate": {
            "status": "passed" if all(checks.values()) else "failed",
            "checks": checks,
            "observed": {
                "maximum_validation_scenario_mean_absolute_ibs_error": mean_absolute_ibs_error,
                "maximum_validation_scenario_single_repeat_absolute_ibs_error": maximum_absolute_ibs_error,
                "maximum_validation_scenario_mean_absolute_time_point_error": mean_absolute_time_error,
            },
            "thresholds": dict(validation_thresholds),
        },
        "scenarios": scenario_results,
        "limitations": [
            "Passing validates the implementation on synthetic independent censoring, not an industrial battery model.",
            "Marginal reverse-KM IPCW can be biased under covariate-dependent or informative censoring.",
            "The stress scenario is diagnostic and is intentionally excluded from the validation gate.",
            "No censored conformal interval, 15-25 year extrapolation, or product-level claim is supported.",
        ],
    }
    return result, pd.DataFrame(detail_rows)
