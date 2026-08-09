"""Private dual-clock cycle/calendar degradation development pipeline.

The model separates elapsed-time and throughput coordinates, learns a
condition-and-duty-cycle prior from reference cells, and updates that prior
from a target cell's short RPT prefix. Target suffix capacity is accepted only
by the later scorer.
"""

from __future__ import annotations

from itertools import product
import json
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from lifetwin.data.snl import DATASET_ID, RPT_TRAJECTORY_COLUMNS
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)
from lifetwin.experiments.snl_rpt_loco import (
    REFERENCE_COLUMNS,
    TARGET_PREFIX_COLUMNS,
    TARGET_TRUTH_COLUMNS,
    _exact_one_sided_sign_flip_p,
    _trajectory_iae,
)
from lifetwin.models.hierarchical_cycle_prior import (
    DualClockKernelPrior,
    dual_clock_prior_coefficients,
    fit_power_condition_prior,
    infer_constant_duty_elapsed_days,
    predict_dual_clock_kernel_prior,
    predict_power_condition_prior,
    prefix_duty_rate_efc_per_day,
)


SCHEMA_VERSION = "lifetwin.private_dual_clock_prior_v3.config.v1"
EXPERIMENT_ID = "private_dual_clock_prior_v3"
MODEL_IDS = (
    "target_prefix_persistence",
    "v1_condition_ridge_delta",
    "v3_dual_clock_kernel_shrinkage",
)
PRIMARY_MODEL_ID = "v3_dual_clock_kernel_shrinkage"
PREDICTION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "model_id",
    "forecast_equivalent_full_cycles",
    "predicted_capacity_retention_pct",
)
DECISION_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "dual_clock_hyperparameters_json",
    "inner_condition_equal_iae_pp",
    "prefix_duty_rate_efc_per_day",
    "nearest_condition_distance",
    "condition_ood_threshold",
    "prefix_residual_rms_pp",
    "evidence_status",
)
SCORE_COLUMNS = (
    "experiment_id",
    "dataset_id",
    "outer_condition_id",
    "cell_id",
    "landmark_visit_count",
    "model_id",
    "future_observation_count",
    "trajectory_iae_pp",
    "trajectory_mae_pp",
    "trajectory_rmse_pp",
    "endpoint_absolute_error_pp",
)


class PrivateDualClockPriorV3Error(ValueError):
    """Raised when the private dual-clock protocol is violated."""


def default_private_dual_clock_prior_v3_config() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "private_only": True,
        "dataset_id": DATASET_ID,
        "landmark_visit_counts": [3, 4],
        "score_end_equivalent_full_cycles": 2500.0,
        "forecast_grid_step_equivalent_full_cycles": 25.0,
        "prediction_clip_pct": [0.0, 110.0],
        "dual_clock_family": {
            "time_exponents": [0.3, 0.5, 0.7],
            "cycle_exponents": [0.7, 1.0],
            "kernel_gammas": [0.3, 1.0, 3.0],
            "coefficient_shrinkages": [1.0, 10.0, 100.0],
            "anchor_weights": [0.25, 0.5, 0.75, 1.0],
            "future_schedule_assumption": "constant_prefix_efc_per_day",
        },
        "selection": {
            "metric": "condition_equal_trajectory_iae_pp",
            "worst_condition_penalty": 0.0,
        },
        "ood": {"cross_condition_quantile": 0.99, "threshold_multiplier": 1.5},
        "uncertainty": {
            "absolute_residual_quantile": 0.9,
            "horizon_bins_efc": [0.0, 500.0, 1000.0, 1500.0, 2500.0],
            "status": "private_development_diagnostic",
        },
    }


def _finite_sequence(values: object, *, name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PrivateDualClockPriorV3Error(f"{name} must be an array")
    converted = tuple(float(value) for value in values)
    if not converted or not all(math.isfinite(value) for value in converted):
        raise PrivateDualClockPriorV3Error(f"{name} is empty or non-finite")
    return converted


def validate_private_dual_clock_prior_v3_config(
    config: Mapping[str, object],
) -> dict[str, object]:
    value = json.loads(
        json.dumps(
            dict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PrivateDualClockPriorV3Error("Private V3 config schema changed")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise PrivateDualClockPriorV3Error("Private V3 experiment identity changed")
    if value.get("private_only") is not True:
        raise PrivateDualClockPriorV3Error("Private V3 must remain private-only")
    landmarks = tuple(int(item) for item in value["landmark_visit_counts"])
    if len(set(landmarks)) != len(landmarks) or min(landmarks) < 3:
        raise PrivateDualClockPriorV3Error("Private V3 landmarks are invalid")
    family = value["dual_clock_family"]
    for name, values in (
        ("time exponents", family["time_exponents"]),
        ("cycle exponents", family["cycle_exponents"]),
    ):
        if not all(0.0 < item <= 2.0 for item in _finite_sequence(values, name=name)):
            raise PrivateDualClockPriorV3Error(f"{name} must lie in (0, 2]")
    for name, values in (
        ("kernel gammas", family["kernel_gammas"]),
        ("coefficient shrinkages", family["coefficient_shrinkages"]),
    ):
        if not all(item > 0.0 for item in _finite_sequence(values, name=name)):
            raise PrivateDualClockPriorV3Error(f"{name} must be positive")
    if not all(
        0.0 <= item <= 1.0
        for item in _finite_sequence(
            family["anchor_weights"], name="anchor weights"
        )
    ):
        raise PrivateDualClockPriorV3Error("Anchor weights must lie in [0, 1]")
    if family.get("future_schedule_assumption") != "constant_prefix_efc_per_day":
        raise PrivateDualClockPriorV3Error("Private V3 schedule assumption changed")
    if float(value["score_end_equivalent_full_cycles"]) <= 0.0:
        raise PrivateDualClockPriorV3Error("Private V3 score end must be positive")
    return value


def _json_text(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _core(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, RPT_TRAJECTORY_COLUMNS].copy()


def _future(
    cell: pd.DataFrame,
    *,
    landmark: int,
    score_end: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = cell.sort_values("visit_index", kind="stable")
    if len(ordered) <= landmark:
        raise PrivateDualClockPriorV3Error("Trajectory lacks a future suffix")
    prefix = ordered.iloc[:landmark].copy()
    future = ordered.loc[
        (ordered["visit_index"] >= landmark)
        & (ordered["equivalent_full_cycles"] <= score_end)
    ].copy()
    if future.empty:
        raise PrivateDualClockPriorV3Error("Trajectory has no scored future")
    return prefix, future


def _prediction_iae(
    prefix: pd.DataFrame,
    future: pd.DataFrame,
    predicted: np.ndarray,
) -> float:
    return _trajectory_iae(
        float(prefix.iloc[-1]["equivalent_full_cycles"]),
        future["equivalent_full_cycles"].to_numpy(dtype=float),
        future["capacity_retention_pct"].to_numpy(dtype=float),
        predicted,
    )


def _selection_objective(
    by_condition: Sequence[float],
    config: Mapping[str, object],
) -> float:
    values = np.asarray(by_condition, dtype=float)
    penalty = float(config["selection"]["worst_condition_penalty"])
    return float(np.mean(values) + penalty * np.max(values))


def _hyperparameters(config: Mapping[str, object]) -> list[dict[str, float]]:
    family = config["dual_clock_family"]
    return [
        {
            "time_exponent": float(time_exponent),
            "cycle_exponent": float(cycle_exponent),
            "gamma": float(gamma),
            "shrinkage": float(shrinkage),
            "anchor_weight": float(anchor_weight),
        }
        for time_exponent, cycle_exponent, gamma, shrinkage, anchor_weight in product(
            family["time_exponents"],
            family["cycle_exponents"],
            family["kernel_gammas"],
            family["coefficient_shrinkages"],
            family["anchor_weights"],
        )
    ]


def _fit_dual(
    references: pd.DataFrame,
    hyperparameters: Mapping[str, object],
) -> DualClockKernelPrior:
    time_exponent = float(hyperparameters["time_exponent"])
    cycle_exponent = float(hyperparameters["cycle_exponent"])
    gamma = float(hyperparameters["gamma"])
    condition_ids: list[str] = []
    vectors: list[np.ndarray] = []
    coefficients: list[np.ndarray] = []
    for condition_id, condition in references.groupby("condition_id", sort=True):
        condition_ids.append(str(condition_id))
        vectors.append(_dual_condition_vector_fast(condition))
        coefficients.append(
            np.mean(
                [
                    nnls(
                        _dual_basis_fast(
                            cell["elapsed_days"].to_numpy(dtype=float),
                            cell["equivalent_full_cycles"].to_numpy(dtype=float),
                            time_exponent=time_exponent,
                            cycle_exponent=cycle_exponent,
                        ),
                        100.0
                        - cell["capacity_retention_pct"].to_numpy(dtype=float),
                    )[0]
                    for _, cell in condition.groupby("cell_id", sort=True)
                ],
                axis=0,
            )
        )
    matrix = np.vstack(vectors)
    center = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    standardized = (matrix - center) / scale
    return DualClockKernelPrior(
        time_exponent=time_exponent,
        cycle_exponent=cycle_exponent,
        gamma=gamma,
        condition_center=tuple(float(value) for value in center),
        condition_scale=tuple(float(value) for value in scale),
        support_condition_ids=tuple(condition_ids),
        support_condition_vectors=tuple(
            tuple(float(value) for value in row) for row in standardized
        ),
        support_coefficients=tuple(
            tuple(float(value) for value in row) for row in coefficients
        ),
    )


def _dual_basis_fast(
    elapsed_days: np.ndarray,
    exposure_efc: np.ndarray,
    *,
    time_exponent: float,
    cycle_exponent: float,
) -> np.ndarray:
    return np.column_stack(
        [
            np.power(elapsed_days / 365.0, time_exponent),
            np.power(exposure_efc / 1000.0, cycle_exponent),
        ]
    )


def _dual_condition_vector_fast(frame: pd.DataFrame) -> np.ndarray:
    first = frame.iloc[0]
    log_duty_rates = []
    for _, cell in frame.groupby("cell_id", sort=True):
        last = cell.sort_values("visit_index", kind="stable").iloc[-1]
        duty_rate = max(
            float(last["equivalent_full_cycles"])
            / max(float(last["elapsed_days"]), 1e-9),
            1e-4,
        )
        log_duty_rates.append(math.log(duty_rate))
    return np.asarray(
        [
            float(first["temperature_c"]),
            float(first["dod_fraction"]),
            float(first["discharge_c_rate"]),
            float(np.mean(log_duty_rates)),
        ],
        dtype=float,
    )


def _predict_dual(
    prefix: pd.DataFrame,
    forecast_efc: np.ndarray,
    model: DualClockKernelPrior,
    hyperparameters: Mapping[str, object],
) -> np.ndarray:
    target = (
        _dual_condition_vector_fast(prefix)
        - np.asarray(model.condition_center, dtype=float)
    ) / np.asarray(model.condition_scale, dtype=float)
    support = np.asarray(model.support_condition_vectors, dtype=float)
    log_weights = -model.gamma * np.sum(np.square(support - target), axis=1)
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    prior = weights @ np.asarray(model.support_coefficients, dtype=float)
    matrix = _dual_basis_fast(
        prefix["elapsed_days"].to_numpy(dtype=float),
        prefix["equivalent_full_cycles"].to_numpy(dtype=float),
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    fade = 100.0 - prefix["capacity_retention_pct"].to_numpy(dtype=float)
    shrinkage = float(hyperparameters["shrinkage"])
    coefficients = nnls(
        np.vstack(
            [matrix, math.sqrt(shrinkage) * np.eye(len(prior), dtype=float)]
        ),
        np.concatenate([fade, math.sqrt(shrinkage) * prior]),
    )[0]
    ordered = prefix.sort_values("visit_index", kind="stable")
    last = ordered.iloc[-1]
    duty_rate = max(
        float(last["equivalent_full_cycles"])
        / max(float(last["elapsed_days"]), 1e-9),
        1e-4,
    )
    forecast_elapsed = forecast_efc / duty_rate
    forecast_matrix = _dual_basis_fast(
        forecast_elapsed,
        forecast_efc,
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    last_matrix = _dual_basis_fast(
        np.asarray([float(last["elapsed_days"])]),
        np.asarray([float(last["equivalent_full_cycles"])]),
        time_exponent=model.time_exponent,
        cycle_exponent=model.cycle_exponent,
    )
    latent_last = float((100.0 - last_matrix @ coefficients)[0])
    residual = float(last["capacity_retention_pct"]) - latent_last
    return (
        100.0
        - forecast_matrix @ coefficients
        + float(hyperparameters["anchor_weight"]) * residual
    )


def _selected_inner_predictions(
    references: pd.DataFrame,
    *,
    landmark: int,
    hyperparameters: Mapping[str, object],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    score_end = float(config["score_end_equivalent_full_cycles"])
    rows: list[dict[str, object]] = []
    for held_condition in sorted(references["condition_id"].unique()):
        training = references.loc[
            references["condition_id"] != held_condition
        ].copy()
        targets = references.loc[
            references["condition_id"] == held_condition
        ].copy()
        model = _fit_dual(training, hyperparameters)
        for cell_id, cell in targets.groupby("cell_id", sort=True):
            prefix, future = _future(cell, landmark=landmark, score_end=score_end)
            forecast = future["equivalent_full_cycles"].to_numpy(dtype=float)
            rows.append(
                {
                    "condition_id": str(held_condition),
                    "cell_id": str(cell_id),
                    "prefix": prefix,
                    "future": future,
                    "predicted": _predict_dual(
                        prefix, forecast, model, hyperparameters
                    ),
                }
            )
    return rows


def _select_hyperparameters(
    references: pd.DataFrame,
    *,
    landmark: int,
    config: Mapping[str, object],
) -> tuple[dict[str, float], float]:
    score_end = float(config["score_end_equivalent_full_cycles"])
    candidates = _hyperparameters(config)
    risks: dict[str, list[float]] = {_json_text(item): [] for item in candidates}
    grouped: dict[tuple[float, float, float], list[dict[str, float]]] = {}
    for candidate in candidates:
        grouped.setdefault(
            (
                candidate["time_exponent"],
                candidate["cycle_exponent"],
                candidate["gamma"],
            ),
            [],
        ).append(candidate)
    for held_condition in sorted(references["condition_id"].unique()):
        training = references.loc[
            references["condition_id"] != held_condition
        ].copy()
        targets = references.loc[
            references["condition_id"] == held_condition
        ].copy()
        for group_candidates in grouped.values():
            model = _fit_dual(training, group_candidates[0])
            for candidate in group_candidates:
                cell_risks = []
                for _, cell in targets.groupby("cell_id", sort=True):
                    prefix, future = _future(
                        cell, landmark=landmark, score_end=score_end
                    )
                    forecast = future["equivalent_full_cycles"].to_numpy(
                        dtype=float
                    )
                    predicted = _predict_dual(
                        prefix, forecast, model, candidate
                    )
                    cell_risks.append(
                        _prediction_iae(prefix, future, predicted)
                    )
                risks[_json_text(candidate)].append(float(np.mean(cell_risks)))
    ranked = sorted(
        (
            _selection_objective(risks[_json_text(candidate)], config),
            _json_text(candidate),
            candidate,
        )
        for candidate in candidates
    )
    objective, _, selected = ranked[0]
    return dict(selected), float(objective)


def _forecast_grid(
    prefix: pd.DataFrame,
    config: Mapping[str, object],
) -> np.ndarray:
    x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
    end = float(config["score_end_equivalent_full_cycles"])
    step = float(config["forecast_grid_step_equivalent_full_cycles"])
    first = math.ceil((x0 + 1e-12) / step) * step
    future = np.arange(first, end + step * 0.5, step, dtype=float)
    return np.concatenate([[x0], future[future > x0]])


def _ood_diagnostic(
    prefix: pd.DataFrame,
    model: DualClockKernelPrior,
    config: Mapping[str, object],
) -> tuple[float, float]:
    _, distances = dual_clock_prior_coefficients(prefix, model)
    nearest = float(np.min(distances))
    support = np.asarray(model.support_condition_vectors, dtype=float)
    nearest_cross = []
    for index in range(len(support)):
        other = np.delete(support, index, axis=0)
        nearest_cross.append(
            float(np.min(np.sqrt(np.sum(np.square(other - support[index]), axis=1))))
        )
    threshold = float(
        np.quantile(
            nearest_cross,
            float(config["ood"]["cross_condition_quantile"]),
            method="higher",
        )
        * float(config["ood"]["threshold_multiplier"])
    )
    return nearest, threshold


def _prefix_residual_rms(
    prefix: pd.DataFrame,
    model: DualClockKernelPrior,
    hyperparameters: Mapping[str, object],
) -> float:
    exposure = prefix["equivalent_full_cycles"].to_numpy(dtype=float)
    predicted = _predict_dual(prefix, exposure, model, hyperparameters)
    observed = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(predicted - observed))))


def predict_private_dual_clock_prior_v3(
    references: pd.DataFrame,
    prefixes: pd.DataFrame,
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Select on reference conditions and predict without suffix truth."""
    frozen = validate_private_dual_clock_prior_v3_config(config)
    if tuple(references.columns) != REFERENCE_COLUMNS:
        raise PrivateDualClockPriorV3Error("Private V3 reference columns changed")
    if tuple(prefixes.columns) != TARGET_PREFIX_COLUMNS:
        raise PrivateDualClockPriorV3Error("Private V3 prefix columns changed")
    prediction_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    lower, upper = (float(value) for value in frozen["prediction_clip_pct"])
    baseline_hyper = {
        "exponent": 0.5,
        "alpha": 1.0,
        "prefix_rate_weight": 0.0,
        "anchor_weight": 1.0,
    }
    for (outer, landmark), target_prefixes in prefixes.groupby(
        ["outer_condition_id", "landmark_visit_count"], sort=True
    ):
        landmark_int = int(landmark)
        reference = references.loc[references["outer_condition_id"] == outer]
        if reference.empty or outer in set(reference["condition_id"]):
            raise PrivateDualClockPriorV3Error("Private V3 outer fold leaked")
        reference_core = _core(reference)
        hyperparameters, inner_risk = _select_hyperparameters(
            reference_core, landmark=landmark_int, config=frozen
        )
        dual_model = _fit_dual(reference_core, hyperparameters)
        baseline_model = fit_power_condition_prior(
            reference_core, exponent=0.5, alpha=1.0
        )
        for cell_id, prefix in target_prefixes.groupby("cell_id", sort=True):
            ordered = _core(prefix).sort_values("visit_index", kind="stable")
            if len(ordered) != landmark_int:
                raise PrivateDualClockPriorV3Error("Private V3 prefix is not exact")
            grid = _forecast_grid(ordered, frozen)
            dual_prediction = _predict_dual(
                ordered, grid, dual_model, hyperparameters
            )
            baseline_prediction = predict_power_condition_prior(
                ordered,
                grid,
                baseline_model,
                prefix_rate_weight=baseline_hyper["prefix_rate_weight"],
                anchor_weight=baseline_hyper["anchor_weight"],
            )
            persistence = np.full_like(
                grid,
                float(ordered.iloc[-1]["capacity_retention_pct"]),
                dtype=float,
            )
            predictions = {
                "target_prefix_persistence": persistence,
                "v1_condition_ridge_delta": baseline_prediction,
                PRIMARY_MODEL_ID: dual_prediction,
            }
            for model_id, values in predictions.items():
                for exposure, predicted in zip(grid, values, strict=True):
                    prediction_rows.append(
                        {
                            "experiment_id": EXPERIMENT_ID,
                            "dataset_id": str(frozen["dataset_id"]),
                            "outer_condition_id": str(outer),
                            "cell_id": str(cell_id),
                            "landmark_visit_count": landmark_int,
                            "model_id": model_id,
                            "forecast_equivalent_full_cycles": float(exposure),
                            "predicted_capacity_retention_pct": float(
                                np.clip(predicted, lower, upper)
                            ),
                        }
                    )
            nearest, threshold = _ood_diagnostic(ordered, dual_model, frozen)
            decision_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_id": str(frozen["dataset_id"]),
                    "outer_condition_id": str(outer),
                    "cell_id": str(cell_id),
                    "landmark_visit_count": landmark_int,
                    "dual_clock_hyperparameters_json": _json_text(hyperparameters),
                    "inner_condition_equal_iae_pp": inner_risk,
                    "prefix_duty_rate_efc_per_day": (
                        prefix_duty_rate_efc_per_day(ordered)
                    ),
                    "nearest_condition_distance": nearest,
                    "condition_ood_threshold": threshold,
                    "prefix_residual_rms_pp": _prefix_residual_rms(
                        ordered, dual_model, hyperparameters
                    ),
                    "evidence_status": (
                        "supported"
                        if nearest <= threshold
                        else "diagnostic_condition_or_duty_ood"
                    ),
                }
            )
    predictions_frame = pd.DataFrame(
        prediction_rows, columns=PREDICTION_COLUMNS
    ).sort_values(
        [
            "outer_condition_id",
            "cell_id",
            "landmark_visit_count",
            "model_id",
            "forecast_equivalent_full_cycles",
        ],
        kind="stable",
        ignore_index=True,
    )
    decisions_frame = pd.DataFrame(
        decision_rows, columns=DECISION_COLUMNS
    ).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count"],
        kind="stable",
        ignore_index=True,
    )
    manifest: dict[str, object] = {
        "schema_version": "lifetwin.private_dual_clock_prior_v3.prediction_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": str(frozen["dataset_id"]),
        "private_only": True,
        "config_sha256": canonical_json_sha256(frozen),
        "reference_rows_sha256": canonical_frame_sha256(
            references, REFERENCE_COLUMNS
        ),
        "target_prefix_rows_sha256": canonical_frame_sha256(
            prefixes, TARGET_PREFIX_COLUMNS
        ),
        "prediction_rows_sha256": canonical_frame_sha256(
            predictions_frame, PREDICTION_COLUMNS
        ),
        "decision_rows_sha256": canonical_frame_sha256(
            decisions_frame, DECISION_COLUMNS
        ),
        "prediction_row_count": len(predictions_frame),
        "decision_row_count": len(decisions_frame),
        "target_truth_argument_accepted": False,
        "target_suffix_rows_present": False,
        "future_elapsed_days_argument_accepted": False,
        "future_schedule_assumption": "constant_prefix_efc_per_day",
        "public_release_permitted": False,
    }
    manifest["manifest_content_sha256"] = canonical_json_sha256(manifest)
    return predictions_frame, decisions_frame, manifest


def _validate_replay(
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> None:
    frozen = validate_private_dual_clock_prior_v3_config(config)
    if tuple(predictions.columns) != PREDICTION_COLUMNS:
        raise PrivateDualClockPriorV3Error("Private V3 prediction columns changed")
    if tuple(decisions.columns) != DECISION_COLUMNS:
        raise PrivateDualClockPriorV3Error("Private V3 decision columns changed")
    if manifest.get("config_sha256") != canonical_json_sha256(frozen):
        raise PrivateDualClockPriorV3Error("Private V3 config hash changed")
    if manifest.get("prediction_rows_sha256") != canonical_frame_sha256(
        predictions, PREDICTION_COLUMNS
    ):
        raise PrivateDualClockPriorV3Error("Private V3 predictions changed after freeze")
    if manifest.get("decision_rows_sha256") != canonical_frame_sha256(
        decisions, DECISION_COLUMNS
    ):
        raise PrivateDualClockPriorV3Error("Private V3 decisions changed after freeze")


def score_private_dual_clock_prior_v3(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    decisions: pd.DataFrame,
    manifest: Mapping[str, object],
    config: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Link frozen V3 predictions to held-condition capacity suffixes."""
    frozen = validate_private_dual_clock_prior_v3_config(config)
    if tuple(truth.columns) != TARGET_TRUTH_COLUMNS:
        raise PrivateDualClockPriorV3Error("Private V3 truth columns changed")
    _validate_replay(predictions, decisions, manifest, frozen)
    score_end = float(frozen["score_end_equivalent_full_cycles"])
    landmarks = tuple(int(item) for item in frozen["landmark_visit_counts"])
    rows: list[dict[str, object]] = []
    for (outer, cell_id), cell in truth.groupby(
        ["outer_condition_id", "cell_id"], sort=True
    ):
        ordered = _core(cell).sort_values("visit_index", kind="stable")
        for landmark in landmarks:
            prefix, future = _future(
                ordered, landmark=landmark, score_end=score_end
            )
            forecast = future["equivalent_full_cycles"].to_numpy(dtype=float)
            actual = future["capacity_retention_pct"].to_numpy(dtype=float)
            x0 = float(prefix.iloc[-1]["equivalent_full_cycles"])
            for model_id in MODEL_IDS:
                curve = predictions.loc[
                    (predictions["outer_condition_id"] == outer)
                    & (predictions["cell_id"] == cell_id)
                    & (predictions["landmark_visit_count"] == landmark)
                    & (predictions["model_id"] == model_id)
                ].sort_values("forecast_equivalent_full_cycles", kind="stable")
                if curve.empty:
                    raise PrivateDualClockPriorV3Error(
                        "Private V3 score curve is missing"
                    )
                predicted = np.interp(
                    forecast,
                    curve["forecast_equivalent_full_cycles"].to_numpy(dtype=float),
                    curve["predicted_capacity_retention_pct"].to_numpy(dtype=float),
                )
                error = predicted - actual
                rows.append(
                    {
                        "experiment_id": EXPERIMENT_ID,
                        "dataset_id": str(frozen["dataset_id"]),
                        "outer_condition_id": str(outer),
                        "cell_id": str(cell_id),
                        "landmark_visit_count": landmark,
                        "model_id": model_id,
                        "future_observation_count": len(future),
                        "trajectory_iae_pp": _trajectory_iae(
                            x0, forecast, actual, predicted
                        ),
                        "trajectory_mae_pp": float(np.mean(np.abs(error))),
                        "trajectory_rmse_pp": float(
                            np.sqrt(np.mean(np.square(error)))
                        ),
                        "endpoint_absolute_error_pp": float(abs(error[-1])),
                    }
                )
    scores = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values(
        ["outer_condition_id", "cell_id", "landmark_visit_count", "model_id"],
        kind="stable",
        ignore_index=True,
    )
    summaries: dict[str, object] = {}
    comparisons: dict[str, object] = {}
    for landmark in landmarks:
        selected = scores.loc[scores["landmark_visit_count"] == landmark]
        model_summary: dict[str, object] = {}
        condition_iae: dict[str, pd.Series] = {}
        for model_id in MODEL_IDS:
            model_scores = selected.loc[selected["model_id"] == model_id]
            condition_means = model_scores.groupby(
                "outer_condition_id", sort=True
            )[
                [
                    "trajectory_iae_pp",
                    "trajectory_mae_pp",
                    "trajectory_rmse_pp",
                    "endpoint_absolute_error_pp",
                ]
            ].mean()
            condition_iae[model_id] = condition_means["trajectory_iae_pp"]
            model_summary[model_id] = {
                "condition_equal_trajectory_iae_pp": float(
                    condition_means["trajectory_iae_pp"].mean()
                ),
                "cell_equal_trajectory_iae_pp": float(
                    model_scores["trajectory_iae_pp"].mean()
                ),
                "condition_equal_trajectory_mae_pp": float(
                    condition_means["trajectory_mae_pp"].mean()
                ),
                "condition_equal_trajectory_rmse_pp": float(
                    condition_means["trajectory_rmse_pp"].mean()
                ),
                "condition_equal_endpoint_absolute_error_pp": float(
                    condition_means["endpoint_absolute_error_pp"].mean()
                ),
            }
        summaries[str(landmark)] = model_summary
        baseline = condition_iae["v1_condition_ridge_delta"]
        candidate = condition_iae[PRIMARY_MODEL_ID]
        improvement = baseline - candidate
        comparisons[str(landmark)] = {
            "baseline_model_id": "v1_condition_ridge_delta",
            "candidate_model_id": PRIMARY_MODEL_ID,
            "absolute_condition_equal_iae_improvement_pp": float(
                baseline.mean() - candidate.mean()
            ),
            "relative_condition_equal_iae_improvement_fraction": float(
                (baseline.mean() - candidate.mean())
                / max(float(baseline.mean()), 1e-12)
            ),
            "improved_condition_fraction": float((improvement > 0.0).mean()),
            "worst_condition_regression_pp": float(
                max(0.0, float((-improvement).max()))
            ),
            "exact_one_sided_condition_sign_flip_p_value": (
                _exact_one_sided_sign_flip_p(improvement.tolist())
            ),
        }
    summary: dict[str, object] = {
        "schema_version": "lifetwin.private_dual_clock_prior_v3.score_summary.v1",
        "experiment_id": EXPERIMENT_ID,
        "dataset_id": str(frozen["dataset_id"]),
        "private_only": True,
        "evidence_role": "outcome_exposed_private_model_development",
        "model_summary_by_landmark": summaries,
        "comparison_vs_v1_condition_prior": comparisons,
        "prediction_manifest_content_sha256": manifest.get(
            "manifest_content_sha256"
        ),
        "mechanism_hypothesis": (
            "Separate elapsed-time and throughput coordinates reduce confounding "
            "between calendar and cycle degradation under unequal duty schedules."
        ),
        "claim_boundary": (
            "Private SNL cycle-aging development only; not independent, field, "
            "Hithium-product, or 15-25 year validation."
        ),
        "public_release_permitted": False,
    }
    summary["score_rows_sha256"] = canonical_frame_sha256(scores, SCORE_COLUMNS)
    summary["summary_content_sha256"] = canonical_json_sha256(summary)
    return scores, summary


def _interval_quantiles(
    inner_predictions: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
) -> list[dict[str, object]]:
    bins = [float(value) for value in config["uncertainty"]["horizon_bins_efc"]]
    quantile = float(config["uncertainty"]["absolute_residual_quantile"])
    records: list[tuple[float, float]] = []
    for record in inner_predictions:
        prefix = record["prefix"]
        future = record["future"]
        horizon = future["equivalent_full_cycles"].to_numpy(dtype=float) - float(
            prefix.iloc[-1]["equivalent_full_cycles"]
        )
        error = np.abs(
            np.asarray(record["predicted"], dtype=float)
            - future["capacity_retention_pct"].to_numpy(dtype=float)
        )
        records.extend(
            (float(horizon_value), float(error_value))
            for horizon_value, error_value in zip(horizon, error, strict=True)
        )
    all_errors = np.asarray([error for _, error in records], dtype=float)
    fallback = float(np.quantile(all_errors, quantile, method="higher"))
    output = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        errors = np.asarray(
            [error for horizon, error in records if lower <= horizon < upper],
            dtype=float,
        )
        output.append(
            {
                "minimum_horizon_efc": lower,
                "maximum_horizon_efc": upper,
                "support_count": len(errors),
                "absolute_error_quantile_pp": (
                    float(np.quantile(errors, quantile, method="higher"))
                    if len(errors) >= 5
                    else fallback
                ),
            }
        )
    return output


def train_private_dual_clock_prior_capsule(
    trajectories: pd.DataFrame,
    config: Mapping[str, object],
    *,
    training_identity: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Train full-reference V3 models and export a JSON-safe private capsule."""
    frozen = validate_private_dual_clock_prior_v3_config(config)
    data = _core(trajectories).sort_values(
        ["condition_id", "cell_id", "visit_index"],
        kind="stable",
        ignore_index=True,
    )
    landmark_models: dict[str, object] = {}
    for landmark in (int(value) for value in frozen["landmark_visit_counts"]):
        hyperparameters, inner_risk = _select_hyperparameters(
            data, landmark=landmark, config=frozen
        )
        model = _fit_dual(data, hyperparameters)
        inner = _selected_inner_predictions(
            data,
            landmark=landmark,
            hyperparameters=hyperparameters,
            config=frozen,
        )
        support_prefixes = data.groupby("cell_id", sort=True).head(landmark)
        thresholds = []
        nearest_values = []
        for _, prefix in support_prefixes.groupby("cell_id", sort=True):
            nearest, threshold = _ood_diagnostic(prefix, model, frozen)
            nearest_values.append(nearest)
            thresholds.append(threshold)
        landmark_models[str(landmark)] = {
            "hyperparameters": hyperparameters,
            "dual_clock_prior": model.to_dict(),
            "inner_condition_equal_iae_pp": inner_risk,
            "condition_ood_threshold": float(max(thresholds)),
            "training_support_nearest_condition_distance_maximum": float(
                max(nearest_values)
            ),
            "diagnostic_interval_quantiles": _interval_quantiles(inner, frozen),
        }
    capsule: dict[str, object] = {
        "schema_version": "lifetwin.private_dual_clock_prior_v3.capsule.v1",
        "model_id": PRIMARY_MODEL_ID,
        "private_only": True,
        "dataset_id": str(frozen["dataset_id"]),
        "config_sha256": canonical_json_sha256(frozen),
        "training_identity": dict(training_identity or {}),
        "condition_fields": [
            "temperature_c",
            "dod_fraction",
            "discharge_c_rate",
            "log_prefix_duty_rate_efc_per_day",
        ],
        "exposure_fields": ["elapsed_days", "equivalent_full_cycles"],
        "target_field": "capacity_retention_pct",
        "default_future_schedule_assumption": "constant_prefix_efc_per_day",
        "explicit_forecast_elapsed_days_supported": True,
        "minimum_prefix_visits": min(
            int(value) for value in frozen["landmark_visit_counts"]
        ),
        "maximum_validated_prefix_visits": max(
            int(value) for value in frozen["landmark_visit_counts"]
        ),
        "prediction_clip_pct": frozen["prediction_clip_pct"],
        "landmark_models": landmark_models,
        "raw_training_rows_in_capsule": False,
        "formal_interval_coverage_claim": False,
        "public_release_permitted": False,
    }
    capsule["capsule_content_sha256"] = canonical_json_sha256(capsule)
    return capsule


def _interval_width(
    horizon: float,
    quantiles: Sequence[Mapping[str, object]],
) -> float:
    for item in quantiles:
        if float(item["minimum_horizon_efc"]) <= horizon < float(
            item["maximum_horizon_efc"]
        ):
            return float(item["absolute_error_quantile_pp"])
    return float(quantiles[-1]["absolute_error_quantile_pp"])


def predict_private_dual_clock_prior_capsule(
    prefix: pd.DataFrame,
    forecast_efc: Sequence[float] | np.ndarray,
    capsule: Mapping[str, object],
    *,
    forecast_elapsed_days: Sequence[float] | np.ndarray | None = None,
    strict_ood: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Predict a private cell using inferred or explicitly planned time coordinates."""
    if capsule.get("schema_version") != (
        "lifetwin.private_dual_clock_prior_v3.capsule.v1"
    ):
        raise PrivateDualClockPriorV3Error("Private V3 capsule schema changed")
    expected_hash = str(capsule.get("capsule_content_sha256", ""))
    hash_input = dict(capsule)
    hash_input.pop("capsule_content_sha256", None)
    if canonical_json_sha256(hash_input) != expected_hash:
        raise PrivateDualClockPriorV3Error("Private V3 capsule content changed")
    ordered = _core(prefix).sort_values("visit_index", kind="stable")
    visit_count = len(ordered)
    landmarks = sorted(int(value) for value in capsule["landmark_models"])
    supported = [value for value in landmarks if value <= visit_count]
    if not supported:
        raise PrivateDualClockPriorV3Error("Private prefix has too few RPT visits")
    selected_landmark = max(supported)
    model_bundle = capsule["landmark_models"][str(selected_landmark)]
    model = DualClockKernelPrior.from_dict(model_bundle["dual_clock_prior"])
    hyperparameters = model_bundle["hyperparameters"]
    forecast = np.asarray(forecast_efc, dtype=float)
    if not np.isfinite(forecast).all() or (
        forecast <= float(ordered.iloc[-1]["equivalent_full_cycles"])
    ).any():
        raise PrivateDualClockPriorV3Error(
            "Private forecast coordinates must be finite and beyond the prefix"
        )
    elapsed = (
        infer_constant_duty_elapsed_days(ordered, forecast)
        if forecast_elapsed_days is None
        else np.asarray(forecast_elapsed_days, dtype=float)
    )
    if elapsed.shape != forecast.shape or not np.isfinite(elapsed).all():
        raise PrivateDualClockPriorV3Error(
            "Forecast elapsed-day coordinates are invalid"
        )
    if (elapsed <= float(ordered.iloc[-1]["elapsed_days"])).any():
        raise PrivateDualClockPriorV3Error(
            "Forecast elapsed days must be beyond the prefix"
        )
    nearest, _ = _ood_diagnostic(
        ordered,
        model,
        {"ood": {"cross_condition_quantile": 0.99, "threshold_multiplier": 1.5}},
    )
    threshold = float(model_bundle["condition_ood_threshold"])
    if strict_ood and nearest > threshold:
        raise PrivateDualClockPriorV3Error(
            "Private target condition or duty cycle is outside capsule support"
        )
    predicted = predict_dual_clock_kernel_prior(
        ordered,
        forecast,
        model,
        shrinkage=float(hyperparameters["shrinkage"]),
        anchor_weight=float(hyperparameters["anchor_weight"]),
        forecast_elapsed_days=elapsed,
    )
    lower_clip, upper_clip = (
        float(value) for value in capsule["prediction_clip_pct"]
    )
    predicted = np.clip(predicted, lower_clip, upper_clip)
    x0 = float(ordered.iloc[-1]["equivalent_full_cycles"])
    widths = np.asarray(
        [
            _interval_width(
                float(exposure - x0),
                model_bundle["diagnostic_interval_quantiles"],
            )
            for exposure in forecast
        ],
        dtype=float,
    )
    result = pd.DataFrame(
        {
            "forecast_elapsed_days": elapsed,
            "forecast_equivalent_full_cycles": forecast,
            "predicted_capacity_retention_pct": predicted,
            "diagnostic_lower_capacity_retention_pct": np.clip(
                predicted - widths, lower_clip, upper_clip
            ),
            "diagnostic_upper_capacity_retention_pct": np.clip(
                predicted + widths, lower_clip, upper_clip
            ),
        }
    )
    metadata = {
        "model_id": PRIMARY_MODEL_ID,
        "selected_landmark_visit_count": selected_landmark,
        "provided_prefix_visit_count": visit_count,
        "future_schedule_source": (
            "constant_prefix_efc_per_day"
            if forecast_elapsed_days is None
            else "explicit_forecast_elapsed_days"
        ),
        "prefix_duty_rate_efc_per_day": prefix_duty_rate_efc_per_day(ordered),
        "evidence_status": (
            "supported"
            if nearest <= threshold and visit_count <= max(landmarks)
            else "extended_prefix_or_condition_diagnostic"
        ),
        "nearest_condition_distance": nearest,
        "condition_ood_threshold": threshold,
        "strict_ood": strict_ood,
        "formal_interval_coverage_claim": False,
    }
    return result, metadata


__all__ = [
    "DECISION_COLUMNS",
    "EXPERIMENT_ID",
    "MODEL_IDS",
    "PREDICTION_COLUMNS",
    "PRIMARY_MODEL_ID",
    "PrivateDualClockPriorV3Error",
    "SCHEMA_VERSION",
    "SCORE_COLUMNS",
    "default_private_dual_clock_prior_v3_config",
    "predict_private_dual_clock_prior_capsule",
    "predict_private_dual_clock_prior_v3",
    "score_private_dual_clock_prior_v3",
    "train_private_dual_clock_prior_capsule",
    "validate_private_dual_clock_prior_v3_config",
]
