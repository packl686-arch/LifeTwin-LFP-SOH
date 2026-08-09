from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from lifetwin.experiments.private_dual_clock_prior_v3 import (
    _core,
    _fit_dual,
    _forecast_grid,
    _future,
    _json_text,
    _predict_dual,
    _prediction_iae,
    _prefix_residual_rms,
    validate_private_dual_clock_prior_v3_config,
)
from lifetwin.models.hierarchical_cycle_prior import (
    fit_power_condition_prior,
    predict_power_condition_prior,
    prefix_duty_rate_efc_per_day,
)
from lifetwin.private_artifacts import atomic_write_json


SIGNATURE_COLUMNS = (
    "temperature_c",
    "dod_fraction",
    "discharge_c_rate",
    "log_prefix_duty_rate",
    "prefix_capacity_slope_pp_per_1000_efc",
    "prefix_linear_residual_rms_pp",
    "v3_prefix_residual_rms_pp",
    "prefix_last_capacity_retention_pct",
    "mean_abs_v3_v1_disagreement_pp",
)
EXPERTS = ("v1", "v3")


def _linear_prefix_features(prefix: pd.DataFrame) -> tuple[float, float]:
    x = prefix["equivalent_full_cycles"].to_numpy(dtype=float) / 1000.0
    y = prefix["capacity_retention_pct"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ coefficients
    return float(coefficients[1]), float(np.sqrt(np.mean(np.square(residual))))


def _mean_absolute_curve_difference(
    exposure: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    if len(exposure) < 2 or float(exposure[-1]) <= float(exposure[0]):
        return float(np.mean(np.abs(first - second)))
    return float(
        np.trapezoid(np.abs(first - second), exposure)
        / (float(exposure[-1]) - float(exposure[0]))
    )


def _signature(
    prefix: pd.DataFrame,
    grid: np.ndarray,
    v1_prediction: np.ndarray,
    v3_prediction: np.ndarray,
    *,
    v3_prefix_residual: float,
) -> dict[str, float]:
    ordered = prefix.sort_values("visit_index", kind="stable")
    slope, residual = _linear_prefix_features(ordered)
    first = ordered.iloc[0]
    values = {
        "temperature_c": float(first["temperature_c"]),
        "dod_fraction": float(first["dod_fraction"]),
        "discharge_c_rate": float(first["discharge_c_rate"]),
        "log_prefix_duty_rate": float(
            math.log(max(prefix_duty_rate_efc_per_day(ordered), 1e-9))
        ),
        "prefix_capacity_slope_pp_per_1000_efc": slope,
        "prefix_linear_residual_rms_pp": residual,
        "v3_prefix_residual_rms_pp": float(v3_prefix_residual),
        "prefix_last_capacity_retention_pct": float(
            ordered.iloc[-1]["capacity_retention_pct"]
        ),
        "mean_abs_v3_v1_disagreement_pp": _mean_absolute_curve_difference(
            grid, v1_prediction, v3_prediction
        ),
    }
    if tuple(values) != SIGNATURE_COLUMNS or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ValueError("Selective V3.1 signature changed or became non-finite")
    return values


def _inner_records(
    references: pd.DataFrame,
    *,
    landmark: int,
    hyperparameters: Mapping[str, object],
    config: Mapping[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    score_end = float(config["score_end_equivalent_full_cycles"])
    baseline_hyper = {
        "prefix_rate_weight": 0.0,
        "anchor_weight": 1.0,
    }
    for held_condition in sorted(references["condition_id"].unique()):
        training = references.loc[
            references["condition_id"] != held_condition
        ].copy()
        targets = references.loc[
            references["condition_id"] == held_condition
        ].copy()
        v3_model = _fit_dual(training, hyperparameters)
        v1_model = fit_power_condition_prior(training, exponent=0.5, alpha=1.0)
        for cell_id, cell in targets.groupby("cell_id", sort=True):
            prefix, future = _future(
                cell, landmark=landmark, score_end=score_end
            )
            future_grid = future["equivalent_full_cycles"].to_numpy(dtype=float)
            signature_grid = _forecast_grid(prefix, config)
            v1_future = predict_power_condition_prior(
                prefix,
                future_grid,
                v1_model,
                prefix_rate_weight=baseline_hyper["prefix_rate_weight"],
                anchor_weight=baseline_hyper["anchor_weight"],
            )
            v3_future = _predict_dual(
                prefix, future_grid, v3_model, hyperparameters
            )
            v1_signature = predict_power_condition_prior(
                prefix,
                signature_grid,
                v1_model,
                prefix_rate_weight=baseline_hyper["prefix_rate_weight"],
                anchor_weight=baseline_hyper["anchor_weight"],
            )
            v3_signature = _predict_dual(
                prefix, signature_grid, v3_model, hyperparameters
            )
            rows.append(
                {
                    "condition_id": str(held_condition),
                    "cell_id": str(cell_id),
                    **_signature(
                        prefix,
                        signature_grid,
                        v1_signature,
                        v3_signature,
                        v3_prefix_residual=_prefix_residual_rms(
                            prefix, v3_model, hyperparameters
                        ),
                    ),
                    "risk__v1": _prediction_iae(prefix, future, v1_future),
                    "risk__v3": _prediction_iae(prefix, future, v3_future),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition_id", "cell_id"], kind="stable", ignore_index=True
    )


def _center_scale(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = frame.loc[:, SIGNATURE_COLUMNS].to_numpy(dtype=float)
    center = np.median(matrix, axis=0)
    mad = 1.4826 * np.median(np.abs(matrix - center), axis=0)
    return center, np.maximum(mad, 1e-3)


def _distances(
    target: Mapping[str, object],
    support: pd.DataFrame,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    target_values = np.asarray(
        [float(target[column]) for column in SIGNATURE_COLUMNS], dtype=float
    )
    matrix = support.loc[:, SIGNATURE_COLUMNS].to_numpy(dtype=float)
    return np.sqrt(np.mean(np.square((matrix - target_values) / scale), axis=1))


def _condition_equal_global_risk(support: pd.DataFrame, expert: str) -> float:
    return float(
        support.groupby("condition_id", sort=True)[f"risk__{expert}"].mean().mean()
    )


def _estimate_risk(
    target: Mapping[str, object],
    support: pd.DataFrame,
    parameters: Mapping[str, object],
) -> tuple[dict[str, float], float]:
    center, scale = _center_scale(support)
    distance = _distances(target, support, center, scale)
    neighbor_count = min(int(parameters["neighbor_count"]), len(support))
    order = np.argsort(distance, kind="stable")[:neighbor_count]
    selected_distance = np.maximum(distance[order], 1e-6)
    weights = 1.0 / selected_distance
    weights /= float(np.sum(weights))
    local_weight = float(parameters["local_weight"])
    penalty = float(parameters["dispersion_penalty"])
    estimated: dict[str, float] = {}
    for expert in EXPERTS:
        values = support.iloc[order][f"risk__{expert}"].to_numpy(dtype=float)
        local_mean = float(np.sum(weights * values))
        local_variance = float(
            np.sum(weights * np.square(values - local_mean))
        )
        local_risk = local_mean + penalty * math.sqrt(max(local_variance, 0.0))
        global_risk = _condition_equal_global_risk(support, expert)
        estimated[expert] = (
            (1.0 - local_weight) * global_risk + local_weight * local_risk
        )
    return estimated, float(distance[order[0]])


def _select_expert(risk: Mapping[str, float], margin: float) -> str:
    return "v3" if float(risk["v3"]) + margin < float(risk["v1"]) else "v1"


def _gate_candidates() -> list[dict[str, object]]:
    return [
        {
            "neighbor_count": neighbor_count,
            "local_weight": local_weight,
            "dispersion_penalty": dispersion_penalty,
            "required_v3_margin_pp": required_margin,
        }
        for neighbor_count, local_weight, dispersion_penalty, required_margin in product(
            (3, 5, 7),
            (0.25, 0.5, 0.75, 1.0),
            (0.0, 0.25, 0.5),
            (0.0, 0.05, 0.1, 0.2),
        )
    ]


def _cross_fitted_gate_risk(
    records: pd.DataFrame,
    parameters: Mapping[str, object],
) -> tuple[float, float, float]:
    condition_errors: list[float] = []
    condition_regressions: list[float] = []
    v3_choices = 0
    row_count = 0
    for held_condition in sorted(records["condition_id"].unique()):
        support = records.loc[records["condition_id"] != held_condition]
        targets = records.loc[records["condition_id"] == held_condition]
        selected_errors = []
        v1_errors = []
        for row in targets.to_dict("records"):
            risk, _ = _estimate_risk(row, support, parameters)
            expert = _select_expert(
                risk, float(parameters["required_v3_margin_pp"])
            )
            selected_errors.append(float(row[f"risk__{expert}"]))
            v1_errors.append(float(row["risk__v1"]))
            v3_choices += int(expert == "v3")
            row_count += 1
        condition_error = float(np.mean(selected_errors))
        condition_v1 = float(np.mean(v1_errors))
        condition_errors.append(condition_error)
        condition_regressions.append(max(0.0, condition_error - condition_v1))
    return (
        float(np.mean(condition_errors)),
        float(np.max(condition_regressions)),
        float(v3_choices / row_count),
    )


def _select_gate(records: pd.DataFrame) -> tuple[dict[str, object], dict[str, float]]:
    ranked = []
    for parameters in _gate_candidates():
        mean_risk, worst_regression, v3_fraction = _cross_fitted_gate_risk(
            records, parameters
        )
        objective = mean_risk + 0.5 * worst_regression
        ranked.append(
            (
                objective,
                mean_risk,
                worst_regression,
                _json_text(parameters),
                parameters,
                v3_fraction,
            )
        )
    objective, mean_risk, worst_regression, _, parameters, v3_fraction = sorted(
        ranked
    )[0]
    return dict(parameters), {
        "selection_objective": float(objective),
        "cross_fitted_condition_equal_iae_pp": float(mean_risk),
        "cross_fitted_worst_condition_regression_vs_v1_pp": float(
            worst_regression
        ),
        "cross_fitted_v3_choice_fraction": float(v3_fraction),
    }


def _curve(
    predictions: pd.DataFrame,
    *,
    outer: str,
    cell_id: str,
    landmark: int,
    model_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = predictions.loc[
        (predictions["outer_condition_id"] == outer)
        & (predictions["cell_id"] == cell_id)
        & (predictions["landmark_visit_count"] == landmark)
        & (predictions["model_id"] == model_id)
    ].sort_values("forecast_equivalent_full_cycles", kind="stable")
    if selected.empty:
        raise ValueError("Missing frozen expert prediction")
    return (
        selected["forecast_equivalent_full_cycles"].to_numpy(dtype=float),
        selected["predicted_capacity_retention_pct"].to_numpy(dtype=float),
    )


def _score_value(
    scores: pd.DataFrame,
    *,
    outer: str,
    cell_id: str,
    landmark: int,
    model_id: str,
) -> float:
    selected = scores.loc[
        (scores["outer_condition_id"] == outer)
        & (scores["cell_id"] == cell_id)
        & (scores["landmark_visit_count"] == landmark)
        & (scores["model_id"] == model_id),
        "trajectory_iae_pp",
    ]
    if len(selected) != 1:
        raise ValueError("Frozen score identity is not unique")
    return float(selected.iloc[0])


def run_exploration(
    input_directory: Path,
    v3_directory: Path,
) -> dict[str, object]:
    references = pd.read_parquet(input_directory / "outer_fold_references.parquet")
    prefixes = pd.read_parquet(input_directory / "target_prefixes.parquet")
    config = validate_private_dual_clock_prior_v3_config(
        json.loads((v3_directory / "private_config.json").read_text(encoding="utf-8"))
    )
    decisions = pd.read_parquet(v3_directory / "model_decisions.parquet")
    predictions = pd.read_parquet(v3_directory / "predictions.parquet")
    scores = pd.read_csv(v3_directory / "scores.csv")
    condition_rows = []
    parameter_counts: dict[str, int] = {}
    for (outer, landmark), target_prefixes in prefixes.groupby(
        ["outer_condition_id", "landmark_visit_count"], sort=True
    ):
        landmark_int = int(landmark)
        reference = _core(
            references.loc[references["outer_condition_id"] == outer]
        )
        outer_decisions = decisions.loc[
            (decisions["outer_condition_id"] == outer)
            & (decisions["landmark_visit_count"] == landmark_int)
        ]
        hyperparameter_text = sorted(
            set(outer_decisions["dual_clock_hyperparameters_json"])
        )
        if len(hyperparameter_text) != 1:
            raise ValueError("Frozen outer V3 hyperparameters are not unique")
        hyperparameters = json.loads(hyperparameter_text[0])
        records = _inner_records(
            reference,
            landmark=landmark_int,
            hyperparameters=hyperparameters,
            config=config,
        )
        gate, gate_diagnostics = _select_gate(records)
        parameter_counts[_json_text(gate)] = parameter_counts.get(
            _json_text(gate), 0
        ) + 1
        cell_rows = []
        for cell_id, prefix_frame in target_prefixes.groupby("cell_id", sort=True):
            prefix = _core(prefix_frame).sort_values("visit_index", kind="stable")
            grid_v1, values_v1 = _curve(
                predictions,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark_int,
                model_id="v1_condition_ridge_delta",
            )
            grid_v3, values_v3 = _curve(
                predictions,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark_int,
                model_id="v3_dual_clock_kernel_shrinkage",
            )
            if not np.array_equal(grid_v1, grid_v3):
                raise ValueError("Frozen expert grids changed")
            decision = outer_decisions.loc[
                outer_decisions["cell_id"] == cell_id
            ]
            if len(decision) != 1:
                raise ValueError("Frozen target decision identity is not unique")
            signature = _signature(
                prefix,
                grid_v1,
                values_v1,
                values_v3,
                v3_prefix_residual=float(
                    decision.iloc[0]["prefix_residual_rms_pp"]
                ),
            )
            estimated_risk, nearest_distance = _estimate_risk(
                signature, records, gate
            )
            selected_expert = _select_expert(
                estimated_risk, float(gate["required_v3_margin_pp"])
            )
            v1_error = _score_value(
                scores,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark_int,
                model_id="v1_condition_ridge_delta",
            )
            v3_error = _score_value(
                scores,
                outer=str(outer),
                cell_id=str(cell_id),
                landmark=landmark_int,
                model_id="v3_dual_clock_kernel_shrinkage",
            )
            cell_rows.append(
                {
                    "selected_expert": selected_expert,
                    "selected_error": v1_error if selected_expert == "v1" else v3_error,
                    "v1_error": v1_error,
                    "v3_error": v3_error,
                    "nearest_inner_distance": nearest_distance,
                    "estimated_v1_risk": estimated_risk["v1"],
                    "estimated_v3_risk": estimated_risk["v3"],
                }
            )
        selected_error = float(np.mean([row["selected_error"] for row in cell_rows]))
        v1_error = float(np.mean([row["v1_error"] for row in cell_rows]))
        v3_error = float(np.mean([row["v3_error"] for row in cell_rows]))
        condition_rows.append(
            {
                "outer_condition_id": str(outer),
                "landmark_visit_count": landmark_int,
                "selected_condition_iae_pp": selected_error,
                "v1_condition_iae_pp": v1_error,
                "v3_condition_iae_pp": v3_error,
                "selected_improvement_vs_v1_pp": v1_error - selected_error,
                "selected_improvement_vs_v3_pp": v3_error - selected_error,
                "v3_choice_fraction": float(
                    np.mean([row["selected_expert"] == "v3" for row in cell_rows])
                ),
                "gate_parameters": gate,
                "gate_selection_diagnostics": gate_diagnostics,
            }
        )
    result_by_landmark = {}
    for landmark in sorted({int(row["landmark_visit_count"]) for row in condition_rows}):
        selected = [row for row in condition_rows if row["landmark_visit_count"] == landmark]
        selected_error = np.asarray(
            [row["selected_condition_iae_pp"] for row in selected], dtype=float
        )
        v1_error = np.asarray([row["v1_condition_iae_pp"] for row in selected])
        v3_error = np.asarray([row["v3_condition_iae_pp"] for row in selected])
        result_by_landmark[str(landmark)] = {
            "selected_condition_equal_iae_pp": float(np.mean(selected_error)),
            "v1_condition_equal_iae_pp": float(np.mean(v1_error)),
            "v3_condition_equal_iae_pp": float(np.mean(v3_error)),
            "selected_improvement_vs_v1_pp": float(np.mean(v1_error - selected_error)),
            "selected_improvement_vs_v3_pp": float(np.mean(v3_error - selected_error)),
            "improved_condition_fraction_vs_v1": float(
                np.mean(selected_error < v1_error)
            ),
            "improved_condition_fraction_vs_v3": float(
                np.mean(selected_error < v3_error)
            ),
            "worst_condition_regression_vs_v1_pp": float(
                max(0.0, float(np.max(selected_error - v1_error)))
            ),
            "worst_condition_regression_vs_v3_pp": float(
                max(0.0, float(np.max(selected_error - v3_error)))
            ),
            "v3_choice_fraction": float(
                np.mean([row["v3_choice_fraction"] for row in selected])
            ),
        }
    return {
        "schema_version": "lifetwin.private_dual_clock_selective_v31.exploration.v1",
        "private_only": True,
        "evidence_role": "outcome_exposed_nested_gate_exploration",
        "gate_selection_objective": (
            "reference-only cross-fitted condition-equal IAE plus 0.5 times "
            "worst condition regression versus V1"
        ),
        "result_by_landmark": result_by_landmark,
        "selected_gate_parameter_counts": parameter_counts,
        "condition_results": condition_rows,
        "claim_boundary": (
            "Retrospective private exploration on the outcome-exposed SNL cohort; "
            "not independent confirmation and not a production gate."
        ),
        "public_release_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-directory", default="artifacts/snl-lfp-rpt-loco-v1"
    )
    parser.add_argument(
        "--v3-directory", default="artifacts/private-dual-clock-prior-v3"
    )
    parser.add_argument(
        "--output",
        default=(
            "artifacts/private-dual-clock-prior-v3-post-outcome-audit/"
            "nested_risk_gate_exploration.json"
        ),
    )
    args = parser.parse_args()
    result = run_exploration(Path(args.input_directory), Path(args.v3_directory))
    atomic_write_json(result, Path(args.output))
    print(json.dumps(result["result_by_landmark"], indent=2))
    print(json.dumps(result["selected_gate_parameter_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
