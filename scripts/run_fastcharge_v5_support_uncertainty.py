from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

import lifetwin.experiments.fastcharge_v5_pairwise as v5
from lifetwin.experiments.fastcharge_safe_prior_v2 import (
    _core_config,
    load_fastcharge_safe_prior_v2_config,
)
from lifetwin.experiments.fastcharge_trajectory_portability import (
    _normalization_capacity,
    _retention,
)
from lifetwin.experiments.nasa_prefix_loco import (
    canonical_frame_sha256,
    canonical_json_sha256,
)


BASE = Path("artifacts/fastcharge-safe-prior-v2")
V5 = Path("artifacts/fastcharge-v5-pairwise-development")
OUTPUT = Path("artifacts/fastcharge-v5-support-uncertainty")
V2_CONFIG = Path("configs/experiments/fastcharge_lfp_safe_prior_v2.json")
PREFIXES = (20, 40, 60, 100)
SCORE_END = 300
COVERAGE = 0.9
SCALE_FLOOR_PP = 0.02
CROSSFIT_COLUMNS = (
    "fold_index",
    "cell_id",
    "prefix_cycle",
    "forecast_cycle",
    "observed_retention_pct",
    "candidate_prediction_pct",
    "fallback_prediction_pct",
    "reference_std_pp",
    "mean_reference_distance",
    "candidate_fallback_disagreement_pp",
)
PREDICTION_COLUMNS = (
    "evidence_role",
    "paper_split",
    "cell_id",
    "prefix_cycle",
    "forecast_cycle",
    "model_id",
    "predicted_capacity_retention_pct",
    "interval_lower_pct",
    "interval_upper_pct",
    "interval_half_width_pp",
    "reference_std_pp",
    "candidate_fallback_disagreement_pp",
    "mean_reference_distance",
    "gate_id",
    "gate_triggered",
    "operational_action",
)


class _ZeroRegressor:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _spec(record: Mapping[str, object]) -> v5.ModelSpec:
    return v5.ModelSpec(
        str(record["model_id"]),
        str(record["family"]),
        tuple(
            (str(key), value)
            for key, value in sorted(record["parameters"].items())
        ),
    )


def _truth(cell: pd.DataFrame, prefix: int) -> np.ndarray:
    normalization = _normalization_capacity(cell)
    return _retention(cell, normalization)[prefix:SCORE_END]


def _weighted_std(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    normalized = np.asarray(weights, dtype=float)
    normalized = normalized / float(np.sum(normalized))
    mean = np.sum(normalized[:, None] * values, axis=0)
    variance = np.sum(normalized[:, None] * np.square(values - mean), axis=0)
    return np.sqrt(np.maximum(variance, 0.0))


def _selected_predictions(
    estimator: object,
    target_prefix: pd.DataFrame,
    reference_cells: Mapping[str, pd.DataFrame],
    reference_resources: Mapping[str, object],
    prefix: int,
    core: Mapping[str, object],
    *,
    reference_count: int,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    candidate_matrix, candidate_weights, audit = (
        v5.pairwise_reference_trajectories(
            estimator,
            target_prefix,
            reference_cells,
            prefix,
            SCORE_END,
            core,
            neighbor_count=reference_count,
            reference_resources=reference_resources,
        )
    )
    candidate = v5.aggregate_reference_trajectories(
        candidate_matrix, candidate_weights, aggregation
    )
    fallback_matrix, fallback_weights, _ = v5.pairwise_reference_trajectories(
        _ZeroRegressor(),
        target_prefix,
        reference_cells,
        prefix,
        SCORE_END,
        core,
        neighbor_count=8,
        reference_resources=reference_resources,
    )
    fallback = v5.aggregate_reference_trajectories(
        fallback_matrix, fallback_weights, "weighted_mean"
    )
    reference_std = _weighted_std(candidate_matrix, candidate_weights)
    return candidate, fallback, reference_std, float(audit["mean_reference_distance"])


def _gate_diagnostics(crossfit: pd.DataFrame) -> pd.DataFrame:
    return crossfit.groupby(["cell_id", "prefix_cycle"], as_index=False).agg(
        mean_reference_distance=("mean_reference_distance", "first"),
        mean_reference_std_pp=("reference_std_pp", "mean"),
        mean_disagreement_pp=(
            "candidate_fallback_disagreement_pp",
            "mean",
        ),
    )


def _thresholds(diagnostics: pd.DataFrame, quantile: float) -> dict[str, float]:
    return {
        "mean_reference_distance": float(
            diagnostics["mean_reference_distance"].quantile(quantile)
        ),
        "mean_reference_std_pp": float(
            diagnostics["mean_reference_std_pp"].quantile(quantile)
        ),
        "mean_disagreement_pp": float(
            diagnostics["mean_disagreement_pp"].quantile(quantile)
        ),
    }


def _gate_mask(
    diagnostics: pd.DataFrame,
    gate_id: str,
    thresholds: Mapping[str, float],
) -> pd.Series:
    if gate_id == "no_gate":
        return pd.Series(False, index=diagnostics.index)
    feature = {
        "distance": "mean_reference_distance",
        "dispersion": "mean_reference_std_pp",
        "disagreement": "mean_disagreement_pp",
    }
    if gate_id.startswith("union_"):
        return (
            (diagnostics["mean_reference_distance"] > thresholds["mean_reference_distance"])
            | (
                diagnostics["mean_reference_std_pp"]
                > thresholds["mean_reference_std_pp"]
            )
            | (
                diagnostics["mean_disagreement_pp"]
                > thresholds["mean_disagreement_pp"]
            )
        )
    stem = gate_id.split("_", maxsplit=1)[0]
    column = feature[stem]
    return diagnostics[column] > thresholds[column]


def _gate_screen(
    crossfit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    diagnostics = _gate_diagnostics(crossfit)
    score_rows: list[dict[str, object]] = []
    for (cell_id, prefix), group in crossfit.groupby(
        ["cell_id", "prefix_cycle"], sort=True
    ):
        observed = group["observed_retention_pct"].to_numpy(dtype=float)
        candidate = group["candidate_prediction_pct"].to_numpy(dtype=float)
        fallback = group["fallback_prediction_pct"].to_numpy(dtype=float)
        score_rows.append(
            {
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix),
                "candidate_mae_pp": v5.trajectory_mae(observed, candidate),
                "fallback_mae_pp": v5.trajectory_mae(observed, fallback),
            }
        )
    score_base = pd.DataFrame(score_rows)
    diagnostics = diagnostics.merge(
        score_base,
        on=["cell_id", "prefix_cycle"],
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    gate_records: dict[str, dict[str, object]] = {
        "no_gate": {"quantile": None, "thresholds": {}}
    }
    for quantile in (0.9, 0.95, 0.99):
        values = _thresholds(diagnostics, quantile)
        suffix = str(quantile).replace("0.", "q")
        for stem in ("distance", "dispersion", "disagreement", "union"):
            gate_records[f"{stem}_{suffix}"] = {
                "quantile": quantile,
                "thresholds": values,
            }
    for gate_id, record in gate_records.items():
        mask = _gate_mask(diagnostics, gate_id, record["thresholds"])
        issued = np.where(
            mask,
            diagnostics["fallback_mae_pp"],
            diagnostics["candidate_mae_pp"],
        )
        rows.append(
            {
                "gate_id": gate_id,
                "fallback_fraction": float(np.mean(mask)),
                "mean_trajectory_mae_pp": float(np.mean(issued)),
                "p90_trajectory_mae_pp": float(np.quantile(issued, 0.9)),
                "maximum_trajectory_mae_pp": float(np.max(issued)),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["mean_trajectory_mae_pp", "p90_trajectory_mae_pp", "gate_id"],
        kind="stable",
        ignore_index=True,
    )
    best = scores.iloc[0]
    no_gate = scores.loc[scores["gate_id"] == "no_gate"].iloc[0]
    # Prefer no gate unless a diagnostic fallback removes at least 0.02 pp mean
    # MAE without increasing the cell-prefix P90 by more than 0.05 pp.
    if (
        float(no_gate["mean_trajectory_mae_pp"])
        <= float(best["mean_trajectory_mae_pp"]) + 0.02
        or float(best["p90_trajectory_mae_pp"])
        > float(no_gate["p90_trajectory_mae_pp"]) + 0.05
    ):
        selected_id = "no_gate"
    else:
        selected_id = str(best["gate_id"])
    return scores, {
        "selected_gate_id": selected_id,
        "selected_gate": gate_records[selected_id],
        "selection_rule": (
            "prefer_no_gate_within_0p02pp_mean_tie; otherwise require_p90_"
            "regression_not_above_0p05pp"
        ),
        "all_gate_records": gate_records,
    }


def _apply_crossfit_gate(
    crossfit: pd.DataFrame,
    gate: Mapping[str, object],
) -> pd.DataFrame:
    result = crossfit.copy()
    diagnostics = _gate_diagnostics(result)
    mask = _gate_mask(
        diagnostics,
        str(gate["selected_gate_id"]),
        gate["selected_gate"]["thresholds"],
    )
    decisions = diagnostics.loc[:, ["cell_id", "prefix_cycle"]].copy()
    decisions["gate_triggered"] = mask.to_numpy(dtype=bool)
    result = result.merge(
        decisions,
        on=["cell_id", "prefix_cycle"],
        validate="many_to_one",
    )
    result["gated_prediction_pct"] = np.where(
        result["gate_triggered"],
        result["fallback_prediction_pct"],
        result["candidate_prediction_pct"],
    )
    result["absolute_residual_pp"] = np.abs(
        result["observed_retention_pct"] - result["gated_prediction_pct"]
    )
    return result


def _scale(frame: pd.DataFrame, method_id: str) -> np.ndarray:
    reference = frame["reference_std_pp"].to_numpy(dtype=float)
    disagreement = frame["candidate_fallback_disagreement_pp"].to_numpy(
        dtype=float
    )
    if method_id == "absolute":
        return np.ones(len(frame), dtype=float)
    if method_id == "reference_scaled":
        return np.maximum(reference, SCALE_FLOOR_PP)
    if method_id == "hybrid_scaled":
        return np.maximum.reduce(
            [reference, disagreement, np.full(len(frame), SCALE_FLOOR_PP)]
        )
    raise v5.FastChargeV5PairwiseError(f"Unknown interval method: {method_id}")


def _cross_conformal_screen(
    gated: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = ("absolute", "reference_scaled", "hybrid_scaled")
    for method_id in methods:
        scored_parts: list[pd.DataFrame] = []
        for cell_id in sorted(gated["cell_id"].unique()):
            target = gated.loc[gated["cell_id"] == cell_id].copy()
            calibration = gated.loc[gated["cell_id"] != cell_id].copy()
            target["scale"] = _scale(target, method_id)
            calibration["scale"] = _scale(calibration, method_id)
            calibration["normalized_residual"] = (
                calibration["absolute_residual_pp"] / calibration["scale"]
            )
            quantile_rows: list[dict[str, object]] = []
            for (prefix, forecast), group in calibration.groupby(
                ["prefix_cycle", "forecast_cycle"], sort=True
            ):
                quantile, level = v5.finite_sample_absolute_quantile(
                    group["normalized_residual"].to_numpy(dtype=float),
                    coverage=COVERAGE,
                )
                quantile_rows.append(
                    {
                        "prefix_cycle": int(prefix),
                        "forecast_cycle": int(forecast),
                        "quantile": quantile,
                        "quantile_level": level,
                    }
                )
            target = target.merge(
                pd.DataFrame(quantile_rows),
                on=["prefix_cycle", "forecast_cycle"],
                validate="one_to_one",
            )
            target["half_width"] = target["quantile"] * target["scale"]
            target["lower"] = np.clip(
                target["gated_prediction_pct"] - target["half_width"], 0.0, 110.0
            )
            target["upper"] = np.clip(
                target["gated_prediction_pct"] + target["half_width"], 0.0, 110.0
            )
            scored_parts.append(target)
        scored = pd.concat(scored_parts, ignore_index=True)
        inside = (
            (scored["observed_retention_pct"] >= scored["lower"])
            & (scored["observed_retention_pct"] <= scored["upper"])
        )
        wis = v5.weighted_interval_score(
            scored["observed_retention_pct"].to_numpy(dtype=float),
            scored["gated_prediction_pct"].to_numpy(dtype=float),
            scored["lower"].to_numpy(dtype=float),
            scored["upper"].to_numpy(dtype=float),
            coverage=COVERAGE,
        )
        rows.append(
            {
                "method_id": method_id,
                "empirical_coverage": float(np.mean(inside)),
                "mean_interval_width_pp": float(
                    np.mean(scored["upper"] - scored["lower"])
                ),
                "mean_weighted_interval_score": float(np.mean(wis)),
                "minimum_prefix_coverage": float(
                    scored.assign(inside=inside)
                    .groupby("prefix_cycle")["inside"]
                    .mean()
                    .min()
                ),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["mean_weighted_interval_score", "mean_interval_width_pp", "method_id"],
        kind="stable",
        ignore_index=True,
    )
    eligible = scores.loc[
        (scores["empirical_coverage"] >= 0.87)
        & (scores["minimum_prefix_coverage"] >= 0.8)
    ]
    selected = eligible.iloc[0] if len(eligible) else scores.iloc[0]
    return scores, {
        "selected_method_id": str(selected["method_id"]),
        "nominal_coverage": COVERAGE,
        "minimum_overall_coverage": 0.87,
        "minimum_prefix_coverage": 0.8,
        "selection_metric": "minimum_cross_conformal_weighted_interval_score",
        "eligible_method_found": bool(len(eligible)),
    }


def _calibration_table(
    gated: pd.DataFrame,
    method_id: str,
) -> pd.DataFrame:
    calibration = gated.copy()
    calibration["scale"] = _scale(calibration, method_id)
    calibration["normalized_residual"] = (
        calibration["absolute_residual_pp"] / calibration["scale"]
    )
    rows: list[dict[str, object]] = []
    for (prefix, forecast), group in calibration.groupby(
        ["prefix_cycle", "forecast_cycle"], sort=True
    ):
        quantile, level = v5.finite_sample_absolute_quantile(
            group["normalized_residual"].to_numpy(dtype=float),
            coverage=COVERAGE,
        )
        rows.append(
            {
                "prefix_cycle": int(prefix),
                "forecast_cycle": int(forecast),
                "method_id": method_id,
                "calibration_cell_count": int(group["cell_id"].nunique()),
                "quantile_level": level,
                "normalized_residual_quantile": quantile,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["prefix_cycle", "forecast_cycle"], kind="stable", ignore_index=True
    )


def _crossfit(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    spec = _spec(selection["selected_model_spec"])
    reference_count = int(selection["selected_reference_count"])
    aggregation = str(selection["selected_aggregation"])
    core = _core_config(load_fastcharge_safe_prior_v2_config(args.v2_config))
    training = pd.read_parquet(args.training_cycles)
    cells = v5._validated_cells(training, required_support=SCORE_END)
    rows: list[dict[str, object]] = []
    folds = v5.deterministic_cell_folds(sorted(cells), fold_count=5)
    for fold_index, (fit_ids, held_out_ids) in enumerate(folds):
        fit_cells = {cell_id: cells[cell_id] for cell_id in fit_ids}
        for prefix in PREFIXES:
            matrix, target, audit = v5.build_pairwise_training_matrix(
                fit_cells, prefix, SCORE_END, core, anchor_stride=20
            )
            v5.assert_pair_fold_firewall(fit_ids, held_out_ids, audit)
            estimator = v5.make_estimator(spec, pairwise=True).fit(matrix, target)
            resources = v5._cell_resources(fit_cells, prefix, core)
            forecast = np.arange(prefix + 1, SCORE_END + 1, dtype=int)
            for cell_id in held_out_ids:
                target_prefix = cells[cell_id].loc[
                    cells[cell_id]["cycle_index"] <= prefix
                ]
                candidate, fallback, reference_std, distance = _selected_predictions(
                    estimator,
                    target_prefix,
                    fit_cells,
                    resources,
                    prefix,
                    core,
                    reference_count=reference_count,
                    aggregation=aggregation,
                )
                observed = _truth(cells[cell_id], prefix)
                for index, cycle in enumerate(forecast):
                    rows.append(
                        {
                            "fold_index": fold_index,
                            "cell_id": cell_id,
                            "prefix_cycle": prefix,
                            "forecast_cycle": int(cycle),
                            "observed_retention_pct": float(observed[index]),
                            "candidate_prediction_pct": float(candidate[index]),
                            "fallback_prediction_pct": float(fallback[index]),
                            "reference_std_pp": float(reference_std[index]),
                            "mean_reference_distance": distance,
                            "candidate_fallback_disagreement_pp": float(
                                abs(candidate[index] - fallback[index])
                            ),
                        }
                    )
        partial = pd.DataFrame(rows, columns=CROSSFIT_COLUMNS)
        _write_csv(partial, output / "crossfit_predictions.partial.csv")
        print(f"completed support/UQ fold {fold_index + 1}/{len(folds)}", flush=True)
    crossfit = pd.DataFrame(rows, columns=CROSSFIT_COLUMNS).sort_values(
        ["cell_id", "prefix_cycle", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    gate_scores, gate = _gate_screen(crossfit)
    gated = _apply_crossfit_gate(crossfit, gate)
    interval_scores, interval = _cross_conformal_screen(gated)
    calibration = _calibration_table(gated, interval["selected_method_id"])
    _write_csv(crossfit, output / "crossfit_predictions.csv")
    _write_csv(gate_scores, output / "support_gate_screen.csv")
    _write_csv(interval_scores, output / "interval_method_screen.csv")
    _write_csv(calibration, output / "calibration_quantiles.csv")
    result = {
        "schema_version": "lifetwin.fastcharge_v5_support_uncertainty.v1",
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "selection_semantic_sha256": canonical_json_sha256(selection),
        "training_cycle_sha256": canonical_frame_sha256(
            training, tuple(training.columns)
        ),
        "crossfit_prediction_sha256": canonical_frame_sha256(
            crossfit, CROSSFIT_COLUMNS
        ),
        "physical_cell_count": len(cells),
        "gate": gate,
        "interval": interval,
        "calibration_sha256": canonical_frame_sha256(
            calibration, tuple(calibration.columns)
        ),
        "evaluation_suffix_used_for_selection": False,
        "formal_cross_domain_coverage_claim": False,
    }
    _write_json(result, output / "development_result.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _target_gate(
    distance: float,
    reference_std: np.ndarray,
    disagreement: np.ndarray,
    gate: Mapping[str, object],
) -> bool:
    row = pd.DataFrame(
        [
            {
                "mean_reference_distance": distance,
                "mean_reference_std_pp": float(np.mean(reference_std)),
                "mean_disagreement_pp": float(np.mean(disagreement)),
            }
        ]
    )
    return bool(
        _gate_mask(
            row,
            str(gate["selected_gate_id"]),
            gate["selected_gate"]["thresholds"],
        ).item()
    )


def _prediction_scale(
    reference_std: np.ndarray,
    disagreement: np.ndarray,
    method_id: str,
) -> np.ndarray:
    frame = pd.DataFrame(
        {
            "reference_std_pp": reference_std,
            "candidate_fallback_disagreement_pp": disagreement,
        }
    )
    return _scale(frame, method_id)


def _predict(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    development = json.loads(Path(args.development_result).read_text(encoding="utf-8"))
    calibration = pd.read_csv(args.calibration)
    spec = _spec(selection["selected_model_spec"])
    reference_count = int(selection["selected_reference_count"])
    aggregation = str(selection["selected_aggregation"])
    method_id = str(development["interval"]["selected_method_id"])
    gate = development["gate"]
    core = _core_config(load_fastcharge_safe_prior_v2_config(args.v2_config))
    training = pd.read_parquet(args.training_cycles)
    prefixes = pd.read_parquet(args.target_prefixes)
    training_cells = v5._validated_cells(training, required_support=SCORE_END)
    rows: list[dict[str, object]] = []
    for prefix in PREFIXES:
        matrix, target, _ = v5.build_pairwise_training_matrix(
            training_cells, prefix, SCORE_END, core, anchor_stride=20
        )
        estimator = v5.make_estimator(spec, pairwise=True).fit(matrix, target)
        resources = v5._cell_resources(training_cells, prefix, core)
        quantiles = calibration.loc[calibration["prefix_cycle"] == prefix].sort_values(
            "forecast_cycle", kind="stable"
        )["normalized_residual_quantile"].to_numpy(dtype=float)
        subset = prefixes.loc[prefixes["prefix_cycle"] == prefix]
        for (paper_split, cell_id), target_prefix in subset.groupby(
            ["paper_split", "cell_id"], sort=True
        ):
            target_prefix = target_prefix.sort_values("cycle_index", kind="stable")
            candidate, fallback, reference_std, distance = _selected_predictions(
                estimator,
                target_prefix,
                training_cells,
                resources,
                prefix,
                core,
                reference_count=reference_count,
                aggregation=aggregation,
            )
            disagreement = np.abs(candidate - fallback)
            triggered = _target_gate(
                distance, reference_std, disagreement, gate
            )
            center = fallback if triggered else candidate
            scale = _prediction_scale(reference_std, disagreement, method_id)
            half_width = quantiles * scale
            lower = np.clip(center - half_width, 0.0, 110.0)
            upper = np.clip(center + half_width, 0.0, 110.0)
            action = "fallback_predict" if triggered else "candidate_predict"
            for index, cycle in enumerate(range(prefix + 1, SCORE_END + 1)):
                rows.append(
                    {
                        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
                        "paper_split": str(paper_split),
                        "cell_id": str(cell_id),
                        "prefix_cycle": prefix,
                        "forecast_cycle": cycle,
                        "model_id": "v5_pairwise_support_uncertainty",
                        "predicted_capacity_retention_pct": float(center[index]),
                        "interval_lower_pct": float(lower[index]),
                        "interval_upper_pct": float(upper[index]),
                        "interval_half_width_pp": float(half_width[index]),
                        "reference_std_pp": float(reference_std[index]),
                        "candidate_fallback_disagreement_pp": float(
                            disagreement[index]
                        ),
                        "mean_reference_distance": distance,
                        "gate_id": str(gate["selected_gate_id"]),
                        "gate_triggered": triggered,
                        "operational_action": action,
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    path = output / "predictions.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    manifest = {
        "schema_version": "lifetwin.fastcharge_v5_support_prediction.v1",
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "selection_semantic_sha256": canonical_json_sha256(selection),
        "development_semantic_sha256": canonical_json_sha256(development),
        "training_cycle_sha256": canonical_frame_sha256(
            training, tuple(training.columns)
        ),
        "target_prefix_sha256": canonical_frame_sha256(
            prefixes, tuple(prefixes.columns)
        ),
        "prediction_sha256": canonical_frame_sha256(
            predictions, PREDICTION_COLUMNS
        ),
        "prediction_row_count": len(predictions),
        "target_suffix_used": False,
    }
    _write_json(manifest, output / "prediction_manifest.json")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _interval_metrics(
    observed: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, float]:
    inside = (observed >= lower) & (observed <= upper)
    wis = v5.weighted_interval_score(
        observed, center, lower, upper, coverage=COVERAGE
    )
    return {
        "trajectory_mae_pp": float(np.mean(np.abs(observed - center))),
        "empirical_coverage": float(np.mean(inside)),
        "mean_interval_width_pp": float(np.mean(upper - lower)),
        "weighted_interval_score": float(np.mean(wis)),
    }


def _score(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    predictions = pd.read_parquet(args.predictions)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if canonical_frame_sha256(predictions, PREDICTION_COLUMNS) != manifest[
        "prediction_sha256"
    ]:
        raise v5.FastChargeV5PairwiseError("V5 support prediction hash mismatch")
    full = pd.read_parquet(args.canonical_cycles)
    cells = v5._validated_cells(full, required_support=SCORE_END)
    rows: list[dict[str, object]] = []
    for (split, cell_id, prefix), group in predictions.groupby(
        ["paper_split", "cell_id", "prefix_cycle"], sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        metrics = _interval_metrics(
            _truth(cells[str(cell_id)], int(prefix)),
            group["predicted_capacity_retention_pct"].to_numpy(dtype=float),
            group["interval_lower_pct"].to_numpy(dtype=float),
            group["interval_upper_pct"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "paper_split": str(split),
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix),
                "gate_triggered": bool(group["gate_triggered"].iloc[0]),
                **metrics,
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["paper_split", "cell_id", "prefix_cycle"],
        kind="stable",
        ignore_index=True,
    )
    v2 = pd.read_parquet(args.v2_predictions)
    v2 = v2.loc[v2["model_id"] == "safe_hard_local_risk_selector"]
    baseline_rows: list[dict[str, object]] = []
    for (split, cell_id, prefix), group in v2.groupby(
        ["paper_split", "cell_id", "prefix_cycle"], sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        baseline_rows.append(
            {
                "paper_split": str(split),
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix),
                **_interval_metrics(
                    _truth(cells[str(cell_id)], int(prefix)),
                    group["predicted_capacity_retention_pct"].to_numpy(dtype=float),
                    group["interval_lower_pct"].to_numpy(dtype=float),
                    group["interval_upper_pct"].to_numpy(dtype=float),
                ),
            }
        )
    baseline = pd.DataFrame(baseline_rows)
    _write_csv(scores, output / "scores.csv")
    _write_csv(baseline, output / "v2_safe_hard_interval_scores.csv")
    overall = {
        metric: float(scores[metric].mean())
        for metric in (
            "trajectory_mae_pp",
            "empirical_coverage",
            "mean_interval_width_pp",
            "weighted_interval_score",
        )
    }
    baseline_overall = {
        metric: float(baseline[metric].mean())
        for metric in (
            "trajectory_mae_pp",
            "empirical_coverage",
            "mean_interval_width_pp",
            "weighted_interval_score",
        )
    }
    minimum_prefix_coverage = float(
        scores.groupby("prefix_cycle")["empirical_coverage"].mean().min()
    )
    result = {
        "schema_version": "lifetwin.fastcharge_v5_support_score.v1",
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "prediction_sha256": manifest["prediction_sha256"],
        "score_sha256": canonical_frame_sha256(scores, tuple(scores.columns)),
        "physical_cell_count": int(scores["cell_id"].nunique()),
        "fallback_cell_prefix_fraction": float(scores["gate_triggered"].mean()),
        "overall": overall,
        "v2_safe_hard": baseline_overall,
        "minimum_prefix_coverage": minimum_prefix_coverage,
        "wis_ratio_vs_v2": (
            overall["weighted_interval_score"]
            / baseline_overall["weighted_interval_score"]
        ),
        "width_ratio_vs_v2": (
            overall["mean_interval_width_pp"]
            / baseline_overall["mean_interval_width_pp"]
        ),
        "h2_development_gate": {
            "minimum_overall_coverage": 0.87,
            "minimum_each_prefix_coverage": 0.8,
            "maximum_wis_ratio_vs_v2": 1.0,
            "maximum_width_ratio_vs_v2": 1.1,
            "interval_subgate_passed": bool(
                overall["empirical_coverage"] >= 0.87
                and minimum_prefix_coverage >= 0.8
                and overall["weighted_interval_score"]
                <= baseline_overall["weighted_interval_score"]
                and overall["mean_interval_width_pp"]
                <= 1.1 * baseline_overall["mean_interval_width_pp"]
            ),
            "online_landmark_update_fraction_requirement_evaluated": False,
            "passed": False,
            "status": "incomplete_online_landmark_requirement_not_evaluated",
        },
        "formal_cross_domain_coverage_claim": False,
    }
    _write_json(result, output / "score_summary.json")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common_selection = str(V5 / "training_cell_cv_selection.json")

    crossfit = sub.add_parser("crossfit")
    crossfit.add_argument("--training-cycles", default=str(BASE / "training_cycles.parquet"))
    crossfit.add_argument("--selection", default=common_selection)
    crossfit.add_argument("--v2-config", default=str(V2_CONFIG))
    crossfit.add_argument("--output-directory", default=str(OUTPUT))
    crossfit.set_defaults(handler=_crossfit)

    predict = sub.add_parser("predict")
    predict.add_argument("--training-cycles", default=str(BASE / "training_cycles.parquet"))
    predict.add_argument("--target-prefixes", default=str(BASE / "target_prefixes.parquet"))
    predict.add_argument("--selection", default=common_selection)
    predict.add_argument("--development-result", default=str(OUTPUT / "development_result.json"))
    predict.add_argument("--calibration", default=str(OUTPUT / "calibration_quantiles.csv"))
    predict.add_argument("--v2-config", default=str(V2_CONFIG))
    predict.add_argument("--output-directory", default=str(OUTPUT))
    predict.set_defaults(handler=_predict)

    score = sub.add_parser("score")
    score.add_argument("--predictions", default=str(OUTPUT / "predictions.parquet"))
    score.add_argument("--manifest", default=str(OUTPUT / "prediction_manifest.json"))
    score.add_argument("--canonical-cycles", default=str(BASE / "canonical_cycles.parquet"))
    score.add_argument("--v2-predictions", default=str(BASE / "predictions.parquet"))
    score.add_argument("--output-directory", default=str(OUTPUT))
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
