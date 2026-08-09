from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

# Keep sklearn's hidden OpenMP work deterministic in restricted Windows runs.
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


DEFAULT_BASE_DIRECTORY = Path("artifacts/fastcharge-safe-prior-v2")
DEFAULT_OUTPUT_DIRECTORY = Path("artifacts/fastcharge-v5-pairwise-development")
DEFAULT_V2_CONFIG = Path("configs/experiments/fastcharge_lfp_safe_prior_v2.json")
PREFIX_CYCLES = (20, 40, 60, 100)
SCORE_END_CYCLE = 300
SCREEN_SCHEMA = "lifetwin.fastcharge_v5_pairwise_screen.v1"
PREDICTION_COLUMNS = (
    "evidence_role",
    "paper_split",
    "cell_id",
    "prefix_cycle",
    "forecast_cycle",
    "model_id",
    "predicted_capacity_retention_pct",
    "reference_cell_ids_json",
    "reference_weights_json",
    "mean_reference_distance",
    "reference_dispersion_mean_pp",
)


PAIR_SCREEN_SPECS = (
    v5.ModelSpec("pairwise_ridge_a10", "ridge", (("alpha", 10.0),)),
    v5.ModelSpec(
        "pairwise_huber_fast",
        "huber",
        (("alpha", 0.0001), ("epsilon", 1.35), ("max_iter", 300)),
    ),
    v5.ModelSpec(
        "pairwise_extra_trees_leaf3_48",
        "extra_trees",
        (("min_samples_leaf", 3), ("n_estimators", 48)),
    ),
    v5.ModelSpec(
        "pairwise_extra_trees_leaf8_48",
        "extra_trees",
        (("min_samples_leaf", 8), ("n_estimators", 48)),
    ),
    v5.ModelSpec(
        "pairwise_extra_trees_leaf16_48",
        "extra_trees",
        (("min_samples_leaf", 16), ("n_estimators", 48)),
    ),
    v5.ModelSpec(
        "pairwise_hist_gbdt_leaf20",
        "hist_gbdt",
        (("min_samples_leaf", 20), ("max_iter", 160)),
    ),
    v5.ModelSpec(
        "pairwise_hist_gbdt_leaf40",
        "hist_gbdt",
        (("min_samples_leaf", 40), ("max_iter", 160)),
    ),
    v5.ModelSpec(
        "pairwise_hist_gbdt_leaf80",
        "hist_gbdt",
        (("min_samples_leaf", 80), ("max_iter", 160)),
    ),
)
REFERENCE_COUNTS = (4, 8, 12, 16)


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


def _spec_record(spec: v5.ModelSpec) -> dict[str, object]:
    return {
        "model_id": spec.model_id,
        "family": spec.family,
        "parameters": spec.parameter_dict(),
    }


def _spec_from_record(record: Mapping[str, object]) -> v5.ModelSpec:
    parameters = tuple(
        (str(key), value) for key, value in sorted(record["parameters"].items())
    )
    return v5.ModelSpec(str(record["model_id"]), str(record["family"]), parameters)


def _truth(cell: pd.DataFrame, prefix_cycle: int) -> np.ndarray:
    normalization = _normalization_capacity(cell)
    return _retention(cell, normalization)[prefix_cycle:SCORE_END_CYCLE]


def _candidate_id(
    model_id: str,
    *,
    reference_count: int | None = None,
    aggregation: str | None = None,
) -> str:
    if reference_count is None:
        return model_id
    return f"{model_id}__k{reference_count}__{aggregation}"


def _score_reference_matrix(
    rows: list[dict[str, object]],
    *,
    fold_index: int,
    prefix_cycle: int,
    cell_id: str,
    model_id: str,
    family: str,
    truth: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
) -> None:
    for count in REFERENCE_COUNTS:
        selected_matrix = matrix[:count]
        selected_weights = weights[:count]
        selected_weights = selected_weights / float(np.sum(selected_weights))
        for aggregation in v5.PAIRWISE_AGGREGATIONS:
            prediction = v5.aggregate_reference_trajectories(
                selected_matrix, selected_weights, aggregation
            )
            rows.append(
                {
                    "fold_scheme": "deterministic_batch_stratified_cell_5fold",
                    "fold_index": fold_index,
                    "prefix_cycle": prefix_cycle,
                    "cell_id": cell_id,
                    "candidate_id": _candidate_id(
                        model_id,
                        reference_count=count,
                        aggregation=aggregation,
                    ),
                    "model_id": model_id,
                    "family": family,
                    "reference_count": count,
                    "aggregation": aggregation,
                    "trajectory_mae_pp": v5.trajectory_mae(truth, prediction),
                }
            )


def _screen(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    score_path = output / "training_cell_cv_scores.csv"
    selection_path = output / "training_cell_cv_selection.json"
    firewall_path = output / "training_pair_firewall_audit.json"
    for path in (score_path, selection_path, firewall_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")

    config = load_fastcharge_safe_prior_v2_config(args.v2_config)
    core = _core_config(config)
    training = pd.read_parquet(args.training_cycles)
    cells = v5._validated_cells(training, required_support=SCORE_END_CYCLE)
    folds = v5.deterministic_cell_folds(sorted(cells), fold_count=5)
    rows: list[dict[str, object]] = []
    firewall_rows: list[dict[str, object]] = []
    for fold_index, (fit_ids, validation_ids) in enumerate(folds):
        fit_cells = {cell_id: cells[cell_id] for cell_id in fit_ids}
        for prefix_cycle in PREFIX_CYCLES:
            pair_x, pair_y, pair_audit = v5.build_pairwise_training_matrix(
                fit_cells,
                prefix_cycle,
                SCORE_END_CYCLE,
                core,
                anchor_stride=20,
            )
            v5.assert_pair_fold_firewall(fit_ids, validation_ids, pair_audit)
            firewall_rows.append(
                {
                    "fold_index": fold_index,
                    "prefix_cycle": prefix_cycle,
                    "fit_cell_count": len(fit_ids),
                    "validation_cell_count": len(validation_ids),
                    "fit_cell_ids": fit_ids,
                    "validation_cell_ids": validation_ids,
                    "pair_training_row_count": pair_audit["training_row_count"],
                    "pair_feature_count": pair_audit["feature_count"],
                    "held_out_cell_in_target_role": False,
                    "held_out_cell_in_reference_role": False,
                }
            )
            pair_models = {
                spec.model_id: v5.make_estimator(spec, pairwise=True).fit(
                    pair_x, pair_y
                )
                for spec in PAIR_SCREEN_SPECS
            }
            direct_x, direct_y, _ = v5.build_direct_training_matrix(
                fit_cells,
                prefix_cycle,
                SCORE_END_CYCLE,
                core,
                anchor_stride=10,
            )
            direct_models = {
                spec.model_id: v5.make_estimator(spec, pairwise=False).fit(
                    direct_x, direct_y
                )
                for spec in v5.DIRECT_MODEL_SPECS
            }
            reference_resources = v5._cell_resources(
                fit_cells, prefix_cycle, core
            )
            for cell_id in validation_ids:
                target_prefix = cells[cell_id].loc[
                    cells[cell_id]["cycle_index"] <= prefix_cycle
                ]
                observed = _truth(cells[cell_id], prefix_cycle)
                baseline_matrix, baseline_weights, _ = (
                    v5.pairwise_reference_trajectories(
                        _ZeroRegressor(),
                        target_prefix,
                        fit_cells,
                        prefix_cycle,
                        SCORE_END_CYCLE,
                        core,
                        neighbor_count=max(REFERENCE_COUNTS),
                        reference_resources=reference_resources,
                    )
                )
                _score_reference_matrix(
                    rows,
                    fold_index=fold_index,
                    prefix_cycle=prefix_cycle,
                    cell_id=cell_id,
                    model_id="fixed_neighbor_delta",
                    family="nonlearned_reference_baseline",
                    truth=observed,
                    matrix=baseline_matrix,
                    weights=baseline_weights,
                )
                for spec in PAIR_SCREEN_SPECS:
                    matrix, weights, _ = v5.pairwise_reference_trajectories(
                        pair_models[spec.model_id],
                        target_prefix,
                        fit_cells,
                        prefix_cycle,
                        SCORE_END_CYCLE,
                        core,
                        neighbor_count=max(REFERENCE_COUNTS),
                        reference_resources=reference_resources,
                    )
                    _score_reference_matrix(
                        rows,
                        fold_index=fold_index,
                        prefix_cycle=prefix_cycle,
                        cell_id=cell_id,
                        model_id=spec.model_id,
                        family=spec.family,
                        truth=observed,
                        matrix=matrix,
                        weights=weights,
                    )
                for spec in v5.DIRECT_MODEL_SPECS:
                    prediction = v5.predict_direct_trajectory(
                        direct_models[spec.model_id],
                        target_prefix,
                        prefix_cycle,
                        SCORE_END_CYCLE,
                        core,
                    )
                    rows.append(
                        {
                            "fold_scheme": (
                                "deterministic_batch_stratified_cell_5fold"
                            ),
                            "fold_index": fold_index,
                            "prefix_cycle": prefix_cycle,
                            "cell_id": cell_id,
                            "candidate_id": spec.model_id,
                            "model_id": spec.model_id,
                            "family": spec.family,
                            "reference_count": 0,
                            "aggregation": "not_applicable",
                            "trajectory_mae_pp": v5.trajectory_mae(
                                observed, prediction
                            ),
                        }
                    )
        partial_scores = pd.DataFrame(rows).sort_values(
            ["candidate_id", "cell_id", "prefix_cycle"],
            kind="stable",
            ignore_index=True,
        )
        _write_csv(partial_scores, output / "training_cell_cv_scores.partial.csv")
        _write_json(
            {
                "schema_version": (
                    "lifetwin.fastcharge_v5_pair_firewall_audit.partial.v1"
                ),
                "completed_fold_count": fold_index + 1,
                "folds": firewall_rows,
            },
            output / "training_pair_firewall_audit.partial.json",
        )
        print(
            f"completed training-only fold {fold_index + 1}/{len(folds)}",
            flush=True,
        )

    scores = pd.DataFrame(rows).sort_values(
        ["candidate_id", "cell_id", "prefix_cycle"],
        kind="stable",
        ignore_index=True,
    )
    _write_csv(scores, score_path)
    summary = (
        scores.groupby(
            [
                "candidate_id",
                "model_id",
                "family",
                "reference_count",
                "aggregation",
            ],
            as_index=False,
        )
        .agg(
            mean_trajectory_mae_pp=("trajectory_mae_pp", "mean"),
            median_trajectory_mae_pp=("trajectory_mae_pp", "median"),
            maximum_cell_prefix_mae_pp=("trajectory_mae_pp", "max"),
        )
        .sort_values(
            ["mean_trajectory_mae_pp", "candidate_id"],
            kind="stable",
            ignore_index=True,
        )
    )
    learned = summary.loc[summary["model_id"] != "fixed_neighbor_delta"]
    selected = learned.iloc[0].to_dict()
    baseline_id = _candidate_id(
        "fixed_neighbor_delta",
        reference_count=8,
        aggregation="weighted_mean",
    )
    baseline_mae = float(
        summary.loc[
            summary["candidate_id"] == baseline_id, "mean_trajectory_mae_pp"
        ].item()
    )
    candidate_id = str(selected["candidate_id"])
    candidate_scores = scores.loc[scores["candidate_id"] == candidate_id]
    baseline_scores = scores.loc[scores["candidate_id"] == baseline_id]
    paired = candidate_scores.merge(
        baseline_scores,
        on=["cell_id", "prefix_cycle", "fold_index", "fold_scheme"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    cell_deltas = (
        paired.assign(
            delta=lambda frame: frame["trajectory_mae_pp_candidate"]
            - frame["trajectory_mae_pp_baseline"]
        )
        .groupby("cell_id")["delta"]
        .mean()
        .to_dict()
    )
    bootstrap = v5.paired_cell_bootstrap(cell_deltas)
    pair_spec = next(
        (
            spec
            for spec in PAIR_SCREEN_SPECS
            if spec.model_id == selected["model_id"]
        ),
        None,
    )
    direct_spec = next(
        (
            spec
            for spec in v5.DIRECT_MODEL_SPECS
            if spec.model_id == selected["model_id"]
        ),
        None,
    )
    selected_spec = pair_spec or direct_spec
    if selected_spec is None:
        raise v5.FastChargeV5PairwiseError("Selected model spec is unavailable")
    relative_improvement = (
        baseline_mae - float(selected["mean_trajectory_mae_pp"])
    ) / baseline_mae
    selection = {
        "schema_version": SCREEN_SCHEMA,
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "selection_data": "41_training_cells_only",
        "selection_protocol": (
            "deterministic_batch_stratified_physical_cell_5fold; held-out cells "
            "excluded from target and reference roles"
        ),
        "training_cycle_sha256": canonical_frame_sha256(
            training,
            tuple(training.columns),
        ),
        "score_table_sha256": canonical_frame_sha256(
            scores, tuple(scores.columns)
        ),
        "screened_pair_specs": [_spec_record(spec) for spec in PAIR_SCREEN_SPECS],
        "screened_direct_specs": [
            _spec_record(spec) for spec in v5.DIRECT_MODEL_SPECS
        ],
        "screened_reference_counts": list(REFERENCE_COUNTS),
        "screened_aggregations": list(v5.PAIRWISE_AGGREGATIONS),
        "selected_candidate_id": candidate_id,
        "selected_model_spec": _spec_record(selected_spec),
        "selected_reference_count": int(selected["reference_count"]),
        "selected_aggregation": str(selected["aggregation"]),
        "selected_mean_trajectory_mae_pp": float(
            selected["mean_trajectory_mae_pp"]
        ),
        "fixed_neighbor_k8_weighted_mean_mae_pp": baseline_mae,
        "relative_improvement_vs_fixed_neighbor": relative_improvement,
        "paired_cell_bootstrap": bootstrap,
        "training_cv_gate": {
            "minimum_relative_improvement": 0.05,
            "bootstrap_upper_delta_must_be_below_pp": 0.0,
            "passed": bool(
                relative_improvement >= 0.05
                and float(bootstrap["upper_delta_mae_pp"]) < 0.0
            ),
        },
        "selection_is_independent_confirmation": False,
        "evaluation_suffix_used_for_selection": False,
    }
    _write_json(selection, selection_path)
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v5_pair_firewall_audit.v1",
            "folds": firewall_rows,
        },
        firewall_path,
    )
    _write_csv(summary, output / "training_cell_cv_summary.csv")
    print(json.dumps(selection, indent=2, ensure_ascii=False))
    return 0


def _predict(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    prediction_path = output / "selected_predictions.parquet"
    manifest_path = output / "selected_prediction_manifest.json"
    for path in (prediction_path, manifest_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    if selection["schema_version"] != SCREEN_SCHEMA:
        raise v5.FastChargeV5PairwiseError("V5 selection schema changed")
    config = load_fastcharge_safe_prior_v2_config(args.v2_config)
    core = _core_config(config)
    training = pd.read_parquet(args.training_cycles)
    prefixes = pd.read_parquet(args.target_prefixes)
    training_cells = v5._validated_cells(training, required_support=SCORE_END_CYCLE)
    spec = _spec_from_record(selection["selected_model_spec"])
    reference_count = int(selection["selected_reference_count"])
    aggregation = str(selection["selected_aggregation"])
    rows: list[dict[str, object]] = []
    for prefix_cycle in PREFIX_CYCLES:
        if spec.model_id.startswith("pairwise_"):
            matrix, target, _ = v5.build_pairwise_training_matrix(
                training_cells,
                prefix_cycle,
                SCORE_END_CYCLE,
                core,
                anchor_stride=20,
            )
            estimator = v5.make_estimator(spec, pairwise=True).fit(matrix, target)
            reference_resources = v5._cell_resources(
                training_cells, prefix_cycle, core
            )
        else:
            matrix, target, _ = v5.build_direct_training_matrix(
                training_cells,
                prefix_cycle,
                SCORE_END_CYCLE,
                core,
                anchor_stride=10,
            )
            estimator = v5.make_estimator(spec, pairwise=False).fit(matrix, target)
            reference_resources = None
        subset = prefixes.loc[prefixes["prefix_cycle"] == prefix_cycle]
        for (paper_split, cell_id), target_prefix in subset.groupby(
            ["paper_split", "cell_id"], sort=True
        ):
            target_prefix = target_prefix.sort_values("cycle_index", kind="stable")
            if int(target_prefix["cycle_index"].max()) != prefix_cycle:
                raise v5.FastChargeV5PairwiseError("Target prefix support changed")
            if spec.model_id.startswith("pairwise_"):
                prediction, audit = v5.predict_pairwise_trajectory(
                    estimator,
                    target_prefix,
                    training_cells,
                    prefix_cycle,
                    SCORE_END_CYCLE,
                    core,
                    aggregation=aggregation,
                    neighbor_count=reference_count,
                    reference_resources=reference_resources,
                )
            else:
                prediction = v5.predict_direct_trajectory(
                    estimator,
                    target_prefix,
                    prefix_cycle,
                    SCORE_END_CYCLE,
                    core,
                )
                audit = {
                    "reference_cell_ids": [],
                    "reference_weights": {},
                    "mean_reference_distance": 0.0,
                    "reference_dispersion_mean_pp": 0.0,
                }
            for forecast_cycle, value in zip(
                range(prefix_cycle + 1, SCORE_END_CYCLE + 1),
                prediction,
                strict=True,
            ):
                rows.append(
                    {
                        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
                        "paper_split": str(paper_split),
                        "cell_id": str(cell_id),
                        "prefix_cycle": prefix_cycle,
                        "forecast_cycle": forecast_cycle,
                        "model_id": selection["selected_candidate_id"],
                        "predicted_capacity_retention_pct": float(value),
                        "reference_cell_ids_json": json.dumps(
                            audit["reference_cell_ids"], separators=(",", ":")
                        ),
                        "reference_weights_json": json.dumps(
                            audit["reference_weights"],
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "mean_reference_distance": float(
                            audit["mean_reference_distance"]
                        ),
                        "reference_dispersion_mean_pp": float(
                            audit["reference_dispersion_mean_pp"]
                        ),
                    }
                )
    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS).sort_values(
        ["paper_split", "cell_id", "prefix_cycle", "forecast_cycle"],
        kind="stable",
        ignore_index=True,
    )
    predictions.to_parquet(prediction_path, index=False)
    manifest = {
        "schema_version": "lifetwin.fastcharge_v5_selected_prediction.v1",
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "selection_semantic_sha256": canonical_json_sha256(selection),
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
        "training_cell_count": int(training["cell_id"].nunique()),
        "target_cell_count": int(prefixes["cell_id"].nunique()),
        "target_suffix_used": False,
        "evaluation_outcomes_used_for_model_selection": False,
        "complete_training_histories_used": True,
        "selected_candidate_id": selection["selected_candidate_id"],
    }
    _write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


def _score(args: argparse.Namespace) -> int:
    output = Path(args.output_directory)
    score_path = output / "selected_scores.csv"
    summary_path = output / "selected_evaluation_summary.json"
    for path in (score_path, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}")
    predictions = pd.read_parquet(args.predictions)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if canonical_frame_sha256(predictions, PREDICTION_COLUMNS) != manifest[
        "prediction_sha256"
    ]:
        raise v5.FastChargeV5PairwiseError("V5 prediction hash mismatch")
    full = pd.read_parquet(args.canonical_cycles)
    cells = v5._validated_cells(full, required_support=SCORE_END_CYCLE)
    rows: list[dict[str, object]] = []
    for (paper_split, cell_id, prefix_cycle, model_id), group in predictions.groupby(
        ["paper_split", "cell_id", "prefix_cycle", "model_id"], sort=True
    ):
        group = group.sort_values("forecast_cycle", kind="stable")
        observed = _truth(cells[str(cell_id)], int(prefix_cycle))
        predicted = group["predicted_capacity_retention_pct"].to_numpy(dtype=float)
        rows.append(
            {
                "paper_split": str(paper_split),
                "cell_id": str(cell_id),
                "prefix_cycle": int(prefix_cycle),
                "model_id": str(model_id),
                "trajectory_mae_pp": v5.trajectory_mae(observed, predicted),
                "trajectory_rmse_pp": float(
                    np.sqrt(np.mean(np.square(observed - predicted)))
                ),
                "endpoint_absolute_error_pp": float(
                    abs(observed[-1] - predicted[-1])
                ),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["paper_split", "cell_id", "prefix_cycle"],
        kind="stable",
        ignore_index=True,
    )
    _write_csv(scores, score_path)
    candidate_id = str(scores["model_id"].iloc[0])
    v2_scores = pd.read_csv(args.v2_scores)
    comparators = v2_scores.loc[
        v2_scores["model_id"].isin(
            ["safe_hard_local_risk_selector", "nearest_neighbor_delta_transfer"]
        ),
        ["paper_split", "cell_id", "prefix_cycle", "model_id", "trajectory_mae_pp"],
    ]
    comparison_records: dict[str, object] = {}
    for comparator_id in (
        "safe_hard_local_risk_selector",
        "nearest_neighbor_delta_transfer",
    ):
        comparator = comparators.loc[
            comparators["model_id"] == comparator_id
        ].drop(columns="model_id")
        paired = scores.merge(
            comparator,
            on=["paper_split", "cell_id", "prefix_cycle"],
            suffixes=("_candidate", "_comparator"),
            validate="one_to_one",
        )
        paired["delta_mae_pp"] = (
            paired["trajectory_mae_pp_candidate"]
            - paired["trajectory_mae_pp_comparator"]
        )
        cell_deltas = paired.groupby("cell_id")["delta_mae_pp"].mean().to_dict()
        comparator_mean = float(paired["trajectory_mae_pp_comparator"].mean())
        candidate_mean = float(paired["trajectory_mae_pp_candidate"].mean())
        split_deltas = {
            str(split): float(group["delta_mae_pp"].mean())
            for split, group in paired.groupby("paper_split", sort=True)
        }
        comparison_records[comparator_id] = {
            "candidate_mean_mae_pp": candidate_mean,
            "comparator_mean_mae_pp": comparator_mean,
            "delta_mae_pp": candidate_mean - comparator_mean,
            "relative_improvement": (
                comparator_mean - candidate_mean
            ) / comparator_mean,
            "paired_cell_bootstrap": v5.paired_cell_bootstrap(cell_deltas),
            "split_mean_delta_mae_pp": split_deltas,
            "candidate_cell_prefix_p90_mae_pp": float(
                np.quantile(paired["trajectory_mae_pp_candidate"], 0.9)
            ),
            "comparator_cell_prefix_p90_mae_pp": float(
                np.quantile(paired["trajectory_mae_pp_comparator"], 0.9)
            ),
        }
    hard = comparison_records["safe_hard_local_risk_selector"]
    summary = {
        "schema_version": "lifetwin.fastcharge_v5_selected_score.v1",
        "evidence_role": v5.DEVELOPMENT_EVIDENCE_ROLE,
        "selected_candidate_id": candidate_id,
        "prediction_sha256": manifest["prediction_sha256"],
        "score_sha256": canonical_frame_sha256(scores, tuple(scores.columns)),
        "physical_cell_count": int(scores["cell_id"].nunique()),
        "cell_prefix_count": len(scores),
        "overall": {
            "trajectory_mae_pp": float(scores["trajectory_mae_pp"].mean()),
            "trajectory_rmse_pp": float(scores["trajectory_rmse_pp"].mean()),
            "endpoint_absolute_error_pp": float(
                scores["endpoint_absolute_error_pp"].mean()
            ),
            "cell_prefix_p90_mae_pp": float(
                np.quantile(scores["trajectory_mae_pp"], 0.9)
            ),
        },
        "by_split": [
            {
                "paper_split": str(split),
                "trajectory_mae_pp": float(group["trajectory_mae_pp"].mean()),
                "cell_count": int(group["cell_id"].nunique()),
            }
            for split, group in scores.groupby("paper_split", sort=True)
        ],
        "by_prefix": [
            {
                "prefix_cycle": int(prefix),
                "trajectory_mae_pp": float(group["trajectory_mae_pp"].mean()),
            }
            for prefix, group in scores.groupby("prefix_cycle", sort=True)
        ],
        "comparisons": comparison_records,
        "h1_development_gate": {
            "minimum_relative_improvement": 0.05,
            "bootstrap_upper_delta_must_be_below_pp": 0.0,
            "maximum_each_author_split_regression_pp": 0.03,
            "maximum_cell_prefix_p90_regression_pp": 0.05,
            "passed": bool(
                float(hard["relative_improvement"]) >= 0.05
                and float(
                    hard["paired_cell_bootstrap"]["upper_delta_mae_pp"]
                )
                < 0.0
                and max(hard["split_mean_delta_mae_pp"].values()) <= 0.03
                and (
                    float(hard["candidate_cell_prefix_p90_mae_pp"])
                    - float(hard["comparator_cell_prefix_p90_mae_pp"])
                )
                <= 0.05
            ),
        },
        "claim_boundaries": {
            "independent_confirmation": False,
            "calendar_aging_validation": False,
            "hithium_product_accuracy": False,
            "fifteen_to_twenty_five_year_accuracy": False,
        },
    }
    _write_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the outcome-exposed FastCharge V5 pairwise development"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    screen = subparsers.add_parser("screen")
    screen.add_argument(
        "--training-cycles",
        default=str(DEFAULT_BASE_DIRECTORY / "training_cycles.parquet"),
    )
    screen.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    screen.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    screen.add_argument("--overwrite", action="store_true")
    screen.set_defaults(handler=_screen)

    predict = subparsers.add_parser("predict")
    predict.add_argument(
        "--training-cycles",
        default=str(DEFAULT_BASE_DIRECTORY / "training_cycles.parquet"),
    )
    predict.add_argument(
        "--target-prefixes",
        default=str(DEFAULT_BASE_DIRECTORY / "target_prefixes.parquet"),
    )
    predict.add_argument(
        "--selection",
        default=str(DEFAULT_OUTPUT_DIRECTORY / "training_cell_cv_selection.json"),
    )
    predict.add_argument("--v2-config", default=str(DEFAULT_V2_CONFIG))
    predict.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)

    score = subparsers.add_parser("score")
    score.add_argument(
        "--canonical-cycles",
        default=str(DEFAULT_BASE_DIRECTORY / "canonical_cycles.parquet"),
    )
    score.add_argument(
        "--predictions",
        default=str(DEFAULT_OUTPUT_DIRECTORY / "selected_predictions.parquet"),
    )
    score.add_argument(
        "--manifest",
        default=str(DEFAULT_OUTPUT_DIRECTORY / "selected_prediction_manifest.json"),
    )
    score.add_argument(
        "--v2-scores",
        default=str(DEFAULT_BASE_DIRECTORY / "scores.csv"),
    )
    score.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    score.add_argument("--overwrite", action="store_true")
    score.set_defaults(handler=_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
