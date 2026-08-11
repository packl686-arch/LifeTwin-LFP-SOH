"""Evaluate fixed P100 Delta-Q challengers on the 41 MATR training cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Mapping, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd

import lifetwin.experiments.fastcharge_v5_pairwise as v5
import lifetwin.experiments.fastcharge_v11_curve_pairwise as v11
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/v11_delta_q_pairwise_development.json"
DEFAULT_TRAINING = ROOT / "artifacts/fastcharge-safe-prior-v2/training_cycles.parquet"
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v11-delta-q-pairwise"
BASELINE_ID = "v5_pairwise_extra_trees_leaf3_k12_weighted_mean"
SCORE_COLUMNS = (
    "fold_scheme",
    "fold_index",
    "held_out_batch_id",
    "cell_id",
    "candidate_id",
    "trajectory_mae_pp",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _model_spec(selection: Mapping[str, object]) -> v5.ModelSpec:
    record = selection["selected_model_spec"]
    return v5.ModelSpec(
        str(record["model_id"]),
        str(record["family"]),
        tuple((str(key), value) for key, value in sorted(record["parameters"].items())),
    )


def _truth(cell: pd.DataFrame) -> np.ndarray:
    normalization = _normalization_capacity(cell)
    return _retention(cell, normalization)[v11.PREFIX_CYCLE : v11.SCORE_END_CYCLE]


def _batch_id(cell_id: str) -> str:
    match = re.match(r"MATR_B(\d+)C", cell_id)
    if match is None:
        raise v5.FastChargeV5PairwiseError(
            f"Cannot derive MATR manufacturing batch from {cell_id}"
        )
    return f"MATR_BATCH_{match.group(1)}"


def _evaluate_folds(
    *,
    scheme: str,
    folds: list[tuple[list[str], list[str]]],
    cells: Mapping[str, pd.DataFrame],
    curve_features: pd.DataFrame,
    core: Mapping[str, object],
    spec: v5.ModelSpec,
    config: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    settings = config["frozen_v5_selection"]
    challengers = {str(item["candidate_id"]): item for item in config["challengers"]}
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for fold_index, (fit_ids, validation_ids) in enumerate(folds):
        fit_cells = {cell_id: cells[cell_id] for cell_id in fit_ids}
        base_x, base_y, base_audit = v5.build_pairwise_training_matrix(
            fit_cells,
            v11.PREFIX_CYCLE,
            v11.SCORE_END_CYCLE,
            core,
            anchor_stride=int(settings["anchor_stride"]),
        )
        v5.assert_pair_fold_firewall(fit_ids, validation_ids, base_audit)
        baseline = v5.make_estimator(
            spec,
            pairwise=True,
            random_state=int(settings["random_state"]),
        ).fit(base_x, base_y)
        scaler = v11.fit_curve_scaler(curve_features, fit_ids)
        curve_x, curve_y, curve_audit = v11.build_curve_pairwise_training_matrix(
            fit_cells,
            curve_features,
            core,
            scaler=scaler,
            anchor_stride=int(settings["anchor_stride"]),
        )
        v11.assert_curve_pair_fold_firewall(
            fit_ids, validation_ids, curve_audit, scaler
        )
        curve_model = v5.make_estimator(
            spec,
            pairwise=True,
            random_state=int(settings["random_state"]),
        ).fit(curve_x, curve_y)
        resources = v5._cell_resources(fit_cells, v11.PREFIX_CYCLE, core)
        held_batches = sorted({_batch_id(cell_id) for cell_id in validation_ids})
        held_batch = held_batches[0] if len(held_batches) == 1 else "mixed"
        audits.append(
            {
                "fold_scheme": scheme,
                "fold_index": fold_index,
                "fit_cell_ids": fit_ids,
                "validation_cell_ids": validation_ids,
                "held_out_batch_id": held_batch,
                "baseline_pair_training_cell_ids": base_audit["target_cell_ids"],
                "curve_pair_training_cell_ids": curve_audit["target_cell_ids"],
                "curve_scaler_fit_cell_ids": list(scaler.cell_ids),
                "held_out_cell_in_pair_role": False,
                "held_out_cell_in_curve_scaler": False,
            }
        )
        for cell_id in validation_ids:
            prefix = (
                cells[cell_id]
                .loc[cells[cell_id]["cycle_index"] <= v11.PREFIX_CYCLE]
                .reset_index(drop=True)
            )
            truth = _truth(cells[cell_id])
            baseline_prediction, _ = v5.predict_pairwise_trajectory(
                baseline,
                prefix,
                fit_cells,
                v11.PREFIX_CYCLE,
                v11.SCORE_END_CYCLE,
                core,
                aggregation=str(settings["aggregation"]),
                neighbor_count=int(settings["reference_count"]),
                reference_resources=resources,
            )
            rows.append(
                {
                    "fold_scheme": scheme,
                    "fold_index": fold_index,
                    "held_out_batch_id": held_batch,
                    "cell_id": cell_id,
                    "candidate_id": BASELINE_ID,
                    "trajectory_mae_pp": v5.trajectory_mae(truth, baseline_prediction),
                }
            )
            for candidate_id in v11.CHALLENGER_IDS:
                challenger = challengers[candidate_id]
                prediction, prediction_audit = v11.predict_curve_pairwise_trajectory(
                    curve_model,
                    cell_id,
                    prefix,
                    fit_cells,
                    curve_features,
                    core,
                    scaler=scaler,
                    geometry_mode=str(challenger["geometry_mode"]),
                    geometry_curve_weight=float(challenger["geometry_curve_weight"]),
                    neighbor_count=int(settings["reference_count"]),
                    aggregation=str(settings["aggregation"]),
                    reference_resources=resources,
                )
                if prediction_audit["target_future_outcomes_used"] is not False:
                    raise v5.FastChargeV5PairwiseError(
                        "V11 prediction audit exposed future outcomes"
                    )
                rows.append(
                    {
                        "fold_scheme": scheme,
                        "fold_index": fold_index,
                        "held_out_batch_id": held_batch,
                        "cell_id": cell_id,
                        "candidate_id": candidate_id,
                        "trajectory_mae_pp": v5.trajectory_mae(truth, prediction),
                    }
                )
        print(f"completed V11 {scheme} fold {fold_index + 1}/{len(folds)}", flush=True)
    return rows, audits


def _candidate_summary(
    scores: pd.DataFrame,
    candidate_id: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    primary = scores.loc[
        scores["fold_scheme"] == "deterministic_batch_stratified_cell_5fold"
    ]
    candidate = primary.loc[primary["candidate_id"] == candidate_id]
    baseline = primary.loc[primary["candidate_id"] == BASELINE_ID]
    paired = candidate.merge(
        baseline,
        on=["fold_scheme", "fold_index", "held_out_batch_id", "cell_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    paired["delta_mae_pp"] = (
        paired["trajectory_mae_pp_candidate"] - paired["trajectory_mae_pp_baseline"]
    )
    deltas = paired.set_index("cell_id")["delta_mae_pp"].to_dict()
    baseline_mean = float(paired["trajectory_mae_pp_baseline"].mean())
    candidate_mean = float(paired["trajectory_mae_pp_candidate"].mean())
    bootstrap = v5.paired_cell_bootstrap(
        deltas,
        repetitions=int(config["selection_protocol"]["bootstrap_repetitions"]),
        confidence=float(config["selection_protocol"]["bootstrap_confidence"]),
    )
    delta_values = paired["delta_mae_pp"].to_numpy(dtype=float)
    batch = scores.loc[scores["fold_scheme"] == "leave_one_manufacturing_batch_out"]
    batch_candidate = batch.loc[batch["candidate_id"] == candidate_id]
    batch_baseline = batch.loc[batch["candidate_id"] == BASELINE_ID]
    batch_paired = batch_candidate.merge(
        batch_baseline,
        on=["fold_scheme", "fold_index", "held_out_batch_id", "cell_id"],
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    batch_paired["delta_mae_pp"] = (
        batch_paired["trajectory_mae_pp_candidate"]
        - batch_paired["trajectory_mae_pp_baseline"]
    )
    batch_deltas = {
        str(batch_id): float(group["delta_mae_pp"].mean())
        for batch_id, group in batch_paired.groupby("held_out_batch_id", sort=True)
    }
    relative = (baseline_mean - candidate_mean) / baseline_mean
    improved_fraction = float(np.mean(delta_values < 0.0))
    p90_delta = float(np.quantile(delta_values, 0.9, method="higher"))
    worst_delta = float(np.max(delta_values))
    gate = config["promotion_gate"]
    checks = {
        "relative_mean_improvement": relative
        >= float(gate["minimum_relative_mean_mae_improvement"]),
        "bootstrap_upper_delta": float(bootstrap["upper_delta_mae_pp"])
        < float(gate["bootstrap_upper_delta_must_be_below_pp"]),
        "improved_cell_fraction": improved_fraction
        >= float(gate["minimum_improved_cell_fraction"]),
        "p90_cell_delta": p90_delta <= float(gate["maximum_p90_cell_delta_mae_pp"]),
        "worst_cell_regression": worst_delta
        <= float(gate["maximum_worst_cell_regression_pp"]),
        "each_batch_holdout_mean_delta": all(
            value <= float(gate["maximum_each_batch_holdout_mean_delta_mae_pp"])
            for value in batch_deltas.values()
        ),
    }
    return {
        "candidate_id": candidate_id,
        "physical_cell_count": len(paired),
        "baseline_mean_trajectory_mae_pp": baseline_mean,
        "candidate_mean_trajectory_mae_pp": candidate_mean,
        "mean_delta_mae_pp": candidate_mean - baseline_mean,
        "relative_mean_mae_improvement": relative,
        "improved_cell_fraction": improved_fraction,
        "p90_cell_delta_mae_pp": p90_delta,
        "worst_cell_regression_pp": worst_delta,
        "paired_cell_bootstrap": bootstrap,
        "batch_holdout_mean_delta_mae_pp": batch_deltas,
        "gate_checks": checks,
        "promotion_gate_passed": bool(all(checks.values())),
    }


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    training_path = Path(args.training_cycles)
    feature_path = Path(args.delta_q_features)
    output = Path(args.output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty V11 output: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if _sha256(feature_path) != str(config["dataset"]["delta_q_feature_file_sha256"]):
        raise v5.FastChargeV5PairwiseError("V11 Delta-Q feature file hash changed")
    selection_path = _resolve(str(config["frozen_v5_selection"]["path"]))
    core_path = _resolve(str(config["frozen_v5_core_config"]["path"]))
    if _sha256(selection_path) != str(config["frozen_v5_selection"]["sha256"]):
        raise v5.FastChargeV5PairwiseError("V11 V5 selection hash changed")
    if _sha256(core_path) != str(config["frozen_v5_core_config"]["sha256"]):
        raise v5.FastChargeV5PairwiseError("V11 V5 core config hash changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    spec = _model_spec(selection)
    if spec.model_id != str(config["frozen_v5_selection"]["model_id"]):
        raise v5.FastChargeV5PairwiseError("V11 baseline model identity changed")
    core = _core_config(load_fastcharge_safe_prior_v2_config(core_path))
    training = pd.read_parquet(training_path)
    if canonical_frame_sha256(training, tuple(training.columns)) != str(
        config["dataset"]["training_cycle_canonical_sha256"]
    ):
        raise v5.FastChargeV5PairwiseError("V11 training-cycle hash changed")
    cells = v5._validated_cells(training, required_support=v11.SCORE_END_CYCLE)
    if len(cells) != int(config["dataset"]["training_cell_count"]):
        raise v5.FastChargeV5PairwiseError("V11 training-cell count changed")
    curve_raw = pd.read_csv(feature_path, usecols=list(v11.CURVE_INPUT_COLUMNS))
    curve_raw = curve_raw.loc[:, v11.CURVE_INPUT_COLUMNS]
    curve_features = v11.validate_curve_features(
        curve_raw, required_cell_ids=sorted(cells)
    )
    primary_folds = v5.deterministic_cell_folds(sorted(cells), fold_count=5)
    batch_folds = v5.batch_holdout_folds(sorted(cells))
    primary_rows, primary_audits = _evaluate_folds(
        scheme="deterministic_batch_stratified_cell_5fold",
        folds=primary_folds,
        cells=cells,
        curve_features=curve_features,
        core=core,
        spec=spec,
        config=config,
    )
    batch_rows, batch_audits = _evaluate_folds(
        scheme="leave_one_manufacturing_batch_out",
        folds=batch_folds,
        cells=cells,
        curve_features=curve_features,
        core=core,
        spec=spec,
        config=config,
    )
    scores = pd.DataFrame(primary_rows + batch_rows, columns=SCORE_COLUMNS).sort_values(
        ["fold_scheme", "fold_index", "cell_id", "candidate_id"],
        kind="stable",
        ignore_index=True,
    )
    summaries = [
        _candidate_summary(scores, candidate_id, config)
        for candidate_id in v11.CHALLENGER_IDS
    ]
    passing = [item for item in summaries if item["promotion_gate_passed"]]
    tie_order = list(config["selection_protocol"]["candidate_tie_break_order"])
    selected = (
        min(
            passing,
            key=lambda item: (
                float(item["candidate_mean_trajectory_mae_pp"]),
                tie_order.index(str(item["candidate_id"])),
            ),
        )["candidate_id"]
        if passing
        else None
    )
    output.mkdir(parents=True, exist_ok=True)
    score_path = output / "training_only_scores.csv"
    firewall_path = output / "fold_firewall_audit.json"
    decision_path = output / "decision.json"
    _write_csv(scores, score_path)
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v11.fold_firewall_audit.v1",
            "folds": primary_audits + batch_audits,
            "held_out_cell_in_pair_role": False,
            "held_out_cell_in_curve_scaler": False,
            "evaluation_cell_outcomes_used": False,
        },
        firewall_path,
    )
    decision = {
        "schema_version": "lifetwin.fastcharge_v11.delta_q_pairwise.result.v1",
        "experiment_id": config["experiment_id"],
        "status": config["status"],
        "config_semantic_sha256": canonical_json_sha256(config),
        "training_cycle_canonical_sha256": canonical_frame_sha256(
            training, tuple(training.columns)
        ),
        "curve_feature_input_sha256": _sha256(feature_path),
        "curve_feature_table_canonical_sha256": canonical_frame_sha256(
            curve_features, v11.CURVE_INPUT_COLUMNS
        ),
        "score_table_sha256": canonical_frame_sha256(scores, SCORE_COLUMNS),
        "physical_training_cell_count": len(cells),
        "primary_prefix_cycle": v11.PREFIX_CYCLE,
        "score_end_cycle": v11.SCORE_END_CYCLE,
        "baseline_id": BASELINE_ID,
        "candidate_summaries": summaries,
        "selected_challenger_for_future_blind_freeze": selected,
        "any_promotion_gate_passed": bool(passing),
        "v5_champion_changed": False,
        "next_action": (
            "freeze_selected_challenger_for_a_new_outcome_blind_queue"
            if selected is not None
            else "retain_v5_and_do_not_advance_delta_q_challengers"
        ),
        "evaluation_cell_outcomes_used": False,
        "future_target_capacity_or_cycle_life_columns_read": False,
        "independent_confirmation": False,
        "claim_boundaries": config["claim_boundaries"],
    }
    _write_json(decision, decision_path)
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v11.artifact_manifest.v1",
            "experiment_id": config["experiment_id"],
            "artifacts": {
                path.name: {
                    "sha256": _sha256(path),
                    "byte_count": path.stat().st_size,
                }
                for path in (score_path, firewall_path, decision_path)
            },
        },
        output / "manifest.json",
    )
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delta_q_features")
    parser.add_argument("--training-cycles", default=str(DEFAULT_TRAINING))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
