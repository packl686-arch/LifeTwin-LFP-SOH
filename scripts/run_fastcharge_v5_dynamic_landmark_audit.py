"""Run the frozen FastCharge V5 dynamic-landmark development audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from lifetwin.experiments import fastcharge_v5_landmark as landmark
from lifetwin.experiments import fastcharge_v5_pairwise as v5


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/experiments/v5_dynamic_landmark_online_update_v1.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v5-dynamic-landmark-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _verify_input(path: Path, expected: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise v5.FastChargeV5PairwiseError(
            f"Frozen dynamic-landmark input hash changed for {path}: {observed}"
        )


def _attach_evaluation_truth(
    predictions: pd.DataFrame, canonical: pd.DataFrame
) -> pd.DataFrame:
    cells = v5._validated_cells(canonical, required_support=300)
    rows: list[pd.DataFrame] = []
    for (cell_id, prefix), group in predictions.groupby(
        ["cell_id", "prefix_cycle"], sort=True
    ):
        ordered = group.sort_values("forecast_cycle", kind="stable").copy()
        expected = np.arange(int(prefix) + 1, 301, dtype=int)
        observed_cycles = ordered["forecast_cycle"].to_numpy(dtype=int)
        if not np.array_equal(observed_cycles, expected):
            raise v5.FastChargeV5PairwiseError(
                f"Evaluation forecast support changed for {cell_id} at P{prefix}"
            )
        cell = cells[str(cell_id)]
        normalization = v5._normalization_capacity(cell)
        truth = v5._retention(cell, normalization)[int(prefix) : 300]
        ordered["observed_retention_pct"] = truth
        rows.append(ordered)
    result = pd.concat(rows, ignore_index=True)
    return result.rename(
        columns={"predicted_capacity_retention_pct": "candidate_prediction_pct"}
    )


def _bootstrap(
    frame: pd.DataFrame,
    *,
    value_column: str,
    seed: int,
    repetitions: int = 10000,
) -> dict[str, float | int]:
    per_cell = frame.groupby("cell_id", sort=True)[value_column].mean()
    return v5.paired_cell_bootstrap(
        {str(cell_id): float(value) for cell_id, value in per_cell.items()},
        repetitions=repetitions,
        random_state=seed,
    )


def _reissue_summary(frame: pd.DataFrame, *, seed: int) -> dict[str, object]:
    transitions: list[dict[str, object]] = []
    for (previous, current), group in frame.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        deltas = group["delta_mae_pp"].to_numpy(dtype=float)
        transitions.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "mean_previous_trajectory_mae_pp": float(
                    group["previous_trajectory_mae_pp"].mean()
                ),
                "mean_current_trajectory_mae_pp": float(
                    group["current_trajectory_mae_pp"].mean()
                ),
                "mean_delta_mae_pp": float(np.mean(deltas)),
                "relative_mae_improvement": float(
                    -np.mean(deltas) / group["previous_trajectory_mae_pp"].mean()
                ),
                "fraction_cells_improved": float(np.mean(deltas < 0.0)),
                "p90_cell_delta_mae_pp": float(np.quantile(deltas, 0.9)),
                "paired_cell_bootstrap": _bootstrap(
                    group, value_column="delta_mae_pp", seed=seed + int(current)
                ),
            }
        )
    return {
        "transitions": transitions,
        "overall_transition_equal": {
            "mean_previous_trajectory_mae_pp": float(
                frame["previous_trajectory_mae_pp"].mean()
            ),
            "mean_current_trajectory_mae_pp": float(
                frame["current_trajectory_mae_pp"].mean()
            ),
            "mean_delta_mae_pp": float(frame["delta_mae_pp"].mean()),
            "fraction_cell_transitions_improved": float(
                np.mean(frame["delta_mae_pp"].to_numpy(dtype=float) < 0.0)
            ),
            "physical_cell_clustered_bootstrap": _bootstrap(
                frame, value_column="delta_mae_pp", seed=seed
            ),
        },
    }


def _nested_summary(frame: pd.DataFrame) -> dict[str, object]:
    transitions: list[dict[str, object]] = []
    for (previous, current), group in frame.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle"], sort=True
    ):
        deltas = group["delta_mae_pp"].to_numpy(dtype=float)
        transitions.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "mean_base_trajectory_mae_pp": float(
                    group["base_trajectory_mae_pp"].mean()
                ),
                "mean_updated_trajectory_mae_pp": float(
                    group["updated_trajectory_mae_pp"].mean()
                ),
                "mean_delta_mae_pp": float(np.mean(deltas)),
                "fraction_cells_improved": float(np.mean(deltas < 0.0)),
                "selected_candidate_counts": dict(
                    sorted(Counter(group["selected_candidate_id"]).items())
                ),
            }
        )
    return {"transitions": transitions}


def _selected_evaluation_scores(
    all_scores: pd.DataFrame, selected: Mapping[int, str]
) -> pd.DataFrame:
    parts = [
        all_scores.loc[
            (all_scores["current_prefix_cycle"] == current)
            & (all_scores["candidate_id"] == candidate_id)
        ]
        for current, candidate_id in sorted(selected.items())
    ]
    return pd.concat(parts, ignore_index=True).sort_values(
        ["current_prefix_cycle", "cell_id"], kind="stable", ignore_index=True
    )


def _selected_summary(
    frame: pd.DataFrame, config: Mapping[str, object]
) -> dict[str, object]:
    minimum_fraction = float(
        config["decision_gates"]["H2_online_update_minimum_fraction_cells_improved"]
    )
    transitions: list[dict[str, object]] = []
    for (previous, current, candidate_id), group in frame.groupby(
        ["previous_prefix_cycle", "current_prefix_cycle", "candidate_id"],
        sort=True,
    ):
        deltas = group["delta_mae_pp"].to_numpy(dtype=float)
        fraction = float(np.mean(deltas < 0.0))
        mean_delta = float(np.mean(deltas))
        transitions.append(
            {
                "previous_prefix_cycle": int(previous),
                "current_prefix_cycle": int(current),
                "candidate_id": str(candidate_id),
                "candidate_family": str(group.iloc[0]["candidate_family"]),
                "physical_cell_count": int(group["cell_id"].nunique()),
                "mean_base_trajectory_mae_pp": float(
                    group["base_trajectory_mae_pp"].mean()
                ),
                "mean_updated_trajectory_mae_pp": float(
                    group["updated_trajectory_mae_pp"].mean()
                ),
                "mean_delta_mae_pp": mean_delta,
                "fraction_cells_improved": fraction,
                "p90_cell_delta_mae_pp": float(np.quantile(deltas, 0.9)),
                "online_update_gate_passed": bool(
                    candidate_id != "no_update"
                    and mean_delta < 0.0
                    and fraction >= minimum_fraction
                ),
            }
        )
    return {
        "transitions": transitions,
        "all_transitions_passed": bool(
            transitions and all(row["online_update_gate_passed"] for row in transitions)
        ),
    }


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    training_path = Path(args.training_crossfit)
    evaluation_path = Path(args.evaluation_predictions)
    canonical_path = Path(args.canonical_cycles)
    output = Path(args.output_directory)

    _verify_input(
        training_path, protocol["inputs"]["training_crossfit_predictions"]["sha256"]
    )
    _verify_input(
        evaluation_path, protocol["inputs"]["public_evaluation_predictions"]["sha256"]
    )
    _verify_input(canonical_path, protocol["inputs"]["canonical_cycles"]["sha256"])

    training = pd.read_csv(training_path)
    evaluation = _attach_evaluation_truth(
        pd.read_parquet(evaluation_path), pd.read_parquet(canonical_path)
    )

    training_scores = landmark.score_residual_candidates(training, protocol)
    training_summary = landmark.summarize_candidate_scores(training_scores, protocol)
    selected = landmark.select_rules(training_summary)
    nested = landmark.nested_selector_audit(training_scores, protocol)

    rule_lookup = {
        rule.candidate_id: rule for rule in landmark.candidate_rules(protocol)
    }
    evaluation_rules = [rule_lookup[value] for value in sorted(set(selected.values()))]
    evaluation_all_selected = landmark.score_residual_candidates(
        evaluation, protocol, rules=evaluation_rules
    )
    evaluation_selected = _selected_evaluation_scores(evaluation_all_selected, selected)

    training_reissue = landmark.score_base_reissues(training, protocol)
    evaluation_reissue = landmark.score_base_reissues(evaluation, protocol)
    interval = json.loads(Path(args.interval_summary).read_text(encoding="utf-8"))
    interval_passed = bool(interval["h2_development_gate"]["interval_subgate_passed"])
    evaluation_update = _selected_summary(evaluation_selected, protocol)
    full_h2_passed = bool(
        interval_passed and evaluation_update["all_transitions_passed"]
    )

    _write_csv(training_scores, output / "training_candidate_cell_scores.csv")
    _write_csv(training_summary, output / "training_candidate_summary.csv")
    _write_csv(nested, output / "training_nested_selector_audit.csv")
    _write_csv(training_reissue, output / "training_base_reissue_scores.csv")
    _write_csv(evaluation_reissue, output / "evaluation_base_reissue_scores.csv")
    _write_csv(evaluation_selected, output / "evaluation_selected_update_scores.csv")

    decision = {
        "schema_version": "lifetwin.fastcharge_v5_dynamic_landmark.result.v1",
        "experiment_id": protocol["experiment_id"],
        "evidence_role": protocol["evidence_role"],
        "protocol_sha256": _sha256(protocol_path),
        "selected_residual_rule_by_current_prefix": {
            str(key): value for key, value in sorted(selected.items())
        },
        "training_nested_selector": _nested_summary(nested),
        "training_base_reissue": _reissue_summary(training_reissue, seed=202608091),
        "public_evaluation_base_reissue": _reissue_summary(
            evaluation_reissue, seed=202608092
        ),
        "public_evaluation_selected_residual_update": evaluation_update,
        "interval_subgate_passed": interval_passed,
        "full_H2_gp_online_landmark_gate_passed": full_h2_passed,
        "decision": (
            "activate_selected_residual_updates"
            if evaluation_update["all_transitions_passed"]
            else "retain_current_prefix_v5_center_without_online_residual_branch"
        ),
        "gp_branch_activated": any(
            value.startswith("gp_") for value in selected.values()
        ),
        "claim_boundary": [
            "Outcome-exposed public development only",
            "Evaluation outcomes never selected the residual rule",
            "No Hithium, calendar-aging, 15-25 year, or production claim",
        ],
    }
    _write_json(decision, output / "decision.json")

    artifacts = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
    _write_json(
        {
            "schema_version": "lifetwin.fastcharge_v5_dynamic_landmark.manifest.v1",
            "experiment_id": protocol["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument(
        "--training-crossfit",
        default=str(
            ROOT
            / "artifacts/fastcharge-v5-support-uncertainty/crossfit_predictions.csv"
        ),
    )
    parser.add_argument(
        "--evaluation-predictions",
        default=str(
            ROOT
            / "artifacts/fastcharge-v5-pairwise-development/selected_predictions.parquet"
        ),
    )
    parser.add_argument(
        "--canonical-cycles",
        default=str(
            ROOT / "artifacts/fastcharge-safe-prior-v2/canonical_cycles.parquet"
        ),
    )
    parser.add_argument(
        "--interval-summary",
        default=str(
            ROOT / "artifacts/fastcharge-v5-support-uncertainty/score_summary.json"
        ),
    )
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
