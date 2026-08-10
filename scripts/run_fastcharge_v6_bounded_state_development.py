"""Run the training-only FastCharge V6 bounded-state challenger audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from lifetwin.experiments import fastcharge_v6_bounded_state as bounded
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/experiments/v6_bounded_state_update_development.json"
)
DEFAULT_INPUT = (
    ROOT / "artifacts/fastcharge-v5-support-uncertainty/crossfit_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v6-bounded-state-development"
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v6_bounded_state.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def _nested_selection_summary(frame: pd.DataFrame) -> dict[str, object]:
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
                "p90_cell_delta_mae_pp": float(np.quantile(deltas, 0.9)),
                "selected_candidate_counts": dict(
                    sorted(Counter(group["selected_candidate_id"]).items())
                ),
            }
        )
    return {"transitions": transitions}


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    input_path = Path(args.training_crossfit)
    output = Path(args.output_directory)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    expected_hash = protocol["inputs"]["training_crossfit_predictions"]["sha256"]
    observed_hash = _sha256(input_path)
    if observed_hash != expected_hash:
        raise FastChargeV5PairwiseError(
            "Training-only bounded-state input hash changed: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    training = pd.read_csv(input_path)
    expected_cells = int(
        protocol["inputs"]["training_crossfit_predictions"][
            "physical_cell_count"
        ]
    )
    observed_cells = int(training["cell_id"].nunique())
    if observed_cells != expected_cells:
        raise FastChargeV5PairwiseError(
            "Training-only bounded-state cell count changed: "
            f"expected {expected_cells}, observed {observed_cells}"
        )

    scores = bounded.score_bounded_state_candidates(training, protocol)
    summary = bounded.summarize_candidate_scores(scores, protocol)
    selected = bounded.select_rules(summary)
    nested = bounded.nested_selector_audit(scores, protocol)
    promotion = bounded.promotion_summary(nested, protocol)

    _write_csv(scores, output / "candidate_cell_scores.csv")
    _write_csv(summary, output / "candidate_summary.csv")
    _write_csv(nested, output / "nested_leave_one_cell_out_scores.csv")

    decision = {
        "schema_version": "lifetwin.fastcharge_v6_bounded_state.result.v1",
        "experiment_id": protocol["experiment_id"],
        "evidence_role": protocol["evidence_role"],
        "protocol_sha256": _sha256(protocol_path),
        "training_input_sha256": observed_hash,
        "runtime_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "implementation": {
            "module_path": str(IMPLEMENTATION_PATH.relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "module_sha256": _sha256(IMPLEMENTATION_PATH),
            "runner_path": str(Path(__file__).resolve().relative_to(ROOT)).replace(
                "\\", "/"
            ),
            "runner_sha256": _sha256(Path(__file__).resolve()),
        },
        "physical_cell_count": observed_cells,
        "candidate_count_including_fallback": len(
            bounded.candidate_rules(protocol)
        ),
        "full_training_nomination_by_current_prefix": {
            str(key): value for key, value in sorted(selected.items())
        },
        "nested_leave_one_cell_out": _nested_selection_summary(nested),
        "promotion_gate": promotion,
        "decision": (
            "nominate_bounded_state_challenger_for_new_outcome_blind_test"
            if promotion["passed"]
            else "retain_frozen_v5_champion"
        ),
        "exposed_81_cell_evaluation_used": False,
        "claim_boundary": [
            "Training-only challenger development",
            "Not an independent confirmation result",
            "No Hithium, calendar-aging, 15-25 year, or production claim",
        ],
    }
    _write_json(decision, output / "decision.json")

    artifacts: dict[str, dict[str, object]] = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            artifacts[path.name] = {
                "sha256": _sha256(path),
                "byte_count": path.stat().st_size,
            }
    _write_json(
        {
            "schema_version": (
                "lifetwin.fastcharge_v6_bounded_state.manifest.v1"
            ),
            "experiment_id": protocol["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(decision, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--training-crossfit", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
