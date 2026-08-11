"""Run nested V7 reissue-aware innovation development and shift audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn

from lifetwin.experiments import fastcharge_v7_reissue_innovation as innovation
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/experiments/v7_reissue_innovation_development.json"
)
DEFAULT_INPUT = (
    ROOT / "artifacts/fastcharge-v5-support-uncertainty/crossfit_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v7-reissue-innovation-development"
PARENT_DECISION = ROOT / "showcase/evidence_v6/gated_state_decision.json"
IMPLEMENTATION_PATH = (
    ROOT
    / "src/lifetwin/experiments/fastcharge_v7_reissue_innovation.py"
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


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    input_path = Path(args.training_crossfit)
    output = Path(args.output_directory)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    expected_parent_hash = protocol["development_lineage"][
        "parent_local_decision_sha256"
    ]
    observed_parent_hash = _sha256(PARENT_DECISION)
    if observed_parent_hash != expected_parent_hash:
        raise FastChargeV5PairwiseError(
            "V7 parent decision hash changed: "
            f"expected {expected_parent_hash}, observed {observed_parent_hash}"
        )
    expected_input_hash = protocol["inputs"]["training_crossfit_predictions"][
        "sha256"
    ]
    observed_input_hash = _sha256(input_path)
    if observed_input_hash != expected_input_hash:
        raise FastChargeV5PairwiseError(
            "V7 training input hash changed: "
            f"expected {expected_input_hash}, observed {observed_input_hash}"
        )
    training = pd.read_csv(input_path)
    expected_cells = int(
        protocol["inputs"]["training_crossfit_predictions"][
            "physical_cell_count"
        ]
    )
    if int(training["cell_id"].nunique()) != expected_cells:
        raise FastChargeV5PairwiseError(
            "V7 training physical-cell count changed"
        )

    cell_scores = innovation.score_innovation_candidates(training, protocol)
    gate_summary = innovation.summarize_innovation_gates(
        cell_scores, protocol
    )
    full_selection = innovation.select_innovation_gates(gate_summary)
    nested = innovation.nested_innovation_gate_audit(cell_scores, protocol)
    batch_holdout = innovation.batch_holdout_innovation_gate_audit(
        cell_scores, protocol
    )
    nomination = innovation.innovation_nomination_summary(
        nested, batch_holdout, protocol
    )

    _write_csv(cell_scores, output / "innovation_cell_scores.csv")
    _write_csv(gate_summary, output / "gate_candidate_summary.csv")
    _write_csv(nested, output / "nested_leave_one_cell_out_gate_scores.csv")
    _write_csv(batch_holdout, output / "leave_one_batch_out_gate_scores.csv")

    nominated = nomination["nominated_current_prefix_cycles"]
    decision = {
        "schema_version": "lifetwin.fastcharge_v7_reissue_innovation.result.v1",
        "experiment_id": protocol["experiment_id"],
        "evidence_role": protocol["evidence_role"],
        "protocol_sha256": _sha256(protocol_path),
        "parent_decision_sha256": observed_parent_hash,
        "training_input_sha256": observed_input_hash,
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
        "physical_cell_count": expected_cells,
        "activation_gate_candidate_count_including_fallback": len(
            innovation.innovation_gates(protocol)
        ),
        "full_training_selected_gate_by_current_prefix": {
            str(key): value for key, value in sorted(full_selection.items())
        },
        "nested_and_batch_future_blind_nomination": nomination,
        "decision": (
            "retain_v5_and_freeze_nominated_v7_transitions_for_future_blind_test"
            if nominated
            else "retain_v5_without_a_v7_blind_candidate"
        ),
        "nominated_current_prefix_cycles": nominated,
        "v5_champion_remains_active": True,
        "v7_innovation_gate_activated": False,
        "exposed_81_cell_evaluation_used": False,
        "claim_boundary": [
            "Outcome-informed development on the same 41 training cells",
            "Nested cell and within-cohort batch audits are not independent confirmation",
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
                "lifetwin.fastcharge_v7_reissue_innovation.manifest.v1"
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
