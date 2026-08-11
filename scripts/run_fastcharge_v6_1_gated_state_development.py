"""Run nested training-only development of the V6.1 abstaining state gate."""

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

from lifetwin.experiments import fastcharge_v6_bounded_state as bounded
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/experiments/v6_1_gated_state_update_development.json"
)
DEFAULT_INPUT = (
    ROOT / "artifacts/fastcharge-v5-support-uncertainty/crossfit_predictions.csv"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v6-1-gated-state-development"
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


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    input_path = Path(args.training_crossfit)
    output = Path(args.output_directory)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    expected_hash = protocol["inputs"]["training_crossfit_predictions"]["sha256"]
    observed_hash = _sha256(input_path)
    if observed_hash != expected_hash:
        raise FastChargeV5PairwiseError(
            "V6.1 gated-state input hash changed: "
            f"expected {expected_hash}, observed {observed_hash}"
        )
    training = pd.read_csv(input_path)
    expected_cells = int(
        protocol["inputs"]["training_crossfit_predictions"][
            "physical_cell_count"
        ]
    )
    if int(training["cell_id"].nunique()) != expected_cells:
        raise FastChargeV5PairwiseError(
            "V6.1 gated-state physical-cell count changed"
        )

    correction_ids = {
        gate.correction_candidate_id
        for gate in bounded.activation_gates(protocol)
        if gate.correction_candidate_id is not None
    }
    rule_lookup = {
        rule.candidate_id: rule for rule in bounded.candidate_rules(protocol)
    }
    correction_rules = [rule_lookup[value] for value in sorted(correction_ids)]
    correction_scores = bounded.score_bounded_state_candidates(
        training, protocol, rules=correction_rules
    )
    activation_table = bounded.build_activation_table(
        training, correction_scores, protocol
    )
    gate_summary = bounded.summarize_activation_gates(
        activation_table, protocol
    )
    full_selection = bounded.select_activation_gates(gate_summary)
    nested = bounded.nested_activation_gate_audit(activation_table, protocol)
    nomination = bounded.activation_nomination_summary(nested, protocol)

    _write_csv(correction_scores, output / "correction_cell_scores.csv")
    _write_csv(activation_table, output / "activation_diagnostics.csv")
    _write_csv(gate_summary, output / "gate_candidate_summary.csv")
    _write_csv(nested, output / "nested_leave_one_cell_out_gate_scores.csv")

    nominated = nomination["nominated_current_prefix_cycles"]
    decision = {
        "schema_version": "lifetwin.fastcharge_v6_1_gated_state.result.v1",
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
        "physical_cell_count": expected_cells,
        "gate_candidate_count_including_fallback": len(
            bounded.activation_gates(protocol)
        ),
        "full_training_selected_gate_by_current_prefix": {
            str(key): value for key, value in sorted(full_selection.items())
        },
        "nested_future_blind_nomination": nomination,
        "decision": (
            "retain_v5_and_freeze_nominated_transitions_for_future_blind_test"
            if nominated
            else "retain_v5_without_a_v6_1_blind_candidate"
        ),
        "nominated_current_prefix_cycles": nominated,
        "v5_champion_remains_active": True,
        "v6_1_gate_activated": False,
        "exposed_81_cell_evaluation_used": False,
        "claim_boundary": [
            "Outcome-informed training-only hypothesis generation",
            "Nested held-out-cell gate audit, not independent confirmation",
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
                "lifetwin.fastcharge_v6_1_gated_state.manifest.v1"
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
