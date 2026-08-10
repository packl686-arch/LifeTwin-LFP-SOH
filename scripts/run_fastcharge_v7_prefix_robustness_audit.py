"""Run the preregistered frozen V7 P100 prefix-robustness audit."""

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

from lifetwin.experiments import fastcharge_v7_prefix_robustness as robustness
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs/experiments/v7_frozen_gate_prefix_robustness_audit.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v7-prefix-robustness-audit"
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v7_prefix_robustness.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_registered_path(path: str) -> Path:
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


def _records_without_nan(frame: pd.DataFrame) -> list[dict[str, object]]:
    normalized = frame.astype(object).where(pd.notna(frame), None)
    return normalized.to_dict(orient="records")


def run(args: argparse.Namespace) -> int:
    protocol_path = Path(args.protocol)
    output = Path(args.output_directory)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    candidate_path = _resolve_registered_path(str(protocol["source_candidate"]["path"]))
    input_path = _resolve_registered_path(str(protocol["input"]["path"]))

    expected_candidate_hash = str(protocol["source_candidate"]["sha256"])
    observed_candidate_hash = _sha256(candidate_path)
    if observed_candidate_hash != expected_candidate_hash:
        raise FastChargeV5PairwiseError(
            "Frozen V7 candidate hash changed: "
            f"expected {expected_candidate_hash}, observed {observed_candidate_hash}"
        )
    expected_input_hash = str(protocol["input"]["sha256"])
    observed_input_hash = _sha256(input_path)
    if observed_input_hash != expected_input_hash:
        raise FastChargeV5PairwiseError(
            "V7 robustness input hash changed: "
            f"expected {expected_input_hash}, observed {observed_input_hash}"
        )

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path)
    baseline, decisions, summaries, audit_decision = (
        robustness.run_frozen_prefix_robustness(frame, protocol, candidate)
    )
    _write_csv(baseline, output / "baseline_cell_scores.csv")
    _write_csv(decisions, output / "perturbation_decisions.csv")
    _write_csv(summaries, output / "scenario_summary.csv")

    result = {
        "schema_version": (
            "lifetwin.fastcharge_v7_frozen_gate_prefix_robustness.result.v1"
        ),
        "experiment_id": protocol["experiment_id"],
        "evidence_role": protocol["evidence_role"],
        "protocol_sha256": _sha256(protocol_path),
        "frozen_candidate_sha256": observed_candidate_hash,
        "input_sha256": observed_input_hash,
        "runtime_versions": {
            "python": (
                f"{sys.version_info.major}.{sys.version_info.minor}."
                f"{sys.version_info.micro}"
            ),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
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
        "baseline": {
            "physical_cell_count": int(baseline["cell_id"].nunique()),
            "activation_count": int(baseline["activated"].sum()),
            "activation_precision": float(
                baseline.loc[baseline["activated"], "gated_delta_mae_pp"].lt(0.0).mean()
            ),
            "mean_all_cell_delta_mae_pp": float(baseline["gated_delta_mae_pp"].mean()),
            "active_max_delta_mae_pp": float(
                baseline.loc[baseline["activated"], "gated_delta_mae_pp"].max()
            ),
        },
        "scenario_summaries": _records_without_nan(summaries),
        **audit_decision,
        "scope": {
            "v7_innovation_layer_only": True,
            "v5_centers_regenerated_under_perturbation": False,
            "independent_confirmation": False,
            "hithium_data_used": False,
            "calendar_or_15_to_25_year_claim": False,
        },
    }
    _write_json(result, output / "decision.json")

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
                "lifetwin.fastcharge_v7_frozen_gate_prefix_robustness.manifest.v1"
            ),
            "experiment_id": protocol["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
