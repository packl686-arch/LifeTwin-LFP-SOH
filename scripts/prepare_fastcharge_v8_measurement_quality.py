"""Compile repeat-measurement CSV data into a V8 noise-quality ledger."""

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

from lifetwin.experiments import fastcharge_v8_measurement_stability as v8
from lifetwin.experiments.fastcharge_v5_pairwise import FastChargeV5PairwiseError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/fastcharge-v8-measurement-quality"
IMPLEMENTATION_PATH = (
    ROOT / "src/lifetwin/experiments/fastcharge_v8_measurement_stability.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FastChargeV5PairwiseError(
            "V8 measurement-quality output directory must be new or empty"
        )
    path.mkdir(parents=True, exist_ok=True)


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    measurements_path = Path(args.measurements)
    output = Path(args.output_directory)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    measurements = pd.read_csv(measurements_path)
    _prepare_output(output)
    scores, ledger, quality = v8.characterize_measurement_noise(measurements, config)
    _write_csv(scores, output / "noise_candidate_scores.csv")
    _write_csv(ledger, output / "noise_ledger.csv")
    ledger_hash = _sha256(output / "noise_ledger.csv")
    score_hash = _sha256(output / "noise_candidate_scores.csv")
    result = {
        "schema_version": ("lifetwin.fastcharge_v8.measurement_quality.result.v1"),
        "experiment_id": config["experiment_id"],
        "config_sha256": _sha256(config_path),
        "measurement_input_sha256": _sha256(measurements_path),
        "measurement_input_row_count": len(measurements),
        "future_outcome_columns_accepted": False,
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
        "noise_candidate_scores_sha256": score_hash,
        "noise_ledger_sha256": ledger_hash,
        "measurement_quality": quality,
        "decision": (
            "measurement_quality_passed_for_outcome_free_stage_b_issuance"
            if bool(quality["measurement_quality_passed"])
            else "measurement_quality_failed_retain_v5_and_stop_before_outcomes"
        ),
        "future_outcome_access_permitted_by_this_result": False,
        "model_accuracy_evidence_created": False,
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
                "lifetwin.fastcharge_v8.measurement_quality.manifest.v1"
            ),
            "experiment_id": config["experiment_id"],
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
