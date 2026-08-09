from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import lifetwin.experiments.private_dual_clock_uncertainty_audit as audit_impl
from lifetwin.experiments.private_dual_clock_uncertainty_audit import (
    audit_private_dual_clock_uncertainty,
)
from lifetwin.private_artifacts import (
    atomic_write_csv,
    atomic_write_json,
    build_completion_manifest,
    exclusive_run_lock,
    file_sha256,
    verify_completion_manifest,
)


DEFAULT_V3_DIRECTORY = Path("artifacts/private-dual-clock-prior-v3")
DEFAULT_DATA_DIRECTORY = Path("artifacts/snl-lfp-rpt-loco-v1")
DEFAULT_OUTPUT_DIRECTORY = Path(
    "artifacts/private-dual-clock-prior-v3-uncertainty-audit"
)


def run(args: argparse.Namespace) -> int:
    v3_directory = Path(args.v3_directory)
    data_directory = Path(args.data_directory)
    output_directory = Path(args.output_directory)
    paths = {
        "cells": output_directory / "cell_uncertainty.csv",
        "conditions": output_directory / "condition_uncertainty.csv",
        "summary": output_directory / "uncertainty_summary.json",
        "complete": output_directory / "run_complete.json",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Refusing to overwrite private uncertainty artifacts: "
            + ", ".join(existing)
        )
    upstream = json.loads(
        (v3_directory / "run_complete.json").read_text(encoding="utf-8")
    )
    verified = verify_completion_manifest(
        v3_directory,
        upstream,
        required_names=[
            "config",
            "predictions",
            "decisions",
            "decisions_csv",
            "manifest",
            "scores",
            "summary",
            "capsule",
        ],
    )
    config = json.loads(
        (v3_directory / "private_config.json").read_text(encoding="utf-8")
    )
    references = pd.read_parquet(data_directory / "outer_fold_references.parquet")
    truth = pd.read_parquet(data_directory / "target_truth.parquet")
    predictions = pd.read_parquet(v3_directory / "predictions.parquet")
    decisions = pd.read_parquet(v3_directory / "model_decisions.parquet")
    prediction_manifest = json.loads(
        (v3_directory / "prediction_manifest.json").read_text(encoding="utf-8")
    )
    cells, conditions, summary = audit_private_dual_clock_uncertainty(
        references,
        truth,
        predictions,
        decisions,
        prediction_manifest,
        config,
    )
    atomic_write_csv(cells, paths["cells"])
    atomic_write_csv(conditions, paths["conditions"])
    atomic_write_json(summary, paths["summary"])
    completion = build_completion_manifest(
        output_directory,
        {name: path for name, path in paths.items() if name != "complete"},
        metadata={
            "experiment_id": "private_dual_clock_prior_v3_uncertainty_audit",
            "upstream_completion_manifest_sha256": verified[
                "manifest_content_sha256"
            ],
            "uncertainty_summary_content_sha256": summary[
                "summary_content_sha256"
            ],
            "implementation_file_sha256": file_sha256(Path(audit_impl.__file__)),
            "runner_file_sha256": file_sha256(Path(__file__)),
            "public_release_permitted": False,
        },
    )
    atomic_write_json(completion, paths["complete"])
    print(json.dumps(summary["summary_by_landmark"], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit private V3 diagnostic intervals and abstention."
    )
    parser.add_argument("--v3-directory", default=str(DEFAULT_V3_DIRECTORY))
    parser.add_argument("--data-directory", default=str(DEFAULT_DATA_DIRECTORY))
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with exclusive_run_lock(Path(args.output_directory)):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
